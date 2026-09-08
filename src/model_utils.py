"""Model loading, prompt construction, and generation with per-token
log-probability / entropy extraction.

This is the only module that talks to an LLM. The proposed single-pass
detector, the 5-call self-consistency baseline, and the interactive demo
all call `generate_with_scores` (or its batched sibling
`generate_batch_with_scores`) -- the difference between them is entirely in
the decoding parameters (greedy vs. sampled, batch size), never in how
logprobs/entropy are computed. Works with any causal-LM checkpoint in
configs/models.yaml, not just Qwen.
"""
import os

# The Xet fast-download backend fails on some networks with a CAS Client
# I/O error; force the plain HTTP downloader before transformers/huggingface_hub
# are imported anywhere in the process.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CFG


@dataclass
class GenerationResult:
    answer_text: str
    token_logprobs: list[float]  # log P(chosen token | context), one per generated token
    token_entropies: list[float]  # full-vocab predictive entropy (nats), one per generated token
    num_new_tokens: int
    token_strs: list[str] = field(default_factory=list)  # decoded piece per kept token, for token-level viz


class LLMGenerator:
    """Generic causal-LM wrapper. Construct with a raw HF repo id, or via
    `LLMGenerator.from_registry("llama-3.2-1b-instruct")` to use a
    configs/models.yaml entry."""

    def __init__(self, model_name: str = CFG.model_name, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Many chat models declare multiple stop-token ids (e.g. Qwen's
        # <|im_end|> AND <|endoftext|>); collect all of them so we correctly
        # strip whichever one ends the turn from the scored tokens.
        stop_ids = set()
        cfg_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if cfg_eos is not None:
            stop_ids.update(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            stop_ids.add(self.tokenizer.pad_token_id)
        self.stop_ids = stop_ids

    @classmethod
    def from_registry(cls, model_key: str, device: str | None = None) -> "LLMGenerator":
        from .registry import resolve_model

        checkpoint = resolve_model(model_key)["checkpoint"]
        return cls(model_name=checkpoint, device=device)

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": CFG.system_prompt},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _token_pieces(self, kept_ids: list[int]) -> list[str]:
        if not kept_ids:
            return []
        tokens = self.tokenizer.convert_ids_to_tokens(kept_ids)
        return [self.tokenizer.convert_tokens_to_string([t]) for t in tokens]

    @torch.no_grad()
    def generate_with_scores(
        self,
        question: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)

        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        out = self.model.generate(**inputs, **gen_kwargs)

        new_token_ids = out.sequences[0][input_len:]
        scores = out.scores  # tuple of length num_new_tokens, each (1, vocab_size) logits

        token_logprobs = []
        token_entropies = []
        kept_ids = []
        for step, logits in enumerate(scores):
            token_id = new_token_ids[step].item()
            if token_id in self.stop_ids:
                break
            log_probs = F.log_softmax(logits[0].float(), dim=-1)
            token_logprobs.append(log_probs[token_id].item())

            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum().item()
            token_entropies.append(entropy)
            kept_ids.append(token_id)

        answer_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            answer_text=answer_text,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            num_new_tokens=len(kept_ids),
            token_strs=self._token_pieces(kept_ids),
        )

    @torch.no_grad()
    def generate_batch_with_scores(
        self,
        questions: list[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        """Batched variant of generate_with_scores for throughput on larger
        runs. Left-pads prompts so every row's generated tokens start at the
        same sequence index. Not used by the DA1/Llama-comparison scripts
        (which need per-example determinism matching the original,
        non-batched run) -- this is for the expanded multi-dataset
        experiments where raw throughput matters more than bit-identical
        reproduction of an earlier single-example run.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if not questions:
            return []

        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            prompts = [self._build_prompt(q) for q in questions]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            out = self.model.generate(**inputs, **gen_kwargs)
            new_token_ids_batch = out.sequences[:, input_len:]  # (batch, num_new_tokens)
            scores = out.scores  # tuple length num_new_tokens, each (batch, vocab)

            results = []
            batch_size = new_token_ids_batch.shape[0]
            for row in range(batch_size):
                token_logprobs, token_entropies, kept_ids = [], [], []
                for step, logits in enumerate(scores):
                    token_id = new_token_ids_batch[row, step].item()
                    if token_id in self.stop_ids:
                        break
                    log_probs = F.log_softmax(logits[row].float(), dim=-1)
                    token_logprobs.append(log_probs[token_id].item())
                    probs = log_probs.exp()
                    token_entropies.append(-(probs * log_probs).sum().item())
                    kept_ids.append(token_id)

                answer_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
                results.append(GenerationResult(
                    answer_text=answer_text,
                    token_logprobs=token_logprobs,
                    token_entropies=token_entropies,
                    num_new_tokens=len(kept_ids),
                    token_strs=self._token_pieces(kept_ids),
                ))
            return results
        finally:
            self.tokenizer.padding_side = prev_padding_side


# Backward-compat alias: the frozen DA1 scripts (scripts/01-03) import
# QwenGenerator by name. Behavior is unchanged -- LLMGenerator is a strict
# generalization (same defaults, same single-example code path).
QwenGenerator = LLMGenerator
