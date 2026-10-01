"""Which proxy labeler should be believed?

The token-F1 and embedding labelers disagree on 22-35% of answers, and several
headline results flip depending on which is used. On SciQ the question is
settleable without human annotation: the reference answer is a short noun
phrase, so "does the generated answer actually contain the correct answer (and
not a distractor)?" is a near-exact criterion. This script builds that
containment label and scores both proxies against it, then re-runs the
detector under the containment label.

Containment rule (normalised: lowercase, punctuation stripped, articles removed):
    grounded   if correct answer appears as a token subsequence of the answer
               and no distractor does
    hallucinated otherwise
Answers containing both a correct answer and a distractor are counted
separately and excluded from the agreement statistics, since the rule is
ambiguous for them.

TruthfulQA is deliberately excluded: its references are full sentences with
many valid paraphrases, so containment is not a valid criterion there.

Usage:
    python scripts/experiment_labeler_adjudication.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.labeling import _normalize
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import bootstrap_auroc_ci
from scripts.experiment_clean_validation import oof_flat

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B", "clean"),
          ("llama-3.2-1b-instruct", "Llama-3.2-1B", "rich"),
          ("smollm2-1.7b-instruct", "SmolLM2-1.7B", "rich")]
NO_LENGTH = [f for f in ALL_RICH_FEATURE_NAMES if f != "answer_length"]


def contains(answer_tokens, ref):
    ref_toks = _normalize(ref)
    if not ref_toks:
        return False
    n = len(ref_toks)
    return any(answer_tokens[i:i + n] == ref_toks for i in range(len(answer_tokens) - n + 1))


def main():
    set_all_seeds()
    df_q = load_subset("sciq", 200, CFG.seed).set_index("qid")
    out = {}
    for key, disp, src in MODELS:
        paths = get_paths(key, "sciq")
        feats = pd.read_csv(paths.features / f"single_pass_features_{src}.csv")
        sem = pd.read_csv(paths.features / "single_pass_features_semantic_labels.csv")[["qid", "semantic_label"]]
        d = feats.merge(sem, on="qid")

        containment, ambiguous = [], []
        for _, row in d.iterrows():
            refs = df_q.loc[row["qid"]]
            toks = _normalize(str(row["generated_answer"]))
            has_correct = contains(toks, refs["best_answer"])
            has_distractor = any(contains(toks, r) for r in refs["incorrect_answers"])
            ambiguous.append(bool(has_correct and has_distractor))
            containment.append(0 if (has_correct and not has_distractor) else 1)
        d["containment_label"] = containment
        d["ambiguous"] = ambiguous

        clean = d[~d["ambiguous"]]
        stats = {
            "n": len(d), "n_ambiguous": int(d["ambiguous"].sum()),
            "containment_halluc_rate": float(d["containment_label"].mean()),
            "tokenf1_halluc_rate": float(d["label"].mean()),
            "embedding_halluc_rate": float(d["semantic_label"].mean()),
            "tokenf1_vs_containment_agreement": float((clean["label"] == clean["containment_label"]).mean()),
            "tokenf1_vs_containment_kappa": float(cohen_kappa_score(clean["label"], clean["containment_label"])),
            "embedding_vs_containment_agreement": float((clean["semantic_label"] == clean["containment_label"]).mean()),
            "embedding_vs_containment_kappa": float(cohen_kappa_score(clean["semantic_label"], clean["containment_label"])),
        }

        # Detector performance under the containment label.
        d_eval = d.rename(columns={"label": "tokenf1_label"}).rename(columns={"containment_label": "label"})
        for fs_name, fs in [("original_3feat", ORIGINAL_THREE_FEATURES),
                            ("all_17feat", ALL_RICH_FEATURE_NAMES),
                            ("all_16feat_no_length", NO_LENGTH)]:
            y, p = oof_flat(d_eval, fs, seed=CFG.seed)
            stats[f"auroc__{fs_name}"] = bootstrap_auroc_ci(y, p)

        pd.DataFrame({"qid": d["qid"], "containment_label": d["label"] if "label" in d and False else containment,
                      "ambiguous": ambiguous}).to_csv(
            paths.features / "single_pass_features_containment_labels.csv", index=False)
        out[key] = stats
        print(f"\n[{disp} / SciQ]  n={stats['n']}  ambiguous={stats['n_ambiguous']}")
        print(f"  hallucination rate: containment={stats['containment_halluc_rate']:.2f}  "
              f"token-F1={stats['tokenf1_halluc_rate']:.2f}  embedding={stats['embedding_halluc_rate']:.2f}")
        print(f"  agreement with containment: token-F1={stats['tokenf1_vs_containment_agreement']:.3f} "
              f"(k={stats['tokenf1_vs_containment_kappa']:.3f})  "
              f"embedding={stats['embedding_vs_containment_agreement']:.3f} "
              f"(k={stats['embedding_vs_containment_kappa']:.3f})")
        for fs_name in ["original_3feat", "all_17feat", "all_16feat_no_length"]:
            c = stats[f"auroc__{fs_name}"]
            print(f"  AUROC under containment label, {fs_name:22s} {c['point_estimate']:.3f} "
                  f"[{c['ci_lower_2.5pct']:.3f}, {c['ci_upper_97.5pct']:.3f}]")

    save_json(out, ROOT / "comparison" / "sciq_labeler_adjudication.json")
    print("\nSaved -> comparison/sciq_labeler_adjudication.json")


if __name__ == "__main__":
    main()
