"""Token-level uncertainty visualization: color each generated token by how
confident the model was in it, so a reader can see *which* tokens drove a
low mean/min logprob rather than just the aggregate number.

Used by both the static per-model analysis reports (scripts/experiment_analysis.py)
and the interactive demo (scripts/demo_app.py) -- one rendering function, two
call sites, so the demo shows exactly the same visualization logic that was
used to build the report figures.
"""
import html
import math


def _prob_to_color(prob: float) -> str:
    """Map a token probability in [0,1] to a red(low)->yellow->green(high)
    background color via HSL hue interpolation."""
    prob = max(0.0, min(1.0, prob))
    hue = prob * 120  # 0=red, 60=yellow, 120=green
    return f"hsl({hue:.0f}, 75%, 80%)"


def render_token_html(
    tokens: list[str],
    token_logprobs: list[float],
    token_entropies: list[float] | None = None,
    suspicious_logprob_threshold: float = -3.0,
) -> str:
    """Return a self-contained HTML snippet: the answer text with each token
    shaded by exp(logprob) (its probability under the model), and tokens
    below `suspicious_logprob_threshold` outlined in red as
    'suspicious/low-confidence'. Hovering a token shows its exact
    logprob/probability/entropy."""
    if not tokens:
        return "<div><em>(empty generation)</em></div>"

    spans = []
    for i, tok in enumerate(tokens):
        lp = token_logprobs[i] if i < len(token_logprobs) else None
        ent = token_entropies[i] if (token_entropies and i < len(token_entropies)) else None
        prob = math.exp(lp) if lp is not None else None

        color = _prob_to_color(prob) if prob is not None else "#eee"
        suspicious = lp is not None and lp <= suspicious_logprob_threshold
        border = "2px solid #d62728" if suspicious else "1px solid transparent"

        title_parts = []
        if lp is not None:
            title_parts.append(f"logprob={lp:.3f}  P={prob:.3f}")
        if ent is not None:
            title_parts.append(f"entropy={ent:.3f} nats")
        if suspicious:
            title_parts.append("SUSPICIOUS: low-confidence token")
        title = html.escape(" | ".join(title_parts))

        safe_tok = html.escape(tok).replace("\n", "<br/>")
        spans.append(
            f'<span title="{title}" style="background:{color}; border:{border}; '
            f'border-radius:3px; padding:1px 2px; margin:0 1px; white-space:pre-wrap; '
            f'font-family:ui-monospace,Consolas,monospace;">{safe_tok}</span>'
        )

    legend = (
        '<div style="margin-top:8px; font-size:12px; color:#666;">'
        '<span style="background:hsl(0,75%,80%); padding:1px 6px; border-radius:3px;">low confidence</span> '
        '<span style="background:hsl(60,75%,80%); padding:1px 6px; border-radius:3px;">medium</span> '
        '<span style="background:hsl(120,75%,80%); padding:1px 6px; border-radius:3px;">high confidence</span> '
        '&nbsp;|&nbsp; red outline = suspicious token (logprob &le; '
        f'{suspicious_logprob_threshold:.1f})</div>'
    )
    return f'<div style="line-height:2.2;">{"".join(spans)}</div>{legend}'


def suspicious_tokens(tokens: list[str], token_logprobs: list[float], threshold: float = -3.0) -> list[dict]:
    """Return the tokens whose logprob is at/below `threshold`, for use in
    plain-text explanations (e.g. the demo's 'why flagged' text)."""
    out = []
    for tok, lp in zip(tokens, token_logprobs):
        if lp <= threshold:
            out.append({"token": tok, "logprob": lp, "probability": math.exp(lp)})
    return out
