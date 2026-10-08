# Olympiad post-agreement motivation (收口)

R1-7B seed 42, OlympiadBench k=4 windowed questions (n=633).

## What this is

After the thinking-side step probes, append trial answers on the
`</think>` suffix of saved Full-CoT / count-bias generations, rebuild the
token–accuracy series with think-then-post alignment, and draw the wide
motivation figure. Final Acc and truncation ΔAcc use the same 633-question
denominator as the curves.

## Entrypoints

| Step | Command |
|---|---|
| Probe post-`</think>` | `bash scripts/olympiad_post_agreement/launch_post_think.sh` (tmux; GPUs 0–3) |
| Or one shard | `python scripts/olympiad_post_agreement/run_post_think_probes.py --shard-id 0 --num-shards 4 --protocol both` |
| Rebuild series | `python scripts/olympiad_post_agreement/rebuild_olympiad_with_post_think.py` |
| Plot | `python scripts/olympiad_post_agreement/plot_olympiad_motivation.py` |

Scratch outputs stay under `tmp/plws_step_probe/` (gitignored): probe jsonl,
`r1_7b_count_bias_s42_token_series_post.json`, and the wide PDF/PNG.

Needs a local PUMA tree at `$PLWS_ROOT/tmp/PUMA/puma` for `gen_trial_answers`,
`separate_steps`, and related helpers.
