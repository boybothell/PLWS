# 评测数据（唯一题序源）

这九份 jsonl 是本机正在跑、也必须在租卡机上使用的评测集。
云端不要再从 Hugging Face 或其它镜像另下同名数据覆盖它们。

| 文件 | 题数 | 来源 |
|---|---:|---|
| `math-500_test.jsonl` | 500 | PUMA `ae8922c6` |
| `olympiadbench_test.jsonl` | 675 | PUMA `ae8922c6` |
| `gpqa-diamond_test.jsonl` | 198 | PUMA `ae8922c6` |
| `aime24_test.jsonl` | 30 | PUMA `ae8922c6` |
| `aime25_test.jsonl` | 30 | PUMA `ae8922c6` |
| `aime26_test.jsonl` | 30 | 本机 Qwen2.5-Math 转写；与 `math-ai/aime26` 题序/LaTeX 一致 |
| `hmmt25_test.jsonl` | 30 | 本机 Qwen2.5-Math 转写；与 `MathArena/hmmt_feb_2025` 题序/LaTeX 一致 |
| `brumo25_test.jsonl` | 30 | `MathArena/brumo_2025` |
| `amc23_test.jsonl` | 40 | 本机 DEER `amc/test.jsonl` 转写 |

`scripts/bootstrap_puma.sh` 会按 `SHA256SUMS` 把这些文件装进 `$PUMA_ROOT/data/`。
续跑依赖 `question_idx`，换一份同条数但不同顺序的数据会把已有 Full-CoT / jobs / shard 接到错题上。
