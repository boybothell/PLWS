# 评测数据（唯一题序源）

这九份受 Git 版本控制的 jsonl 是本机和租卡机的唯一内容、格式与题序源。
clone/pull 本仓库后数据已经齐全；云端不需要再从 Hugging Face、PUMA 上游
或其它镜像下载同名数据。

现行新跑只用 MATH-500、OlympiadBench、GPQA-Diamond、AIME25、HMMT25。
AIME24、AIME26、BRUMO25、AMC23 仍钉在仓库里，只服务已有产物续跑和哈希核验，不再入新队列。

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

`scripts/bootstrap_puma.sh` 会把仓库 `data/` 原样安装到
`$PUMA_ROOT/data/`，随后使用 `SHA256SUMS` 逐文件校验；重复执行会恢复被改动
或覆盖的数据。`$PUMA_ROOT/data/` 是运行副本，不是另一个数据来源。

正式续跑不要设置 `PLWS_DATA_ROOT`，让运行时默认读取 `$PUMA_ROOT/data/`。
如确需覆盖该变量，目录内九份文件必须与本目录逐字节一致，预检会按 SHA256
拒绝不一致的内容。

续跑依赖 `question_idx`。换一份同条数但内容、LaTeX 或顺序不同的数据，会让
已有 Full-CoT、jobs 和 shard 接到错题上；只检查题数不能证明可以复用。
