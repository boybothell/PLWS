# 脚本入口

不要按文件名猜主入口。`scripts/` 同时保留正式流水线、报告脚本和历史分析；
租卡只使用本页“正式入口”。

## 正式租卡入口

| 用途 | 入口 |
|---|---|
| 安装固定 PUMA 补丁和九份评测数据 | `bootstrap_puma.sh` |
| 环境与资产预检 | `preflight_rental_server.sh` |
| 单个大模型 cell | `run_large_model_cell.sh` |
| 多卡 PUMA/PLWS fill | `run_contest_fill_queue.py`（租卡必须传现行六集和 `--seeds 42,0,1`） |
| 只采 Full-CoT 后停 | 同上，加 `--fullcot-only`（不跑 PUMA / dense / PLWS） |
| 刷新缺格清单 | `report_large_model_rental_inventory.py`（现行范围报缺；传输核验仍扫历史 180 格） |
| 核对已落盘判分旗标 | `audit_grader_flags.py`（默认只报；`--fix` 只把 False 升 True） |

判分后端是硬依赖，不是可选项。缺 `antlr4-python3-runtime` / `latex2sympy2` 时
`math_grader` 无法判定 LaTeX 等价，所有符号答案会静默算错，Acc 整体偏低。
`run_contest_fill_queue.py`、`score_leftover_suppress.py`、
`report_fullcot_puma_plws.py` 开跑前都会 `require_grader()` 自检并直接报错。
换机器或重装环境后，先跑一遍 `audit_grader_flags.py` 再信任旧格子。
历史小模型怎么扫、`--fix` 何时能用，见 `docs/GRADER_ACCURACY.md`。

单格内部调用链：

```text
run_contest_prereq_cell.sh
 ├── run_puma_aligned_sample.sh
 ├── run_puma_official.sh
 ├── run_dense_trials_model.sh
 │    └── 复用 PUMA 已生成的试答，只补 embedding-filter 跳过的步骤
 └── detach_export_leftover_jobs.sh
run_contest_plws_cell.sh
 └── score_leftover_suppress.py
run_deer_official.sh
```

这些原子脚本是正式实现，不要另复制一套参数。

租卡 fill 队列下的 vLLM `gpu_memory_utilization`（包装脚本先 export 0.97）：

| 阶段 | 占用比 | 接线 |
|---|---|---|
| Full-CoT | 0.97 | 跟着队列环境 |
| PUMA | 0.90 | `PUMA_VLLM_GPU_MEMORY_UTILIZATION` |
| dense | 0.90 | `DENSE_VLLM_GPU_MEMORY_UTILIZATION`，不继承 0.97 |
| 窗后压 | 0.97 | 读队列环境 |

详见 `docs/RENTAL_SERVER.md`。dense 不得再用窗后压那档 0.97。

`run_dense_trials_model.sh` 的 dense 轨迹仍覆盖 Full-CoT 的每个推理步骤，但
不再重跑 PUMA 已生成的重叠切点。它冻结
`puma_trial_reuse_plan.json`，复用 PUMA 非 `skipped` 行，只对缺步调用
`gen_trial_answers.py`，最后按 `(question_idx, stopped_len)` 合并并严格核验完整
覆盖。旧版已落盘的 dense shard 优先保留，可从中断处升级续跑。

## 其他脚本

- `report_*`、`analyze_*`、`score_*`：结果重算和论文分析，保留用于复现，不是
  GPU 调度入口。
- `run_*_queue.py`：本机历史或小模型队列。其 GPU 池、模型范围和 parked 状态
  不应直接带到租卡机。
- `launch_*`、`stop_*`、`wait_*`：机器相关 tmux 包装；正式租卡命令写在
  `docs/RENTAL_SERVER.md`。
- 云服务器拉远程：见 `docs/RENTAL_SERVER.md`「云服务器如何拉远程」。当前
  AutoDL 用 `ghproxy.net` 做 `git fetch`，不要改 `origin` URL。

主表协议只能使用 `puma-fullcot-32k-v2`、`firstwin`、`lexicon=core`。旧
low/mix/high 分跑、16K greedy DEER、历史 leftover 变体不得混入主表。
