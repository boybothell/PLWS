# 脚本入口

不要按文件名猜主入口。`scripts/` 同时保留正式流水线、报告脚本和历史分析；
租卡只使用本页“正式入口”。

## 正式租卡入口

| 用途 | 入口 |
|---|---|
| 安装固定 PUMA 补丁和九份评测数据 | `bootstrap_puma.sh` |
| 环境与资产预检 | `preflight_rental_server.sh` |
| 单个大模型 cell | `run_large_model_cell.sh` |
| 多卡 PUMA/PLWS fill | `run_contest_fill_queue.py`（租卡必须传现行五集和 `--seeds 42,0,1`） |
| 刷新缺格清单 | `report_large_model_rental_inventory.py`（现行范围报缺；传输核验仍扫历史 180 格） |

单格内部调用链：

```text
run_contest_prereq_cell.sh
 ├── run_puma_aligned_sample.sh
 ├── run_puma_official.sh
 ├── run_dense_trials_model.sh
 └── detach_export_leftover_jobs.sh
run_contest_plws_cell.sh
 └── score_leftover_suppress.py
run_deer_official.sh
```

这些原子脚本是正式实现，不要另复制一套参数。

## 其他脚本

- `report_*`、`analyze_*`、`score_*`：结果重算和论文分析，保留用于复现，不是
  GPU 调度入口。
- `run_*_queue.py`：本机历史或小模型队列。其 GPU 池、模型范围和 parked 状态
  不应直接带到租卡机。
- `launch_*`、`stop_*`、`wait_*`：机器相关 tmux 包装；正式租卡命令写在
  `docs/RENTAL_SERVER.md`。

主表协议只能使用 `puma-fullcot-32k-v2`、`firstwin`、`lexicon=core`。旧
low/mix/high 分跑、16K greedy DEER、历史 leftover 变体不得混入主表。
