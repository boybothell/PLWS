# PLWS

PLWS 是本仓库的主方法：在第一扇连续同答窗后继续推理，并压制 CORE
反思词。源码、配置与实验产物按可复现边界分离。

本机进度看 `STATUS.md`，流水看 `docs/log.md`（都不进 git）。
跨项目核验与教训写在工作区 `.cursor/rules/experiment-status.mdc`，不写在本文件。

## 快速入口

- 租卡部署与单格运行：[`docs/RENTAL_SERVER.md`](docs/RENTAL_SERVER.md)
- Full-CoT 公平协议：[`docs/FULLCOT_PROTOCOL.md`](docs/FULLCOT_PROTOCOL.md)
- 产物目录与断点同步：[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)
- 脚本分层与正式入口：[`scripts/README.md`](scripts/README.md)

## 方法边界

- **PLWS**：第一扇同答窗后继续推理，屏蔽 Wait / Alternatively / Hmm。新代码在 `src/plws/`。
- **PUMA / DEER**：冻结对照。代码和结果不得删除或覆盖，也不是 PLWS 的别名。
- **`attn_early_exit`**：保留的兼容包。现有脚本可以继续原样导入它。

## 数据与产物

- `data/`：九份评测 jsonl，题序唯一源。`bootstrap_puma.sh` 会装进 `$PUMA_ROOT/data/`。
- `samples/`：只读输入。
- `results/`：唯一机器结果根；规范见 [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)。
- `tables/`、`figures/`：本地生成的人读结果，默认不进 Git。
- `configs/`：模型、数据集和方法默认配置。

## 默认窗口定义

```text
k = 4
最早允许停下的步骤 MSS = 10
高把握阈值 tau = 0.995
高把握回落容差 eps = 0.03
```

窗的 H / M / L 只是第一扇窗的把握标签，写在 `src/plws/window.py`，不是三种跑法。
词表在 `src/plws/lexicon.py`。规范 jobs 是 `jobs/firstwin.jsonl`。
路径和 manifest/status 协议在 `src/plws/artifacts.py` 与 `src/plws/paths.py`。
