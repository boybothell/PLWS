# PLWS

PLWS 是本仓库的主方法。源码、配置与实验产物已按可复现边界整理；历史产物保留在结构化归档中。

本机进度看 `STATUS.md`，流水看 `docs/log.md`（都不进 git）。
跨项目核验与教训写在工作区 `.cursor/rules/experiment-status.mdc`，不写在本文件。

## 方法边界

- **PLWS**：第一扇同答窗后继续推理，屏蔽 Wait / Alternatively / Hmm。新代码在 `src/plws/`。
- **PUMA / DEER**：冻结对照。代码和结果不得删除或覆盖，也不是 PLWS 的别名。
- **`attn_early_exit`**：保留的兼容包。现有脚本可以继续原样导入它。

## 数据与产物

- `samples/`：只读输入。
- `results/`：唯一机器结果根；规范见 [`results/README.md`](results/README.md)。
- `tables/`：面向人的结果表。当前主表是 [`tables/firstwin_wait/fullcot_puma_plws.md`](tables/firstwin_wait/fullcot_puma_plws.md)。
- `figures/`：面向人的成品图。
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
