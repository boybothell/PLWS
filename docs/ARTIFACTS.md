# 产物目录与同步边界

`results/` 是唯一机器结果根，`samples/` 是只读 Full-CoT 输入；两者都不进
Git。租卡机恢复任务时必须单独同步它们，不能只执行 `git clone`。

## 当前布局

```text
samples/<model>/<dataset>/seed_<seed>/       # Full-CoT，只读
results/
├── baselines/
│   ├── puma/                                # PUMA
│   ├── deer/puma_fullcot_32k_v2/            # 主表 DEER
│   ├── answer_convergence/puma_fullcot_32k_v2/
│   ├── dynasor/puma_fullcot_32k_v2/
│   └── official/
├── upstream/dense_trials/                   # PLWS 冻结试答轨迹
├── runs/plws/window_first/                   # PLWS jobs、shards、manifest
├── reports/                                  # 可重建的机器汇总
├── cache/                                    # 可安全重建
├── archive/                                  # 只读历史，不得作新主表
├── registry/
└── runs/machines/<机名>/<队列名>/            # 工位账本，见下方
```

这里的 `results/baselines/` 是**机器产物目录**。仓库顶层 `baselines/` 是额外
对比基线的源码/适配目录，两者不要混淆。PLWS 源码仍在 `src/plws/` 与 `scripts/`；
PUMA 源码仍由 `scripts/bootstrap_puma.sh` 从固定 commit 重建，不在顶层
`baselines/` 再复制一份。

## 保存规则

- `baselines/`、`upstream/`、`runs/` 是断点和证据，不得覆盖或按类别删除。
- 只有 `cache/` 可以直接清空。
- `tables/` 是人读报告；Git 只保留租卡交接快照。
- 每个 cell 的完成状态以规范产物和 manifest 为准，不以日志退出码为准。
- Acc 旗标可能是「判分失败」而不是答错。沿用历史格子（含小模型）先按
  [`GRADER_ACCURACY.md`](GRADER_ACCURACY.md) 重判，再出表。
- seed 42 的平铺产物不能代替其他 seed。

## 多机账本

格子产物仍写上面的规范路径。调度账本（`status.json`、`events.jsonl`、attempt
日志）按机器分开，避免合并时盖住另一台的队列指针。

工位置 `PLWS_MACHINE=<机名>`。未另指定时，账本落在：

```text
results/runs/machines/<机名>/<队列名>/
```

`CONTEST_FILL_RUN_ROOT` / `EXTRA_BASELINE_RUN_ROOT` 若自己指定，必须仍在
`results/runs/machines/<机名>/` 下面，否则拒绝启动。不设 `PLWS_MACHINE`
时维持本机历史根（`contest_fill`、`extra_baseline_fill` 等）。

拷回本机：已齐格子进规范产物目录；账本整棵放进
`results/runs/machines/<机名>/`，只覆盖这个机名。不要对拷整棵 `results/runs/`。
任务谁跑什么不写在这里。

## 租卡机最小同步集

先在本机按缺格清单确定目标 cell，再只同步这些 cell 的已有前缀：

```text
samples/<model>/<dataset>/seed_<seed>/
results/baselines/puma/<对应目录>/
results/upstream/dense_trials/dense_G_<model>/<dataset>/seed_<seed>/
results/runs/plws/window_first/k_4/lexicon_core/<model>/<dataset>/seed_<seed>/
results/baselines/deer/puma_fullcot_32k_v2/<model>/<dataset>/seed_<seed>/
results/baselines/answer_convergence/puma_fullcot_32k_v2/<model>/<dataset>/seed_<seed>/
results/baselines/dynasor/puma_fullcot_32k_v2/<model>/<dataset>/seed_<seed>/
```

后两项只有对应 runner 已实现且 cell 已经生成时才同步。同步应保留目录层级，
并在租卡机运行清单脚本重新扫描。不要把 191 GB 本机产物提交到 Git，也不要同步
`results/archive/`。

四个大模型的本机已有产物打在 `transfer/large_model_rental_artifacts_20260916.tar.gz`：
Full-CoT、PUMA、dense、PLWS jobs/shard；不含 archive、不含小模型、不含 DEER（本机没有）。
在 `PLWS_ROOT` 解包后运行：

```bash
"$PLWS_PY" scripts/report_large_model_rental_inventory.py \
  --expect-transfer manifests/large_model_transfer_20260916.json
```

验证 Full-CoT/PUMA/dense/PLWS 至少为 94/85/87/79 格后再开队列；这是历史
4×9×5 传输包完整性，不是现行新跑格子数。已齐格会自动跳过。新队列只领
MATH / Olympiad / GPQA / AIME25 / HMMT25 的 seed 42/0/1。
