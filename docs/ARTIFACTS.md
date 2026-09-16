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
│   └── official/
├── upstream/dense_trials/                   # PLWS 冻结试答轨迹
├── runs/plws/window_first/                   # PLWS jobs、shards、manifest
├── reports/                                  # 可重建的机器汇总
├── cache/                                    # 可安全重建
├── archive/                                  # 只读历史，不得作新主表
└── registry/
```

## 保存规则

- `baselines/`、`upstream/`、`runs/` 是断点和证据，不得覆盖或按类别删除。
- 只有 `cache/` 可以直接清空。
- `tables/` 是人读报告；Git 只保留租卡交接快照。
- 每个 cell 的完成状态以规范产物和 manifest 为准，不以日志退出码为准。
- seed 42 的平铺产物不能代替其他 seed。

## 租卡机最小同步集

先在本机按缺格清单确定目标 cell，再只同步这些 cell 的已有前缀：

```text
samples/<model>/<dataset>/seed_<seed>/
results/baselines/puma/<对应目录>/
results/upstream/dense_trials/dense_G_<model>/<dataset>/seed_<seed>/
results/runs/plws/window_first/k_4/lexicon_core/<model>/<dataset>/seed_<seed>/
results/baselines/deer/puma_fullcot_32k_v2/<model>/<dataset>/seed_<seed>/
```

同步应保留目录层级，并在租卡机运行清单脚本重新扫描。不要把 191 GB 本机
产物提交到 Git，也不要同步 `results/archive/`。

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
