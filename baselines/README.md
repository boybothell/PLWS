# 额外基线代码入口

完整的方法配置、上游 pin、复用边界、产物格式及本机/A800 部署差异统一写在：

[`../docs/BASELINE_ALIGNMENT_AND_CLOUD_RUNBOOK.md`](../docs/BASELINE_ALIGNMENT_AND_CLOUD_RUNBOOK.md)

这里不再重复方法说明。以后只放三种额外对比基线的代码：

```text
baselines/
  deer/
  answer_convergence/
  dynasor/
```

- PLWS 是主方法，唯一实现仍在 `src/plws/` 与 `scripts/`。
- PUMA 由 `scripts/bootstrap_puma.sh` 按固定 commit + patch 重建，不复制源码。
- Full-CoT 是参考轨迹，不单建方法目录。
- `results/baselines/` 是机器产物目录，与本目录不是一回事。

当前代码状态：

- PUMA、PLWS、统一-host DEER：已有可用入口。
- Answer Convergence：核心 runner 已有，但云端 wrapper 仍需移除本机 venv 路径。
- Dynasor：canonical offline runner 尚未实现。
- PUMA：正式新产物入表前还需实现逐题剩余 32K 预算限制。
- `configs/deploy/{local_48g,a800_80g}.toml` 尚未实现。

在代码真正落地前不创建空方法目录，也不因文档中出现计划路径就认定入口可用。
