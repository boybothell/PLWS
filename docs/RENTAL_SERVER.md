# 租卡服务器运行手册

本手册只覆盖四个大模型、九个数据集、五个 seed 的 PUMA、PLWS、DEER。
当前缺格见
[`large_model_rental_missing.md`](../tables/firstwin_wait/large_model_rental_missing.md)，
机器清单见
[`large_model_rental_inventory.json`](../manifests/large_model_rental_inventory.json)。

## 1. 克隆与环境

推荐目录：

```text
work/
├── plws/
└── PUMA/
```

```bash
git clone <PLWS_REPO_URL> plws
cd plws

# 指向租卡镜像中能导入 vllm、torch、transformers 的 Python。
export PLWS_PY=/path/to/vllm-env/bin/python
uv pip install --python "$PLWS_PY" -e . --no-deps
# 需要在服务器跑测试时再安装：
uv pip install --python "$PLWS_PY" -r requirements-dev.txt

# 从固定上游提交创建带 puma-fullcot-32k-v2 补丁的 PUMA。
bash scripts/bootstrap_puma.sh
```

每个 tmux 会话都设置：

```bash
export PLWS_ROOT="$PWD"
export PUMA_ROOT="$(cd ../PUMA && pwd)"
export PLWS_MODELS_ROOT=/path/to/models
export PLWS_PY=/path/to/vllm-env/bin/python
```

模型目录名见 `configs/models.toml`。模型根目录可以通过
`PLWS_MODELS_ROOT` 修改，不要求复刻本机绝对路径。

## 2. 同步断点

Git 不包含 `samples/` 和 `results/`。如需续跑本机部分 cell，按
[`ARTIFACTS.md`](ARTIFACTS.md) 的最小同步集，用 rsync 或对象存储单独同步。
不同步也能从零运行，但不会继承已完成的 Full-CoT、dense 或 PLWS shard。

同步后重扫：

```bash
"$PLWS_PY" scripts/report_large_model_rental_inventory.py
```

## 3. 预检

昂贵任务开始前必须运行：

```bash
bash scripts/preflight_rental_server.sh
```

它检查四个模型、九份数据、PUMA 补丁、关键入口和实际 Python import。预检
失败时不要启动生成。

## 4. 运行一个 cell

统一入口为 `scripts/run_large_model_cell.sh`。例如在一张 80 GB 卡上跑
Qwen3-30B-A3B、AIME24、seed 0 的三种方法：

```bash
tmux new-session -d -s plws_q30_aime24_s0 \
  "cd '$PLWS_ROOT' && \
   PLWS_PY='$PLWS_PY' PUMA_ROOT='$PUMA_ROOT' \
   PLWS_MODELS_ROOT='$PLWS_MODELS_ROOT' \
   MODEL_TAG=qwen3_30b_a3b DATASET=aime24 SEED=0 \
   GPU=0 PLWS_TP=1 STAGES=puma,plws,deer \
   bash scripts/run_large_model_cell.sh"
tmux ls
```

`STAGES` 可取：

- `puma`：补 Full-CoT、PUMA、dense 和 firstwin jobs。
- `plws`：先补上述前置，再续跑 PLWS shard。
- `deer`：只跑统一 host 的 DEER。
- `puma,plws,deer`：顺序跑完整 cell。

一张 80 GB 卡能否容纳 38K 上下文必须以 smoke/预检后的实际加载为准。放不下时
使用 `GPU=0,1 PLWS_TP=2`，不能缩短 32K host 预算。

## 5. 多卡调度

PUMA/PLWS 多卡任务复用通用 fill queue，不再为模型或 GPU 编号复制脚本：

```bash
MODELS=qwen3_30b_a3b,r1_32b,qwen3_32b,qwq_32b
DATASETS=math-500,olympiadbench,gpqa-diamond,aime24,aime25,aime26,brumo25,hmmt25,amc23

tmux new-session -d -s plws-large-fill \
  "cd '$PLWS_ROOT' && \
   PLWS_PY='$PLWS_PY' PUMA_ROOT='$PUMA_ROOT' \
   PLWS_MODELS_ROOT='$PLWS_MODELS_ROOT' PLWS_LARGE_TP=1 \
   '$PLWS_PY' scripts/run_contest_fill_queue.py \
   --gpus 0,1,2,3 --models '$MODELS' --datasets '$DATASETS'"
tmux ls
```

该队列按规范产物重扫、动态领卡并串行冷加载。`PLWS_LARGE_TP=1` 只适用于实际
能容纳 38K host 的 80 GB 卡；否则改为 2，并保证 GPU 池能分成双卡 lane。
DEER 使用上一节的统一单格入口另建 tmux 池，避免和 PUMA/PLWS 同时冷加载。

停止时先向对应 tmux 会话发送 `Ctrl-C`，不要直接杀 vLLM worker。Full-CoT 和
DEER 整批通常没有题级 checkpoint；PLWS 以完整 `shard_*.jsonl` 为断点。

## 6. 完成核验

```bash
"$PLWS_PY" scripts/report_large_model_rental_inventory.py
"$PLWS_PY" -m pytest -q
```

只有规范产物完整且 manifest 的模型、数据集、seed、协议与预算一致，cell 才算
完成。日志退出码、旧 Markdown 数字和其他 seed 的产物都不能代替核验。
