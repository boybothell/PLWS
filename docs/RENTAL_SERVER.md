# 租卡服务器运行手册

本手册只覆盖四个大模型的 PUMA、PLWS、DEER。
现行新跑是六集 × 第一波三个 seed：

```text
DATASETS=math-500,olympiadbench,gpqa-diamond,aime25,hmmt25,amc23
SEEDS=42,0,1
```

`123` / `7` 等第一波齐了再排。不要新开 `aime24` / `aime26` / `brumo25`。
仓库里仍钉着九份 jsonl；其中这三份只为续跑已有产物和校验哈希。
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

# 从固定上游提交创建带 puma-fullcot-32k-v2 补丁的 PUMA，
# 并把仓库自带的九份评测 jsonl 原样装进 PUMA/data 后校验 SHA256。
# clone/pull 后数据已经齐全；不要另下 Hugging Face 或同名官方集覆盖。
# 新队列用现行六集；其余三份只服务已有产物续跑/核验。
bash scripts/bootstrap_puma.sh
```

### 云服务器如何拉远程

不要改 `git config`，也不要把 `origin` 永久改成镜像。部分 AutoDL 机房
`/etc/network_turbo` 会写「该地区学术加速暂未支持」，并提示 GitHub 走
`https://ghproxy.com/`；实测该地址对 git 协议经常不返回分支。当前可用的是
`ghproxy.net`：

```bash
# 只更新 origin/main，不改 remote URL
git fetch https://ghproxy.net/https://github.com/boybothell/PLWS.git \
  +main:refs/remotes/origin/main
git rebase origin/main
```

直连 `github.com` 的 `git fetch` / `ls-remote` 若卡在 TCP，先换上面这条。
`source /etc/network_turbo` 只在脚本真正导出 `http_proxy` 时有用；若脚本只
打印「暂未支持」，不要指望它能拉 GitHub。用完镜像后继续让 `origin` 指向
官方 `https://github.com/boybothell/PLWS.git`。

每个 tmux 会话都设置：

```bash
export PLWS_ROOT="$PWD"
export PUMA_ROOT="$(cd ../PUMA && pwd)"
export PLWS_MODELS_ROOT=/path/to/models
export PLWS_PY=/path/to/vllm-env/bin/python
```

模型目录名见 `configs/models.toml`。模型根目录可以通过
`PLWS_MODELS_ROOT` 修改，不要求复刻本机绝对路径。九份评测 jsonl 的唯一源是
本仓库 `data/`，`bootstrap_puma.sh` 会将其安装到 `$PUMA_ROOT/data/`；
正式续跑不要设置 `PLWS_DATA_ROOT`。如确需覆盖该变量，目标文件必须与仓库
锁定文件逐字节一致。详见 [`data/README.md`](../data/README.md)。
本机绝对路径写进 gitignored 的 `.env` 或 `tmp/`，不要写进本手册。

## 2. 同步断点

Git 不包含 `samples/` 和 `results/`。先把
`large_model_rental_artifacts_20260916.tar.gz` 放到服务器；它包含四个大模型
本机已有的 Full-CoT、PUMA、dense、PLWS jobs 和 shard。

先校验、解包，再按规范产物重扫：

```bash
echo \
  "e7d96190fd6c26f8719420e8bb7dd8f1c7477bfdfde0a1c93743b449de89a5ca  large_model_rental_artifacts_20260916.tar.gz" \
  | sha256sum -c -
tar -xzf large_model_rental_artifacts_20260916.tar.gz

"$PLWS_PY" scripts/report_large_model_rental_inventory.py \
  --expect-transfer manifests/large_model_transfer_20260916.json
```

校验通过时至少应识别出历史包基线：Full-CoT 94/180、PUMA 85/180、dense
87/180、PLWS 79/180。这是 4×9×5 传输包的完整性，不是现行新跑格子数。
低于任一项说明包未完整解到 `PLWS_ROOT`，此时禁止启动 GPU 任务。高于这些
数目表示服务器已有新增进度，可以继续使用。缺格清单按现行 4×5×3 报要跑的格。

可直接续跑的 PLWS 是 R1-32B OlympiadBench s0（496/560）和 MATH-500
s1（40/272）。另有七格 Qwen3-30B jobs 已齐但尚未计分：MATH s42/s0、
OlympiadBench s42、GPQA s42/s0/s1/s123。队列会从这些 shard/jobs 接着跑，
不会重做已完成 cell。

包内附带的缺格快照生成较早；解包后必须执行上面的重扫命令，以服务器实际产物
覆盖它。不要按旧 `status.json` 或压缩包内旧 Markdown 手工建任务。

## 3. 预检

昂贵任务开始前必须运行：

```bash
bash scripts/preflight_rental_server.sh
```

它检查四个模型、PUMA 补丁、关键入口和实际 Python import，并按仓库
`data/SHA256SUMS` 校验九份锁定数据、`$PUMA_ROOT/data/` 及运行时实际数据路径。
这会同时锁定内容与题序，不再只检查条数。预检失败时不要启动生成。
新队列用现行六集。

## 4. 运行一个 cell

统一入口为 `scripts/run_large_model_cell.sh`。例如在一张 80 GB 卡上跑
Qwen3-30B-A3B、AIME25、seed 0 的三种方法：

```bash
tmux new-session -d -s plws_q30_aime25_s0 \
  "cd '$PLWS_ROOT' && \
   PLWS_PY='$PLWS_PY' PUMA_ROOT='$PUMA_ROOT' \
   PLWS_MODELS_ROOT='$PLWS_MODELS_ROOT' \
   MODEL_TAG=qwen3_30b_a3b DATASET=aime25 SEED=0 \
   GPU=0 PLWS_TP=1 STAGES=puma,plws,deer \
   bash scripts/run_large_model_cell.sh"
tmux ls
```

`STAGES` 可取：

- `puma`：补 Full-CoT、PUMA、dense 和 firstwin jobs。
- `plws`：先补上述前置，再续跑 PLWS shard。
- `deer`：只跑统一 host 的 DEER。
- `puma,plws,deer`：顺序跑完整 cell。

前置内部顺序是 Full-CoT → PUMA → dense → firstwin jobs。dense 会直接复用
PUMA `trial_answers.json` 中已经生成的切点，只补 PUMA embedding filter 跳过的
步骤，不再对重叠切点做第二次 GPU 试答。合并后必须逐步覆盖完整 Full-CoT
轨迹；复用来源和步数写在 dense shard 下的计划与结果 JSON 中。

一张 80 GB 卡能否容纳 38K 上下文必须以 smoke/预检后的实际加载为准。放不下时
使用 `GPU=0,1 PLWS_TP=2`，不能缩短 32K host 预算。

## 5. 多卡调度

PUMA/PLWS 多卡任务复用通用 fill queue，不再为模型或 GPU 编号复制脚本：

```bash
MODELS=qwen3_30b_a3b,r1_32b,qwen3_32b,qwq_32b
DATASETS=math-500,olympiadbench,gpqa-diamond,aime25,hmmt25,amc23
SEEDS=42,0,1

tmux new-session -d -s plws-large-fill \
  "cd '$PLWS_ROOT' && \
   PLWS_PY='$PLWS_PY' PUMA_ROOT='$PUMA_ROOT' \
   PLWS_MODELS_ROOT='$PLWS_MODELS_ROOT' PLWS_LARGE_TP=1 \
   VLLM_GPU_MEMORY_UTILIZATION=0.97 \
   VLLM_MAX_NUM_SEQS=8 \
   VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
   '$PLWS_PY' scripts/run_contest_fill_queue.py \
   --gpus 0,1,2,3 --models '$MODELS' \
    --datasets '$DATASETS' --seeds '$SEEDS'"
tmux ls
```

只采齐 Full-CoT、不跑 PUMA / dense / PLWS 时，在同一条命令末尾加
`--fullcot-only`。格子已有规范 `answers.json` 的会直接记成功；队列在全部
sample 齐后退出。

该队列按规范产物重扫、动态领卡并串行冷加载。`PLWS_LARGE_TP=1` 只适用于实际
能容纳 38K host 的 80 GB 卡；否则改为 2，并保证 GPU 池能分成双卡 lane。
DEER 使用上一节的统一单格入口另建 tmux 池，避免和 PUMA/PLWS 同时冷加载。
队列启动时会逐格调用 `fill_task_complete`：包内已齐的 PUMA/PLWS 自动记为
succeeded，不进入 GPU pending；半截 shard 只补缺少的 uid。

停止时先向对应 tmux 会话发送 `Ctrl-C`，不要直接杀 vLLM worker。Full-CoT 和
DEER 整批通常没有题级 checkpoint；PLWS 以完整 `shard_*.jsonl` 为断点。

租卡 fill 包装脚本（本机 `tmp/run_q30_dynamic_queue.sh` 同类）会先
`export VLLM_GPU_MEMORY_UTILIZATION=0.97`。各阶段实际占用比如下，不要混用：

| 阶段 | `gpu_memory_utilization` | 谁定的 |
|---|---|---|
| Full-CoT（`run_vllm.py`） | 0.97 | 采样脚本不改写，跟着队列环境 |
| PUMA 试答 + 前缀续写 | 0.90 | `run_puma_official.sh` 用 `PUMA_VLLM_GPU_MEMORY_UTILIZATION` 强制 |
| dense 补缺步 | 0.90 | `run_dense_trials_model.sh` 用 `DENSE_VLLM_GPU_MEMORY_UTILIZATION` 强制；不得继承窗后压的 0.97 |
| 窗后压 PLWS | 0.97 | `score_leftover_suppress.py` 读环境；脚本默认 0.88，队列给 0.97 |

0.6B 冗余检测写死 0.30，与上表无关。Olympiad 等长前缀 dense 在 0.97 下会在
logits 排序时 OOM；PUMA / dense 必须保持 0.90。窗后压仍要 0.97 才能放下
38K KV。不经 fill 包装单独跑时：Full-CoT 回到 `run_vllm.py` 默认 0.85，窗后压
回到 0.88。

## 6. 完成核验

```bash
"$PLWS_PY" scripts/report_large_model_rental_inventory.py
"$PLWS_PY" -m pytest -q
```

只有规范产物完整且 manifest 的模型、数据集、seed、协议与预算一致，cell 才算
完成。日志退出码、旧 Markdown 数字和其他 seed 的产物都不能代替核验。
