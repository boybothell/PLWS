# AC / Dynasor 交接：本机 vs 四张 5090

日期：2026-09-22。  
任务清单：`results/runs/extra_baseline_fill/handoff_5090_split_20260922.json`。

本机继续跑 Nemotron 全部未齐格子，以及全部模型未齐的 AIME25。  
5090 只跑 **非 Nemotron** 的 MATH-500 / GPQA-Diamond / OlympiadBench。  
AMC23 已齐，不要转、不要重跑。AIME25 不要给 5090。Nemotron 权重不在那台机器上，一格都不要排。

## 1. 分工

| 机器 | 范围 | 格数 |
|---|---|---|
| 本机 6+7 | 11 模型 AIME25 未齐 | 31 |
| 本机 6+7 | Nemotron 的 GPQA + MATH + Olympiad | 18 |
| 本机合计 | | 49 |
| 5090 ×4 | 其余 10 模型 × 三长集 × seed 42/0/1 × AC+Dynasor | **180** |

10 个远程模型：`r1_1p5b` `r1_7b` `r1_14b` `r1_llama_8b` `qwen3_4b` `qwen3_8b` `qwen3_30b_a3b` `qwen3_32b` `qwq_32b` `r1_32b`。

已齐格以规范产物为准，禁止重跑。半成品带 `records/` 的续同一目录。

## 2. 协议：两边必须逐字段相同

`puma-fullcot-32k-v2`。不要改这些：

- checkpoint 目录名（见下表）和 tokenizer
- 数据集文件与题序（仓库 `data/` → `bootstrap_puma.sh`）
- prompt：PUMA `default`，数据集专用 instruction
- 32768 / 2048 / 3072 / `max_model_len=37888`
- 交付采样只读该 checkpoint 的 `generation_config.json`
- AC：句子切点、greedy probe、k=10
- Dynasor：offline mid，chunk 64，certainty 3
- 统一 grader

方法内部参数不要为了 5090 去调。

## 3. 5090 部署：只动 TP，不改利用率

四张 32GB 合计 128GB，大于本机两张 L40 TP=2 的 96GB。  
`gpu_memory_utilization` 保持 **0.90**，`max_num_seqs` 保持 **64**。不要用降利用率或改预算来解决 OOM。

部署 profile：`configs/deploy/rtx5090_32g.toml`

| 模型 | 本机 TP | 5090 TP | 说明 |
|---|---|---|---|
| 1.5B / 7B / Llama-8B / 4B / 8B | 1 | 1 | 单卡 |
| R1-14B | 1 | 1 | 与本机相同；若 32GB 装不下 KV，只把该模型 TP 调到 2 |
| 30B / 32B / QwQ / R1-32B | 2 | **4** | 四卡一格，总显存比本机更宽 |

不要套 `local_48g` 的 TP=2 去只占两张 5090 跑 32B。两张 32GB 小于两张 48GB。  
不要套 `a800_80g` 的 TP=1。

冷加载整机串行：同时只装 1 个新引擎。已经 generate 的格子可以并行。  
长任务一律 tmux。

## 4. 远程要有的东西

### 代码与数据

按 `docs/RENTAL_SERVER.md` 第 1 节：clone 本仓库、独立 venv、`bootstrap_puma.sh`。  
判分三包必须装，否则 Acc 会 silently 偏低。

```bash
export PLWS_ROOT="$PWD"
export PUMA_ROOT="$(cd ../PUMA && pwd)"
export PLWS_MODELS_ROOT=/path/to/models
export PLWS_PY=/path/to/vllm-env/bin/python
export PLWS_DEPLOY_PROFILE=rtx5090_32g
export PYTHONPATH="${PLWS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
```

vLLM / torch 需要能跑 Blackwell（CUDA 12.8+）。这是运行时，不是协议。

### 权重目录名（根目录用 `PLWS_MODELS_ROOT`）

| tag | 目录名 |
|---|---|
| `r1_1p5b` | `DeepSeek-R1-Distill-Qwen-1.5B` |
| `r1_7b` | `DeepSeek-R1-Distill-Qwen-7B` |
| `r1_14b` | `DeepSeek-R1-Distill-Qwen-14B` |
| `r1_llama_8b` | `DeepSeek-R1-Distill-Llama-8B` |
| `r1_32b` | `DeepSeek-R1-Distill-Qwen-32B` |
| `qwen3_4b` | `Qwen3-4B` |
| `qwen3_8b` | `Qwen3-8B` |
| `qwen3_30b_a3b` | `Qwen3-30B-A3B-Thinking-2507` |
| `qwen3_32b` | `Qwen3-32B` |
| `qwq_32b` | `QwQ-32B` |

不要拷 `Llama-3.1-Nemotron-Nano-8B-v1`。

### 冻结 Full-CoT（必须）

AC / Dynasor 只重放已有轨迹。每个远程格子要有：

```text
samples/<model_tag>/<dataset>/seed_<seed>/answers.json
samples/<model_tag>/<dataset>/seed_<seed>/sample_meta.json
```

`sample_meta.json` 必须是 `puma-fullcot-32k-v2`，prompt version `default`，预算 32768/2048/37888。缺 `answers.json` 的格子不要开。

本机打包（10 模型 × 三长集 × 三 seed，不含 Nemotron、不含 AIME/AMC）：

```bash
cd "$PLWS_ROOT"
tar -cvf extra_ac_dyn_samples_long3.tar \
  samples/r1_1p5b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/r1_7b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/r1_14b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/r1_llama_8b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/r1_32b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/qwen3_4b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/qwen3_8b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/qwen3_30b_a3b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/qwen3_32b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1} \
  samples/qwq_32b/{gpqa-diamond,math-500,olympiadbench}/seed_{42,0,1}
```

解到远程 `$PLWS_ROOT`，保持相对路径。不要整份拷 `results/`。不要拷 `results/archive/`。

## 5. 远程怎么起

队列会跳过已齐格。这 180 格目前全空，从零开即可。

```bash
cd "$PLWS_ROOT"
tmux new-session -d -s plws-ac-dyn-5090
tmux send-keys -t plws-ac-dyn-5090 "cd \"$PLWS_ROOT\" && \\
  export PLWS_ROOT=\"$PLWS_ROOT\" PUMA_ROOT=\"$PUMA_ROOT\" \\
         PLWS_MODELS_ROOT=\"$PLWS_MODELS_ROOT\" PLWS_PY=\"$PLWS_PY\" \\
         PLWS_DEPLOY_PROFILE=rtx5090_32g \\
         PYTHONPATH=\"${PLWS_ROOT}/src\${PYTHONPATH:+:\$PYTHONPATH}\" && \\
  \"\$PLWS_PY\" scripts/run_extra_baseline_queue.py \\
    --gpus 0,1,2,3 \\
    --models r1_1p5b,r1_7b,r1_14b,r1_llama_8b,qwen3_4b,qwen3_8b,qwen3_32b,qwq_32b,qwen3_30b_a3b,r1_32b \\
    --datasets gpqa-diamond,math-500,olympiadbench \\
    --seeds 42,0,1 \\
    --methods answer_convergence,dynasor" Enter
```

调度两卡空闲时优先领 TP>=2。四卡空闲时会先领 30B/32B/QwQ 的 TP=4 格。小模型等没有大格或卡不够四张时再上。这是现成逻辑，不要为了「先跑小的」去改方法参数。

若希望小模型先出数：先把 `--models` 收成 `r1_1p5b,r1_7b,r1_llama_8b,qwen3_4b,qwen3_8b`，齐了再开 14B 和大模型。协议不变。

产物写在：

```text
results/baselines/answer_convergence/puma-fullcot-32k-v2/<model>/<dataset>/seed_<seed>/
results/baselines/dynasor/puma-fullcot-32k-v2/<model>/<dataset>/seed_<seed>/
```

齐的标志：`manifest.json` + `final_answers.jsonl`，题数等于该格 Full-CoT，`protocol_id` 对，AC `threshold=10`，Dynasor mid/64/3。

## 6. 本机留下的（不要给 5090）

- 正在跑的 `ansconv__qwq_32b__aime25__s1`：等落盘，不要杀快写完的格。
- 其余 AIME25 未齐（含 14B Dynasor s1 的 16/30，续同一目录）。
- Nemotron 全部未齐：AIME 还剩 AC s1 + Dynasor 三 seed；GPQA/MATH/Olympiad 两方法 × 三 seed 共 18 格。
- QwQ Olympiad leftover 在 0+2，与这条交接无关，不要动。

建议本机当前格齐了之后：6、7 先两路 TP=1 收完 Nemotron，再合成 TP=2 继续 AIME。Nemotron 必须在本机停卡前齐。

## 7. 回传

只回收远程新写的 `results/baselines/{answer_convergence,dynasor}/puma-fullcot-32k-v2/` 下上述 10 模型、三长集目录。  
合进本机前按 `extra_baseline_complete` 扫一遍，已齐本机格不得覆盖。  
回传后不要在远程继续写同一格，除非本机明确说缺 records。
