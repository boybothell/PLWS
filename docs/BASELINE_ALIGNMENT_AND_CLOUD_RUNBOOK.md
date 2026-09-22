# 四个早停基线与 PLWS 的统一对齐规范

更新：2026-09-17

本文规定 PUMA、DEER、Answer Convergence、Dynasor 与 PLWS 在本项目中的统一
实验口径、逐模型配置、可复用产物和云端单格执行要求。本文不规定运行顺序和 GPU
排队顺序。

额外对比基线的目录骨架、上游 pin 和现行入口见
[`../baselines/README.md`](../baselines/README.md)；本文只写跨方法的统一口径。

## 1. 结论先行

1. 所有进入同一公平主表的新产物统一使用 `puma-fullcot-32k-v2`。
2. PUMA、PLWS、Answer Convergence 和本文定义的 Dynasor offline replay 可以复用
   同一条规范 Full-CoT 轨迹。
3. DEER 的正式结果必须在线运行，不能把普通 Full-CoT 上已有的 `Wait` 切点重放结果
   当成正式 DEER。原因是 DEER 在一次 probe 未过门后会把 `Wait` 插回上下文，后续
   轨迹已经改变。离线 Wait replay 只作诊断或消融。
4. Dynasor 的 probe 不写回主推理上下文，因此在固定 Full-CoT 轨迹上按 token chunk
   重放，与“probe 后丢弃 probe、继续原轨迹”的算法语义一致。在线运行与离线重放的
   实际墙钟不同，但停止判断和交付前缀可以一致。
5. 九模型主表已经包含 Qwen3-30B-A3B-Thinking-2507。扩展覆盖新增的是
   Qwen3-32B 和 QwQ-32B；两者未齐前不进入主表。
6. 当前已有的 `$PUMA_ROOT/baselines/run_dynasor.py` 不能直接产出正式主表结果：
   它使用自己的 system prompt，`max_model_len` 不是 37888，而且数学答案一致性被
   简化成字符串相等。正式 Dynasor 需要新的 canonical offline runner。
7. 云端只 clone 本仓库。PLWS 继续以 `src/plws/` + `scripts/` 为唯一实现，PUMA
   继续由固定 commit + patch 重建；只有 DEER、Answer Convergence、Dynasor
   这三种额外对比基线在 `baselines/` 内分目录收口。见第 16 节。
8. 本机 48 GB × 11 模型与云端 A800 80 GB × 四个大模型属于不同**部署 profile**。
   允许不同的只有 TP、显存利用率、并发和卡池；host 协议与方法内部参数两边逐字段
   相同。见第 17 节。

## 2. 统一 host 协议

### 2.1 不随方法变化的外部条件

- 协议 ID：`puma-fullcot-32k-v2`
- checkpoint 和 tokenizer：同一模型行必须使用第 4 节中的同一目录
- 数据：同一 dataset、同一 split、同一题序、同一数据文件 SHA256
- seed：同一 cell 使用同一 seed
- prompt：PUMA `default` 的数据集 instruction 和完全相同的模型 chat template
- 主生成预算：32768 token
- prompt 预留：3072 token
- 截断思考后的 answer-fix：最多 2048 token
- vLLM `max_model_len`：37888
- 交付推理和最终答案解码：读取该 checkpoint 的 `generation_config.json`
- grader、答案抽取、Acc 和 token 定义：同一数据集必须一致

从中间前缀继续的方法只有：

```text
remaining_main_tokens = 32768 - prefix_tokens
```

不能在前缀后重新获得 32768 token。只有主预算确实耗尽且尚未收口时，才允许额外
最多 2048 token 的 answer-fix。

### 2.2 保留为方法内部机制的参数

以下参数不能为了表面“统一”而抹平：

- PUMA 的 embedding filter、confidence gate、连续次数和 trial 解码
- DEER 的 ATP、阈值、最多检查次数、`think_ratio`、置信聚合和 Qwen3 安全门
- Answer Convergence 的句子切点、greedy probe 和连续同答次数
- Dynasor 的 chunk 大小、certainty window、probe suffix 和不确定词门
- PLWS 的第一扇同答窗、`k=4` 和 CORE suppression

这些 probe/trial 可以使用不同于交付生成的解码，因为它们是方法本身的一部分；但
必须单独记录 probe/trial token、调用次数和延迟。

### 2.3 prompt 的唯一构造

所有方法都复用 `$PUMA_ROOT/puma/prompt_utils.py` 的 `default` 语义：

- 数学集：`INSTRUCTION_MATH`
- GPQA-Diamond：`INSTRUCTION_GPQA`
- instruction 与问题放在同一个 user message
- DeepSeek-R1-Distill、QwQ 和普通模型使用各自标准 chat template
- Qwen3-4B、Qwen3-8B、Qwen3-32B 使用 `enable_thinking=True`
- Qwen3-30B-A3B-Thinking-2507 是 always-thinking 模型，不传
  `enable_thinking`
- Nemotron 在最前面增加 system message `detailed thinking on`

正式统一表不使用 `aligned_with_deer`，也不使用 Dynasor 现有移植脚本中的通用
system prompt。

## 3. 当前覆盖范围

### 3.1 新开数据与 seed

当前允许新开或续跑的数据集：

- `math-500`
- `olympiadbench`
- `gpqa-diamond`
- `aime25`
- `amc23`

先覆盖 seed `42,0,1`；seed `123,7` 属于后续扩展。已有 `aime24`、`aime26`、
`brumo25`、`hmmt25` 产物保留，但不因为本文重新开跑。

### 3.2 模型集合

九模型主表：

1. `r1_7b`
2. `nemotron_8b`
3. `r1_14b`
4. `r1_1p5b`
5. `r1_llama_8b`
6. `r1_32b`
7. `qwen3_4b`
8. `qwen3_8b`
9. `qwen3_30b_a3b`

扩展模型：

10. `qwen3_32b`
11. `qwq_32b`

## 4. 逐模型固定配置

`top_k=-1` 表示 checkpoint 未给出 top-k，vLLM 中关闭 top-k 截断。

下表最后两列是**部署量**，不是算法参数：`local_48g` 是本机 48 GB 卡的稳妥值，
`a800_80g` 是云端 A800 80 GB 的目标值。两者由部署 profile 决定，可按实际加载调整
并记入 manifest 的 `deployment`；其余列是算法语义，两边必须相同。见第 17 节。
`a800_80g` 只放四个大模型，其它行标为 `—`。

| tag | 精确 checkpoint 目录 | 交付采样 `temp/top_p/top_k` | chat/thinking | PUMA conf | DEER profile | `local_48g` | `a800_80g` |
|---|---|---|---|---|---|---|---|
| `r1_1p5b` | `DeepSeek-R1-Distill-Qwen-1.5B` | `0.6/0.95/-1` | DeepSeek-R1 标准模板 | `DS-7B.conf` | standard | TP=1 | — |
| `r1_7b` | `DeepSeek-R1-Distill-Qwen-7B` | `0.6/0.95/-1` | DeepSeek-R1 标准模板 | `DS-7B.conf` | standard | TP=1 | — |
| `r1_llama_8b` | `DeepSeek-R1-Distill-Llama-8B` | `0.6/0.95/-1` | DeepSeek-R1/Llama 模板 | `DS-7B.conf` | standard | TP=1 | — |
| `nemotron_8b` | `Llama-3.1-Nemotron-Nano-8B-v1` | `0.6/0.95/-1` | 增加 `detailed thinking on` | `Nemotron.conf` | standard | TP=1 | — |
| `r1_14b` | `DeepSeek-R1-Distill-Qwen-14B` | `0.6/0.95/-1` | DeepSeek-R1 标准模板 | `DS-14B.conf` | standard | TP=1 | — |
| `r1_32b` | `DeepSeek-R1-Distill-Qwen-32B` | `0.6/0.95/-1` | DeepSeek-R1 标准模板 | `DS-32B.conf` | standard | TP=2 | TP=1 |
| `qwen3_4b` | `Qwen3-4B` | `0.6/0.95/20` | `enable_thinking=True` | `DS-7B.conf` | qwen3 | TP=1 | — |
| `qwen3_8b` | `Qwen3-8B` | `0.6/0.95/20` | `enable_thinking=True` | `DS-7B.conf` | qwen3 | TP=1 | — |
| `qwen3_30b_a3b` | `Qwen3-30B-A3B-Thinking-2507` | `0.6/0.95/20` | always-thinking | `Q30B-T.conf` | qwen3 | TP=2 | TP=1 |
| `qwen3_32b` | `Qwen3-32B` | `0.6/0.95/20` | `enable_thinking=True` | `DS-32B.conf` | qwen3 | TP=2 | TP=1 |
| `qwq_32b` | `QwQ-32B` | `0.6/0.95/40` | QwQ pure-thinking 模板 | `DS-32B.conf` | standard | TP=2 | TP=1 |

配置映射的原则：

- PUMA 只发布了若干模型规模配置，因此当前项目按已锁定映射使用，不按测试集调参。
- `qwen3_4b/qwen3_8b` 和所有 8B 以下或近似 7B 的未单列模型使用
  `DS-7B.conf`。
- `r1_14b` 使用 `DS-14B.conf`。
- `r1_32b/qwen3_32b/qwq_32b` 使用 `DS-32B.conf`。
- 30B thinking MoE 使用专用 `Q30B-T.conf`。
- Qwen3 的 PUMA conf 映射与 DEER profile 是两套独立机制，不能因为 PUMA 使用
  `DS-32B.conf` 就把 Qwen3-32B 的 DEER profile 改成 standard。

## 5. Full-CoT 参考轨迹

Full-CoT 不是第五个早停基线，而是统一参考和可复用的 frozen trajectory。

规范产物位置：

```text
samples/<model_tag>/<dataset>/seed_<seed>/answers.json
samples/<model_tag>/<dataset>/seed_<seed>/sample_meta.json
```

可复用前必须验证：

- `sample_meta.json` 的 protocol ID 是 `puma-fullcot-32k-v2`
- model tag、checkpoint、dataset、seed、题数一致
- prompt version 为 `default`
- 主预算、answer-fix 和 `max_model_len` 分别为 32768、2048、37888
- 采样来自对应 checkpoint 的 `generation_config.json`
- `answers.json` 完整落盘；中断且无完整文件的半批不能复用

## 6. PUMA

### 6.1 正式算法

PUMA 使用同一条 Full-CoT：

1. 对推理 step 做 embedding redundancy filter。
2. 在保留的 step 上生成 trial answer。
3. 根据置信度和连续条件选停止/压缩位置。
4. 从 PUMA 选定前缀重新生成交付答案。

这不是简单截断 Full-CoT；第 4 步是模型重新生成，因此必须继续使用对应模型自己的
交付采样配置。每题实际生成上限必须是：

```text
min(4096, 32768 - reasoning_prefix_tokens)
```

不能对每个前缀无条件再给 4096 token。

### 6.2 全模型共同参数

- `CONFIDENCE_THRESHOLD=0.98`
- `EPSILON=0.03`
- `CONSECUTIVE=2`
- `MIN_STOP_STEP=10`
- embedding model：
  `qwen3-embedding-redundancy-detector-0.6B`
- `SIMILARITY_THRESHOLD=0.35`
- `CONSECUTIVE_REDUNDANCY_MIN_STEP=50`
- `FORCED_STOP_MIN_CONFIDENCE=0.8`
- trial：最多 30 token，sampling，`temperature=0.6`，`top_p=0.95`
- final regeneration：内部上限 4096 token，同时受统一 host 剩余预算约束

各 conf 的 `CONSECUTIVE_REDUNDANCY_STOP`：

| conf | 值 |
|---|---:|
| `DS-7B.conf` | 1 |
| `DS-14B.conf` | 3 |
| `DS-32B.conf` | 4 |
| `Nemotron.conf` | 1 |
| `Q30B-T.conf` | 0，关闭该 forced stop |

云端必须通过 `plws.puma_official_conf` 物化 `_local.conf`，把 embedding checkpoint
改为云端 `PLWS_MODELS_ROOT` 下的本地路径，并把 seed 写入配置。禁止按正式测试集
另选 PUMA 参数。

### 6.3 复用与入口

单格入口：

```text
scripts/run_puma_official.sh
```

该入口当前能运行既有 PUMA 流水，但 `gen_prefixed_answers.py` 仍把同一个
`max_tokens=4096` 用于整个 batch，尚未逐题执行上述剩余预算上限。新产物进入公平
主表前必须先补这个 guard；历史产物至少验证每题
`reasoning_prefix_tokens + count_generated_tokens <= 32768`。

PUMA 可以直接复用规范 Full-CoT。已有 PUMA cell 只有同时存在完整
`statistics.json`、`prefixed_answers.json`、输入 Full-CoT 哈希和实际 `_local.conf`
哈希一致时，才算可复用。早期 manifest 若缺协议、采样或预算字段，须先补验证，
不能仅凭目录存在进入主表。

## 7. DEER

### 7.1 正式算法必须在线

正式 DEER 状态机：

1. 主推理生成到 `Wait` ATP。
2. 临时诱导 trial answer，probe 不直接写入主上下文。
3. 若置信度严格大于 0.95，进入交付答案阶段。
4. 若未通过，把 `Wait` 插回主上下文并继续生成。
5. 最多检查 10 次；达到 DEER 自己的检查上限不等于耗尽 host，仍可用 32K 主预算
   的剩余部分完成回答。

由于第 4 步改变后续上下文，普通 Full-CoT 的后续 token 不再是正式 DEER 后续轨迹。
所以 Full-CoT 上的 Wait probe 重放只能标成 `DEER offline diagnostic/ablation`。

### 7.2 固定内部参数

- ATP：`Wait`
- `threshold=0.95`，判断为严格 `conf > 0.95`
- `max_judge_steps=10`
- `prob_check_max_tokens=20`
- probe decoding：greedy
- `points=1`

standard profile，适用于 DeepSeek-R1-Distill、Nemotron 和 QwQ：

- `think_ratio=0.6`
- confidence policy：`avg1`，答案 token 概率算术平均
- 不要求 probe 生成 `</think>`

qwen3 profile，适用于所有 `qwen3_*`，包括 Qwen3-30B-Thinking：

- `think_ratio=0.8`
- confidence policy：`avg2`，几何平均
- 只有 probe 确实生成 `</think>` 才允许置信门通过

### 7.3 复用与入口

单格入口：

```text
scripts/run_deer_official.sh
```

正式产物根：

```text
results/baselines/deer/puma_fullcot_32k_v2/
```

已经跑过的 DEER 可以复用，但必须逐 cell 验证 manifest 中的：

- protocol、model、dataset、seed 和题数
- 32768/2048/37888 host 预算
- `prompt_version=default`
- checkpoint generation sampling
- standard/qwen3 profile、threshold、max checks 和 probe decoding
- 每行没有错误占位，且每行 family policy 与 manifest 一致

“aligned v2”只是该产物目录的历史简称，正式文档和表头应写
`DEER (puma-fullcot-32k-v2)`，不要把 “aligned v2” 当成新算法名。

## 8. Answer Convergence

### 8.1 正式 offline replay

该方法本来就是在完整 CoT 的累计句子前缀上做分析，因此直接复用规范 Full-CoT：

1. 用固定 NLTK punkt 版本切分 `reasoning`，保留源文本。
2. 构造每个累计句子前缀。
3. 在每个前缀后追加 end-of-reasoning/boxed answer suffix。
4. greedy 生成 answer probe。
5. 提取规范答案；连续 10 个 probe 的答案字符串完全相同就停止。
6. 若直到最后一个句子前缀仍未收敛，使用最后一个完整前缀的 greedy probe。

固定参数：

- `k=10`
- probe `temperature=0`
- probe 最多 100 token
- `stop=["\n"]`
- `presence_penalty=1.0`
- 判等：答案抽取后的精确字符串相等

这里不能把 `k` 改成与 PLWS 一样的 4；`k=10` 是 Answer Convergence 的方法参数。

### 8.2 复用与入口

单格入口：

```text
scripts/run_answer_convergence_cell.sh
```

它可直接复用规范 Full-CoT，不复用 PUMA trial，因为句子切点、suffix、greedy 解码和
连续窗口都不同。云端要显式设置 `PY`、`MODEL`、`MODEL_TAG`、`DATASET`、`SEED`、
`GPU` 和 `OUT`，不要依赖 wrapper 中本机 venv 的默认路径。

已有结果只有 `manifest.json`、`final_answers.jsonl`、输入 sample 哈希、NLTK
资源版本和题数都一致时才能复用。

官方实现固定到：

```text
launchnlp/reasoning_earlystop
commit 0ff83811e409bc28cf2e06b5cd51ac4eac553f80
```

## 9. Dynasor

### 9.1 本项目采用 frozen-trajectory offline replay

Dynasor 每生成固定数量的 proposal token 就分叉做一次 probe；probe 结果不写回
proposal 上下文。因而可把规范 Full-CoT 当作 proposal trajectory：

1. 用原 checkpoint tokenizer 对 `reasoning` 重新编码。
2. 在 token 位置 64、128、192……构造累计前缀；自然结束后的边界不再 probe。
3. 每个边界附加官方 math probe suffix：

```text
... Oh, I suddenly got the answer to the whole problem, **Final Answer**

\[ \boxed{
```

4. probe 最多 20 token，使用该模型 `generation_config.json` 的
   `temperature/top_p/top_k`。
5. 从第一个未配对 `}` 前提取候选答案。
6. 候选必须非空；完整 probe 文本不能包含
   `wait/hold/but/okay/no/hmm`，不区分大小写。
7. 最近 3 次候选都 certain，且用官方 `math_equal` 属于同一数学等价类时停止。
8. 早停时交付冻结前缀和最后一个 probe 答案；未早停时交付原 Full-CoT。

固定配置：

- effort：`mid`
- certainty threshold：3
- chunk size：64 token
- probe max tokens：20
- 数学判等：官方 `math_equal`，不能降级成字符串相等

GPQA 仍使用相同 64/3 日程和 certainty gate，但答案先统一抽取为
`A/B/C/D` 再判等。

### 9.2 为什么可复用 Full-CoT

Dynasor 的 probe 是旁路分叉，失败后直接丢弃；它不像 DEER 那样向主上下文插入
额外 token。只要：

- Full-CoT 的 prompt、checkpoint、seed 和采样就是统一 host 版本；
- chunk 按 token ID 而不是字符长度切；
- probe 不污染 frozen trajectory；
- 前缀后不重置 32K 主预算；

offline replay 给出的停止位置与同一条 proposal trajectory 的在线执行一致。云端
实际运行可以因 batching/RNG 调度采到另一条同分布轨迹，因此逐题文本未必与另一次
online run 相同；这不影响本项目把 frozen-trajectory replay 定义为主比较实现，但
manifest 必须明确写 `execution_mode=frozen_trajectory_offline_replay`，不能冒充
逐请求在线复现。

### 9.3 当前入口状态

官方参考固定到：

```text
hao-ai-lab/Dynasor
commit 0d2f1b93a60e031cfc7fb43843d2b736b225960c
```

当前 `$PUMA_ROOT/baselines/run_dynasor.py` 和
`$PUMA_ROOT/baselines/dynasor/eval_dynasor.py` 只可作开发参考，不是正式入口。
正式入口应新增为：

```text
baselines/dynasor/run_cell.sh
baselines/dynasor/runner.py
```

在 runner 完成并通过 prompt、token boundary、`math_equal`、预算和 manifest 测试
前，Dynasor 产物不得进入主表。

## 10. PLWS

正式 PLWS 只有一套：

- method：`firstwin`
- `k=4`
- lexicon：`core`
- 锁点：每步试答出现第一扇连续 4 步同答窗，使用窗末前缀
- high/mix/low 只是窗的把握标签，不是三个方法
- 无窗题直接交付 Full-CoT
- 有窗题从窗末继续，主预算只剩 `32768 - prefix_tokens`
- continuation 中通过统一 `suppress_bad_words("core")` 抑制 CORE
- 交付推理和最终答案都读取模型自己的 generation config
- 只有真正耗尽主预算而未收口时才使用 2048 answer-fix

PLWS 可以复用：

- 规范 Full-CoT
- 同 cell 的 PUMA trial/dense probe
- 已导出的 firstwin jobs
- 完整且 manifest 一致的 leftover shard

PLWS 不能复用：

- 其它 seed 的 seed-42 平铺产物
- 没有第一扇窗却人工构造的前缀
- `leftover_suppress` 旧名下语义不明或非 CORE 的结果
- `results/archive/` 中的历史结果作为新主表来源

## 11. 复用矩阵

| 方法 | 规范 Full-CoT | PUMA trial/dense | 自身旧结果 | 正式执行形态 |
|---|---|---|---|---|
| PUMA | 直接复用 | 自己生成/续用同配置 trial | manifest 与配置齐全才复用 | Full-CoT 后压缩并重生成 |
| DEER | 不复用为正式结果 | 不复用 | 合法 online cell 可复用 | online 状态机 |
| Answer Convergence | 直接复用 | 不复用 | 输入哈希与参数齐全才复用 | frozen CoT 句子前缀重放 |
| Dynasor | 直接复用 | 不复用 | canonical runner 产物才复用 | frozen CoT token-chunk 重放 |
| PLWS | 直接复用 | 直接复用同 cell probes | 合法 shard 可续用 | 第一扇窗后 CORE suppression |

## 12. 计量口径

主报告至少同时保留：

1. `delivery_tokens`：用户真正收到的 reasoning + final answer。
2. `probe_trial_tokens`：PUMA/PLWS trial、DEER probe、Answer Convergence probe、
   Dynasor probe。
3. `total_method_tokens`：该方法实际需要生成的全部 token；不能漏掉被丢弃的 probe。
4. `latency`：主生成和 probe/trial 分项记录。

当前主表历史列名采用 `delivery_plus_boxed_trial_v1` 时，不得与只算 delivery 的外部
数字混报。offline replay 为准备轨迹而预先跑完整 Full-CoT 的实际云端成本另记为
`replay_preparation_cost`；算法模拟成本只累计到虚拟停止点的 proposal token 和所有
已执行 probe。论文中必须同时说明两者，避免把离线预计算成本隐藏成零。

Acc 使用同一 grader。所有模型、方法、seed 必须按预先声明的规则聚合；禁止按正式
测试集为每个方法挑最好 seed 或参数。

## 13. 云端产物与 manifest 最低要求

每个 cell 的 manifest 至少记录：

- schema version、method、execution mode、method source commit
- protocol ID
- model tag、精确 checkpoint 路径或 revision、tokenizer revision
- dataset、split、dataset SHA256、seed、题数
- prompt version、task instruction、chat-template family、thinking 开关
- delivery sampling 的 `temperature/top_p/top_k`
- 每个 probe/trial 阶段的完整采样参数
- 32768/2048/3072/37888 四个 host 数值
- 方法内部参数和所有本项目适配项
- 输入 Full-CoT/PUMA/jobs 的路径与 SHA256
- TP、dtype、vLLM 版本、GPU 型号只作部署审计，不作为算法配置
- 每题 delivery/probe token、停止原因和 cell 完整度

缺上述身份字段的历史产物可以保留，但必须先由验证脚本结合机器产物补齐证据，不能
直接进入公平主表。

## 14. 云端部署约束

- 模型统一放在云端 `PLWS_MODELS_ROOT`，目录名必须与第 4 节一致。
- 使用项目独立 uv venv，不把云端依赖装进其它项目环境。
- 任何长任务都在 tmux 中启动。
- 动态领卡；TP 由部署 profile 给：48 GB 卡上 32B/30B 为 2，A800 80 GB 为 1。
- 同一台使用 HDD 权重的机器同一时间只允许一个新 vLLM 引擎冷加载。
- `gpu_memory_utilization`、`max_num_seqs` 和 batch size 可按硬件调整，但必须记录；
  不得修改 prompt、采样、预算或方法阈值来解决 OOM。
- generate 完成后立即显式 shutdown vLLM，再做 CPU 汇总和写盘。
- 完整性以规范产物和 manifest 为准，不以 wrapper 退出码或 tmux 是否还在为准。

## 15. 正式入口清单

| 对象 | 现行入口 | 计划入口 | 状态 |
|---|---|---|---|
| Full-CoT | `scripts/run_contest_prereq_cell.sh` → `run_puma_aligned_sample.sh` | 保持现有 | 已有 |
| PUMA | `scripts/run_puma_official.sh` | 保持现有 | 已有；新跑前补逐题 remaining-budget guard |
| DEER | `scripts/run_deer_official.sh` | `baselines/deer/run_cell.sh` | 已有，已有部分合法结果 |
| DEER 官方对照 | `scripts/run_deer_github_official*.sh` | `baselines/deer/run_upstream_cell.sh` | 仅本机；写死 `repos/DEER`，云端不可用 |
| Answer Convergence | `scripts/run_answer_convergence_cell.sh` | `baselines/answer_convergence/run_cell.sh` | 已有；现行 wrapper 写死本机 venv |
| Dynasor offline | 无 | `baselines/dynasor/run_cell.sh` | 待实现和验证 |
| PLWS | `scripts/run_contest_plws_cell.sh`、`scripts/score_leftover_suppress.py` | 保持现有 | 已有 |
| 多卡填格 | `scripts/run_contest_fill_queue.py` | 不变 | 已有 |

三种额外基线的计划入口落地时，老脚本保留为转发 shim；现有 Full-CoT、PUMA、
PLWS 的代码和正在运行的队列路径不移动、不复制。

本文只锁定“每个 cell 怎么跑、用什么配置、什么能复用”；运行顺序、卡池和调度队列
另行制定。

## 16. 仓库自洽与目录规范

云端只 clone 本仓库，因此判据是：**除了模型权重和 `samples/` / `results/`，
不允许依赖任何本机才有的目录。**

现有两种实现保持原位：

1. PLWS 完整实现在 `src/plws/` 与 `scripts/`，它是本方法，不是 `baselines/`
   下面再复制的一种基线。
2. PUMA 由 `scripts/bootstrap_puma.sh` 从固定 commit 重建，再应用
   `vendor/patches/puma-fullcot-32k-v2.patch` 和本仓库锁定数据；不复制第二份源码。

Full-CoT 是 PUMA 生成的统一参考轨迹，不另建源码目录。注意
`results/baselines/puma/` 是机器产物路径，与源码目录 `baselines/` 含义不同。

只有 DEER、Answer Convergence、Dynasor 三种额外对比基线按下列骨架收口：

```text
baselines/<method>/
  README.md        # 算法、固定内部参数、复用边界、产物根、入口
  UPSTREAM.toml    # 上游 url + commit + patch（纯自研写 none）
  run_cell.sh      # 唯一单格入口
  runner.py        # 适配层，import plws.*
```

共享不变量只有一份，优先复用 `src/plws/`；确有多个额外基线共用的 shell 逻辑
时才建 `baselines/_common/`。方法目录内不得复制：host 预算常量与断言、
prompt/chat template 构造、交付采样读取、manifest schema、数据路径与 SHA256、
答案抽取与 grader。复制其中任一项即视为破坏可比性，不论当时数值是否相同。

已知依赖关系（避免重复 clone）：

- 正式 DEER 跑在打过补丁的 PUMA `baselines/deer/`，不需要第二份 DEER checkout。
- Dynasor 的 `dynasor_utils` 与官方 `math_equal` 都在 PUMA pin 的 tracked 文件里，
  也不需要另 clone。
- 官方 DEER（16K greedy 对照）需要 `iie-ycx/DEER` `c9dd19f`，只服务独立对照表。

## 17. 部署 profile 与算法配置分离

`PLWS_DEPLOY_PROFILE` 选择 `configs/deploy/<profile>.toml`。两份目标 profile：
`local_48g`（本机 11 模型）与 `a800_80g`（云端只跑 `qwen3_30b_a3b`、`r1_32b`、
`qwen3_32b`、`qwq_32b`）。

| profile 可以规定 | profile 不得触碰 |
|---|---|
| `models_root`、python、`PUMA_ROOT` | 32768 / 2048 / 3072 / 37888 |
| GPU 池、模型白名单 | prompt 版本与数据集 instruction |
| 每模型 tensor parallel size | 交付推理与终答采样（只能来自 checkpoint） |
| `gpu_memory_utilization`、`max_num_seqs`、batch | 各方法内部阈值、连续次数、检查上限 |
| 冷加载串行开关 | grader、Acc 与 token 定义 |

因此 manifest 分两块：

- `host_protocol`：两边逐字段相同，不同即不得同表。
- `deployment`：TP、dtype、`gpu_memory_utilization`、vLLM 版本、GPU 型号，
  只作部署审计，允许不同。

硬约束：不得用改 prompt、采样、预算或方法阈值的方式解决 OOM；装不下就提高 TP 或
降低并发。`configs/models.toml` 只保留模型身份（目录名、family），TP 归 profile。
