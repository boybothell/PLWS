# 四个大模型租卡缺格清单

机器产物快照：2026-09-16T13:53:09+08:00

## 范围与口径

- 模型：Qwen3-30B-A3B、R1-Distill-Qwen-32B、Qwen3-32B、QwQ-32B。
- 数据集：MATH-500、OlympiadBench、GPQA-Diamond、AIME24、AIME25、AIME26、BRUMO25、HMMT25、AMC23。
- seeds：42、0、1、123、7。每种方法总计 4 × 9 × 5 = 180 格。
- 公平宿主：`puma-fullcot-32k-v2`；Full-CoT 主预算 32768，answer-fix 2048，prompt 预留 3072，`max_model_len=37888`。
- PUMA、PLWS、DEER 分别检查规范产物；归档、旧协议和半截 `answers.json` 不算完成。

## 缺格总览

- Qwen3-30B-A3B：Full-CoT 宿主未齐 3/45；PUMA 缺 10/45；dense 缺 8/45；PLWS 缺 15/45；DEER 缺 45/45。
- R1-Distill-Qwen-32B：Full-CoT 宿主未齐 7/45；PUMA 缺 8/45；dense 缺 8/45；PLWS 缺 9/45；DEER 缺 45/45。
- Qwen3-32B：Full-CoT 宿主未齐 33/45；PUMA 缺 34/45；dense 缺 34/45；PLWS 缺 34/45；DEER 缺 45/45。
- QwQ-32B：Full-CoT 宿主未齐 43/45；PUMA 缺 43/45；dense 缺 43/45；PLWS 缺 43/45；DEER 缺 45/45。

## Qwen3-30B-A3B

### PUMA
- MATH-500：s0、s1、s123、s7（还缺 Full-CoT）
- OlympiadBench：s0、s1、s123、s7（还缺 Full-CoT）
- GPQA-Diamond：s123、s7（还缺 Full-CoT）

### PLWS
- MATH-500：s42（已写 0/328）、s0（已写 0/349）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- OlympiadBench：s42（已写 0/634）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- GPQA-Diamond：s42（已写 0/183）、s0（已写 0/187）、s1（已写 0/185）、s123（已写 0/184）、s7（还没有 jobs）

### DEER
- 全部缺：9 个数据集 × 5 seeds，共 45 格；seeds 为 42、0、1、123、7。

## R1-Distill-Qwen-32B

### PUMA
- MATH-500：s0、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- OlympiadBench：s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- GPQA-Diamond：s123（还缺 Full-CoT）、s7（还缺 Full-CoT）

### PLWS
- MATH-500：s1（已写 40/272）、s123（还没有 jobs）、s7（还没有 jobs）
- OlympiadBench：s0（已写 496/560）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- GPQA-Diamond：s123（还没有 jobs）、s7（还没有 jobs）

### DEER
- 全部缺：9 个数据集 × 5 seeds，共 45 格；seeds 为 42、0、1、123、7。

## Qwen3-32B

### PUMA
- MATH-500：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- OlympiadBench：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- GPQA-Diamond：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME24：s42（还缺 Full-CoT）、s1、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME26：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AMC23：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）

### PLWS
- MATH-500：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- OlympiadBench：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- GPQA-Diamond：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME24：s42（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME26：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AMC23：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）

### DEER
- 全部缺：9 个数据集 × 5 seeds，共 45 格；seeds 为 42、0、1、123、7。

## QwQ-32B

### PUMA
- MATH-500：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- OlympiadBench：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- GPQA-Diamond：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME24：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AIME26：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- BRUMO25：s42（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- HMMT25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）
- AMC23：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）、s123（还缺 Full-CoT）、s7（还缺 Full-CoT）

### PLWS
- MATH-500：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- OlympiadBench：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- GPQA-Diamond：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME24：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AIME26：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- BRUMO25：s42（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- HMMT25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）
- AMC23：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）、s123（还没有 jobs）、s7（还没有 jobs）

### DEER
- 全部缺：9 个数据集 × 5 seeds，共 45 格；seeds 为 42、0、1、123、7。

## 租卡执行方案

1. 先把本机 `samples/`、PUMA、dense、jobs、PLWS shard 和 DEER 规范目录同步到租卡机；以本清单对应 JSON 重扫，禁止按旧 `status.json` 直接重跑。
2. A800-80GB 先对每个 checkpoint 做单题 TP=1 预检：真实导入、`max_model_len=37888`、32K 生成和写盘都通过后才放全量。若某模型单卡因 KV 余量不足，只把该模型回退 TP=2，不整队统一 TP=2。
3. 第一批先收可续的 PLWS：R1-32B Olympiad s0 496/560、MATH s1 40/272；再收 Qwen3-30B 已有 jobs 的 7 格。这些格子不需要重做 Full-CoT。
4. 第二批补缺 Full-CoT/PUMA，并立即产出 dense、第一扇 k=4 窗口 jobs；同一格 jobs 一齐就进入 PLWS 动态池。
5. DEER 是独立方法池，四个模型当前 180 格全缺。保留各模型族自己的 think_ratio、置信聚合和退出机制；probe 成本单独记录。
6. 空闲 GPU 动态领下一格；所有模型冷加载整机串行。完整性看规范产物与 manifest，不看 wrapper 退出码。

## 本机分工

- 本机不再启动上述四个模型的 TP=2 任务；已有 R1-32B Olympiad s0 保留 496/560，MATH s1 保留 40/272。
- 当前只用空闲的 GPU 3、4 跑单卡模型；GPU 5、6 已被其他用户占用，不抢占。
- 第一优先：14B 主三集 seed 7。先并行续 GPQA 120/178 和 MATH 0/414，再补 Olympiad 的 Full-CoT/PUMA/PLWS。
- 第二优先：1.5B 主三集五 seeds；第三优先：Nemotron、4B、8B 主三集 seed 7；第四优先：Llama-8B 主三集与 AIME26/AMC23 缺格。
- 上述 PUMA/PLWS 收齐后，本机单卡继续补小模型 DEER；不把租卡上的四个大模型 DEER 混回本机。

机器可读清单：`manifests/large_model_rental_inventory.json`。
