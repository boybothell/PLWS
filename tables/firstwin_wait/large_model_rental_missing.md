# 四个大模型租卡缺格清单

机器产物快照：2026-09-16T16:22:50+08:00

## 范围与口径

- 模型：Qwen3-30B-A3B、R1-Distill-Qwen-32B、Qwen3-32B、QwQ-32B。
- 现行新跑：MATH-500、OlympiadBench、GPQA-Diamond、AIME25、HMMT25；第一波 seed 42、0、1。每种方法总计 4 × 5 × 3 = 60 格。
- 第一波齐了再排 seed 123、7。不再新开 AIME24、AIME26、BRUMO25、AMC23；已有产物保留，不删、不重跑、不入新队列。
- 传输包仍按历史 4 × 9 × 5 = 180 格核验，避免漏解已有前缀。
- 公平宿主：`puma-fullcot-32k-v2`；Full-CoT 主预算 32768，answer-fix 2048，prompt 预留 3072，`max_model_len=37888`。
- PUMA、PLWS、DEER 分别检查规范产物；归档、旧协议和半截 `answers.json` 不算完成。主表分数线先不动。

## 缺格总览（现行 5 集 × 3 seed）

- Qwen3-30B-A3B：Full-CoT 宿主未齐 0/15；PUMA 缺 4/15；dense 缺 3/15；PLWS 缺 9/15；DEER 缺 15/15。
- R1-Distill-Qwen-32B：Full-CoT 宿主未齐 1/15；PUMA 缺 2/15；dense 缺 2/15；PLWS 缺 3/15；DEER 缺 15/15。
- Qwen3-32B：Full-CoT 宿主未齐 12/15；PUMA 缺 12/15；dense 缺 12/15；PLWS 缺 12/15；DEER 缺 15/15。
- QwQ-32B：Full-CoT 宿主未齐 15/15；PUMA 缺 15/15；dense 缺 15/15；PLWS 缺 15/15；DEER 缺 15/15。

## Qwen3-30B-A3B

### PUMA
- MATH-500：s0、s1
- OlympiadBench：s0、s1

### PLWS
- MATH-500：s42（已写 0/328）、s0（已写 0/349）、s1（还没有 jobs）
- OlympiadBench：s42（已写 0/634）、s0（还没有 jobs）、s1（还没有 jobs）
- GPQA-Diamond：s42（已写 0/183）、s0（已写 0/187）、s1（已写 0/185）

### DEER
- 现行范围全部缺：5 个数据集 × 3 seeds，共 15 格；seeds 为 42、0、1。

## R1-Distill-Qwen-32B

### PUMA
- MATH-500：s0
- OlympiadBench：s1（还缺 Full-CoT）

### PLWS
- MATH-500：s1（已写 40/272）
- OlympiadBench：s0（已写 496/560）、s1（还没有 jobs）

### DEER
- 现行范围全部缺：5 个数据集 × 3 seeds，共 15 格；seeds 为 42、0、1。

## Qwen3-32B

### PUMA
- MATH-500：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- OlympiadBench：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- GPQA-Diamond：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- AIME25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）

### PLWS
- MATH-500：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- OlympiadBench：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- GPQA-Diamond：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- AIME25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）

### DEER
- 现行范围全部缺：5 个数据集 × 3 seeds，共 15 格；seeds 为 42、0、1。

## QwQ-32B

### PUMA
- MATH-500：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- OlympiadBench：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- GPQA-Diamond：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- AIME25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）
- HMMT25：s42（还缺 Full-CoT）、s0（还缺 Full-CoT）、s1（还缺 Full-CoT）

### PLWS
- MATH-500：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- OlympiadBench：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- GPQA-Diamond：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- AIME25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）
- HMMT25：s42（还没有 jobs）、s0（还没有 jobs）、s1（还没有 jobs）

### DEER
- 现行范围全部缺：5 个数据集 × 3 seeds，共 15 格；seeds 为 42、0、1。

## 租卡执行方案

1. 先把本机 `samples/`、PUMA、dense、jobs、PLWS shard 和 DEER 规范目录同步到租卡机；以本清单对应 JSON 重扫，禁止按旧 `status.json` 直接重跑。
2. A800-80GB 先对每个 checkpoint 做单题 TP=1 预检：真实导入、`max_model_len=37888`、32K 生成和写盘都通过后才放全量。若某模型单卡因 KV 余量不足，只把该模型回退 TP=2，不整队统一 TP=2。
3. 第一波只领现行五集的 seed 42、0、1。已齐格跳过；可续 PLWS 优先 R1-32B Olympiad s0 496/560、MATH s1 40/272，再收 Qwen3-30B 已有 jobs 且落在现行范围内的格子。这些格子不需要重做 Full-CoT。
4. 第二批补缺 Full-CoT/PUMA，并立即产出 dense、第一扇 k=4 窗口 jobs；同一格 jobs 一齐就进入 PLWS 动态池。
5. DEER 是独立方法池，只排现行 60 格。保留各模型族自己的 think_ratio、置信聚合和退出机制；probe 成本单独记录。
6. 空闲 GPU 动态领下一格；所有模型冷加载整机串行。完整性看规范产物与 manifest，不看 wrapper 退出码。
7. seed 123/7 等第一波齐了再开。AIME24 / AIME26 / BRUMO25 / AMC23 不入队。

## 本机分工

- 本机不再启动上述四个模型的 TP=2 任务；已有 R1-32B Olympiad s0 保留 496/560，MATH s1 保留 40/272，转租卡续。
- 已在飞的 14B Olympiad Full-CoT s7 等落盘，不再新开 seed 7。
- 本机新开只按现行五集、先三个 seed；AIME24 / AIME26 / BRUMO25 / AMC23 不再补。
- 不把租卡上的四个大模型 DEER 混回本机。

机器可读清单：`manifests/large_model_rental_inventory.json`。
