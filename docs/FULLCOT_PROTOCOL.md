# 唯一生成协议：PUMA Full-CoT 32K

本仓库所有 Full-CoT、PUMA、PLWS、DEER 公平对比和词表消融统一使用
`puma-fullcot-32k-v2`。不再把历史 pilot 的 16K context、锁窗后 1024
token 或 1024-token 终答当作正式协议。

四个基线与 PLWS 的逐方法、逐模型配置及复用边界见
[`BASELINE_ALIGNMENT_AND_CLOUD_RUNBOOK.md`](BASELINE_ALIGNMENT_AND_CLOUD_RUNBOOK.md)。

## 规范

| 项 | 唯一值 |
|---|---:|
| 原始 prompt 后的主生成预算 | 32768 token |
| 截断思考的 answer-fix | 2048 token |
| prompt 预留 | 3072 token |
| vLLM `max_model_len` | 37888 token |
| 交付推理/终答采样 | 模型 `generation_config.json`；7B/8B/14B 为 temperature 0.6、top-p 0.95、top-k 关闭，Qwen3 使用其自己的 top-k |
| Full-CoT prompt | PUMA `default` 的数据集专用 instruction + chat template；Nemotron 加 thinking system prompt，Qwen3 开启 thinking |

`max_model_len=32768+3072+2048`。3072 来自现有题集实际 prompt 审计：
最长为 2774 token，留有额外边界余量。未来
`scripts/run_puma_aligned_sample.sh` 对 `PUMA/puma/run_vllm.py` 的调用也使用
该值。

## 适用方法

以后进入公平主表的 PUMA、PLWS、DEER、Answer Convergence、ThinkBrake 及新增
方法，都必须保持本页的外部 host 条件不变。本项目不另开官方复现表。

### 同一赛道，各自跑法

公平比较统一的是外部实验条件，不是所有数值参数：

- 统一：checkpoint/tokenizer、数据与 seed、prompt/chat template、交付推理和
  终答采样、host token/context 上限、grader、统计口径。
- 保留：方法自己的退出信号、阈值、连续次数、检查频率与上限、`think_ratio`、
  probe/trial prompt 与解码、置信度聚合、模型族安全门。
- probe/trial 的解码可以不同于 host 采样，因为它属于方法机制；但其 token 和
  延迟必须单列，并进入总计算成本。
- 达到方法内部上限只代表主动退出，终答继续使用32K主预算余量；只有真正耗尽
  32K且未生成 `</think>`，才启用额外2048 answer-fix。
- 内部参数优先采用方法推荐配置；如需选择，只能在独立 dev set 上按预先声明的
  规则完成，然后冻结到全部正式测试集。

### DEER 的统一-host内部配置

- 公共 host：PUMA prompt、模型 generation config、32768主预算、2048截断修复、
  37888 context、相同 seed/grader。
- R1-Distill 与 Nemotron：`think_ratio=0.6`、算术平均置信度 `avg1`。
- Qwen3：`think_ratio=0.8`、几何平均置信度 `avg2`，probe 必须正常生成
  `</think>` 才允许置信退出。
- 全部模型：`threshold=0.95`、ATP=`Wait`、最多10次置信检查；probe 使用 greedy。

DEER 和 Answer Convergence 的旧 aligned 产物已经删除；统一-host代码与完成门
已经验证。对应全量重跑暂缓。之后新增、补格或重跑的 cell 必须使用本协议，不同
协议的产物不得混合汇总。

## PLWS 续写预算

PLWS 不是在锁窗后重新获得 32768 token。设原始 Full-CoT 从 prompt 开始
到锁窗前缀已经生成 `L` 个 token：

1. 压词续写最多生成 `32768-L` 个 token。
2. 若在该预算内自然生成 `</think>`，终答只能使用 32768 主预算中的剩余
   token。
3. 若用尽主预算仍没有 `</think>`，才与官方 Full-CoT 一样强制补
   `</think>`，并使用额外 2048-token answer-fix。
4. 禁止再设置独立的“剩窗 1024 token”上限。
5. 锁窗前缀、续写和终答都计入交付 token；探针成本另列，不混入交付
   token。

## 产物要求

正式 manifest 和分数行必须记录：

- `protocol_id=puma-fullcot-32k-v2`
- `fullcot_generation_tokens=32768`
- `truncated_answer_fix_tokens=2048`
- `max_model_len=37888`
- 是否自然收口、实际 answer budget、finish/stop reason

完成检查除题数外必须校验协议身份。旧产物只有在思考和终答均于旧上限前
自然结束时才可迁移为已验证；触碰旧上限的题必须删除并按本协议重跑。

## 旧协议处置

- Qwen3 的旧 PLWS 分数统一删除：旧 wrapper 漏传预算，实际继承了
  16384 context、1024 剩窗和 1024 终答，不能进入对比。
- 7B/8B/14B 旧 PLWS 分数全部删除并重跑。除 token 上限外，旧 scorer
  还误用了通用 instruction 和固定 `top_k=30`，因此即使没有触顶也不兼容。
- 历史 smoke、screen 和 full 词表消融输出删除；修复完成前不恢复消融。
- 旧 aligned DEER / Answer Convergence 输出已删除。两者 v2 runner、manifest
  和完成门已验证；DEER 另校验模型族内部策略，Answer Convergence 仅接受真实带
  v2 metadata 的 Full-CoT host，禁止给旧 sample 补标签冒充 v2。
