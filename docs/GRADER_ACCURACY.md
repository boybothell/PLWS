# 判分后端与已落盘 Acc 核验

主表 Acc 是旗标均值：PUMA `statistics.json` 的 `original_correct` /
`compressed_correct`，以及窗后压 shard 的 `new_gold_ok`。旗标一旦写成
False，事后看不出来是「答错」还是「判分失败」。这次更新把判分失败从 Acc
里拆出去，并给出对**已落盘小模型格子**的核验入口。

换机器、重装环境、或要沿用历史 7B / 8B / 14B / 1.5B 分数时，先读本文再
改表。不要只看旧 `tables/`。

## 为什么要更新

2026-09-17 租卡核 R1-32B MATH-500 seed 0 窗后压时，报表给出 44.40%，同格
Full-CoT / PUMA 约 90%。答案文本大多对，错的是旗标。顺着这条往下挖，是
三个会把「判分失败」写成「答错」的洞，**不是 R1 独有，也不是 30B 才有**。
小模型走同一套 `PUMA/puma/math_grader.py`、同一套 jobs 导出、同一套
`score_leftover_suppress.py`。

### 1. LaTeX 后端缺失（主因，Acc 系统性偏低）

`check_is_correct` 在 `antlr4` / `latex2sympy2` 不在时不会报错，只是退回
字符串比较。下面这些数学上相等的答案会全部判错：

```text
\dfrac{19}{2}          vs  9.5
0.09                   vs  \frac{9}{100}
\dfrac{-p^2+1}{3}      vs  \frac{1-p^{2}}{3}
256(4-\pi)             vs  1024-256\pi
```

MATH / Olympiad / AIME / HMMT 这类格子受影响最大；GPQA 选择题几乎不受。
本机当时唯一的 `PLWS_PY` 缺这三包，本机判过的 Full-CoT / PUMA / 窗后压
一律偏低。迁过来的历史格也有同一类漏判，说明原机环境同样不完整。

租卡机 `uv pip install -e . --no-deps` **不会**自动装这三包，所以「代码
是新的、Python 里仍没有后端」会再发一次。

### 2. `multiprocessing.Pool` + 超时子进程

grader 的符号比较会再 spawn 一个 3 秒超时子进程。`Pool` 的 worker 是
daemonic，不允许再开子进程，抛 `AssertionError`。调用方原来写成
`except Exception: gold_ok = False`，于是难比较的题被静默算错。并行判
Full-CoT sample、DEER 回填都走这条。

### 3. jobs 无 gold 仍落盘，续跑按 UID 不重判

export 只从 PUMA `statistics.json` 或 `dense_g` 取 `ground_truth`。R1 的
`dense_g` 没有这个字段，PUMA 未齐就导出时 `gt=null`。score 对空 gold
走异常分支写成 `new_gold_ok=False`，续跑只认 UID + 协议，jobs 后来补上
gold 也不会改旧旗标。R1 MATH-500 seed 0 的 229 条属于这一类。

### 本机已核实的大模型数字（供对照）

在补齐后端后重判，只把 stored False 且现判 True 的旗标上调（见下方原则）：

| 旗标 | 受影响格 | 上调条数 |
|---|---:|---:|
| `original_correct` | 14 | 125 |
| `compressed_correct` | 15 | 126 |
| `new_gold_ok` | 8 | 51 |

mean@3 大约上浮 1.5–2.5pp（如 Qwen3-30B MATH-500 Full-CoT 94.47% →
95.93%，R1-32B MATH-500 Full-CoT 89.80% → 91.40%）。方法间相对高低大体
仍在，个别行胜者会翻转。QwQ-32B 在对面机，本机未改它的共享盘产物。

**小模型格子本机没有扫。** 拉到这份更新的人必须在自己的 `results/` 上跑
下一节，不能用大模型涨幅去外推 7B / 8B。

## 怎么更新（代码）

| 位置 | 作用 |
|---|---|
| `src/plws/grading.py` | 唯一判分入口。`require_grader()` 用只有符号后端才能过的等价对自检；空 gold 抛 `MissingGold`；`grade_many` 用 `ProcessPoolExecutor` |
| `pyproject.toml` | 钉死 `antlr4-python3-runtime==4.11.1`、`latex2sympy2==1.9.1`、`word2number`（跟上游 DEER） |
| `scripts/export_leftover_suppress_jobs.py` | 某集缺 gold 整集拒写，退出码非 0 |
| `scripts/score_leftover_suppress.py` | 开跑前自检；空 gold 拒跑；记录 `gt` / `gold_error`；jobs gold 变了则 CPU 重判旧 shard |
| `scripts/run_contest_fill_queue.py` | 占卡前 `require_grader()` |
| `scripts/report_fullcot_puma_plws.py` | 出表前自检；并行判分改 `ProcessPoolExecutor`（函数名 `grade_deer_item` 是历史名，Full-CoT sample 判分也走它） |
| `scripts/audit_grader_flags.py` | 重判已落盘旗标；默认只报；`--fix` 只升不降 |
| `tests/test_grading.py` | 把上面三个洞钉成回归 |

### 正式 fill 不再静默复用旧判分

`run_puma_official.sh` 无论刚生成还是复用已有 `statistics.json`，都会调用
`python -m plws.puma_grading --fix`。只有统一 grader 重判无错误、无
True→False 待人工项后，才写 `statistics.grader.json`。fill 的 prereq
完成判断要求该凭证与当前 statistics 的大小和 mtime 一致；对面服务器重写
statistics 后凭证会失效，该格会重新进入 prereq，而不是直接沿用旧 Acc。

窗后压也不再只在 gold 变化时重判。`score_leftover_suppress.py` 恢复已有
shard 时重判本 shard 全部答案，只安全提升 False→True；缺 `gt` /
`gold_error` 的旧脚本行不再算 protocol-valid 完成。判分异常或存量
True→False 会直接阻断，不自动降级。

`check_is_correct` 返回 True 是「两式相等」的正证据；返回 False 仍可能是
3 秒超时。因此 `--fix` **只把 False 升 True**，True→False 只进 `review`，
不自动改。

## 别人拉代码后先做什么

### 1. 装后端（`--no-deps` 的机器必做）

```bash
export PLWS_ROOT=/path/to/plws
export PUMA_ROOT=/path/to/PUMA
export PLWS_PY=/path/to/vllm-env/bin/python

"$PLWS_PY" -m pip install \
  "antlr4-python3-runtime==4.11.1" \
  "latex2sympy2==1.9.1" \
  word2number
```

自检必须过，过不了不要跑队列、不要改表：

```bash
PYTHONPATH="$PLWS_ROOT/src" "$PLWS_PY" -c \
  "from plws.grading import require_grader; require_grader(); print('grader ok')"
```

失败信息会点名哪组等价式判错。`scripts/preflight_rental_server.sh` 现在也
跑同一条自检。

### 2. 先报告、再决定是否改盘

默认只打印，退出码 1 表示有 stale，**不写盘**：

```bash
# 历史小模型（按本机实际有的 tag 删）
PYTHONPATH="$PLWS_ROOT/src" "$PLWS_PY" scripts/audit_grader_flags.py \
  --models r1_7b,nemotron_8b,r1_14b,r1_1p5b,r1_llama_8b,qwen3_4b,qwen3_8b \
  --datasets math-500,olympiadbench,gpqa-diamond,aime24,aime25,hmmt25,amc23 \
  --seeds 42,0,1,123,7

# 四个大模型
PYTHONPATH="$PLWS_ROOT/src" "$PLWS_PY" scripts/audit_grader_flags.py
```

读每一行：

| 输出 | 含义 | 怎么处理 |
|---|---|---|
| `OK` | 现 grader 与落盘旗标一致 | 不用动 |
| `promotable=N` | 当初写成错、现判对 | 确认环境自检已过后再 `--fix` |
| `review=N` | 当初写成对、现判错 | **不要 --fix 降级**；人工看答案。可能是超时，也可能是提取到了整段原文 |
| `no_gold=N` | 这一格没有可用 gold | 先补齐 PUMA `statistics.json` 再重导 jobs，不要在空 gold 上重打分 |
| `grader_err=N` | 单条比较崩溃 | 先修环境，不要当答错 |

确认 `require_grader` 已过、且 `promotable` 不是对面正在写的半截格，再改盘
（每文件留一份 `*.bak_grader_audit`）：

```bash
PYTHONPATH="$PLWS_ROOT/src" "$PLWS_PY" scripts/audit_grader_flags.py \
  --models r1_7b,nemotron_8b,r1_14b \
  --fix
```

改完必须再跑一遍不加 `--fix`，应看到 `0 stale`。然后再重跑
`report_fullcot_puma_plws.py` 出表。不要先出表再审计。

### 3. 不需要重跑 GPU 的情况

只修旗标：答案文本、token、窗位置都没变。`--fix` 是 CPU 重判。

需要重跑的只有：jobs 当时 `gt=null` 且你打算作废旧 shard、按新 jobs 重打
窗后压。缺 gold 的 jobs 现在 export 会拒写，PUMA 齐了会自动再导。

对面机子如果只采 Full-CoT / dense、不跑 PUMA official、不跑
`score_leftover_suppress.py`，那边环境缺包不影响；Acc 以本机（或任何已自检
的机器）事后审计为准。PUMA official **本身就会写** `original_correct` /
`compressed_correct`，那一步已经是打分。

## 共享盘注意

`samples/`、`results/` 若链到同一份 autodl-fs：只改本机负责的
`(model, dataset, seed)`。对面的半截 PUMA / shard 不要 `--fix`。QwQ 的格
留给跑 QwQ 的那台，或等那格写完再在已自检的机器上审计。
