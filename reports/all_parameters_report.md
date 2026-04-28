Now I have all the data. Let me compile the full parameter documentation.

---

# 项目全部参数配置整理

## 一、候选策略配置总览

| 配置名 | EMA | Pivot min | Pivot max | RR | 数据文件 | 备注 |
|-------|-----|-----------|-----------|-----|----------|------|
| `aggressive_ema34_A` | 34 | 3 | 20 | 2.2 | BTCUSDT_5m_2024.csv | 激进型，低EMA |
| `stable_ema34_B` | 34 | 3 | 20 | 2.0 | BTCUSDT_5m_2024.csv | 稳健型，较低RR |
| `robust_ema55` | 55 | 3 | 20 | 2.2 | BTCUSDT_5m_2024.csv | 稳健型，高EMA |
| `robust_ema55_pivot_wide_9_35` | 55 | 9 | 35 | 2.2 | BTCUSDT_5m_2024_2025.csv | 宽pivot，**缺 atr_period** |

---

## 二、每个策略的完整参数明细

### 2.1 aggressive_ema34_A

```yaml
# --- Stochastic ---
k_period: 14
d_period: 3
smooth: 1
oversold: 15
overbought: 85

# --- Pivot ---
left_bars: 4
right_bars: 3
min_separation: 3
max_separation: 20
strict: true

# --- Trend ---
ema_period: 34
lookback_bars: 24
min_close_ratio: 0.80

# --- Risk ---
atr_period: 14
stop_buffer: 0.0
rr_target: 2.2

# --- Setup ---
setup_max_bars: 12
replace_same_side_setup: true
invalidate_on_stop_anchor_break: true

# --- Backtest ---
initial_cash: 100000
fee_per_trade: 0.0
slippage: 0.0
allow_short: true

# --- Data ---
data_path: data/raw/BTCUSDT_5m_2024.csv
```

---

### 2.2 stable_ema34_B

```yaml
# --- Stochastic ---
k_period: 14
d_period: 3
smooth: 1
oversold: 15
overbought: 85

# --- Pivot ---
left_bars: 4
right_bars: 3
min_separation: 3
max_separation: 20
strict: true

# --- Trend ---
ema_period: 34
lookback_bars: 24
min_close_ratio: 0.80

# --- Risk ---
atr_period: 14
stop_buffer: 0.0
rr_target: 2.0          # <-- 唯一与 A 不同的参数

# --- Setup ---
setup_max_bars: 12
replace_same_side_setup: true
invalidate_on_stop_anchor_break: true

# --- Backtest ---
initial_cash: 100000
fee_per_trade: 0.0
slippage: 0.0
allow_short: true

# --- Data ---
data_path: data/raw/BTCUSDT_5m_2024.csv
```

---

### 2.3 robust_ema55

```yaml
# --- Stochastic ---
k_period: 14
d_period: 3
smooth: 1
oversold: 15
overbought: 85

# --- Pivot ---
left_bars: 4
right_bars: 3
min_separation: 3
max_separation: 20
strict: true

# --- Trend ---
ema_period: 55         # <-- 与 ema34 系列不同
lookback_bars: 24
min_close_ratio: 0.80

# --- Risk ---
atr_period: 14
stop_buffer: 0.0
rr_target: 2.2

# --- Setup ---
setup_max_bars: 12
replace_same_side_setup: true
invalidate_on_stop_anchor_break: true

# --- Backtest ---
initial_cash: 100000
fee_per_trade: 0.0
slippage: 0.0
allow_short: true

# --- Data ---
data_path: data/raw/BTCUSDT_5m_2024.csv
```

---

### 2.4 robust_ema55_pivot_wide_9_35

```yaml
# --- Stochastic ---
k_period: 14
d_period: 3
smooth: 1
oversold: 15
overbought: 85

# --- Pivot ---
left_bars: 4
right_bars: 3
min_separation: 9         # <-- 更宽的 pivot 间距下限
max_separation: 35        # <-- 更宽的 pivot 间距上限
strict: true

# --- Trend ---
ema_period: 55
lookback_bars: 24
min_close_ratio: 0.80

# --- Risk ---
stop_buffer: 0.0
rr_target: 2.2
# ⚠️ 缺少 atr_period（源码默认值 14）

# --- Setup ---
setup_max_bars: 12
replace_same_side_setup: true
invalidate_on_stop_anchor_break: true

# --- Backtest ---
initial_cash: 100000
fee_per_trade: 0.0
slippage: 0.0
allow_short: true

# --- Data ---
data_path: data/raw/BTCUSDT_5m_2024_2025.csv
```

---

## 三、全局固定参数（所有策略共享）

以下参数在 4 个 yaml 中值完全相同：

| 分组 | 参数 | 固定值 |
|------|------|--------|
| stochastic | `k_period` | 14 |
| stochastic | `d_period` | 3 |
| stochastic | `smooth` | 1 |
| stochastic | `oversold` | 15 |
| stochastic | `overbought` | 85 |
| pivots | `left_bars` | 4 |
| pivots | `right_bars` | 3 |
| pivots | `strict` | true |
| trend | `lookback_bars` | 24 |
| trend | `min_close_ratio` | 0.80 |
| risk | `stop_buffer` | 0.0 |
| setup | `setup_max_bars` | 12 |
| setup | `replace_same_side_setup` | true |
| setup | `invalidate_on_stop_anchor_break` | true |
| backtest | `initial_cash` | 100000 |
| backtest | `fee_per_trade` | 0.0 |
| backtest | `slippage` | 0.0 |
| backtest | `allow_short` | true |

**结论：18 个参数中，14 个在所有策略间完全一致。**

---

## 四、各策略差异对比（高亮不同之处）

| 参数 | aggressive_ema34_A | stable_ema34_B | robust_ema55 | robust_ema55_pivot_wide_9_35 |
|------|:-:|:-:|:-:|:-:|
| `trend.ema_period` | **34** | **34** | **55** | **55** |
| `pivots.min_separation` | 3 | 3 | 3 | **9** |
| `pivots.max_separation` | 20 | 20 | 20 | **35** |
| `risk.rr_target` | 2.2 | **2.0** | 2.2 | 2.2 |
| `risk.atr_period` | 14 | 14 | 14 | **⚠️ 缺失** |
| `data.path` | `...2024.csv` | `...2024.csv` | `...2024.csv` | **`...2024_2025.csv`** |

**关键差异总结：**

- **EMA 维度**：ema34 系列（A + B） vs ema55 系列（robust + wide）——趋势过滤灵敏度不同，ema34 更灵敏、ema55 更平滑
- **Pivot 间距维度**：标准间距 (3-20) vs 宽间距 (9-35) ——wide 版本要求 pivot 之间至少相隔 9 根 K 线，更严格地筛选背离
- **RR 维度**：仅 stable_ema34_B 使用 2.0，其余均为 2.2 ——B 版本更保守
- **风险提示**：`robust_ema55_pivot_wide_9_35` 的 yaml 缺少 `risk.atr_period` 字段，运行时 signal_builder.py:68 会触发 `KeyError`

---

## 五、Python 源码中的隐含参数

以下参数在源码中硬编码了默认值，但不出现在 yaml 中（或 yaml 中的值等于源码默认值）：

### 5.1 signal_builder.py — setup 生命周期默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `setup_max_bars` | `:56` | `12` | 等待触发的最大 K 线数，超时失效 |
| `replace_same_side_setup` | `:57` | `True` | 新 setup 是否覆盖同向旧 setup |
| `invalidate_on_stop_anchor_break` | `:58` | `True` | stop_anchor 被击穿时是否废止 setup |

### 5.2 features/indicators.py — 指标默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `stochastic.k_period` | `:24` | `14` | Stochastic %K 周期 |
| `stochastic.d_period` | `:25` | `3` | Stochastic %D 平滑周期 |
| `stochastic.smooth` | `:26` | `1` | raw %K 的滚动均值窗口 |
| `atr.period` | `:9` | `14` | ATR 计算周期 |

### 5.3 features/trend_filter.py — 趋势过滤默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `lookback_bars` | `:25` | `20` | prior trend 回看 K 线数（yaml 中设为 24） |
| `min_close_ratio` | `:26` | `0.6` | prior trend 判定所需的最小比例（yaml 中设为 0.80） |

### 5.4 features/divergence.py — 背离检测默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `pivots.strict` | `:94` | `True` | pivot 检测是否严格模式（必须严格大于/小于邻域） |

### 5.5 strategy/risk_model.py — 止损默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `buffer` | `:4, :9` | `0.0` | stop buffer 比例（percentage-based） |

### 5.6 backtest/custom_engine.py — 回测默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `initial_cash` | `:20` | `100000` | 初始资金 |
| `qty` | `:21` | `1.0` | **固定交易数量（不同于 vectorbt 引擎的百分比模式）** — yaml 中无此参数 |

### 5.7 backtest/event_engine.py — 回测默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `initial_cash` | `:29` | `100000` | 初始资金 |
| `fee_per_trade` | `:30` | `0.0` | 每笔手续费 |
| `slippage` | `:31` | `0.0` | 滑点 |
| `allow_short` | `:32` | `True` | 是否允许做空 |

### 5.8 strategy/cost_model.py — 成本模型默认值

| 参数 | 代码位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `fees` | `:2` | `0.0` | 手续费（与 backtest 中的 fee_per_trade 是独立字段） |
| `slippage` | `:2` | `0.0` | 滑点 |

---

### 5.9 硬编码且完全不可配置的逻辑参数

以下逻辑在源码中直接写死，无法通过 yaml 调整：

| 逻辑 | 代码位置 | 硬编码值/规则 |
|------|---------|--------------|
| 背离确认 bar 位置 | `divergence.py:142` | `confirm_pos = pos2 + right_bars`（pivot2 确认后 right_bars 根 K 线确认） |
| Bullish 背离条件 | `divergence.py:141` | `low[idx2] < low[idx1]`（价格 lower low）+ `osc[idx2] > osc[idx1]`（osc higher low）+ `osc[idx1] <= oversold` |
| Bearish 背离条件 | `divergence.py:203` | `high[idx2] > high[idx1]`（价格 higher high）+ `osc[idx2] < osc[idx1]`（osc lower high）+ `osc[idx1] >= overbought` |
| Bullish trigger_price | `divergence.py:165` | `high.loc[idx2]`（pivot2 处 K 线的最高价） |
| Bearish trigger_price | `divergence.py:227` | `low.loc[idx2]`（pivot2 处 K 线的最低价） |
| Bullish stop_anchor | `divergence.py:166` | `low.loc[idx2]`（pivot2 处 K 线的最低价） |
| Bearish stop_anchor | `divergence.py:228` | `high.loc[idx2]`（pivot2 处 K 线的最高价） |
| Progressive pullback 通道检查 | `divergence.py:147-150, 209-211` | pivot1 → `inside_or_below_high`（bullish）/ `inside_or_above_low`（bearish）；pivot2 → `below_channel`（bullish）/ `above_channel`（bearish） |
| Setup 触发方向 | `signal_builder.py:134` | Long: `High > trigger_price`；Short: `Low < trigger_price` |
| Setup 失效 (stop_anchor) | `signal_builder.py:126,152` | Long: `Low < stop_anchor`；Short: `High > stop_anchor` |
| Stop 价格计算 | `signal_builder.py:137,163` | Long stop = `stop_anchor - stop_buffer`；Short stop = `stop_anchor + stop_buffer` |
| Target 价格计算 | `signal_builder.py:139,165` | Long target = `entry + rr_target * risk`；Short target = `entry - rr_target * risk`（risk = entry与stop的距离） |
| custom_engine 仓位管理 | `custom_engine.py:21` | **固定 qty=1.0**，不支持百分比仓位，不支持同时持有多仓 |
| 同 bar 不触发规则 | `signal_builder.py:116-117` | 背离确认 bar 只创建 setup，触发必须从下一根 K 线开始 |

---

### ⚠️ 已知问题

`robust_ema55_pivot_wide_9_35.yaml` 缺少 `risk.atr_period` 字段。`signal_builder.py:68` 直接通过 `risk_cfg["atr_period"]` 访问（不是 `.get()`），会触发 `KeyError`。虽然 `atr` 值当前仅用于 features 输出（不影响交易逻辑），但会导致该配置在调用 `build_signals()` 时崩溃。
