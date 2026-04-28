# TPE 第一阶段参数搜索可行性报告

> 生成时间：2026-04-28  
> 目标：在 18 参数策略空间中，通过两阶段 TPE 搜索定位鲁棒参数组合。

---

## 1. 18 参数方案完整清单

本项目策略模块由 `features/divergence.py`、`features/trend_filter.py`、`strategy/signal_builder.py` 与 `backtest/event_engine.py` 共同构成，共涉及 **19 个可调参数**（若将 3 个布尔/规则开关视为固定执行规则，则为 **18 个核心数值参数**）。

### 1.1 数值参数（16 个）

| 模块 | 参数 | 类型 | 默认值 | 物理意义 | Phase 1 |
|------|------|------|--------|----------|---------|
| `stochastic` | `k_period` | int | 14 | 随机指标 K 线周期 | ✅ 搜索 |
| `stochastic` | `d_period` | int | 3 | 随机指标 D 线周期 | ✅ 搜索 |
| `stochastic` | `smooth` | int | 1 | K 线平滑窗口 | ✅ 搜索 |
| `stochastic` | `oversold` | int | 20 | 超卖阈值 | ✅ 搜索 |
| `stochastic` | `overbought` | int | 80 | 超买阈值 | ✅ 搜索 |
| `pivots` | `left_bars` | int | 3 |  pivot 左侧确认 bar 数 | ✅ 搜索 |
| `pivots` | `right_bars` | int | 3 | pivot 右侧确认 bar 数 | ✅ 搜索 |
| `pivots` | `min_separation` | int | 3 | 两 pivot 最小间距 | ✅ 搜索 |
| `pivots` | `max_separation` | int | 20 | 两 pivot 最大间距 | ✅ 搜索 |
| `trend` | `ema_period` | int | 55 | EMA 通道周期 | ✅ 搜索 |
| `trend` | `lookback_bars` | int | 12 | 趋势回看 bar 数 | ❌ 固定 24 |
| `trend` | `min_close_ratio` | float | 1.0 | 趋势确认最小 close 比例 | ✅ 搜索 |
| `risk` | `atr_period` | int | 14 | ATR 计算周期 | ❌ 固定 14 |
| `risk` | `stop_buffer` | float | 0.0 | 止损缓冲点数 | ❌ 固定 0.0 |
| `risk` | `rr_target` | float | 2.0 | 目标风险收益比 | ✅ 搜索 |
| `setup` | `setup_max_bars` | int | 12 | setup 最大等待 bar 数 | ❌ 固定 12 |

### 1.2 布尔/规则参数（3 个）

| 参数 | 默认值 | 说明 | Phase 1 |
|------|--------|------|---------|
| `pivots.strict` | `True` | 严格 pivot（必须严格高低点） | ❌ 固定 |
| `setup.replace_same_side_setup` | `True` | 同向 setup 是否覆盖 | ❌ 固定 |
| `setup.invalidate_on_stop_anchor_break` | `True` | stop_anchor 击穿即失效 | ❌ 固定 |

**Phase 1 搜索参数合计：12 个；固定参数合计：7 个。**

---

## 2. 两阶段搜索策略设计

### Phase 1 — 核心参数锚定（本阶段）
- **搜索范围**：12 个核心数值参数（见上表）。
- **采样器**：Optuna `TPESampler(multivariate=True, seed=42, n_startup_trials=100)`。
- **剪枝器**：`MedianPruner`，要求每个 trial 按训练季度通过 `trial.report()` 汇报中间结果。
- **数据划分**：
  - 训练集：2024-Q1 ~ 2025-Q2（6 个季度，约 15.5 万根 5m K 线）
  - 验证集：2025-Q3（1 个季度，用于阶段 2 初筛）
  - 测试集：2025-Q4（hold-out，阶段 2 结束后最终评估）
- **评分函数**：多目标加权，避免单一指标过拟合：
  - `expectancy_R × 0.30`
  - `clipped_profit_factor × 0.25`（PF 封顶 5）
  - `quarterly_stability × 0.20`（正收益季度占比）
  - `trade_frequency_score × 0.10`（trades / 50 封顶）
  - `drawdown_penalty × -0.15`（`abs(max_drawdown) / 25`）
- **硬性过滤**：
  - `total_trades < 10` → 直接返回 `-9999`
  - 任意季度 `trades < 2` → 扣分
  - `PF > 5.0` 且 `trades < 30` → 扣分（疑似过拟合）
  - cost 后总收益为负 → 直接淘汰

### Phase 2 — 扩展与稳健性验证（后续）
1. **参数扩展**：释放 `lookback_bars`、`setup_max_bars`、`stop_buffer`、`atr_period`，甚至 `strict` 布尔开关，进行第二轮 TPE 局部搜索。
2. **Walk-forward**：采用滚动 6 季度训练 + 1 季度验证，验证参数时间稳定性。
3. **Cost 模式细化**：在 cost 与 no-cost 之间做 Pareto 分析，选择 cost 鲁棒且 no-cost 优异的组合。
4. **多品种外推**：将最优参数映射到 ETH、SOL 等品种，检验通用性。
5. **蒙特卡洛扰动**：对最优参数施加 ±5% 扰动，评估策略敏感度。

---

## 3. 计算量估算

### 3.1 单核估算

| 项目 | 估算 |
|------|------|
| 单 trial 平均耗时 | 0.8 ~ 1.5 s（含 6+1 个季度切片、信号构建、回测、评分） |
| `n_startup_trials=100` | 纯随机探索，TPE 尚未介入，耗时与常规 trial 相同 |
| 500 trials 总耗时 | **约 7 ~ 13 分钟** |
| SQLite I/O 开销 | 可忽略（本地文件，< 1 ms / trial） |

### 3.2 多核估算

Optuna 原生支持 `n_jobs` 并行。由于 trial 之间无状态依赖，可线性扩展：

| 核心数 | 预估耗时 | 说明 |
|--------|----------|------|
| 1 | ~10 min | 基准 |
| 4 | ~3 min | 推荐开发调试 |
| 8 | ~1.5 min | 推荐完整搜索 |
| 16 | ~1 min | 受 SQLite 写锁轻微限制 |

> 若使用 `JournalStorage`（文件锁）替代 `RDBStorage`，可进一步降低多核锁竞争。

---

## 4. 过拟合风险及缓解措施

### 4.1 主要风险源

| 风险 | 表现 | 根因 |
|------|------|------|
| 高 PF + 低样本 | 某季度 PF > 10，但 trades < 5 | pivot 间距过宽或阈值过于极端 |
| 参数耦合过拟合 | `oversold`/`overbought` 与 `k_period` 联动，只在特定行情生效 | 搜索空间存在强相关性 |
| 样本外崩塌 | 训练集 6 季度优异，验证集 2025-Q3 收益为负 | 对 2024 年行情过度拟合 |
| 幸存者偏差 | MedianPruner 剪掉大量差 trial，最终 Top-10 集中在局部极值 | 剪枝器本身不改变过拟合本质 |

### 4.2 缓解措施

1. **硬性过滤（Hard Thresholds）**
   - `total_trades < 10` → `-9999`，杜绝无统计意义的组合。
   - 单季度 `trades < 2` → 扣分，避免"靠运气"的季度。
   - `PF > 5.0` 且 `trades < 30` → 扣分，抑制极端 PF。
   - cost 后收益为负 → 直接淘汰，确保实盘可行性。

2. **复合评分（Composite Score）**
   - 不直接使用 `profit_factor` 或 `total_return` 作为目标，而是将 `expectancy_R`、`quarterly_stability`、`drawdown` 综合加权，降低单一指标操纵空间。

3. **季度级 MedianPruner**
   - 要求 trial 在 6 个训练季度逐季汇报，若早期季度已显著劣于历史中位数，则提前剪枝，减少无效计算。

4. **留一法验证（Hold-out）**
   - 2025-Q4 完全不参与搜索与剪枝，仅用于最终报告，作为样本外真实性能。

5. **Top-50 参数分布分析**
   - 输出 `tpe_top50.csv`，观察最优参数是否聚集在搜索边界；若边界聚集严重，则 Phase 2 需要扩展边界。

---

## 5. 本阶段固定参数说明

以下 7 个参数在 Phase 1 中固定为默认值，理由如下：

| 参数 | 固定值 | 理由 |
|------|--------|------|
| `strict` | `True` | 严格 pivot 已在前序批次中验证优于非严格模式 |
| `replace_same_side_setup` | `True` | 覆盖逻辑可减少 stale setup，属于执行规则而非调优对象 |
| `invalidate_on_stop_anchor_break` | `True` | 结构破坏即失效是策略核心逻辑，不宜开关 |
| `setup_max_bars` | `12` | 等待 12 根 5m K 线 ≈ 1h，已在前序实验中为合理中值 |
| `stop_buffer` | `0.0` | 先定位核心参数，buffer 在 Phase 2 作为微调项释放 |
| `lookback_bars` | `24` | 24 根 5m ≈ 2h，是趋势过滤的常用默认值 |
| `atr_period` | `14` | 经典 14 周期 ATR，Phase 2 可微调 |

---

## 6. 第二阶段扩展路径

### 6.1 参数扩展
- **释放数值参数**：`lookback_bars`（12~48）、`setup_max_bars`（6~24）、`stop_buffer`（0~2×ATR）、`atr_period`（10~20）。
- **释放布尔参数**：`strict`（True/False）、`replace_same_side_setup`、`invalidate_on_stop_anchor_break`。
- **扩展搜索边界**：若 Phase 1 最优值频繁落在边界（如 `oversold=10` 或 `rr_target=3.5`），Phase 2 应外扩边界并重新采样。

### 6.2 Walk-forward 验证
- 采用 6-Q 滚动窗口：训练 [T-6, T-1]，验证 [T]，测试 [T+1]。
- 对 Phase 1 Top-10 参数逐窗口执行 walk-forward，计算 **WF 成功率**（参数在 ≥70% 窗口中 cost 收益为正）。

### 6.3 Cost 模式细化
- Phase 1 已强制 cost 收益为正；Phase 2 可引入 **Cost-NoCost 散点图**，寻找帕累托前沿。
- 引入 `fee_per_trade` 与 `slippage` 的敏感性分析（±50% 扰动）。

### 6.4 多品种与多时间框架
- 将 Phase 1 最优参数映射到 ETHUSDT、SOLUSDT 的 5m/15m 数据。
- 若跨品种表现稳定（正收益季度占比 > 50%），则参数具备较高通用性。

### 6.5 输出交付物（Phase 2）
- `reports/tpe_phase2_summary.md`
- `reports/tpe_phase2_wf_report.csv`
- `reports/tpe_phase2_best_params.json`
- `reports/tpe_phase2_cost_sensitivity.csv`

---

## 7. 结论

Phase 1 的 12 参数 TPE 搜索在计算上完全可行（单核 < 15 min，8 核 < 2 min）。通过复合评分、硬性过滤、季度级剪枝与 hold-out 测试集的四层防护，可将过拟合风险控制在可接受范围。预期交付物可直接支撑 Phase 2 的扩展与稳健性验证。
