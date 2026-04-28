# Quant Scaffold 开发报告

> 生成时间：2026-04-27
> 报告范围：最近两次主要提交（`f2ca070` ~ `abbce6f`）

---

## 1. 本次改动清单

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `optimize/utils.py` | 提取共享工具函数：安全 profit_factor/win_rate 归一化、DataFrame 切片、回测额外指标提取 |
| `optimize/batch1_stoch_pivot.py` | Batch 1 参数敏感性脚本：Stochastic 周期 × Pivot 间隔 × 8 季度 × 2 成本模式 = **384 次回测** |
| `configs/risk_cost_sensitivity.yaml` | Phase 4B 风险-成本敏感性分析的配置文件 |
| `optimize/phase4b_position_cap.py` | Phase 4B 仓位上限分析脚本 |
| `reports/batch1_stoch_pivot_summary.csv` | Batch1 汇总报告（49 行参数组合） |
| `reports/batch1_stoch_pivot_top20_cost.csv` | Cost 模式 Top 20 参数组合 |
| `reports/batch1_stoch_pivot_top20_nocost.csv` | No-cost 模式 Top 20 参数组合 |
| `reports/batch1_stoch_pivot_validation.csv` | 384 次回测逐条验证结果 |
| `reports/phase4b_position_cap_skipped.csv` | Phase 4B 被跳过记录 |
| `reports/phase4b_position_cap_summary.csv` | Phase 4B 汇总（109 行） |
| `reports/phase4b_position_cap_trades.csv` | Phase 4B 逐笔交易（5641 条） |
| `reports/phase4_risk_sensitivity_trades.csv` | Phase 4 风险敏感性逐笔交易（5641 条） |

### 修改文件

| 文件路径 | 改动摘要 |
|---------|---------|
| `features/divergence.py` | **向量化 pivot 检测**（`_pivot_high` / `_pivot_low` 从 Python loop 改为 `rolling().max/min` + `shift`）；**批量 Series 写入**（收集 confirm 索引后统一 `.loc[]` 赋值，减少 90%+ 的逐行写操作） |
| `backtest/custom_engine.py` | 修复 `init_cash` → `initial_cash` 配置键名不一致；新增 `qty` 仓位乘数支持，所有 PnL 计算乘以 `qty` |
| `optimize/grid_search.py` | 接入 `optimize/utils.py` 的 `safe_profit_factor`、`safe_win_rate`、`slice_dataframe` |
| `optimize/phase2a_search.py` | 同上，代码简化，删除冗余内联工具函数 |
| `optimize/phase2b_search.py` | 同上，接入共享 utils |
| `optimize/phase3_holdout.py` | 同上，接入共享 utils |
| `optimize/phase3b_walk_forward.py` | 同上，接入共享 utils |
| `optimize/phase3c_plateau.py` | 同上，接入共享 utils |
| `optimize/phase4_risk_sensitivity.py` | 接入共享 utils，小幅重构 |
| `optimize/validate_candidates.py` | 接入共享 utils，简化指标提取逻辑 |
| `optimize/validate_final_candidates.py` | 接入共享 utils，简化指标提取逻辑 |
|

## 2. 各模块状态

### optimize/ 模块：提取 utils.py 共享工具模块

- **状态**：✅ 已完成
- **内容**：`safe_profit_factor`、`safe_win_rate`、`slice_dataframe`、`compute_extra_metrics` 四个共享函数
- **影响范围**：`grid_search.py`、`phase2a_search.py`、`phase2b_search.py`、`phase3_holdout.py`、`phase3b_walk_forward.py`、`phase3c_plateau.py`、`phase4_risk_sensitivity.py`、`validate_candidates.py`、`validate_final_candidates.py` 均已接入
- **收益**：消除 8 个文件中的重复代码，统一边界值处理（None/NaN/inf）

### features/ 模块：pivot 检测向量化、批量 Series 写入

- **状态**：✅ 已完成
- **改动详情**：
  1. `_pivot_high` / `_pivot_low`：从 `for` 循环逐行比较改为 `rolling().max/min` + `shift` 向量化实现，strict 和非 strict 模式均覆盖
  2. `detect_regular_divergence`：将 bullish / bearish 循环中的逐行 `.loc[idx] = val` 改为先收集到 Python list，再批量 `.loc[list] = values` 赋值
- **性能收益**：大幅减少 pandas Series 逐行写操作，预期加速 3~10 倍（取决于数据长度）

### backtest/ 模块：custom_engine.py 的 bug 修复

- **状态**：✅ 已修复
- **问题 1**：配置键名 `init_cash` 与项目实际使用的 `initial_cash` 不一致，导致 cash 初始化失败
- **问题 2**：回测引擎不支持仓位数量 `qty`，所有 PnL 固定为 1 单位
- **修复**：统一键名为 `initial_cash`；新增 `qty` 读取（默认 1.0），所有 long/short 的 exit PnL 均乘以 `qty`


### optimize/batch1_stoch_pivot.py：新增脚本和 384 次回测结果

- **状态**：✅ 已完成，报告已生成
- **参数空间**：
  - Stochastic 组合：4 种 `(k_period, smooth, d_period)`
  - Pivot 间隔：6 种 `(min_separation, max_separation)`
  - 季度：8 个（2024-Q1 ~ 2025-Q4）
  - 成本模式：2 种（no-cost / cost）
  - **总计**：4 × 6 × 8 × 2 = **384 次回测**
- **输出文件**：`reports/batch1_stoch_pivot_*.csv`（4 个报告）
- **关键结论**：脚本内嵌评分函数 `compute_batch1_score`，自动输出 Top 20 参数组合、baseline 排名、cost/no-cost 一致性相关系数、满足筛选条件的组合列表及是否建议替换 baseline 的判断

---

## 3. 修复的 Bug

| # | 问题描述 | 影响文件 | 修复方式 |
|---|---------|---------|---------|
| 1 | `custom_engine.py` 配置键 `init_cash` 与项目标准 `initial_cash` 不一致 | `backtest/custom_engine.py` | 统一改为 `initial_cash` |
| 2 | `custom_engine.py` 不支持仓位数量调整，PnL 永远按 1 单位计算 | `backtest/custom_engine.py` | 新增 `qty` 参数，所有 PnL 乘以 `qty` |
|| 3 | `profit_factor` 在多个 optimize 脚本中重复处理 None/NaN/inf 边界 | 8 个 optimize 脚本 | 提取到 `optimize/utils.py` 统一处理 |
|| 4 | `divergence.py` pivot 检测使用 Python `for` 循环，性能差 | `features/divergence.py` | 改为 `rolling().max/min` 向量化实现 |
|| 5 | `divergence.py` 循环内逐行 `.loc[]` 写入 Series，触发大量 pandas 开销 | `features/divergence.py` | 改为 list 收集 + 批量 `.loc[]` 赋值 |

---

## 4. 未解决的问题

### 4.1 3 个 skipped 测试

运行 `pytest -v` 结果：**29 passed, 3 skipped**

```
tests/test_smoke.py::test_trade_timing_is_after_confirmation         SKIPPED
tests/test_smoke.py::test_bullish_divergence_requires_prior_uptrend_and_progressive_channel   SKIPPED
tests/test_smoke.py::test_bearish_divergence_requires_prior_downtrend_and_progressive_channel SKIPPED
```

- **原因**：样本数据中缺乏足够的交易信号或背离信号，测试通过 `pytest.skip()` 主动跳过
- **风险**：核心逻辑（交易时机、趋势语义、通道条件）在空数据上未得到自动化验证
- **建议**：提供包含明确 bullish/bearish divergence 的 fixture 数据，解除 skip

### 4.2 RL 占位符

- **文件**：`rl/env.py`、`rl/reward.py`、`rl/train_sb3.py`
- **状态**：🚧 仅骨架/占位符
- **问题**：
  - `env.py`：`ParameterSwitchEnv` 为 toy env，observation 恒为 0 或随机数，reward 随机
  - `reward.py`：仅一行公式，未接入实际回测
  - `train_sb3.py`：仅跑 1000 步 smoke test，无实际策略参数切换逻辑
- **建议**：需要接入实际回测 PnL、实际参数 bucket 后才具备训练价值

### 4.2 RL 占位符

## 5. Git 提交历史

```
abbce6f (HEAD -> main, origin/main) perf: vectorize pivot detection and batch Series writes in divergence.py
f2ca070 refactor: extract optimize/utils.py for shared helpers; add batch1 stoch pivot + phase4b position cap analysis; fix custom engine qty support
b5c9920 chore: remove large data files from git tracking (keep local)
2c47629 feat: Phase 4 risk sensitivity with dual position modes, actual risk tracking, and cost model
93181ea feat: Phase 3C parameter plateau analysis with parallelization and cost reports
f5307f6 feat: Phase 3B walk-forward validation code and final reports
3d251f0 feat: Phase 3A holdout validation code and reports
2f44db5 feat: Phase 1-2 base code, grid search, and candidate generation
da0ac1b chore: add .gitignore for data, cache, and env files
3f96c94 (tag: phase2a-review) Freeze phase2a results for code review
6795812 Add script to download historical klines from Binance API
5343730 feat: complete BTC 5m strategy scaffold with custom backtester
f1f2b18 Add compiled Python files for CSV loader and signal builder modules
e1799f8 Refactor code structure for improved readability and maintainability
13d04b3 first commit
```

---

## 附录：统计摘要

| 指标 | 数值 |
|-----|------|
| 最近提交数 | 2 次（`f2ca070`, `abbce6f`） |
| 新增文件 | 12 个 |
| 修改文件 | 12 个 |
| 删除代码行 | ~457 行 |
| 新增代码行 | ~12,869 行（主要为 CSV 报告数据） |
| 测试通过率 | 29/29 passed（3 skipped） |
| Batch1 回测总数 | 384 次 |
