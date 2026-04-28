# robust_ema55 项目进度报告

> 生成时间：2026-04-27

---

## 1. 项目概述

本项目是一个基于 **BTCUSDT 5分钟K线** 的量化交易策略研究与回测框架，核心策略为 **Robust EMA55 背离策略**。策略通过检测价格与 Stochastic 指标的常规背离（Regular Divergence），结合 EMA55 趋势过滤和通道过滤，生成多空交易信号。项目涵盖从数据下载、特征工程、参数优化到回测验证的完整流程。

**当前状态**：Batch 2B 参数优化已完成，新增 pivot-wide 候选参数 (9/35)。

---

## 2. 已完成工作

### 2.1 策略基础搭建（Phase 1-2）
- ✅ 完整 Python 回测框架：`backtest/`（custom_engine、event_engine、metrics）
- ✅ 特征工程模块：`features/`（divergence.py、indicators.py、trend_filter.py）
- ✅ 信号构建：`strategy/signal_builder.py`（setup 生命周期、确认/触发/超时/失效）
- ✅ 数据管道：Binance K线下载脚本 + CSV 加载器

### 2.2 参数搜索与验证（Phase 2-3）
- ✅ **Phase 2A/B**：网格搜索 + Optuna 搜索，确定 baseline 候选参数
- ✅ **Phase 3A-C**：Holdout 验证、Walk-forward 验证、参数平台期分析
- ✅ **Batch 1**：Stochastic × Pivot 间隔敏感性分析（384 次回测）
- ✅ **Batch 2B**：Pivot 宽区间网格验证（见下方关键数据）

### 2.3 性能优化与 Bug 修复
- ✅ `features/divergence.py`：pivot 检测向量化（rolling + shift），批量 Series 写入
- ✅ `backtest/custom_engine.py`：修复 `initial_cash` 键名不一致，新增 `qty` 仓位乘数支持
- ✅ 提取 `optimize/utils.py` 共享工具模块，消除 8 个文件中的重复代码

---

## 3. 进行中工作

| 工作项 | 状态 | 说明 |
|--------|------|------|
| Batch 2B 候选参数确认 | 🔄 待决策 | (9,35) 已验证等价于 (9,42)，但需决定最终候选集 |

---

## 4. 待办项

1. **决定 Batch 2B 最终候选参数**：当前新增 `robust_ema55_pivot_wide_9_35.yaml`，需确认是否替换或并行测试 baseline

---

## 5. 关键数据

### 5.1 候选配置文件

`configs/candidates/` 目录下现有 4 个候选配置：

| 配置文件 | EMA周期 | min_sep | max_sep | RR | 说明 |
|---------|---------|---------|---------|-----|------|
| `robust_ema55.yaml` | 55 | 3 | 20 | 2.2 | Baseline 基准 |
| `robust_ema55_pivot_wide_9_35.yaml` | 55 | 9 | 35 | 2.2 | **Batch 2B 新增候选** |
| `aggressive_ema34_A.yaml` | 34 | 3 | 20 | 2.2 | 激进变体 |
| `stable_ema34_B.yaml` | 34 | 3 | 20 | 2.0 | 稳健变体 |

### 5.2 Batch 2B Pivot 网格摘要（前5行，no-cost 模式）

| min_sep | max_sep | positive_ratio | avg_quarter_return | avg_profit_factor | batch2b_score | 备注 |
|---------|---------|---------------|-------------------|------------------|---------------|------|
| 3 | 20 | 0.625 | 1.42% | 1.72 | **0.367** | Baseline |
| 5 | 25 | 0.875 | 1.74% | 1.78 | 0.432 | — |
| 5 | 30 | 0.875 | 1.85% | 1.75 | 0.436 | — |
| 5 | 35 | 0.875 | 2.25% | 1.89 | **0.458** | 最优组合 |
| 5 | 40 | 0.875 | 2.25% | 1.89 | **0.458** | 与 35 等价 |

**结论**：max_separation >= 35 时表现稳定提升，且 35/40/45/50 等价。

### 5.3 Batch 2B 等价性分析（前10行）

在 cost 和 no-cost 模式下，**max_separation 在 35~50 区间内完全等价**：

| cost_mode | min_sep | equivalent_max_sep_values | count | score |
|-----------|---------|---------------------------|-------|-------|
| cost | 5 | 35,40,45,50 | 4 | 0.370 |
| cost | 6 | 35,40,45,50 | 4 | 0.370 |
| cost | 7 | 35,40,45,50 | 4 | 0.395 |
| cost | 8 | 35,40,45,50 | 4 | 0.386 |
| cost | 9 | 35,40,45,50 | 4 | 0.408 |
| cost | 10 | 35,40,45,50 | 4 | 0.383 |
| cost | 11 | 35,40,45,50 | 4 | 0.383 |
| cost | 12 | 35,40,45,50 | 4 | 0.372 |
| no-cost | 5 | 35,40,45,50 | 4 | 0.458 |
| no-cost | 6 | 35,40,45,50 | 4 | 0.458 |

### 5.4 9/35 vs 9/42 周期级等价验证（前10行）

| cost_mode | period | return_equal | trades_equal | pf_equal | dd_equal | all_equal |
|-----------|--------|-------------|--------------|----------|----------|-----------|
| no-cost | 2024-Q1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2024-Q2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2024-Q3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2024-Q4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2025-Q1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2025-Q2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2025-Q3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| no-cost | 2025-Q4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| cost | 2024-Q1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| cost | 2024-Q2 | ✅ | ✅ | ✅ | ✅ | ✅ |

**结论**：(9,35) 与 (9,42) 在全部 8 个季度、两种成本模式下**完全等价**，因此选择 35 作为 max_separation 更保守且与网格结果一致。

---

## 6. 下一步建议

1. **候选参数锁定**：`robust_ema55_pivot_wide_9_35.yaml` 已通过等价性验证，建议将其纳入最终候选集，与 baseline (3/20) 并行对比
2. **测试补强**：补充 fixture 数据，解除 3 个 skipped 测试，确保核心逻辑（交易时机、趋势过滤、通道条件）通过自动化验证

---

## 附录：最近 Git 提交

```
d83eb16 Add Batch 2B pivot-wide candidate and equivalence report
fc62f90 fix: add explicit 9/35 vs 9/42 period-level equivalence validation
6651108 feat: Batch 2B pivot grid validation - confirm max_sep>=35 equivalence, add candidate (9/35)
1203034 feat: Batch 2 Optuna pivot search - best params (9,42)
0718421 fix: add safe access with default for initial_cash in custom_engine.py
abbce6f perf: vectorize pivot detection and batch Series writes in divergence.py
f2ca070 refactor: extract optimize/utils.py; add batch1 stoch pivot + phase4b position cap
b5c9920 chore: remove large data files from git tracking (keep local)
2c47629 feat: Phase 4 risk sensitivity with dual position modes, actual risk tracking
93181ea feat: Phase 3C parameter plateau analysis with parallelization and cost reports
f5307f6 feat: Phase 3B walk-forward validation code and final reports
3d251f0 feat: Phase 3A holdout validation code and reports
2f44db5 feat: Phase 1-2 base code, grid search, and candidate generation
```
