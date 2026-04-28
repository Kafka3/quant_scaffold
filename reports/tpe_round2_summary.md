# TPE Round 2 — 审计收口报告

**日期**: 2026-04-28
**审计者**: Hermes Agent (on behalf of project lead)

---

## 审计结论

**TPE Round 2 is exploratory only and is not used to replace the current primary candidate.**

---

## 1. 季度范围

| 项目 | 实际值 | 要求值 |
|------|--------|--------|
| 运行季度 | **2024-Q1 ~ 2025-Q2（6 个季度）** | 任务文档要求 8 个季度（2024-Q1 ~ 2025-Q4） |
| 缺少 | 2025-Q3, 2025-Q4 | — |

**原因**: TPE Round 2 是训练探索用途，设计上只包含 2024 全年 + 2025 前半段作为训练集，2025 后半段预留给 Phase 3A 样本外验证。这是有意的设计约束，不是 bug。

**结论**: ✅ 不需要补跑 8 季度版本。Round 2 的结果不作为最终候选排序依据。

---

## 2. 参数审计

### Seed 789 最佳参数校验

| 参数 | 报告值 | 实际值 | 匹配 |
|------|--------|--------|------|
| k_period | 12 | 12 | ✅ |
| d_period | 2 | 2 | ✅ |
| oversold | 14 | 14 | ✅ |
| overbought | **80** | **80** | ✅ |

**overbought = 80 搜索空间检查**:
- 代码: `trial.suggest_int("overbought", oversold + 65, 90)`
- 当 oversold = 14，下界 = 79，上界 = 90
- **80 ∈ [79, 90]** ✅ 完全在搜索范围内
- 搜索空间未被改动

**结论**: ✅ 报告展示 `KDJ (12, 2, 14-80)` 正确，无需修正。

---

## 3. 三个 Seed 汇总

| 指标 | Seed 42 | Seed 123 | **Seed 789** |
|------|---------|----------|-------------|
| Best Score | 0.5134 | 0.4929 | **0.5267** |
| 总交易数 | 143 | 105 | **166** |
| 季均回报 | 2.17% | 2.17% | **2.90%** |
| Avg PF | 2.20 | 2.49 | 2.25 |
| 最差回撤 | -2.11% | **-1.52%** | -3.05% |

---

## 4. 当前候选决策

| 层级 | 候选 | 状态 |
|------|------|------|
| **Primary Candidate** | **tpe_trial_490** | ✅ 保留 |
| **Secondary Candidate** | **tpe_trial_262** | ✅ 保留 |
| TPE Round 2 Seeds | seed 42 / 123 / 789 | 探索性结果，不替换主候选 |

**配置摘要**:

### tpe_trial_490 (Primary)
| 模块 | 参数 |
|------|------|
| Stochastic | k=21, d=5, smooth=3, oversold=17, overbought=75 |
| Pivots | L=6, R=5, min_sep=15, max_sep=31 |
| Trend | EMA=82, lookback=24, mcr=0.512 |
| Risk | RR=3.14 |

### tpe_trial_262 (Secondary)
| 模块 | 参数 |
|------|------|
| Stochastic | k=16, d=5, smooth=3, oversold=14, overbought=82 |
| Pivots | L=6, R=5, min_sep=15, max_sep=30 |
| Trend | EMA=77, lookback=24, mcr=0.688 |
| Risk | RR=3.43 |

---

## 5. 下一阶段禁止事项

- ❌ 不继续 TPE
- ❌ 不做 RL
- ❌ 不修改策略核心逻辑
