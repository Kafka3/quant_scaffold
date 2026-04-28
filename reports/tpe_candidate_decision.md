# TPE Candidate Decision

## Candidate Comparison

### current_baseline
- **score**: 0.334
- **return**: 7.18%
- **profit factor**: 1.29
- **正季度比**: 0.62
- **risk flags**: COST_FRAGILE + QUARTER_UNSTABLE

### pivot_wide_9_35
- **score**: 0.408
- **return**: 11.47%
- **profit factor**: 1.55
- **正季度比**: 0.88
- **risk flags**: QUARTER_UNSTABLE

### tpe_trial_9 (Rank #1 in top50)
- **score**: 0.900
- **return**: 11.05%
- **profit factor**: 5.00
- **正季度比**: 1.00
- **risk flags**: LOW_TRADE_COUNT
- **不做主候选原因**: 全周期仅 **39 笔交易**，交易频率过低，不满足实盘最低样本量要求。

### tpe_trial_490 (Primary Candidate)
- **score**: 0.637
- **return**: 39.72%
- **profit factor**: 2.69
- **正季度比**: 1.00
- **risk flags**: 无
- **做主候选原因**: 在 cost 模式下验证 score 最高，全季度正收益，无风险 flag，交易笔数充足（141 笔）。

### tpe_trial_262 (Secondary Candidate)
- **score**: 0.643
- **return**: 26.78%
- **profit factor**: 3.10
- **正季度比**: 1.00
- **risk flags**: 无
- **做次候选原因**: score 接近 primary，pf 更高（3.10），作为备选可对冲单一参数过拟合风险。

## Decision

- **暂不替换** `configs/candidates/robust_ema55.yaml`
- Primary 候选: `configs/candidates/tpe_trial_490.yaml`
- Secondary 候选: `configs/candidates/tpe_trial_262.yaml`
- **下一步**: 对 tpe_trial_490 和 tpe_trial_262 做风险复测（walk-forward / 压力测试），确认无 hidden fragility 后再决定是否提升为新的 baseline。
