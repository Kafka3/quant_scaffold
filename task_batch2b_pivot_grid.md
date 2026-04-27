# Batch 2B Pivot Grid Search 任务文档

## 1. 任务目标

- 用**完整 Grid Search** 取代 TPE，在固定离散空间中遍历 min_separation / max_separation 的所有组合。
- 验证 `max_separation >= 35` 的**等价区间**（是否不敏感）。
- 产出 `candidate_pivot_wide (9/35)` 作为新候选，与 baseline_current (3/20)、baseline_batch1 (8/40)、candidate_tpe (9/42) 做横向对比。
- **cost 版本为主**，no-cost 仅作辅助。

## 2. 约束（不要做）

- [ ] 不改 `robust_ema55` 主配置
- [ ] 不接入 RL
- [ ] 不接入 HMM/Regime
- [ ] 不改策略逻辑
- [ ] 不改背离 / entry / stop / target
- [ ] 不做 Optuna
- [ ] 暂不替换 baseline，仅把 (9/35) 提升为**新候选**

## 3. 先修代码问题

### 3.1 问题 1：删除无效约束

**文件：** `optimize/batch2_optuna_pivot.py`  
**动作：** 找到 `max_sep >= min_sep + 10` 的约束并**删除**。该约束在离散小空间中无效，且已决定不再使用 TPE。

### 3.2 问题 2：改为完整 Grid Search + 去重

- 不再使用 TPE / Optuna。
- 遍历搜索空间内所有合法组合。
- 对同一组 (min_sep, max_sep) 在一次运行中**只计算一次**，避免重复。

## 4. 搜索空间

固定参数保持 baseline 不变，仅搜索以下两个 pivot 参数：

```
min_separation: [5, 6, 7, 8, 9, 10, 11, 12]   # 8 个值
max_separation: [25, 30, 35, 40, 45, 50]        # 6 个值
```

组合数：`8 × 6 = 48` 组。

每组运行：8 个季度 × 2 版 (cost / no-cost) = **768 次验证**。

## 5. 必须包含的对照参数并标记

以下 4 组对照必须出现在结果中。若其坐标**不在上述网格中**，需**额外加入运行**：

| 标签 | min_separation | max_separation | 说明 |
|------|----------------|----------------|------|
| `baseline_current` | 3 | 20 | 当前 baseline |
| `baseline_batch1` | 8 | 40 | Batch 1 最优 |
| `candidate_tpe` | 9 | 42 | Batch 2 TPE 候选 |
| `candidate_wide_simple` | 9 | 35 | **本批次新候选** |

## 6. 新增脚本

**文件：** `optimize/batch2b_pivot_grid.py`

脚本职责：
1. 加载数据与配置（保持 `robust_ema55` 固定参数不变）。
2. 遍历搜索空间 + 补充对照组。
3. 对每组 (min_sep, max_sep) 执行 8 季度回测（cost / no-cost 各一次）。
4. 按评分规则打分，生成所有输出文件。
5. 执行等价区间分析。

## 7. 输出文件

全部放置于 `reports/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `batch2b_pivot_grid_validation.csv` | 逐条验证明细（768 行级别） |
| `batch2b_pivot_grid_summary.csv` | 每组参数汇总 + 评分 + 标记位 |
| `batch2b_pivot_grid_top20_cost.csv` | cost 版 Top 20 |
| `batch2b_pivot_grid_top20_nocost.csv` | no-cost 版 Top 20 |
| `batch2b_pivot_grid_equivalence.csv` | 等价区间分析（max_sep >= 35 是否等价） |

## 8. 数据字段定义

### 8.1 summary.csv 字段

```
cost_mode
min_separation
max_separation
total_periods
positive_periods
positive_ratio
total_return_sum
avg_quarter_return
median_quarter_return
min_quarter_return
total_trades_sum
min_trades_per_period
avg_trades_per_period
avg_profit_factor
median_profit_factor
min_profit_factor
max_quarter_drawdown_worst
long_trades_total
short_trades_total
batch2b_score
is_baseline_current
is_baseline_batch1
is_candidate_tpe
is_candidate_wide_simple
```

### 8.2 equivalence.csv 字段

```
cost_mode
min_separation
result_signature        # 由以下字段哈希生成：positive_ratio, total_return_sum, total_trades_sum,
                        # avg_profit_factor, min_profit_factor, max_quarter_drawdown_worst,
                        # long_trades_total, short_trades_total
equivalent_max_separation_values   # 等价的 max_sep 列表（如 "35,40,45,50"）
count
score
total_trades_sum
avg_profit_factor
min_profit_factor
positive_ratio
```

**目的：** 确认 `max_sep >= 35` 时，同一 `min_sep` 下不同 `max_sep` 的结果签名是否相同，从而验证不敏感假设。

## 9. 评分规则

```python
if min_trades_per_period < 8:
    score = -9999
else:
    score = (
        0.25 * min(avg_pf, 5) / 5
        + 0.20 * positive_ratio
        + 0.20 * min(min_pf, 5) / 5
        + 0.15 * clip(avg_quarter_return / 5, -1, 1)
        + 0.10 * min(min_trades / 20, 1)
        - 0.10 * abs(max_drawdown_worst / 20)
    )
```

- `avg_pf` = avg_profit_factor
- `min_pf` = min_profit_factor
- `clip(x, -1, 1)` 表示限制在 [-1, 1] 区间
- 以 **cost 版本** 为主进行排序和决策，no-cost 仅作辅助观察。

## 10. 运行前检查

```bash
python -m py_compile optimize/batch2b_pivot_grid.py
pytest -q
```

两项必须全部通过，方可进入正式运行。

## 11. 运行命令

```bash
python optimize/batch2b_pivot_grid.py --data data/raw/BTCUSDT_5m_2024_2025.csv
```

## 12. 完成后必须反馈的 12 项

请在任务完成后，以结构化形式输出以下结论：

1. **是否成功运行**（是 / 否，若有异常请附日志摘要）
2. **是否完成全部 768 次验证**（实际运行数 / 768）
3. **cost Top 20**（列出前 20 的 min_sep / max_sep / score）
4. **no-cost Top 20**（列出前 20 的 min_sep / max_sep / score）
5. **baseline_current (3/20)** 在 cost / no-cost 中的**排名**和 **score**
6. **baseline_batch1 (8/40)** 在 cost / no-cost 中的**排名**和 **score**
7. **candidate_tpe (9/42)** 在 cost / no-cost 中的**排名**和 **score**
8. **candidate_wide_simple (9/35)** 在 cost / no-cost 中的**排名**和 **score**
9. **max_sep >= 35 是否等价**（引用 `equivalence.csv` 中的签名重复情况与数据）
10. **最稳定的 min_sep 是多少**（在高分组中出现频率最高、且等价区间最宽的 min_sep）
11. **最稳定的 max_sep 区间是多少**（等价区间内表现一致的最小 max_sep 范围）
12. **是否建议：**
    - [ ] 保留 3/20 为 baseline
    - [ ] 升级到 8/40
    - [ ] 升级到 9/35
    - [ ] 或继续保持 robust_ema55 不变（仅新增候选，不替换 baseline）

## 13. 验收标准

- [ ] `optimize/batch2b_pivot_grid.py` 脚本存在且编译通过
- [ ] `pytest -q` 无失败
- [ ] 5 个输出文件全部生成且字段完整
- [ ] 768 次验证全部完成（或明确说明缺失原因）
- [ ] 4 个对照参数在 summary 中正确标记
- [ ] equivalence.csv 包含 max_sep >= 35 的等价区间分析数据
- [ ] 12 项反馈结论已输出
