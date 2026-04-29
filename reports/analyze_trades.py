#!/usr/bin/env python3
import csv, math, statistics
from collections import defaultdict

FILE = "/Volumes/t7/quant_scaffold/reports/phase5_pre_live_1pct_10x_trades.csv"

rows = []
with open(FILE, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        period = row['period'].strip()
        if period in ('2024-2025-Full',):
            continue
        rows.append(row)

print(f"Total individual trades (excl full-period rows): {len(rows)}\n")

def f(v):
    try: return float(v)
    except: return float('nan')

print("="*60)
print("1. TOTAL TRADES")
print("="*60)
print(f"   {len(rows)}")

print("\n" + "="*60)
print("2. STOP_DISTANCE_PCT")
print("="*60)
vals = sorted([f(r['stop_distance_pct']) for r in rows])
n = len(vals)
print(f"   min:  {vals[0]:.6f}")
print(f"   25%:  {vals[n//4]:.6f}")
print(f"   50%:  {vals[n//2]:.6f}")
print(f"   75%:  {vals[3*n//4]:.6f}")
print(f"   max:  {vals[-1]:.6f}")

print("\n" + "="*60)
print("3. NOTIONAL")
print("="*60)
ns = [f(r['notional']) for r in rows]
mn = sum(ns)/len(ns)
sn = sorted(ns)
medn = sn[len(sn)//2]
print(f"   mean:   ${mn:,.2f}")
print(f"   median: ${medn:,.2f}")

print("\n" + "="*60)
print("4. COST (fees + slippage_cost)")
print("="*60)
cs = [f(r['fees'])+f(r['slippage_cost']) for r in rows]
mc = sum(cs)/len(cs)
sc = sorted(cs)
medc = sc[len(sc)//2]
print(f"   mean:   ${mc:,.2f}")
print(f"   median: ${medc:,.2f}")

print("\n" + "="*60)
print("5. COST / NOTIONAL %")
print("="*60)
ratios = []
for r in rows:
    c = f(r['fees'])+f(r['slippage_cost'])
    nv = f(r['notional'])
    if not math.isnan(c) and not math.isnan(nv) and nv!=0:
        ratios.append(c/nv*100)
print(f"   mean:   {sum(ratios)/len(ratios):.4f}%")

print("\n" + "="*60)
print("6. WIN RATES")
print("="*60)
gw = sum(1 for r in rows if f(r['gross_pnl'])>0)
nw = sum(1 for r in rows if f(r['net_pnl'])>0)
print(f"   Gross PnL: {gw}/{len(rows)} = {gw/len(rows)*100:.2f}%")
print(f"   Net PnL:   {nw}/{len(rows)} = {nw/len(rows)*100:.2f}%")

print("\n" + "="*60)
print("7. PNL SUMS")
print("="*60)
print(f"   sum gross_pnl: ${sum(f(r['gross_pnl']) for r in rows):,.2f}")
print(f"   sum net_pnl:   ${sum(f(r['net_pnl']) for r in rows):,.2f}")

print("\n" + "="*60)
print("8. R-MULTIPLE BY SIGN")
print("="*60)
pos = [f(r['r_multiple']) for r in rows if f(r['r_multiple'])>0]
neg = [f(r['r_multiple']) for r in rows if f(r['r_multiple'])<0]
print(f"   mean positive R: {sum(pos)/len(pos):.4f}  (n={len(pos)})" if pos else "   no positive R")
print(f"   mean negative R: {sum(neg)/len(neg):.4f}  (n={len(neg)})" if neg else "   no negative R")

print("\n" + "="*60)
print("9. BY EXIT_REASON")
print("="*60)
br = defaultdict(lambda: {'c':0,'w':0,'np':0.0})
for r in rows:
    rea = r['exit_reason']
    br[rea]['c'] += 1
    if f(r['net_pnl'])>0: br[rea]['w'] += 1
    br[rea]['np'] += f(r['net_pnl'])
for rea in sorted(br):
    d=br[rea]
    wr=d['w']/d['c']*100
    print(f"   {rea:>12s}: count={d['c']:3d}  wins={d['w']:3d} ({wr:5.2f}%)  net_pnl=${d['np']:>10,.2f}")

print("\n" + "="*60)
print("10. BY SIDE")
print("="*60)
bs = defaultdict(lambda: {'c':0,'np':0.0})
for r in rows:
    s = r['side']
    bs[s]['c'] += 1
    bs[s]['np'] += f(r['net_pnl'])
for s in sorted(bs):
    d=bs[s]
    print(f"   {s:>6s}: count={d['c']:3d}  net_pnl=${d['np']:>10,.2f}")

print("\n" + "="*60)
print("11. BY PERIOD (QUARTER)")
print("="*60)
bp = defaultdict(lambda: {'c':0,'np':0.0,'gp':0.0})
for r in rows:
    p = r['period']
    bp[p]['c'] += 1
    bp[p]['np'] += f(r['net_pnl'])
    bp[p]['gp'] += f(r['gross_pnl'])
for p in sorted(bp):
    d=bp[p]
    print(f"   {p:>10s}: count={d['c']:3d}  net_pnl=${d['np']:>10,.2f}  gross_pnl=${d['gp']:>10,.2f}")

print("\n" + "="*60)
print("12. CORRELATION: stop_distance_pct vs r_multiple")
print("="*60)
pairs = [(f(r['stop_distance_pct']), f(r['r_multiple'])) for r in rows
         if not math.isnan(f(r['stop_distance_pct'])) and not math.isnan(f(r['r_multiple']))]
k = len(pairs)
mx = sum(x for x,_ in pairs)/k
my = sum(y for _,y in pairs)/k
num = sum((x-mx)*(y-my) for x,y in pairs)
den = math.sqrt(sum((x-mx)**2 for x,_ in pairs)*sum((y-my)**2 for _,y in pairs))
corr = num/den if den!=0 else 0
print(f"   Pearson r: {corr:.6f}  (n={k})")
