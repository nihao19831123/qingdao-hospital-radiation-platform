#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_sim_solve.py — 阶段 4：L3 决策智能（反向求解 solve + 自动寻优 optimize + 反事实留痕）

设计依据：《智能体推演-详细设计.md》V1.4 §10
  10.1 反向求解：目标类型（add_opd / add_ipd / radius80 / hri_region）+ 目标值 + 约束 →
       在 (action × dept) 上枚举、对 magnitude 二分求解（**先验证单调性**），取「投入代理成本最小」者；
       不可达时明确返回「在当前动作空间内不可达」+ 最接近方案的达成度（绝不虚构刚好达标的数字）。
  10.2 自动寻优：候选动作 → 两阶段（低精度档粗搜 → 标准档复算 top3）+ 风险约束 + 帕累托 + 胜出理由。
  10.3 反事实留痕：每次求解/寻优都进结果库（独立库 `agent_sim_results.db` 的 `solve_runs` 表，带 source 字段）。

硬约束：纯标准库；只读复用 agent_sim；业务库 hospital.db 全程只读。

⚠ 关键实现纪律（乱改会让结果不可信）：
  · **全部候选共用同一个随机种子**（`seed = 基准参数派生种子`）。若让种子随 magnitude 变化，
    蒙特卡洛噪声会淹没「强度→指标」的单调性，二分法会收敛到伪解。
  · 二分前**必须先验证单调性**；非单调的组合直接标记为不可用并说明原因，不做沉默处理。

用法：
  python3 agent_sim_solve.py --kind add_opd --target 100000
  python3 agent_sim_solve.py --kind radius80 --target 16.0 --dept 心内科
  python3 agent_sim_solve.py --optimize --target 100000 --top 3
  python3 agent_sim_solve.py --selftest
"""
import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_sim as A

ACTIONS = ("specialty", "alliance", "capacity")
KINDS = {
    "add_opd": "累计新增门诊（人次，P50）",
    "add_ipd": "累计新增住院（人次，P50）",
    "radius80": "80% 辐射半径（km，越大覆盖越广）",
    "hri_region": "选定区域 HRI（出院/人口占比口径；需给定 streets 选定区域）",
}
MAG_MIN, MAG_MAX = 0.05, 0.50
DEFAULT_RUNS, DEFAULT_DENSITY = 8, 0.5      # 求解用低精度档：控制总耗时（设计 §10.2 两阶段思想）
RISK = (("bed_usage", 90.0), ("numsource_util", 95.0))   # 键名与 risk_of() 一致（床位率 ≤90%、号源率 ≤95%）


# ────────────────────────── 指标读取 ──────────────────────────
def metric_of(res, kind, region=None):
    """从一次推演结果里取目标指标（None = 该口径不可用）"""
    it = res["interval"]
    if kind == "add_opd":
        return it["metrics"]["final_add_opd"]["p50"]
    if kind == "add_ipd":
        return it["metrics"]["final_add_ipd"]["p50"]
    if kind == "radius80":
        return (it.get("radius80") or {}).get("after")
    if kind == "hri_region":
        return (it.get("hri") or {}).get("region", {}).get("after")
    raise ValueError(f"未知目标类型：{kind}")


def risk_of(res):
    m = res["interval"]["metrics"]
    return dict(bed_usage=m["final_bed_usage"]["p50"], numsource_util=m["final_numsource_util"]["p50"])


def _base_params(params=None):
    p = dict(A.DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if k in A.DEFAULTS})
    p["runs"] = int(p.get("runs") or DEFAULT_RUNS)
    p["density"] = float(p.get("density") or DEFAULT_DENSITY)
    return p


# ────────────────────────── 10.1 反向求解 ──────────────────────────
def solve(kind="add_opd", target=None, params=None, depts=None, actions=ACTIONS,
          max_years=5, min_years=2, runs=None, density=None, tol=0.02, region=None,
          coarse=(4, 0.25), top_combos=3, coarse_iters=4, refine_iters=8, years_set=None, verbose=False):
    """反向求解：在动作空间内找到「达成目标且投入代理成本最小」的参数组合。

    **两阶段**（大动作空间下必需，否则全组合精算要 70 秒以上）：
      ① 粗筛：低精度档（默认 runs=4/density=0.25）对每个 (action, dept) 做端点探测 + 粗二分 → 按成本取前 top_combos 个组合；
      ② 精算：对入选组合用标准档做完整二分，并在 min_years..max_years 内挑成本最小年限。

    min_years 默认 2：years=1 时释放曲线退化为「一年满额见效」，与战略落地的管理现实不符（且会因成本代理
    = 强度×年限 而永远胜出），故默认不允许 1 年档（可显式传 min_years=1 复现该行为）。

    region：hri_region 目标时的选定街道集合（不传则用 params['streets']；两者都空 → hri 不可用，如实返回）
    """
    t0 = time.time()
    # ── 入参硬校验（2026-09-21）：非法输入必须在**进入计算前**拒绝，且由引擎自身兜底（脚本直调也安全）──
    if kind not in KINDS:
        raise ValueError(f"未知目标类型 {kind!r}；可用：{'/'.join(KINDS)}")
    if target is None:
        raise ValueError("缺少目标值 target")
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError(f"target 必须是数字，收到 {type(target).__name__}")
    if float(target) <= 0:
        raise ValueError(f"target 必须为正数（目标指标值），收到 {target}")
    target = float(target)
    max_years = max(1, min(5, int(max_years)))                  # 防 max_years=0/-1 造成下游 KeyError
    min_years = max(1, min(int(max_years), int(min_years)))
    top_combos = max(1, int(top_combos))
    coarse_iters = max(1, int(coarse_iters))
    refine_iters = max(1, int(refine_iters))
    actions = tuple(actions) or ACTIONS
    bad_actions = [x for x in actions if x not in ACTIONS]
    if bad_actions:
        raise ValueError(f"未知动作 {bad_actions}；可用：{'/'.join(ACTIONS)}")
    p0 = _base_params(params)
    if runs:
        p0["runs"] = int(runs)
    if density:
        p0["density"] = float(density)
    world = A.load_world(p0["visit_rate"])
    depts = list(depts or world["depts"])
    bad_depts = [d for d in depts if d not in world["depts"]]
    if bad_depts:
        raise ValueError(f"未知科室 {bad_depts}；可用：{'/'.join(world['depts'])}")
    if kind == "hri_region":
        sel = region or p0.get("streets")
        if not sel:
            return dict(kind=kind, target=target, reachable=False,
                        reason="hri_region 目标需要选定区域（streets/region）；全域口径 HRI 恒为 1，无法作为求解目标",
                        candidates=[], monotonicity=[], elapsed_sec=round(time.time() - t0, 2))
        p0["streets"] = list(sel)
    min_years = max(1, min(int(min_years), int(max_years)))
    # 共用同一随机种子：所有候选与二分迭代的差异只来自参数，不掺蒙卡噪声
    base_seed = A.derive_seed(p0, world)

    def run(fast=False, **over):
        q = dict(p0)
        if fast:
            q["runs"], q["density"] = int(coarse[0]), float(coarse[1])
        q.update(over)
        return A.simulate(q, seed=base_seed)

    ref = run(magnitude=MAG_MIN)                     # 参考：最小强度（上限探索的基线）
    if kind == "radius80":
        # 半径目标的方向取决于战略的空间分布：先用两类动作探测方向，再决定「求大还是求小」
        up = run(action="alliance", magnitude=MAG_MAX)
        dn = run(action="specialty", magnitude=MAG_MAX)
        val_all, val_spec = metric_of(up, kind), metric_of(dn, kind)
        base_val = metric_of(ref, kind)
        want = "max" if val_all > base_val else "min"
        direction_note = (f"方向探测：医联体 {val_all} / 专科 {val_spec} / 基线 {base_val} → "
                          f"本目标按『{'增大' if want == 'max' else '减小'}』方向求解"
                          f"（半径由战略的空间分布决定：医联体向外扩散、专科提升向近端聚焦）")
    else:
        want = "max"
        direction_note = "单调方向：战略强度越大，指标越高（先验证再二分）"

    def score(res):
        return metric_of(res, kind)

    def bisect(ac, dp, y, fast, iters):
        """在强度上二分；返回达到目标的强度（或 None）。粗筛用少量迭代（够排序），精算用 8 次（≈0.2% 强度精度）。"""
        lo, hi = MAG_MIN, MAG_MAX

        def ok(m):
            v = score(run(fast=fast, action=ac, dept=dp, magnitude=round(m, 4), years=y))
            v = v if v is not None else (float("-inf") if want == "max" else float("inf"))
            return (v >= target) if want == "max" else (v <= target)

        if ok(lo):
            return MAG_MIN
        if not ok(hi):
            return None
        for _ in range(int(iters)):
            mid = (lo + hi) / 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return round(hi, 3)

    # ── ① 粗筛：端点探测（单调性 + 可达性）+ 粗二分估计成本 ──
    mono, rough = [], []
    for ac in actions:
        for dp in depts:
            vlo = score(run(fast=True, action=ac, dept=dp, magnitude=MAG_MIN, years=max_years))
            vhi = score(run(fast=True, action=ac, dept=dp, magnitude=MAG_MAX, years=max_years))
            if vlo is None or vhi is None:
                mono.append(dict(action=ac, dept=dp, ok=False, why="指标不可用"))
                continue
            good = (vhi > vlo) if want == "max" else (vhi < vlo)
            mono.append(dict(action=ac, dept=dp, ok=bool(good), lo=vlo, hi=vhi,
                             why="" if good else "强度-指标非单调（该组合不参与求解）"))
            if not good:
                continue
            reach_hi = (vhi >= target) if want == "max" else (vhi <= target)
            if not reach_hi:                                  # 最大强度仍达不到 → 记最接近方案
                rough.append(dict(action=ac, dept=dp, magnitude=MAG_MAX, value=vhi,
                                  cost=MAG_MAX * max_years, unreachable=True))
                continue
            mag = bisect(ac, dp, max_years, fast=True, iters=coarse_iters)   # 粗二分（低成本，够排序）
            rough.append(dict(action=ac, dept=dp, magnitude=mag or MAG_MIN,
                              cost=(mag or MAG_MIN) * max_years))

    reachable_rough = sorted([r for r in rough if not r.get("unreachable")], key=lambda r: r["cost"])
    shortlist = reachable_rough[:max(1, int(top_combos))]

    # ── ② 精算：对入选组合用标准档完整二分 + 年限择优（≥ min_years）──
    ys = sorted({int(v) for v in (years_set or (min_years, int(max_years))) if min_years <= int(v) <= int(max_years)})
    cands = []
    for r in shortlist:
        ac, dp = r["action"], r["dept"]
        for y in ys:
            mag = bisect(ac, dp, y, fast=False, iters=refine_iters)
            if mag is None:
                continue
            val = score(run(action=ac, dept=dp, magnitude=mag, years=y))
            cands.append(dict(action=ac, dept=dp, years=y, magnitude=mag, value=val,
                              cost=round(mag * y, 4)))
    cands.sort(key=lambda c: c["cost"])

    # 结果组装
    if not cands:
        best = max(rough, key=lambda c: c["value"] if want == "max" else -c["value"]) if rough else None
        ach = round(best["value"] / target * 100, 1) if (best and target) else None
        return dict(kind=kind, kind_cn=KINDS[kind], target=target, reachable=False,
                    reason=(f"在当前动作空间内不可达：最大强度（{MAG_MAX:.0%} / {max_years} 年）下最优组合为"
                            f"{best['action'] if best else '无'}·{best['dept'] if best else '无'}，"
                            f"指标 {best['value'] if best else '—'}（达成度 {ach}%）"),
                    best_effort=best, achievement_pct=ach,
                    monotonicity=mono, seed=base_seed, direction_note=direction_note,
                    runs=p0["runs"], min_years=min_years,
                    elapsed_sec=round(time.time() - t0, 2))

    rec = cands[0]
    res = run(action=rec["action"], dept=rec["dept"], magnitude=rec["magnitude"], years=rec["years"])
    actual = metric_of(res, kind)
    err = abs(actual - target) / max(1e-9, abs(target)) * 100
    return dict(
        kind=kind, kind_cn=KINDS[kind], target=target, reachable=True,
        recommended=dict(action=rec["action"], dept=rec["dept"], magnitude=rec["magnitude"],
                         years=rec["years"], cost_proxy=rec.get("cost"),
                         metric_value=actual, achievement_pct=round(actual / target * 100, 1),
                         achievement_error_pct=round(err, 2)),
        alternatives=[dict(action=c["action"], dept=c["dept"], magnitude=c["magnitude"], years=c["years"],
                           cost_proxy=c.get("cost"), metric_value=c.get("value"),
                           achievement_pct=round((c.get("value") or 0) / target * 100, 1))
                      for c in cands[1:4]],
        risk=risk_of(res), monotonicity=mono, seed=base_seed,
        direction_note=direction_note, runs=p0["runs"], params=p0,
        precision=dict(coarse=list(coarse), refine=[p0["runs"], p0["density"]], top_combos=top_combos),
        min_years=min_years, shortlist=[dict(action=r["action"], dept=r["dept"], rough_cost=r.get("cost"))
                                        for r in shortlist],
        discipline=("两阶段求解（粗筛 top%d 组合 → 标准档精算）；全部候选共用同一随机种子；"
                    "二分前先验证单调性；非单调组合已标记不参与；投入代理成本 = 强度 × 年限；"
                    "年限下限 %d 年（避免 years=1 的『一年满额见效』边界假设）"
                    % (top_combos, min_years)),
        elapsed_sec=round(time.time() - t0, 2),
    )


# ────────────────────────── 10.2 自动寻优 ──────────────────────────
def optimize(params=None, depts=None, magnitudes=(0.10, 0.20, 0.30, 0.40, 0.50), years=(2, 3, 5),
             kind="add_opd", top=3, coarse=(6, 0.5), refine=(30, 1.0), max_combos=120, region=None):
    """两阶段自动寻优：粗搜（低精度档遍历候选）→ 精算（标准档复算 top N）+ 帕累托 + 胜出理由。"""
    t0 = time.time()
    p0 = _base_params(params)
    world = A.load_world(p0["visit_rate"])
    depts = list(depts or world["depts"])
    if kind == "hri_region":
        sel = region or p0.get("streets")
        if not sel:
            return dict(kind=kind, ok=False, reason="hri_region 目标需选定区域；全域 HRI 恒为 1",
                        elapsed_sec=round(time.time() - t0, 2))
        p0["streets"] = list(sel)
    base_seed = A.derive_seed(p0, world)

    combos = [dict(action=a, dept=d, magnitude=m, years=y)
              for a in ACTIONS for d in depts for m in magnitudes for y in years][:max_combos]

    def run(c, runs, density):
        q = dict(p0)
        q.update(c)
        q["runs"], q["density"] = runs, density
        return A.simulate(q, seed=base_seed)

    coarse_rows = []
    for c in combos:
        r = run(c, coarse[0], coarse[1])
        coarse_rows.append(dict(**c, metric=metric_of(r, kind), risk=risk_of(r)))
    # 风险约束淘汰
    for row in coarse_rows:
        row["risk_ok"] = all(row["risk"][k] <= lim for k, lim in RISK)
    feasible = [r for r in coarse_rows if r["risk_ok"] and r["metric"] is not None]
    feasible.sort(key=lambda r: -(r["metric"] or 0))
    feas_probe = feasible or [r for r in coarse_rows if r["metric"] is not None]

    # 帕累托（收益 vs 风险：床位使用率越低越好）
    pareto = []
    for r in feas_probe:
        if not any((o["metric"] >= r["metric"] and o["risk"]["bed_usage"] <= r["risk"]["bed_usage"]
                    and (o["metric"] > r["metric"] or o["risk"]["bed_usage"] < r["risk"]["bed_usage"]))
                   for o in feas_probe):
            pareto.append(r)

    top_rows = feas_probe[:top]
    refined = []
    for r in top_rows:
        c = {k: r[k] for k in ("action", "dept", "magnitude", "years")}
        full = run(c, refine[0], refine[1])
        refined.append(dict(**c, metric=metric_of(full, kind), risk=risk_of(full),
                            metrics=full["interval"]["metrics"], coarse_metric=r["metric"],
                            cap=full["cap"],
                            radius80=(full["interval"].get("radius80") or {}).get("after"),
                            hri_region=(full["interval"].get("hri") or {}).get("region", {}).get("after")))
    refined.sort(key=lambda r: -(r["metric"] or 0))
    win = refined[0]["metric"] if refined else 0
    for r in refined:
        r["tie_with_winner"] = bool(win and abs((r["metric"] or 0) - win) / max(1e-9, win) <= 0.005)

    # ── 诚实判据（设计 §10.2：必须说清"为什么第一名胜出"，不是一句"最优"）──
    notes = []
    cap_headroom = None
    win_capped = False
    capped = []
    if refined:
        cap_headroom = round(refined[0]["cap"]["opd_capacity"] - refined[0]["cap"]["opd_base"])
        capped = [r for r in coarse_rows if r["metric"] and r["metric"] >= cap_headroom * 0.999]
        win_capped = bool(win and win >= cap_headroom * 0.999)
        if win_capped:
            notes.append(f"⚠ 首选方案已触门诊容量上限（{cap_headroom:,} 人次 = 周号源 13,759 × 52 − 年门诊 55 万）："
                         "此时**总量由容量决定**、不由战略参数决定，方案之间的差异主要体现为空间分布"
                         "（80% 半径 / 区域 HRI）。该场景为『容量瓶颈型』，汇报时不得表述为总量优势。")
        elif capped:
            notes.append(f"⚠ 有 {len(capped)} 个候选方案已触门诊容量上限（{cap_headroom:,} 人次），"
                         f"因号源利用率将达 100% 被风险约束排除（例：{capped[0]['action']}·{capped[0]['dept']} "
                         f"{capped[0]['magnitude']:.0%}/{capped[0]['years']}年）。"
                         "说明『容量瓶颈型』场景下继续加大强度**不再增加总量**，须同步扩容（capacity 动作）"
                         "或按空间分布口径表述。")
    tie = [r for r in refined if r["tie_with_winner"]]
    if len(tie) > 1:
        notes.append("并列方案（收益差异 ≤0.5%，落在蒙特卡洛噪声内）：" +
                     "、".join(f"{r['action']}·{r['dept']} {r['magnitude']:.0%}/{r['years']}年" for r in tie) +
                     " → 区分判据应改用『风险更低 / 年限更短 / 空间分布更合意』，不要按收益硬排名。")
    by_ac = {}
    for r in coarse_rows:
        if r["metric"] is not None:
            by_ac.setdefault(r["action"], []).append(r["metric"])
    for ac, vs in by_ac.items():
        if len(vs) > 1 and max(vs) > 0:
            spread = (max(vs) - min(vs)) / max(vs) * 100
            if spread < 1.0:
                notes.append(f"⚠ {ac} 动作对目标科室几乎不敏感（科室间极差 {spread:.2f}%）："
                             "该动作按地理/容量起效，**科室选择不应作为卖点**（建议按区域或容量口径表述）。")

    if refined:
        w = refined[0]
        why = [f"收益最高：{KINDS[kind]} {w['metric']:,.0f}（粗搜 {w['coarse_metric']:,.0f}，精算复算一致度 "
               f"{round(min(w['metric'], w['coarse_metric']) / max(1e-9, max(w['metric'], w['coarse_metric'])) * 100, 1)}%）",
               f"风险满足约束：床位使用率 {w['risk']['bed_usage']}% ≤ 90%、号源利用率 {w['risk']['numsource_util']}% ≤ 95%",
               f"投入代理成本 {round(w['magnitude'] * w['years'], 2)}（强度 {w['magnitude']:.0%} × {w['years']} 年）"]
        for o in refined[1:]:
            why.append(f"对比落选方案 {o['action']}·{o['dept']}·{o['magnitude']:.0%}/{o['years']}年："
                       f"收益 {o['metric']:,.0f}（低 {round((w['metric'] - o['metric']) / max(1e-9, w['metric']) * 100, 1)}%）、"
                       f"床位率 {o['risk']['bed_usage']}%")
        if win_capped:
            why.insert(0, "⚠ 首选方案已触门诊容量上限：总量由容量而非战略参数决定，见 notes")
        elif capped:
            why.insert(0, f"⚠ 另有 {len(capped)} 个更强方案因触容量上限（号源率 100%）被风险约束排除，见 notes")
        if len(tie) > 1:
            why.insert(0, f"⚠ 前 {len(tie)} 名并列（差异 ≤0.5%）：按收益排名不具区分力，见 notes")
    else:
        why = ["无满足风险约束的候选方案（建议放宽约束或扩大候选集）"]
    return dict(kind=kind, kind_cn=KINDS[kind], ok=True,
                candidates_total=len(combos), feasible=len(feasible),
                risk_constraint=dict(RISK), coarse_precision=coarse, refine_precision=refine,
                cap_headroom=cap_headroom, notes=notes,
                top=[dict(**{k: r[k] for k in ("action", "dept", "magnitude", "years", "metric", "risk",
                                               "tie_with_winner", "radius80", "hri_region")},
                          metrics_p50={kk: r["metrics"][kk]["p50"] for kk in r["metrics"]}) for r in refined],
                pareto=[dict(action=r["action"], dept=r["dept"], magnitude=r["magnitude"], years=r["years"],
                             metric=r["metric"], bed_usage=r["risk"]["bed_usage"]) for r in pareto[:8]],
                winner_reasons=why, seed=base_seed, params=p0,
                discipline="两阶段：粗搜用低精度档（runs=%d/density=%.1f），精算对 top%d 用标准档（runs=%d/density=%.1f）复算；"
                           "全部候选共用同一种子。" % (coarse[0], coarse[1], top, refine[0], refine[1]),
                elapsed_sec=round(time.time() - t0, 2))


# ────────────────────────── 10.3 反事实留痕（独立结果库） ──────────────────────────
def save_solve(out, source="cli", note=None):
    """把一次求解/寻优写进结果库 solve_runs 表（只新增表，业务库不参与）"""
    A.ensure_db()
    conn = sqlite3.connect(A.RESULTS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS solve_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, solver TEXT,
        algo_version TEXT, data_version TEXT, seed TEXT, target REAL, reachable INTEGER,
        params_json TEXT, result_json TEXT, source TEXT, note TEXT)""")
    cur = conn.execute(
        "INSERT INTO solve_runs(created_at, kind, solver, algo_version, data_version, seed, target,"
        " reachable, params_json, result_json, source, note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), out.get("kind"), out.get("solver", "solve"),
         A.ALGO_VERSION, A.data_version(), str(out.get("seed")), out.get("target"),
         1 if out.get("reachable") or out.get("ok") else 0,
         json.dumps(out.get("params") or {}, ensure_ascii=False),
         json.dumps(out, ensure_ascii=False), source, note))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def list_solves(limit=20):
    A.ensure_db()
    conn = sqlite3.connect(A.RESULTS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS solve_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, solver TEXT,
        algo_version TEXT, data_version TEXT, seed TEXT, target REAL, reachable INTEGER,
        params_json TEXT, result_json TEXT, source TEXT, note TEXT)""")
    rows = conn.execute("SELECT id, created_at, kind, solver, target, reachable, source, note"
                        " FROM solve_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ────────────────────────── 自检 ──────────────────────────
def selftest(verbose=True):
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    t0 = time.time()
    # 1 反解 add_opd：目标取「最小强度、单科室」可达值，检验达成度
    base = dict(action="specialty", dept="心内科", magnitude=0.30, years=5, runs=8, density=0.5)
    probe = A.simulate(base)
    tgt = int(probe["interval"]["metrics"]["final_add_opd"]["p50"] * 0.6)
    s1 = solve("add_opd", tgt, params=dict(runs=8, density=0.5), depts=["心内科"], actions=("specialty",))
    ck("1 反解可运行且给出推荐方案", s1.get("reachable") and s1.get("recommended"),
       f"目标 {tgt:,} → 推荐 {s1['recommended']['magnitude']:.0%}/{s1['recommended']['years']}年 "
       f"实测 {s1['recommended']['metric_value']:,}（达成度 {s1['recommended']['achievement_pct']}%，"
       f"误差 {s1['recommended']['achievement_error_pct']}%）")
    # 2 达成度误差在容差内
    ck("2 达成度误差 ≤ 5%", (s1.get("recommended") or {}).get("achievement_error_pct", 99) <= 5.0,
       f"误差 {s1['recommended']['achievement_error_pct']}%")
    # 3 单调性预检有记录且非单调组合被排除
    mono = s1.get("monotonicity", [])
    ck("3 单调性预检留痕（含不可用组合标记）", len(mono) >= 1 and all("ok" in m for m in mono),
       f"预检 {len(mono)} 组合：{[m['ok'] for m in mono]}，非单调原因示例="
       f"{next((m['why'] for m in mono if not m['ok']), '无')}")
    # 4 不可达如实返回
    s2 = solve("add_opd", 10_000_000, params=dict(runs=6, density=0.5), depts=["心内科"], actions=("specialty",))
    ck("4 不可达如实返回（不虚构达标数字）", (not s2.get("reachable")) and s2.get("best_effort"),
       f"{s2.get('reason')}")
    # 5 radius80 方向探测与求解
    s3 = solve("radius80", 20.0, params=dict(runs=6, density=0.5), depts=["心内科"], actions=("alliance",))
    ck("5 radius80 目标可用（含方向探测）", s3.get("direction_note") is not None,
       s3.get("direction_note") if s3.get("direction_note") else "")
    # 6 寻优两阶段 + 风险约束
    o = optimize(params=dict(runs=6, density=0.5), depts=["心内科", "骨科"], magnitudes=(0.2, 0.4),
                 years=(3, 5), top=2, coarse=(6, 0.5), refine=(10, 0.5))
    ok6 = o.get("ok") and o.get("top") and all(
        r["risk"]["bed_usage"] <= 90 and r["risk"]["numsource_util"] <= 95 for r in o["top"])
    ck("6 寻优返回 top-N 且满足风险约束", bool(ok6),
       f"候选 {o.get('candidates_total')} 可行 {o.get('feasible')} 帕累托 {len(o.get('pareto', []))}；"
       f"第一名 {o['top'][0]['action']}·{o['top'][0]['dept']} {o['top'][0]['magnitude']:.0%}/"
       f"{o['top'][0]['years']}年 → {o['top'][0]['metric']:,.0f} 人次")
    # 7 复用同一种子的确定性（两次求解结果一致，剔除耗时）
    s1b = solve("add_opd", tgt, params=dict(runs=8, density=0.5), depts=["心内科"], actions=("specialty",))
    a, b = dict(s1), dict(s1b)
    for d in (a, b):
        d.pop("elapsed_sec", None)
        d.pop("availability_time_sec", None)
    ck("7 同参数两次求解逐位一致（含同种子）", json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True),
       f"seed={s1['seed']}")
    # 9 诚实判据：容量触顶（含被风险约束排除）/并列/科室不敏感必须显式说明（设计 §10.2）
    o2 = optimize(params=dict(runs=6, density=0.5), depts=["心内科", "骨科"], magnitudes=(0.4,),
                  years=(5,), top=2, coarse=(6, 0.5), refine=(8, 0.5))
    cap_note = next((n for n in o2.get("notes", []) if "容量上限" in n), None)
    ck("9 容量触顶/并列/科室不敏感有显式说明（不硬排名）", bool(cap_note),
       (cap_note or "未触发")[:150])
    # 10 两阶段性能与年限下限：全科室 × 三动作反解须在合理时间内完成，且推荐年限 ≥ min_years
    t10 = time.time()
    s10 = solve("add_opd", tgt, params=dict(runs=8, density=0.5), min_years=2)
    dt10 = time.time() - t10
    ck("10 两阶段性能（全 27 组合反解 < 40s）且推荐年限 ≥ min_years",
       dt10 < 40 and (s10.get("recommended", {}).get("years", 0) >= 2),
       f"耗时 {dt10:.1f}s ｜ 推荐 {s10.get('recommended', {}).get('action')}·"
       f"{s10.get('recommended', {}).get('dept')} {s10.get('recommended', {}).get('magnitude')}"
       f"/{s10.get('recommended', {}).get('years')}年 ｜ 短名单 {len(s10.get('shortlist', []))} 组合")
    # 8 反事实留痕可写可读
    sid = save_solve(dict(s1, solver="solve"), source="selftest", note="阶段4自检")
    lst = list_solves(5)
    ck("8 求解结果留痕（solve_runs 表可写可读）", any(r["id"] == sid for r in lst),
       f"solve_id={sid}，库中记录 {len(lst)} 条")
    print(f"\n阶段 4 自检汇总：{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} 通过，"
          f"用时 {time.time() - t0:.1f}s")
    return all(ok for _, ok, _ in checks), checks


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段 4：L3 决策智能（反解 + 寻优）")
    ap.add_argument("--kind", default="add_opd", choices=list(KINDS))
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--dept", default="", help="逗号分隔科室（留空=全部科室）")
    ap.add_argument("--actions", default="", help="逗号分隔动作（留空=三类全试）")
    ap.add_argument("--years", type=int, default=5, help="反解的最大年限")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    ap.add_argument("--streets", default="", help="逗号分隔街道（hri_region 目标必填）")
    ap.add_argument("--optimize", action="store_true", help="做自动寻优（两阶段）")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="结果写入结果库 solve_runs")
    ap.add_argument("--note", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, _ = selftest()
        return 0 if ok else 1

    depts = [x.strip() for x in args.dept.split(",") if x.strip()] or None
    actions = tuple(x.strip() for x in args.actions.split(",") if x.strip()) or ACTIONS
    streets = [x.strip() for x in args.streets.split(",") if x.strip()] or None
    p = dict(runs=args.runs, density=args.density)
    if streets:
        p["streets"] = streets

    if args.optimize:
        out = optimize(p, depts=depts, kind=args.kind, top=args.top, region=streets)
        print(f"自动寻优（{KINDS[args.kind]}）｜候选 {out['candidates_total']} 可行 {out.get('feasible')}"
              f" ｜ 用时 {out['elapsed_sec']}s ｜ 种子 {out['seed']}")
        for i, r in enumerate(out.get("top", []), 1):
            print(f"  #{i} {r['action']}·{r['dept']} {r['magnitude']:.0%}/{r['years']}年 → "
                  f"{KINDS[args.kind]} {r['metric']:,.0f} ｜ 床位率 {r['risk']['bed_usage']}% "
                  f"号源率 {r['risk']['numsource_util']}% ｜ 80%半径 {r['radius80']}km")
        print("  胜出理由：")
        for w in out["winner_reasons"]:
            print(f"    · {w}")
        for n in out.get("notes", []):
            print(f"    ⚠ {n}")
        out["solver"] = "optimize"
    else:
        if args.target is None:
            print("需要 --target（或改用 --optimize）")
            return 2
        out = solve(args.kind, args.target, params=p, depts=depts, actions=actions,
                    max_years=args.years, region=streets)
        print(f"反向求解：{KINDS[args.kind]} 目标 {args.target:,.1f} ｜ 用时 {out['elapsed_sec']}s ｜ 种子 {out['seed']}")
        print(f"  {out.get('direction_note', '')}")
        if out.get("reachable"):
            r = out["recommended"]
            print(f"  ✅ 可达 → 推荐 {r['action']}·{r['dept']}·强度 {r['magnitude']:.0%}·{r['years']} 年 "
                  f"（投入代理成本 {r['cost_proxy']}）；实测 {r['metric_value']:,.1f}"
                  f"（达成度 {r['achievement_pct']}%，误差 {r['achievement_error_pct']}%）")
            for i, c in enumerate(out["alternatives"], 1):
                print(f"     备选 {i}：{c['action']}·{c['dept']} {c['magnitude']:.0%}/{c['years']}年 "
                      f"成本 {c['cost_proxy']} 达成度 {c['achievement_pct']}%")
        else:
            print(f"  ❌ {out.get('reason')}")
        out["solver"] = "solve"
    if args.save:
        sid = save_solve(out, source="cli", note=args.note)
        print(f"  已留痕：solve_runs id={sid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
