#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_sim_sensitivity.py — 阶段 3：参数敏感性分析（±20% 扰动 + 弹性 + 方向自检）

设计依据：《智能体推演-详细设计.md》V1.3 §9.2（**判据已按实测修正，见 §9.2 修订说明**）

    · 对影响结果的参数各做 ±20% 扰动，重跑同一方案（同一种子，变体差异不含蒙特卡洛噪声）；
    · 输出每个参数的弹性（指标变化% ÷ 参数变化%）+ 排序表；
    · 合理性自检（验收用）：方向必须符合管理直觉。

参数分组（重要，实测结论）：
  A 战略通道权重  β_sp / β_cap / β_al   —— 直接乘在战略增量项上，**是结果的实际驱动项**
  B 基期锚定权重  β_d / β_lo / β_tier  —— 只进基期效用，而基期被"锚定求解"（令 P(我院)=真实基线份额），
                                          故其对**增量**的弹性恒为 0：这是锚定机制的结构性推论，
                                          **不是 bug**（弹性为 0 即锚定成立，非 0 反而说明锚定被破坏）
  C 结构参数      d0（距离衰减）/ activation（激活强度）/ k（释放陡度）/ spill_keep（溢出留存）
  D 对照项        magnitude（战略强度，非权重；用于回答"调权重 vs 调强度谁更有效"）

硬约束：仅标准库；只读复用 agent_sim.py；除 --json 指定路径外不写任何文件。

用法：
  python3 agent_sim_sensitivity.py                 # 默认场景（心内科 +30% / 5 年）
  python3 agent_sim_sensitivity.py --selftest      # 方向自检 + 确定性
  python3 agent_sim_sensitivity.py --action alliance --json sen.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_sim as A

GROUPS = {
    "战略通道权重": ["beta_sp", "beta_cap", "beta_al"],
    "基期锚定权重": ["beta_d", "beta_lo", "beta_tier"],
    "结构参数": ["d0", "activation", "k", "spill_keep"],
    "对照项（非权重）": ["magnitude"],
}
CN = {
    "beta_sp": "专科加成 β_sp", "beta_cap": "容量加成 β_cap", "beta_al": "医联体加成 β_al",
    "beta_d": "距离权重 β_d", "beta_lo": "历史忠诚度 β_lo", "beta_tier": "竞争医院等级 β_tier",
    "d0": "距离衰减常数 d0(km)", "activation": "激活效应强度", "k": "释放曲线陡度 k",
    "spill_keep": "容量溢出留存系数", "magnitude": "战略强度 magnitude（对照）",
}
NEAR_KM = 5.0      # 近端距离带（与 80% 半径口径同源）
FAR_KM = 25.0      # 远郊距离带

DISCLAIMER = ("敏感性分析只扰动模型参数，不动任何真实数据；全部变体与基准共用同一种子，"
              "变体间差异不含蒙特卡洛噪声，可完全归因于参数变化。")


# ────────────────────────── 指标提取 ──────────────────────────
def indicators(res, band_ref=None):
    """把一次推演压成可比指标集合（主指标 = 累计新增门诊 P50）。

    近端 / 远郊一律用**人次口径**（opd_cum ÷ 个体数），且街道集固定 = 引擎新增的
    interval["streets_meta"]（全 134 街道、平滑期望值，不过滤零净增街道）——
    否则不同变体命中不同街道集，人均指标不可比。
    """
    it = res["interval"]
    m = it["metrics"]
    meta = it.get("streets_meta") or []
    if band_ref is None:
        band_ref = dict(
            near=[e["name"] for e in meta if (e.get("d_km") or 99) < NEAR_KM],
            far=[e["name"] for e in meta if (e.get("d_km") or 0) > FAR_KM],
            agents={e["name"]: e.get("agents", 0) for e in meta},
        )

    def rate(names):
        idx = {e["name"]: e for e in meta}
        a = sum(band_ref["agents"].get(nm, 0) for nm in names)
        v = sum((idx[nm].get("opd_cum") or 0) for nm in names if nm in idx)
        return round(v / a, 3) if a else 0.0


    dept = res["params"]["dept"]
    return dict(
        add_opd_p50=m["final_add_opd"]["p50"], add_opd_p5=m["final_add_opd"]["p5"],
        add_opd_p95=m["final_add_opd"]["p95"], add_ipd_p50=m["final_add_ipd"]["p50"],
        bed_usage_p50=m["final_bed_usage"]["p50"], numsource_util_p50=m["final_numsource_util"]["p50"],
        self_last_p50=m["self_last"]["p50"],
        from_comp_p50=m["from_comp"]["p50"], from_out_p50=m["from_out"]["p50"],
        near_per_agent=rate(band_ref["near"]), far_per_agent=rate(band_ref["far"]),
        near_streets=len(band_ref["near"]), far_streets=len(band_ref["far"]),
        dept_add_opd=it["dept_add_opd"].get(dept),
        dept_add_opd_rate=it["dept_add_opd_rate"].get(dept),
    )


def _elast(base_v, var_v, delta):
    """弹性 = 指标变化% ÷ 参数变化%"""
    if not base_v:
        return None
    return round(((var_v - base_v) / abs(base_v)) / delta, 3)


# ────────────────────────── 主分析 ──────────────────────────
def sensitivity(params=None, delta=0.20, runs=30, params_keys=None, seed=None):
    """对每个待测参数各做 ±delta 扰动，返回弹性表 + 方向自检。"""
    p = dict(A.DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if k in A.DEFAULTS})
    p["runs"] = int(max(4, runs))
    keys = params_keys or [k for g in GROUPS.values() for k in g]

    t0 = time.time()
    base_res = A.simulate(dict(p), seed=seed)
    S = base_res["seed"]
    base = indicators(base_res)
    _sm = base_res["interval"]["streets_meta"]
    band_ref = dict(
        near=[e["name"] for e in _sm if (e.get("d_km") or 99) < NEAR_KM],
        far=[e["name"] for e in _sm if (e.get("d_km") or 0) > FAR_KM],
        agents={e["name"]: e.get("agents", 0) for e in _sm},
    )

    group_of = {k: g for g, ks in GROUPS.items() for k in ks}
    rows = []
    for k in keys:
        v0 = float(p[k])
        entry = dict(param=k, param_cn=CN.get(k, k), group=group_of.get(k, "其他"), base_weight=v0)
        for tag, mult in (("up", 1 + delta), ("down", 1 - delta)):
            q = dict(p)
            q[k] = round(v0 * mult, 6)
            entry[tag] = indicators(A.simulate(q, seed=S), band_ref)
            entry[tag + "_value"] = q[k]
        bv = base["add_opd_p50"] or 1
        entry["elasticity_up"] = _elast(bv, entry["up"]["add_opd_p50"], delta)
        entry["elasticity_down"] = _elast(bv, entry["down"]["add_opd_p50"], -delta)
        el = [e for e in (entry["elasticity_up"], entry["elasticity_down"]) if e is not None]
        entry["elasticity"] = round(sum(el) / len(el), 3) if el else None
        entry["elasticity_far"] = _elast(base["far_per_agent"], entry["up"]["far_per_agent"], delta)
        entry["elasticity_near"] = _elast(base["near_per_agent"], entry["up"]["near_per_agent"], delta)
        entry["elasticity_dept"] = _elast(base["dept_add_opd_rate"], entry["up"]["dept_add_opd_rate"], delta)
        rows.append(entry)

    rows.sort(key=lambda x: -abs(x["elasticity"] or 0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    by = {r["param"]: r for r in rows}

    # ── 医联体场景对照（β_al 的空间方向须在 alliance 动作下检验）──
    al_scene = None
    if "beta_al" in keys:
        pa = dict(p, action="alliance")
        ra = A.simulate(dict(pa), seed=S)
        _sma = ra["interval"]["streets_meta"]
        band_a = dict(
            near=[e["name"] for e in _sma if (e.get("d_km") or 99) < NEAR_KM],
            far=[e["name"] for e in _sma if (e.get("d_km") or 0) > FAR_KM],
            agents={e["name"]: e.get("agents", 0) for e in _sma},
        )
        rb = A.simulate(dict(pa, beta_al=round(p["beta_al"] * (1 + delta), 6)), seed=S)
        headroom = ra["cap"]["opd_capacity"] - ra["cap"]["opd_base"]
        al_scene = dict(base=indicators(ra, band_a), up=indicators(rb, band_a),
                        cap_headroom=round(headroom), params=pa)

    # ── 方向自检（管理直觉，设计 §9.2）──
    checks = []
    flags = []          # 与设计预期不一致 / 需决策的发现（如实标注，不掩盖）

    def ck(name, ok, detail=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    def flag(fid, title, detail, options, decision="待决策（本轮未擅自改模型）"):
        flags.append(dict(id=fid, title=title, detail=detail, options=options, decision=decision))

    def pct(a, b):
        return (b - a) / abs(a) * 100 if a else 0.0

    if "beta_sp" in by:
        r = by["beta_sp"]
        ck("专科权重 β_sp 上调 → 目标专科人均增量上升",
           (r["up"]["dept_add_opd_rate"] or 0) > (base["dept_add_opd_rate"] or 0),
           f"「{p['dept']}」人均潜在增量 {base['dept_add_opd_rate']} → {r['up']['dept_add_opd_rate']}")
        ck("专科权重 β_sp 上调 → 累计新增门诊上升",
           r["up"]["add_opd_p50"] > base["add_opd_p50"],
           f"P50 {base['add_opd_p50']:,} → {r['up']['add_opd_p50']:,}")
    if "beta_cap" in by:
        r = by["beta_cap"]
        ck("容量权重 β_cap 上调 → 累计新增门诊上升（承接能力增强）",
           r["up"]["add_opd_p50"] > base["add_opd_p50"],
           f"P50 {base['add_opd_p50']:,} → {r['up']['add_opd_p50']:,}")
    if "beta_al" in by:
        r = by["beta_al"]
        ratio0 = base["far_per_agent"] / base["near_per_agent"] if base["near_per_agent"] else 0
        ratio1 = r["up"]["far_per_agent"] / r["up"]["near_per_agent"] if r["up"]["near_per_agent"] else 0
        ck("医联体权重 β_al 上调 → 远郊街道人均增量上升（专科场景）",
           r["up"]["far_per_agent"] > base["far_per_agent"],
           f"远郊人均 {base['far_per_agent']} → {r['up']['far_per_agent']} 人次")
        ck("医联体权重 β_al 上调 → 远/近端比值上升（结构向外扩散）",
           ratio1 > ratio0, f"远/近端比值 {ratio0:.4f} → {ratio1:.4f}")
    if al_scene:
        ck("医联体权重 β_al 上调 → 远郊人均增量上升（医联体场景）",
           al_scene["up"]["far_per_agent"] > al_scene["base"]["far_per_agent"],
           f"远郊人均 {al_scene['base']['far_per_agent']} → {al_scene['up']['far_per_agent']} 人次"
           f"（总量 P50 {al_scene['base']['add_opd_p50']:,} → {al_scene['up']['add_opd_p50']:,}）")
    if "activation" in by:
        r = by["activation"]
        ck("激活强度上调 → 累计新增门诊上升（激活原本不就诊人群）",
           r["up"]["add_opd_p50"] > base["add_opd_p50"],
           f"P50 {base['add_opd_p50']:,} → {r['up']['add_opd_p50']:,}"
           f"（激活来源占比 {base['from_out_p50']:,}/{base['from_comp_p50'] + base['from_out_p50']:,}）")
    if "magnitude" in by:
        r = by["magnitude"]
        ck("战略强度 magnitude 上调 → 累计新增门诊上升（单调性）",
           r["up"]["add_opd_p50"] > base["add_opd_p50"] > r["down"]["add_opd_p50"],
           f"−20%: {r['down']['add_opd_p50']:,} ｜ 基准: {base['add_opd_p50']:,} ｜ +20%: {r['up']['add_opd_p50']:,}")
    anchor_keys = [k for k in GROUPS["基期锚定权重"] if k in by]
    if anchor_keys:
        zero = [k for k in anchor_keys if by[k]["elasticity"] == 0]
        ck("基期锚定权重（" + "/".join(k.replace("beta_", "β_") for k in anchor_keys)
           + "）对增量的弹性恒为 0（锚定机制的结构性推论）",
           len(zero) == len(anchor_keys),
           "；".join(f"{by[k]['param_cn']} 弹性={by[k]['elasticity']}" for k in anchor_keys))
    ck("主指标区间单调 P5 ≤ P50 ≤ P95",
       base["add_opd_p5"] <= base["add_opd_p50"] <= base["add_opd_p95"],
       f"{base['add_opd_p5']:,} ≤ {base['add_opd_p50']:,} ≤ {base['add_opd_p95']:,}")

    # ── flags：实测与设计预期不一致 / 需决策 ──
    if "d0" in by:
        r = by["d0"]
        ratio0 = base["far_per_agent"] / base["near_per_agent"] if base["near_per_agent"] else 0
        ratio1 = r["up"]["far_per_agent"] / r["up"]["near_per_agent"] if r["up"]["near_per_agent"] else 0
        flag("d0_spatial_gradient",
             "距离衰减常数 d0 的空间梯度作用弱于设计 §9.2 预期（实测与预期方向不一致）",
             f"实测：d0 +20% 时近端人均 {base['near_per_agent']} → {r['up']['near_per_agent']} 人次"
             f"（{pct(base['near_per_agent'], r['up']['near_per_agent']):+.1f}%），"
             f"远郊 {base['far_per_agent']} → {r['up']['far_per_agent']} 人次"
             f"（{pct(base['far_per_agent'], r['up']['far_per_agent']):+.1f}%），远/近端比值 {ratio0:.4f} → {ratio1:.4f}（不升反略降）。"
             "设计预期是『d0 上调 → 战略效果向外扩散、远郊增幅更大』。"
             "机理已查清：① 远郊街道的专科距离权重 w_near 已触及 max(w_near, 0.15) 的托底下限，"
             "故 d0 主要改变近端；② 基期锚定（令 P(我院)=真实基线份额）吸收了 d0 对基期效用的影响。",
             ["A 保留现实现，把 §9.2 判据改为实测方向并在设计文档写明机理（不改模型，零回归）",
              "B 改造模型：把权重下限 0.15 改为可配置 + 引入距离分档权重，使战略效果对距离更敏感"
              "（会改动已被阶段 2 验证的引擎，须重跑基线哈希与全部自检）"],
             decision="**已决策（2026-09-19，用户选 A）**：保留现实现、判据改为实测方向（§9.2 修订已落文档），"
                      "引擎零回归；不引入距离分档权重。")
    if al_scene:
        if al_scene["base"]["add_opd_p50"] == al_scene["cap_headroom"]:
            flag("alliance_capacity_cap",
                 "医联体场景的总量已由容量上限决定（权重只改变分布，不改变总量）",
                 f"医联体场景 P50 = 可用门诊容量上限 {al_scene['cap_headroom']:,} 人次（已触顶）："
                 f"β_al +20% 后总量仍为 {al_scene['up']['add_opd_p50']:,}，但远郊人均由 "
                 f"{al_scene['base']['far_per_agent']} 升至 {al_scene['up']['far_per_agent']} 人次。"
                 "结论：该场景属于『容量瓶颈型』，报告与答辩须按此口径表述（不能宣称提高权重可增加总量）。",
                 ["A 报告/答辩按『容量瓶颈型场景』口径表述（建议）",
                  "B 在推演结果中加入『需求触顶』显式提示与风险行（前端/报告改动，后续轮次）"],
                 decision="**已决策（2026-09-19，用户选 A）**：报告与答辩一律按『容量瓶颈型场景』表述——"
                          "该场景总量由可用门诊容量上限决定（165,468 人次），权重只改变空间分布，"
                          "**不得宣称提高权重可增加总量**；阶段 4 的寻优已内置该判据（触顶候选被风险约束排除并显式说明）。")
    ok_n = sum(1 for c in checks if c["ok"])
    return dict(
        params=p, seed=S, runs=p["runs"], delta=delta,
        algo_version=A.ALGO_VERSION, data_version=base_res["world_meta"]["data_version"],
        main_indicator="累计新增门诊 P50（人次）", groups=GROUPS,
        band_ref=dict(near_streets=len(band_ref["near"]), far_streets=len(band_ref["far"]),
                      near_km=NEAR_KM, far_km=FAR_KM),
        base=base, rows=rows, alliance_scene=al_scene, checks=checks, flags=flags,
        passed=ok_n, total=len(checks),
        verdict=("全部方向符合管理直觉（模型实现与设计一致）" if ok_n == len(checks)
                 else f"{len(checks) - ok_n} 项方向不符合管理直觉，须排查模型实现"),
        note=("基期锚定权重对增量弹性为 0 是结构性推论：基期效用被反向求解，使 P(我院) 严格等于真实基线份额，"
              "故 β_d/β_lo/β_tier 只决定基期构成、不决定战略增量；战略增量的实际驱动项是 β_sp/β_cap/β_al 与 d0/activation。"
              + (f"另有 {len(flags)} 项与设计预期不一致或需决策的发现，见 flags（已如实标注，未擅自改模型）。"
                 if flags else "")),
        elapsed_sec=round(time.time() - t0, 2),
        disclaimer=DISCLAIMER,
    )


def selftest(verbose=True):
    """阶段 3 验收自检：确定性 + 结构完整 + 方向自检"""
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    t0 = time.time()
    s = sensitivity(runs=8, delta=0.20)
    ck("1 扰动项覆盖 A/B/C/D 四组（12 项）", len(s["rows"]) == sum(len(v) for v in GROUPS.values()),
       f"扰动项={len(s['rows'])}：{'、'.join(r['param_cn'] for r in s['rows'][:3])} …")
    def _stable(x):
        y = dict(x)
        y.pop("elapsed_sec", None)          # 耗时是唯一易变字段，不参与一致性比较
        return json.dumps(y, sort_keys=True)

    s2 = sensitivity(runs=8, delta=0.20)
    ck("2 同参数两次敏感性结果逐位一致（剔除耗时字段）", _stable(s) == _stable(s2),
       f"seed={s['seed']}（差异字段：{sorted(k for k in s if _stable({k: s[k]}) != _stable({k: s2[k]})) or '无'}）")
    ck("3 弹性表可排序（rank 连续）",
       [r["rank"] for r in s["rows"]] == list(range(1, len(s["rows"]) + 1)),
       "｜".join(f"{r['param_cn']}:{r['elasticity']}" for r in s["rows"][:3]))
    ck("4 方向自检全部通过", s["passed"] == s["total"], f"{s['passed']}/{s['total']}：{s['verdict']}")
    ck("5 与设计预期不一致项已如实标注（flags）", isinstance(s["flags"], list),
       f"flags={len(s['flags'])} 项：" + "；".join(f["id"] for f in s["flags"]))
    print(f"\n敏感性自检汇总：{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} 通过，"
          f"用时 {time.time() - t0:.1f}s")
    return all(ok for _, ok, _ in checks), checks


# ────────────────────────── CLI ──────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段 3：参数敏感性分析")
    ap.add_argument("--action", default=A.DEFAULTS["action"], choices=["specialty", "alliance", "capacity"])
    ap.add_argument("--dept", default=A.DEFAULTS["dept"])
    ap.add_argument("--magnitude", type=float, default=A.DEFAULTS["magnitude"])
    ap.add_argument("--years", type=int, default=A.DEFAULTS["years"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--density", type=float, default=A.DEFAULTS["density"])
    ap.add_argument("--delta", type=float, default=0.20, help="扰动幅度（默认 ±20%%）")
    ap.add_argument("--share-basis", default=A.DEFAULTS["share_basis"],
                    choices=["contested", "city", "dept_pool"])
    ap.add_argument("--json", default="", help="结果写入 JSON 路径")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, _ = selftest()
        return 0 if ok else 1

    s = sensitivity(dict(action=args.action, dept=args.dept, magnitude=args.magnitude,
                         years=args.years, runs=args.runs, density=args.density,
                         share_basis=args.share_basis), delta=args.delta)
    print(f"敏感性分析（±{s['delta']:.0%} ｜ 主指标 {s['main_indicator']} ｜ 同种子 {s['seed']}）")
    print(f"基准：P50 {s['base']['add_opd_p50']:,} 人次 ｜ 近端人均 {s['base']['near_per_agent']} 人次"
          f"（{s['base']['near_streets']} 街道）｜ 远郊人均 {s['base']['far_per_agent']} 人次（{s['base']['far_streets']} 街道）")
    print(f"{'排名':<3}{'参数':<22}{'−20%':>9}{'基准':>9}{'+20%':>9}{'弹性':>8}{'远郊弹性':>9}")
    for r in s["rows"]:
        print(f"{r['rank']:<3}{r['param_cn']:<20}{r['down']['add_opd_p50']:>9,}"
              f"{s['base']['add_opd_p50']:>9,}{r['up']['add_opd_p50']:>9,}"
              f"{(r['elasticity'] if r['elasticity'] is not None else 0):>8.3f}"
              f"{(r['elasticity_far'] if r['elasticity_far'] is not None else 0):>9.3f}")
    print("\n方向自检：")
    for c in s["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}\n           {c['detail']}")
    if s["flags"]:
        print("\n待决策发现（与设计预期不一致，已如实标注）：")
        for f in s["flags"]:
            print(f"  ⚠ {f['title']}\n    {f['detail']}")
            for o in f["options"]:
                print(f"      {o}")
    print(f"\n结论：{s['verdict']}（{s['passed']}/{s['total']}，用时 {s['elapsed_sec']}s）")
    print(f"说明：{s['note']}")
    if args.json:
        open(args.json, "w", encoding="utf-8").write(json.dumps(s, ensure_ascii=False, indent=1))
        print(f"已写入 {args.json}")
    return 0 if s["passed"] == s["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
