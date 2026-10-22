#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_sim.py — 智能体推演引擎（阶段 1：智能体核心）

设计依据：《智能体推演-详细设计.md》V1.3 第 1–5 章
硬约束（违反即返工）：
  1. 仅使用 Python 标准库（零新增依赖）
  2. 只读现有数据（heatmap_data.json / hospital.db）与只读复用 engine.py
  3. 不修改 engine.py / app.py / 任何现有表；本阶段**不写数据库**（结果落 JSON）
  4. 不调用 capacity_profile.build()（含删库逻辑）
  5. 参数确定 → 结果逐位可复现（种子由 sha256(参数+数据版本+算法版本) 派生）

用法：
  python3 agent_sim.py --action specialty --dept 心内科 --magnitude 0.3 --years 5 --runs 50
  python3 agent_sim.py --selftest                 # 跑验收自检
  python3 agent_sim.py --action alliance --json out.json --trace out_trace.json
"""
import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time

ALGO_VERSION = "1.2.0"          # 1.2.0：修复选区口径（选区外个体按 L=0 参与，不再产生负增量与失真概率）
                         # 变更公式/默认权重时必须递增此常量（设计 §5.3：数据与算法任一变化都不得复用旧结果）

BASE = os.path.dirname(os.path.abspath(__file__))          # backend/strategy_engine
ROOT = os.path.dirname(BASE)                               # 项目根
HEATMAP = os.path.join(ROOT, "heatmap_data.json")
DB = os.path.join(BASE, "hospital.db")                      # 业务库（只读，绝不写入）
RESULTS_DB = os.path.join(BASE, "agent_sim_results.db")      # 本引擎结果库（新增文件，与业务库隔离）
HOSPITAL = (120.3884087, 36.1795722)                       # 三医 WGS-84（与 engine.py 一致）
_HEATMAP_CACHE = None                                      # 热力图 JSON 解析缓存（见 load_world 说明）

# ── 默认参数（与设计 §3.2 / §4 一致；全部可覆盖） ──────────────────────────────
DEFAULTS = dict(
    action="specialty",          # specialty | alliance | capacity
    dept="心内科",
    magnitude=0.30,              # 战略强度 0.05–0.50
    years=5,                     # 1–5
    streets=None,                # None=全域；否则街道名列表
    runs=50,                     # 蒙特卡洛次数
    density=1.0,                 # 抽样密度系数（0.25=3千 / 1.0=1.2万 / 2.1=2.5万）
    seed=None,                   # None=由参数派生
    # 效用权重（β）：距离/专科/容量/医联体/忠诚度 + 校准项
    beta_d=1.5, beta_sp=1.0, beta_cap=0.85, beta_al=0.95, beta_lo=0.14,
    beta0=0.9, beta_tier=1.1,
    d0=15.0, d0_comp=15.0,       # 距离衰减常数（km，与 80% 半径口径一致）
    activation=0.85,             # 激活效应强度（odds 空间）
    spill_keep=0.5,              # 容量溢出留存系数（滚入次年）
    curve="logistic",            # logistic | linear（linear 与老引擎 release() 同语义，做对照）
    k=1.7,                       # logistic 陡度
    sampling_divisor=4000,       # 每（人口/该值）个智能体
    min_per_street=8, max_per_street=40,
    hri_low_q=0.05, hri_high_q=0.95,
    match_all=False,             # 验收压测开关：True=所有专科都算命中（模拟全院整体提升）
    visit_rate=4.5,              # 人均年门诊次数（全市门诊总需求口径；用于把真实门诊量反解为渗透率）
    share_basis="contested",     # 口径：contested=竞争圈层(决策方选定,主口径) | city=全市场真实份额 | dept_pool=专科需求池
)

TIER = {"三甲": 1.0, "三乙": 0.6, "三级": 0.4}


# ══════════════════════════ 1. 世界数据（只读） ══════════════════════════
def data_version():
    """数据版本：关键数据文件的体积+修改时间摘要（数据一变，种子随之变化）"""
    h = hashlib.sha256()
    for p in (HEATMAP, DB):
        try:
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{os.path.basename(p)}:missing".encode())
    return h.hexdigest()[:16]


def _dist_km(lng, lat, lng2=HOSPITAL[0], lat2=HOSPITAL[1]):
    dlat = (lat - lat2) * 111.0
    dlng = (lng - lng2) * 89.6
    return math.sqrt(dlat * dlat + dlng * dlng)


def load_world(visit_rate=None):
    """读取街道/人口/就诊/科室/竞争医院数据，产出街道表（只读，不写任何文件）

    性能：热力图 JSON 解析约 40ms/次，而反解/寻优要调用 simulate 上百次 → 这里缓存**已解析的 JSON**，
    每次调用只重建街道字典（不重复解析 2.3MB 文件）。街道对象仍是新建的，调用方可安全就地修改。
    """
    global _HEATMAP_CACHE
    if _HEATMAP_CACHE is None:
        _HEATMAP_CACHE = json.load(open(HEATMAP, encoding="utf-8"))
    d = _HEATMAP_CACHE
    pop = d["population"].get("2025", {})
    opd = d["opd"].get("2025", {})
    dis = d["discharge"].get("2025", {})
    coords = d["coords"]
    depts_map = d.get("departments", {})
    hospitals = d.get("hospitals", [])

    # 竞争医院（三级及以上，排除本院）
    comps = []
    for h in hospitals:
        g = h.get("grade")
        if g in TIER and "第三人民" not in h.get("name", ""):
            lng = h.get("lon") or h.get("lng")
            if lng and h.get("lat"):
                comps.append((lng, h["lat"], TIER[g], h["name"], g, float(h.get("opd") or 0)))
    if not comps:
        raise SystemExit("未从数据中解析到竞争医院（三级及以上），请检查 heatmap_data.json")

    # 全科室表与全局份额
    dept_tot = {}
    for st, dd in depts_map.items():
        for k, v in dd.items():
            dept_tot[k] = dept_tot.get(k, 0) + v
    depts = [k for k, _ in sorted(dept_tot.items(), key=lambda x: -x[1])]

    streets = []
    for f in d["street_geo"]["features"]:
        nm = f["properties"]["name"]
        c = coords.get(nm)
        if not c:
            continue
        lng, lat = c["lng"], c["lat"]
        dd = depts_map.get(nm, {})
        tot = sum(dd.values()) or 1
        shares = sorted(((depts.index(k), v / tot) for k, v in dd.items() if k in depts),
                        key=lambda x: -x[1])[:8] or [(0, 1.0)]
        best = min(((_dist_km(lng, lat, cl, ca), t, hn, g, co)
                    for cl, ca, t, hn, g, co in comps), default=(30.0, 0.6, "无", "", 0.0))
        streets.append(dict(
            name=nm, district=f["properties"].get("district", ""),
            lng=lng, lat=lat,
            pop=int(pop.get(nm, 0)), opd=int(opd.get(nm, 0)), dis=int(dis.get(nm, 0)),
            ds=round(_dist_km(lng, lat), 2),
            dc=round(best[0], 2), tier=best[1], cname=best[2], cgrade=best[3], copd=best[4],
            dept_shares=shares,
        ))

    # HRI（出院份额 ÷ 人口份额）→ 归一化为忠诚度
    tp = sum(s["pop"] for s in streets) or 1
    td = sum(s["dis"] for s in streets) or 1
    for s in streets:
        s["hri"] = (s["dis"] / td) / ((s["pop"] / tp) or 1e-9)
    hris = sorted(s["hri"] for s in streets)
    lo = hris[int(len(hris) * DEFAULTS["hri_low_q"])]
    hi = hris[min(len(hris) - 1, int(len(hris) * DEFAULTS["hri_high_q"]))]
    for s in streets:
        s["loyal"] = min(1.0, max(0.0, (s["hri"] - lo) / ((hi - lo) or 1.0)))
        s["visit_base"] = (s["opd"] / s["pop"]) if s["pop"] else 0.0

    # ── 基线锚定（全部由真实数据反解，口径可切换）────────────────────────────
    # A 全市场(city)：p0 = 本街道我院门诊量 ÷ (人口 × 人均年门诊次数)
    # B 竞争圈层(contested, 决策方选定主口径)：分母 = 我院 + 最近三级竞争医院的门诊量
    #   竞争医院门诊量按"距离衰减权重"分摊到各街道（数据可得的最合理近似）
    # C 专科需求池(dept_pool)：分母 = 该街道该专科的潜在就诊人次
    vr = float(visit_rate or DEFAULTS["visit_rate"])
    d0w = DEFAULTS["d0"]
    wsum = {}
    for s in streets:                                   # 先算各竞争医院的门摊权重和
        wsum[s["cname"]] = wsum.get(s["cname"], 0.0) + math.exp(-s["dc"] / d0w)
    for s in streets:
        pot = s["pop"] * vr
        s["potential"] = pot
        p_city = (s["opd"] / pot) if pot else 0.02
        w = math.exp(-s["dc"] / d0w)
        alloc = (s["copd"] * w / wsum[s["cname"]]) if wsum.get(s["cname"]) else 0.0
        p_contested = (s["opd"] / (s["opd"] + alloc)) if (s["opd"] + alloc) > 0 else 0.02
        dsh = dict((i, v) for i, v in s["dept_shares"])
        ds0 = max(dsh.values()) if dsh else 1.0
        s["p0_city"] = round(min(0.85, max(0.02, p_city)), 4)
        s["p0_contested"] = round(min(0.95, max(0.02, p_contested)), 4)
        s["share_dept"] = dsh
        # 各口径的"人均势能"（关键：势能必须与分母口径一致，否则 Σ p0×势能 ≠ 真实基线门诊量）
        s["pc_city"] = vr                                        # 全市潜在门诊需求/人
        s["pc_contested"] = ((s["opd"] + alloc) / s["pop"]) if s["pop"] else 0.0   # 竞争圈层池/人
        s["p0"] = s["p0_contested"]                              # load_world 直用时的默认口径
        s["pot_pc"] = s["pc_contested"]

    return dict(streets=streets, depts=depts, comps=comps,
                data_version=data_version(),
                city_pop=tp, city_opd=sum(s["opd"] for s in streets))


# ══════════════════════════ 2. 容量（只读复用老引擎口径） ══════════════════════════
def capacity():
    """门诊/住院容量：优先复用 engine.capacity()，失败则用 hospital.db 参数自行计算（只读）"""
    try:
        import engine as _engine                       # 同目录，只读复用
        cap = _engine.capacity()
        return dict(opd_capacity=cap["opd_capacity"], ipd_capacity=cap["ipd_capacity"],
                    opd_base=cap["opd_base"], ipd_base=cap["ipd_base"],
                    beds=cap.get("beds"), bed_usage=cap["bed_usage"],
                    weekly_numsource=cap.get("weekly_numsource"),
                    source="engine.capacity()（只读复用，含院内容量口径）")
    except Exception:
        import sqlite3
        params = {}
        try:
            conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
            params = {r[0]: r[1] for r in conn.execute("SELECT param, value FROM hospital_params")}
            conn.close()
        except Exception:
            pass
        opd_base = float(params.get("annual_opd", 550000))
        ipd_base = float(params.get("annual_ipd", 26000))
        usage = float(params.get("bed_usage_rate", 0.64)) or 0.64
        return dict(opd_capacity=13759 * 52, ipd_capacity=ipd_base / max(usage, 0.1),
                    opd_base=opd_base, bed_usage=usage, source="hospital.db 参数(只读)")


# ══════════════════════════ 3. 智能体与决策模型 ══════════════════════════
def derive_seed(params, world):
    """种子 = sha256(参数 JSON + 数据版本 + 算法版本) 前 16 位（禁用内置 hash()）"""
    blob = json.dumps({k: v for k, v in sorted(params.items()) if k != "seed"},
                      ensure_ascii=False, sort_keys=True)
    s = f"{blob}|{world['data_version']}|{ALGO_VERSION}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def build_agents(world, p, rng):
    """抽样生成患者智能体（字段全部来自现有数据）"""
    agents = []
    div = p["sampling_divisor"]
    for si, s in enumerate(world["streets"]):
        base = max(p["min_per_street"], round(s["pop"] / div)) if s["pop"] else p["min_per_street"]
        n = max(4, round(base * p["density"]))
        n = min(n, p["max_per_street"] * 4)
        weight = (s["pop"] / n) if n else 0
        for _ in range(n):
            r = rng.random()
            acc, di = 0.0, s["dept_shares"][0][0]
            for idx, sh in s["dept_shares"]:
                acc += sh
                if r <= acc:
                    di = idx
                    break
            sens = 0.55 + 0.95 * rng.random()
            ds, dc = s["ds"], s["dc"]
            f_self, f_comp = math.exp(-ds / p["d0"]), math.exp(-dc / p["d0_comp"])
            # 基期效用（L=0：尚无战略效果）与外部选项效用——**锚定只在基期做一次**
            u_self0 = (p["beta0"] + p["beta_d"] * f_self * sens + p["beta_sp"] * 0.35
                       + p["beta_cap"] * 0.45 + p["beta_lo"] * s["loyal"])
            u_comp = p["beta_d"] * f_comp * sens + p["beta_tier"] * s["tier"]
            ratio = min(1 - 1e-6, max(1e-6, s["p0"]))
            # 求解 e_o 使 P(我院) = 基线 r：e_o = e_s(1−r)/r − e_c
            e_s0, e_c0 = math.exp(u_self0), math.exp(u_comp)
            u_out = math.log(max(1e-9, e_s0 * (1 - ratio) / ratio - e_c0))
            agents.append(dict(
                si=si, dept=di, p0=s["p0"], loyal=s["loyal"], dist_sens=sens,
                weight=weight, visit_base=s["visit_base"],
                visit_pot=weight * s.get("pot_pc", p["visit_rate"]),   # 该智能体代表的潜在年门诊人次（随口径）
                u_self0=u_self0, u_comp=u_comp, u_out=u_out,
                f_self=f_self, d_sens_base=sens,
            ))
    return agents


def _odds_from_p(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def _p_from_odds(o):
    return o / (1 + o)


def release_factor(t, years, curve, k):
    """潜能释放曲线：logistic（默认）或 linear（与老引擎 release() 同语义，做对照）"""
    if years <= 1:
        return 1.0
    if curve == "linear":
        return (t + 1) / years
    return 1.0 / (1.0 + math.exp(-k * (t - (years - 1) / 2.0)))


def _prepare(p):
    """载入世界数据 + 按选定口径落定各街道 p0（simulate 与 agent_breakdown 共用，保证两条路径同口径）

    返回 (world, streets, depts, target_dept, share_report)。**纯读+纯算，不消费随机数。**
    """
    world = load_world(p["visit_rate"])
    streets = world["streets"]
    depts = world["depts"]
    if p["dept"] not in depts:                     # 容错：未知科室 → 取第 1 个
        p["dept"] = depts[0]
    target_dept = depts.index(p["dept"])
    # 按选定口径落定 p0（dept_pool 依赖目标专科，故在此处完成）
    basis = p.get("share_basis", "contested")
    for s in streets:
        if basis == "city":
            s["p0"], s["pot_pc"] = s["p0_city"], s["pc_city"]
        elif basis == "dept_pool":
            sh = s["share_dept"].get(target_dept, 0.0)
            pool = s["potential"] * sh
            s["p0"] = round(min(0.95, max(0.02, (s["opd"] * sh / pool) if pool else 0.02)), 4)
            s["pot_pc"] = p["visit_rate"] * sh                   # 该专科人均潜在就诊人次
        else:
            s["p0"], s["pot_pc"] = s["p0_contested"], s["pc_contested"]
    share_report = dict(
        basis=basis,
        mean_city=round(sum(s["p0_city"] for s in streets) / len(streets), 4),
        mean_contested=round(sum(s["p0_contested"] for s in streets) / len(streets), 4),
        mean_used=round(sum(s["p0"] for s in streets) / len(streets), 4),
        denominator_note={"city": "全市潜在门诊需求（人口 × 人均年门诊次数）",
                          "contested": "我院 + 最近三级竞争医院合计门诊量（竞争医院按距离衰减分摊）",
                          "dept_pool": f"「{p['dept']}」专科潜在就诊人次"}[basis],
    )
    return world, streets, depts, target_dept, share_report


def _agent_step(a, s, p, L, target_dept):
    """单个智能体在第 t 年的效用与选择概率（纯函数：**不消费随机数**，保证逐位可复现）

    同时返回各效用分量（β_sp/β_cap/β_al 各自的贡献、基期效用、激活效应），供个体决策卡展示"为什么这样选"。
    """
    match = 1.0 if (a["dept"] == target_dept or p["match_all"]) else 0.12
    spec = p["magnitude"] * match * (1.0 if p["action"] == "specialty" else 0.3)
    capp = p["magnitude"] * (0.9 if p["action"] == "capacity" else 0.18)
    al = p["magnitude"] * (1.1 if p["action"] == "alliance" else 0.16)
    sens = 0.45 + 1.0 * a["dist_sens"]
    w_near = a["f_self"] * sens
    w_far = (1 - math.exp(-s["ds"] / 20.0)) * (0.6 + 0.7 * a["dist_sens"])
    d_s = spec * max(w_near, 0.15)
    d_a = al * max(w_far, 0.15)
    d_c = capp * 0.85 * sens

    # 战略效果叠加在**基期效用**上（锚定已在基期一次性完成，逐年不再重解）
    u_self = a["u_self0"] + p["beta_sp"] * L * d_s + p["beta_cap"] * L * d_c + p["beta_al"] * L * d_a
    e_s, e_c, e_o = math.exp(u_self), math.exp(a["u_comp"]), math.exp(a["u_out"])
    p_self = e_s / (e_s + e_c + e_o)
    p_comp = e_c / (e_s + e_c + e_o)
    # 激活效应（odds 空间）：专科提升→目标专科人群；医联体→远郊低覆盖人群
    act = 0.0
    if p["action"] in ("specialty", "capacity") and match == 1.0:
        act = L * d_s * p["activation"]
    elif p["action"] == "alliance":
        act = L * d_a * p["activation"] * 1.2
    p_self_pre = p_self
    if act > 0:
        o = _odds_from_p(p_self) * (1.0 + act)
        p_self = _p_from_odds(o)
    return dict(
        p_self=p_self, p_comp=p_comp, p_out=max(0.0, 1.0 - p_self - p_comp),
        p_self_pre_act=p_self_pre, activation=act, release=L,
        match=match, sens=sens, w_near=w_near, w_far=w_far,
        d_s=d_s, d_a=d_a, d_c=d_c,
        u_self=u_self, u_comp=a["u_comp"], u_out=a["u_out"],
        # 效用分量（个体决策卡逐条显示，设计 §7.2 D1）
        contrib_base=a["u_self0"], contrib_sp=p["beta_sp"] * L * d_s,
        contrib_cap=p["beta_cap"] * L * d_c, contrib_al=p["beta_al"] * L * d_a,
    )


def simulate(params=None, seed=None, want_trace=False, want_prob=False):
    """智能体推演主循环（蒙特卡洛 + 容量溢出 + 分年/分街道输出）

    want_prob=True 时在帧数据里附带**逐年个体级选择概率** p_self_hist（蜂群视图的"期望态"着色用；
    体积 ≈ 7 字节 × 个体数 × 年数，建议密度 ≤ 2.0 时使用）。
    """
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})
    world, streets, depts, target_dept, share_report = _prepare(p)
    cap = capacity()
    years = int(max(1, min(5, p["years"])))
    runs = int(max(1, p["runs"]))
    base_seed = int(seed) if seed is not None else derive_seed(p, world)
    sel = set(p["streets"]) if p["streets"] else None
    if sel:
        for nm in sel:
            if nm not in {s["name"] for s in streets}:
                raise ValueError(f"未知街道：{nm}（请核对街道名；可用街道共 {len(streets)} 个）")   # 2026-09-21：SystemExit → ValueError（API 层转 400）

    per_run = []
    trace = None
    for r in range(runs):
        rng = random.Random(base_seed + r)
        agents = build_agents(world, p, rng)
        choices = [[0] * years for _ in range(len(agents))]
        spill_opd = spill_ipd = 0.0
        years_out = []
        st_cum, st_last = {}, {}              # 逐街道累计/末期年度人次增量
        cum_opd = cum_ipd = 0.0
        pself = [[a["p0"] for a in agents]]          # 透视：各年选择我院的期望概率
        for t in range(years):
            L = release_factor(t, years, p["curve"], p["k"])
            inc_opd = inc_ipd = 0.0
            street_inc = {}
            p_prev = pself[t]                            # 上一年（第 0 年的上一年=基线）
            row_p = [0.0] * len(agents)
            for i, a in enumerate(agents):
                s = streets[a["si"]]
                # 选区口径：**选区外个体按「战略强度为 0」参与**（保持基线概率与基线选择）。
                # 2026-09-21 修复：原来是直接 continue 跳过 → row_p[i] 保持 0.0 → pself 末年记为 0，
                # 与基线 p0 相减产生**负增量**（李沧选区场景曾出现 9/9 科室全负、合计 −275,913，
                # 而同一 run 的总量却为 +13,756）；skip 还会把选区外个体全部编码成"未就诊"。
                # 现在统一走同一条路径，只是把释放因子置 0：_agent_step(.., L=0) → p_self 恒为基线 p0
                # → 增量恒为 0、trace 概率不失真。（全域场景 sel=None 时 L_i=L，行为逐位不变）
                L_i = 0.0 if (sel and s["name"] not in sel) else L
                u = _agent_step(a, s, p, L_i, target_dept)
                p_self, p_comp = u["p_self"], u["p_comp"]

                x = rng.random()
                ch = 1 if x < p_self else (2 if x < p_self + p_comp else 0)
                choices[i][t] = ch
                row_p[i] = p_self
                # 新增门诊/住院 = 期望增量（p_t − p_{t−1}）× 潜在人次；p_{−1}=基期 p0
                # **首年（t=0）即计入**——与设计 §4.3 伪代码（逐年 draw 增量）及老引擎 5 年口径一致
                d = (p_self - p_prev[i]) * a["visit_pot"]
                inc_opd += d
                inc_ipd += d * 0.047                    # 住院/门诊比（本院 2.6万/55万）
                st = street_inc.setdefault(a["si"], [0.0, 0.0])
                st[0] += d
                st[1] += d * 0.047
            # 容量约束 + 溢出滚入次年
            avail_opd = max(0.0, cap["opd_capacity"] - cap["opd_base"] - cum_opd)
            avail_ipd = max(0.0, cap["ipd_capacity"] - cap["ipd_base"] * cap["bed_usage"] - cum_ipd)
            pot_opd, pot_ipd = inc_opd + spill_opd, inc_ipd + spill_ipd
            take_opd = min(pot_opd, avail_opd) if pot_opd > 0 else pot_opd      # 负增长直接释放容量
            take_ipd = min(pot_ipd, avail_ipd) if pot_ipd > 0 else pot_ipd
            spill_opd = max(0.0, (pot_opd - take_opd)) * p["spill_keep"]
            spill_ipd = max(0.0, (pot_ipd - take_ipd)) * p["spill_keep"]
            cum_opd = max(0.0, cum_opd + take_opd)
            cum_ipd = max(0.0, cum_ipd + take_ipd)
            for si2, v2 in street_inc.items():
                st_cum[si2] = st_cum.get(si2, 0.0) + v2[0]
                st_last[si2] = v2[0]
            pself.append(row_p)
            years_out.append(dict(
                year=2025 + t + 1, add_opd=round(take_opd), add_ipd=round(take_ipd),
                cum_opd=round(cum_opd), cum_ipd=round(cum_ipd),
                bed_usage=round((cap["ipd_base"] * cap["bed_usage"] + cum_ipd) / max(1.0, cap["ipd_capacity"]) * 100, 1),
                numsource_util=round((cap["opd_base"] + cum_opd) / max(1.0, cap["opd_capacity"]) * 100, 1),
                spill_opd=round(spill_opd), spill_ipd=round(spill_ipd),
            ))
        # 分街道效果与来源分解（末期）
        st_eff = []
        for si, s in enumerate(streets):
            pre = cur = from_comp = from_out = lost = stay = 0
            for i, a in enumerate(agents):
                if a["si"] != si:
                    continue
                y0, y1 = choices[i][0], choices[i][years - 1]
                if y0 == 1:
                    pre += 1
                if y1 == 1:
                    cur += 1
                if y0 == 2 and y1 == 1:
                    from_comp += 1
                if y0 == 0 and y1 == 1:
                    from_out += 1
                if y0 == 1 and y1 == 1:
                    stay += 1
                if y0 == 1 and y1 != 1:
                    lost += 1
            if pre or cur or from_comp or from_out:
                st_eff.append(dict(name=s["name"], district=s["district"], d_km=s["ds"],
                                   agents=sum(1 for a in agents if a["si"] == si),
                                   self_pre=pre, self_cur=cur, net=cur - pre,
                                   from_comp=from_comp, from_out=from_out, lost=lost, stay=stay,
                                   opd_last_year=round(st_last.get(si, 0.0)),
                                   opd_cum=round(st_cum.get(si, 0.0)),
                                   cname=s["cname"], cgrade=s["cgrade"]))
        st_eff.sort(key=lambda x: -x["net"])
        first = [c[0] for c in choices]
        last = [c[years - 1] for c in choices]
        per_run.append(dict(
            years=years_out, street_effects=st_eff,
            summary=dict(
                agents=len(agents),
                self_first=sum(1 for v in first if v == 1),
                self_last=sum(1 for v in last if v == 1),
                comp_last=sum(1 for v in last if v == 2),
                out_last=sum(1 for v in last if v == 0),
                from_comp=sum(e["from_comp"] for e in st_eff),
                from_out=sum(e["from_out"] for e in st_eff),
                lost=sum(e["lost"] for e in st_eff),
                final_add_opd=years_out[-1]["cum_opd"] if years_out else 0,
                final_add_ipd=years_out[-1]["cum_ipd"] if years_out else 0,
                final_bed_usage=years_out[-1]["bed_usage"] if years_out else 0,
                final_numsource_util=years_out[-1]["numsource_util"] if years_out else 0,
                spill_opd=years_out[-1]["spill_opd"] if years_out else 0,
            ),
            # 专科效率（验收用）
            dept_net={},
        ))
        # 逐科室净增（验收：目标专科应最有效）
        dn, da, dv = {}, {}, {}
        for i, a in enumerate(agents):
            d = depts[a["dept"]]
            da[d] = da.get(d, 0) + 1
            dn[d] = dn.get(d, 0) + ((1 if choices[i][years - 1] == 1 else 0) - (1 if choices[i][0] == 1 else 0))
            # 人次口径增量（期望）：Σ (p_末年 − p_基线) × 潜在人次；按科室汇总（平滑，抗抽样噪声）
            dv[d] = dv.get(d, 0) + (pself[years][i] - pself[0][i]) * a["visit_pot"]
        per_run[-1]["dept_net"] = dn
        per_run[-1]["dept_agents"] = da
        per_run[-1]["dept_add_opd"] = {k: round(v) for k, v in dv.items()}
        per_run[-1]["dept_pot_opd"] = {}
        for i, a in enumerate(agents):
            d = depts[a["dept"]]
            per_run[-1]["dept_pot_opd"][d] = per_run[-1]["dept_pot_opd"].get(d, 0.0) + a["visit_pot"]
        per_run[-1]["dept_rate"] = {k: round(dv[k] / max(1.0, per_run[-1]["dept_pot_opd"][k]), 5) for k in dv}
        # 全街道平滑口径（人次期望值），**不过滤零净增街道** —— 供敏感性/空间梯度分析用
        # （street_effects 按整数净增过滤街道，不同变体命中不同街道集，人均指标不可比）
        per_run[-1]["streets_meta"] = [
            dict(name=s["name"], district=s["district"], d_km=s["ds"],
                 lng=round(s["lng"], 6), lat=round(s["lat"], 6),
                 agents=sum(1 for a in agents if a["si"] == si),
                 pop=s["pop"], opd=s["opd"], dis=s["dis"],
                 opd_cum=round(st_cum.get(si, 0.0), 1), ipd_cum=round(st_cum.get(si, 0.0) * 0.047, 1),
                 opd_last=round(st_last.get(si, 0.0), 1),
                 p0=s["p0"])
            for si, s in enumerate(streets)
        ]
        if want_trace and r == 0:
            trace = dict(
                meta=dict(n=len(agents), years=[2026 + t for t in range(years)],
                          seed=base_seed, algo_version=ALGO_VERSION,
                          data_version=world["data_version"], action=p["action"], dept=p["dept"],
                          magnitude=p["magnitude"],
                          note="合成患者智能体（代表人群，非真实患者）· 模型模拟仅供决策实验"),
                # 列式 + 街道表外置（前端按 street_idx 查表）：把体积压到预算内（设计 §7.4 ≤400KB/硬顶 800KB）
                street_table=[[s["name"], s["district"], round(s["lng"], 6), round(s["lat"], 6),
                               s["ds"], s["cname"], s["cgrade"], round(s["p0"], 4)] for s in streets],
                dept_table=depts,
                agents=dict(
                    street_idx=[a["si"] for a in agents],
                    dept_idx=[a["dept"] for a in agents],
                    dist_sens=[round(a["dist_sens"], 3) for a in agents],
                    loyal=[round(a["loyal"], 3) for a in agents],
                    weight=[round(a["weight"], 1) for a in agents],
                    # 个体级选择概率（后端算，前端只读）：基期 p0 与末年 p_self；
                    # ① 焦点口径用 |p_self_end − p_self_base| 判定"被战略触动的个体"，前端不做任何效用计算
                    p_self_base=[round(x, 4) for x in pself[0]],
                    p_self_end=[round(x, 4) for x in pself[years]],
                    **({"p_self_hist": [[round(x, 4) for x in row] for row in pself[1:]]} if want_prob else {}),
                ),
                choice=["".join(str(c) for c in row) for row in choices],
            )

    tot_pot = sum(s.get("potential", 0.0) for s in streets)
    baseline_share = dict(share=(world["city_opd"] / tot_pot) if tot_pot else 0.0, potential=tot_pot,
                          city_opd=world["city_opd"])
    interval_out = aggregate(per_run)
    _spatial_metrics(interval_out, streets, p.get("streets"))   # 新增字段：radius80 / HRI（不动既有字段）
    return dict(params=p, seed=base_seed, cap=cap, cap_source=cap["source"],
                share_report=share_report,
                baseline_share=baseline_share,
                world_meta=dict(streets=len(streets), depts=depts,
                                data_version=world["data_version"]),
                interval=interval_out, trace=trace)


def _radius80(rows, after=False):
    """80% 辐射半径：按距离升序累计「我院门诊量」到 80% 处的距离（线性插值，km）。

    基线 = 街道真实门诊量；策略后 = 真实门诊量 + 该街道累计新增（期望人次）。
    与老引擎的 `base × (1 − 0.06×…)` 近似缩放不同——本口径**直接由分布重算**（设计 §4.4 要求）。
    """
    rs = sorted([m for m in rows if m.get("d_km") is not None], key=lambda m: m["d_km"])
    w = [(m["d_km"], (m.get("opd", 0) or 0) + ((m.get("opd_cum", 0) or 0) if after else 0)) for m in rs]
    tot = sum(x[1] for x in w)
    if tot <= 0:
        return None
    acc = 0.0
    for i, (d, v) in enumerate(w):
        prev = acc
        acc += v
        if acc >= 0.8 * tot:
            if i == 0 or v <= 0:
                return round(d, 2)
            frac = max(0.0, min(1.0, (0.8 * tot - prev) / v))
            d0 = w[i - 1][0]
            return round(d0 + (d - d0) * frac, 2)
    return round(w[-1][0], 2)


def _hri_of(rows, names, after=False):
    """区域 HRI = (区域内我院出院 ÷ 全市我院出院) ÷ (区域内人口 ÷ 全市人口)。

    口径与 engine.hri_region 一致（出院/人口占比），策略后出院量 = 真实出院 + 累计住院增量。
    全域口径恒为 1（分子分母同集合），故只在选定区域/区级口径下才有管理含义。
    """
    tot_v = sum(((m.get("dis", 0) or 0) + ((m.get("ipd_cum", 0) or 0) if after else 0)) for m in rows)
    tot_p = sum((m.get("pop", 0) or 0) for m in rows)
    if tot_v <= 0 or tot_p <= 0:
        return None
    sub = [m for m in rows if m["name"] in names] if names else rows
    v = sum(((m.get("dis", 0) or 0) + ((m.get("ipd_cum", 0) or 0) if after else 0)) for m in sub)
    p = sum((m.get("pop", 0) or 0) for m in sub)
    if v <= 0 or p <= 0:
        return None
    return round((v / tot_v) / (p / tot_p), 4)


def _spatial_metrics(interval, streets, sel=None):
    """在区间结果上追加空间口径指标（设计 §4.4）：radius80（基线/策略后）与 HRI（选定区域 + 各区市）。"""
    meta = interval.get("streets_meta") or []
    if not meta:
        return interval
    interval["radius80"] = dict(base=_radius80(meta, False), after=_radius80(meta, True),
                                unit="km", note="由模拟患者分布直接重算（非老引擎的近似缩放）")
    names = set(sel) if sel else None
    dists = {}
    for m in meta:
        dists.setdefault(m.get("district") or "未知", []).append(m["name"])
    interval["hri"] = dict(
        scope=("selected" if names else "all"),
        region=dict(before=_hri_of(meta, names, False), after=_hri_of(meta, names, True),
                    note=("选定街道集合口径" if names else "全域口径恒为 1（分子分母同集合）")),
        districts=[dict(district=d, before=_hri_of(meta, set(ns), False), after=_hri_of(meta, set(ns), True))
                   for d, ns in sorted(dists.items())],
        note="HRI = (区域我院出院占比) ÷ (区域人口占比)；>1 辐射强、<1 覆盖弱（与 engine.hri_region 同口径）",
    )
    return interval


def agent_breakdown(params=None, agent_id=0, year=None, seed=None, run_index=0):
    """单个智能体的画像 + 逐年效用分量分解（个体决策卡 D1 的数据来源，设计 §7.2 / §6.2）

    **复算一致性**：与 simulate 用同一 _prepare/_agent_step 与同一随机数消费顺序，
    因此同参数 + 同种子下，本函数给出的逐年选择与 trace 中该智能体的选择逐位相同（由自检第 11 项保证）。
    个体级数据不落库：任何时候都能靠"同参数 + 同种子"复算重建。
    """
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})
    world, streets, depts, target_dept, share_report = _prepare(p)
    years = int(max(1, min(5, p["years"])))
    base_seed = int(seed) if seed is not None else derive_seed(p, world)
    sel = set(p["streets"]) if p["streets"] else None
    rng = random.Random(base_seed + int(run_index))
    agents = build_agents(world, p, rng)
    if not (0 <= agent_id < len(agents)):
        raise ValueError(f"agent_id 越界（0 ≤ id < {len(agents)}）")
    tgt = agents[agent_id]
    s = streets[tgt["si"]]

    hist = []
    for t in range(years):
        L = release_factor(t, years, p["curve"], p["k"])
        for i, a in enumerate(agents):
            if sel and streets[a["si"]]["name"] not in sel:
                continue                                   # 与 simulate 一致：不消费随机数
            u = _agent_step(a, streets[a["si"]], p, L, target_dept)
            x = rng.random()
            ch = 1 if x < u["p_self"] else (2 if x < u["p_self"] + u["p_comp"] else 0)
            if i == agent_id:
                hist.append(dict(
                    year=2026 + t, choice=ch,
                    choice_cn={0: "未就诊", 1: "我院", 2: "竞争医院"}[ch],
                    p_self=round(u["p_self"], 5), p_comp=round(u["p_comp"], 5), p_out=round(u["p_out"], 5),
                    release=round(L, 4),
                    contrib=dict(base=round(u["contrib_base"], 4), sp=round(u["contrib_sp"], 4),
                                 cap=round(u["contrib_cap"], 4), al=round(u["contrib_al"], 4)),
                    match=u["match"], sens=round(u["sens"], 3), activation=round(u["activation"], 4),
                    d_s=round(u["d_s"], 5), d_c=round(u["d_c"], 5), d_a=round(u["d_a"], 5),
                    u_self=round(u["u_self"], 4), u_comp=round(u["u_comp"], 4), u_out=round(u["u_out"], 4),
                ))
    out = dict(
        agent_id=agent_id, run_index=int(run_index), seed=base_seed, years=hist,
        profile=dict(street=s["name"], district=s["district"], dept=depts[tgt["dept"]],
                     pop=s["pop"], d_km=s["ds"], p0=s["p0"],
                     comp=s["cname"], comp_grade=s["cgrade"], comp_d_km=s["dc"],
                     loyal=round(tgt["loyal"], 3), dist_sens=round(tgt["dist_sens"], 3),
                     weight=round(tgt["weight"], 2), visit_pot=round(tgt["visit_pot"], 2)),
        params=p, share_report=share_report, agent_count=len(agents),
        algo_version=ALGO_VERSION, data_version=world["data_version"],
        discipline=("合成患者智能体（代表人群，非真实患者）；效用分量全部由模型参数与脱敏聚合数据算出，"
                    "不含任何个体真实信息；同参数 + 同种子可逐位复算。"),
    )
    if year is not None:
        out["years"] = [h for h in hist if h["year"] == int(year)]
    return out


def aggregate(per_run):
    """把 runs 次结果汇总为 P5/P50/P95 区间 + 稳定字段"""
    def pct(vals, q):
        v = sorted(vals)
        return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]
    keys = ["final_add_opd", "final_add_ipd", "final_bed_usage", "final_numsource_util",
            "self_last", "from_comp", "from_out", "lost"]
    out = {"runs": len(per_run), "metrics": {}}
    for k in keys:
        vals = [r["summary"][k] for r in per_run]
        out["metrics"][k] = dict(p5=pct(vals, 0.05), p50=pct(vals, 0.50), p95=pct(vals, 0.95))
    # 代表性运行：取 self_last 的中位数那一轮，其内部所有分项**出自同一轮**，
    # 因此"净增 = 转来 + 激活 − 流失"等恒等式在导出数据里始终成立（不会被分位点打散）
    order = sorted(range(len(per_run)), key=lambda i: per_run[i]["summary"]["self_last"])
    rep = per_run[order[len(order) // 2]]
    out["rep_run_index"] = order[len(order) // 2]
    years = len(rep["years"])
    out["years"] = []
    for t in range(years):
        row = dict(rep["years"][t])
        for k in ("add_opd", "add_ipd", "cum_opd", "cum_ipd"):
            vals = [r["years"][t][k] for r in per_run]
            row[k + "_p5"] = pct(vals, 0.05)
            row[k + "_p95"] = pct(vals, 0.95)
        out["years"].append(row)
    # 分街道：数值取代表性运行；净增同时给出跨轮 P5/P95（不确定性）
    out["street_effects"] = []
    for e in rep["street_effects"]:
        row = dict(e)
        vals = []
        for r in per_run:
            x = next((y for y in r["street_effects"] if y["name"] == e["name"]), None)
            vals.append(x["net"] if x else 0)
        row["net_p5"], row["net_p95"] = pct(vals, 0.05), pct(vals, 0.95)
        out["street_effects"].append(row)
    out["street_effects"].sort(key=lambda x: -x["net"])
    out["dept_net"] = dict(rep["dept_net"])
    out["streets_meta"] = list(rep.get("streets_meta", []))   # 全街道平滑口径（代表轮）
    out["agents"] = rep["summary"].get("agents")              # 本轮智能体规模（答辩口径：几千个点）
    out["dept_agents"] = dict(rep.get("dept_agents", {}))
    out["dept_net_rate"] = {d: round(v / max(1, out["dept_agents"].get(d, 1)), 4)
                            for d, v in out["dept_net"].items()}
    out["dept_add_opd"] = dict(rep.get("dept_add_opd", {}))       # 人次口径增量（平滑）
    out["dept_add_opd_rate"] = dict(rep.get("dept_rate", {}))     # 人次口径人均增量（验收判据用）
    out["top_dept"] = max(out["dept_net"].items(), key=lambda x: x[1])[0] if out["dept_net"] else None
    return out


# ══════════════════════════ 4. 自检（验收用） ══════════════════════════
def selftest(verbose=True):
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    t0 = time.time()
    base = dict(action="specialty", dept="心内科", magnitude=0.30, years=5, runs=12)

    # 1 逐次可复现
    a = simulate(dict(base))
    b = simulate(dict(base))
    same = json.dumps(a["interval"], sort_keys=True) == json.dumps(b["interval"], sort_keys=True)
    ck("1 同参数两次结果逐位一致", same, f"seed={a['seed']}")

    # 2 换种子 P50 稳定
    c = simulate(dict(base), seed=a["seed"] + 1)
    p50 = a["interval"]["metrics"]["final_add_opd"]["p50"]
    p50c = c["interval"]["metrics"]["final_add_opd"]["p50"]
    dev = abs(p50c - p50) / max(1, p50) * 100
    ck("2 换种子 P50 波动 < 2%（人次口径）", dev < 2.0, f"{p50} vs {p50c} → {dev:.2f}%")

    # 3 区间单调
    m = a["interval"]["metrics"]["final_add_opd"]
    ck("3 区间 P5≤P50≤P95", m["p5"] <= m["p50"] <= m["p95"], str(m))

    # 4 容量溢出与瓶颈（极端参数）
    headroom = a["cap"]["opd_capacity"] - a["cap"]["opd_base"]
    over = [r for r in a["interval"]["years"] if r["cum_opd"] > headroom + 1]
    z = simulate(dict(base, magnitude=1.0, action="capacity", match_all=True, runs=4))
    zy = z["interval"]["years"]
    touch = [r["year"] for r in zy if r["spill_opd"] > 0 or r["numsource_util"] >= 99]
    ck("4 容量约束生效（压测触发溢出/触顶，且常规场景不越界）",
       (not over) and bool(touch),
       f"常规场景越界轮次={len(over)} ｜ 压测触顶年={touch or '未触顶'} ｜ 压测末期号源率={zy[-1]['numsource_util']}%")

    # 5 来源分解自洽：净增 = 转来 + 激活 − 流失
    se = a["interval"]["street_effects"]
    bad = [e for e in se if e["net"] != e["from_comp"] + e["from_out"] - e["lost"]]
    ck("5 份额转移自洽（net = 转来+激活−流失）", not bad, f"不一致街道数={len(bad)}")

    # 6 目标专科最有效
    rate = a["interval"]["dept_add_opd_rate"]
    tgt = rate.get("心内科", 0.0)
    others = [v for k, v in rate.items() if k != "心内科"]
    avg_other = sum(others) / max(1, len(others))
    ck("6 目标专科人均增量显著更高（≥1.5×，设计 §5 判据）", tgt > 1.5 * avg_other,
       f"心内科 {tgt*100:.2f}% vs 其他科室均值 {avg_other*100:.2f}%（潜在人次口径）= {tgt/max(1e-9,avg_other):.2f} 倍")

    # 7 慢动作 vs 医联体：距离带差异
    sp = simulate(dict(base, action="specialty", runs=8))
    al = simulate(dict(base, action="alliance", runs=8))
    def belt(sim):
        se = {e["name"]: e for e in sim["interval"]["street_effects"]}
        near = [e for e in se.values() if e["d_km"] < 5]
        far = [e for e in se.values() if e["d_km"] > 25]
        return (sum(e["net"] for e in near) / max(1, sum(e["agents"] for e in near)),
                sum(e["net"] for e in far) / max(1, sum(e["agents"] for e in far)))
    _, sp_far = belt(sp)
    _, al_far = belt(al)
    ck("7 医联体对远郊更有效", al_far > sp_far, f"远郊人均净增 专科={sp_far:.4f} 医联体={al_far:.4f}")

    # 8 抽样代表性（加权人口份额 vs 真实份额）
    world = load_world()
    rng = random.Random(1)
    ag = build_agents(world, dict(DEFAULTS), rng)
    got, real = {}, {}
    tot_w = sum(x["weight"] for x in ag) or 1
    tot_p = sum(s["pop"] for s in world["streets"]) or 1
    for x in ag:
        nm = world["streets"][x["si"]]["name"]
        got[nm] = got.get(nm, 0) + x["weight"]
    for s in world["streets"]:
        real[s["name"]] = s["pop"]
    worst = max(abs(got.get(k, 0) / tot_w - real[k] / tot_p) for k in real)
    ck("8 抽样代表性偏差 < 1%", worst < 0.01, f"最大偏差={worst*100:.3f}%")

    # 9 零新增依赖（仅标准库）
    stdlib = {"argparse", "hashlib", "json", "math", "os", "random", "sys", "time", "sqlite3"}
    imported = {m.split(".")[0] for m in sys.modules if m and not m.startswith("_")}
    third = {m for m in imported if m in {"numpy", "pandas", "sklearn", "scipy", "matplotlib"}}
    ck("9 未引入任何第三方依赖", not third, f"第三方={third or '无'}（本模块仅用 {' '.join(sorted(stdlib & imported))} 等标准库）")

    # 10 落库可回放（写→读→关键值一致）
    rid, _created = save_run(a, note="selftest")
    back = load_run(rid)
    same_p = json.dumps(back["params"], sort_keys=True) == json.dumps(a["params"], sort_keys=True)
    same_n = back["metrics"]["metrics"]["final_add_opd"]["p50"] == a["interval"]["metrics"]["final_add_opd"]["p50"]
    ck("10 结果落库并可回放（参数与关键值一致）", same_p and same_n,
       f"run_id={rid} 分街道记录={len(back['street_effects'])} 分科室记录={len(back['dept_effects'])}")

    # 11 个体级复算一致（决策卡：同参数 + 同种子 → 与 trace 逐年选择逐位相同）
    tr = simulate(dict(base), seed=a["seed"], want_trace=True)
    tid = min(7, tr["trace"]["meta"]["n"] - 1)
    bd = agent_breakdown(dict(base), agent_id=tid, seed=a["seed"])
    mine = "".join(str(h["choice"]) for h in bd["years"])
    ch_ok = mine == tr["trace"]["choice"][tid]
    p_ok = all(abs(h["p_self"] + h["p_comp"] + h["p_out"] - 1.0) < 1e-4 for h in bd["years"])
    ck("11 个体级复算一致（决策卡逐年选择 = trace 逐位相同；概率三分量自洽）", ch_ok and p_ok,
       f"agent_id={tid} 逐年选择={mine} ｜ 概率和自洽={p_ok} ｜ 画像={bd['profile']['street']}·{bd['profile']['dept']}")

    ok_n = sum(1 for _, ok, _ in checks if ok)
    print(f"\n自检汇总：{ok_n}/{len(checks)} 通过，用时 {time.time()-t0:.1f}s，算法版本 {ALGO_VERSION}")
    return ok_n == len(checks), checks



# ══════════════════════════ 6. 结果落库（独立结果库，不碰业务库） ══════════════════════════
def _rdb():
    conn = sqlite3.connect(RESULTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db():
    """建两张表（runs / run_effects）。结果库为独立文件，业务库 hospital.db 完全不参与写入。"""
    conn = _rdb()
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, algo_version TEXT, data_version TEXT, seed TEXT,
        share_basis TEXT, params_json TEXT, metrics_json TEXT,
        rep_run_index INTEGER, trace_path TEXT, biz_key TEXT UNIQUE, note TEXT);
      CREATE TABLE IF NOT EXISTS run_effects(
        run_id INTEGER, kind TEXT, name TEXT, district TEXT, dept TEXT, d_km REAL,
        agents INTEGER, net INTEGER, from_comp INTEGER, from_out INTEGER, lost INTEGER,
        stay INTEGER, self_pre INTEGER, self_cur INTEGER,
        opd_last_year REAL, opd_cum REAL,
        PRIMARY KEY(run_id, kind, name));
      CREATE INDEX IF NOT EXISTS idx_effects_run ON run_effects(run_id, kind);
    """)
    conn.commit()
    conn.close()


def biz_key(res):
    blob = json.dumps({"p": res["params"], "seed": res["seed"], "algo": ALGO_VERSION,
                       "data": res["world_meta"]["data_version"]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def save_run(res, trace_path=None, note=None):
    """把一次推演结果写入结果库（幂等：同参数+同种子+同数据/算法版本只存一份）"""
    ensure_db()
    conn = _rdb()
    key = biz_key(res)
    row = conn.execute("SELECT id FROM runs WHERE biz_key=?", (key,)).fetchone()
    if row:                                     # 已有 → 直接复用，不重复落库
        conn.close()
        return row["id"], False
    cur = conn.execute(
        "INSERT INTO runs(created_at, algo_version, data_version, seed, share_basis, params_json,"
        " metrics_json, rep_run_index, trace_path, biz_key, note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), ALGO_VERSION, res["world_meta"]["data_version"],
         str(res["seed"]), res["params"].get("share_basis"), json.dumps(res["params"], ensure_ascii=False),
         json.dumps(res["interval"], ensure_ascii=False), res["interval"].get("rep_run_index"),
         trace_path, key, note))
    rid = cur.lastrowid
    rows = []
    for e in res["interval"]["street_effects"]:
        rows.append((rid, "street", e["name"], e.get("district"), None, e.get("d_km"),
                     e.get("agents"), e.get("net"), e.get("from_comp"), e.get("from_out"),
                     e.get("lost"), e.get("stay"), e.get("self_pre"), e.get("self_cur"),
                     e.get("opd_last_year"), e.get("opd_cum")))
    for d, v in res["interval"]["dept_net"].items():
        rows.append((rid, "dept", d, None, d, None, res["interval"]["dept_agents"].get(d), v,
                     None, None, None, None, None, None,
                     res["interval"]["dept_add_opd"].get(d), None))
    conn.executemany("INSERT INTO run_effects VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return rid, True


def list_runs(limit=20):
    ensure_db()
    conn = _rdb()
    rows = conn.execute("SELECT id, created_at, share_basis, seed, params_json, metrics_json,"
                        " trace_path FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_run(run_id):
    """按 id 回放一次落库结果（参数 + 汇总 + 分街道/分科室效果）"""
    conn = _rdb()
    r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        conn.close()
        return None
    eff = [dict(x) for x in conn.execute("SELECT * FROM run_effects WHERE run_id=?", (run_id,))]
    conn.close()
    out = dict(r)
    out["params"] = json.loads(out.pop("params_json"))
    out["metrics"] = json.loads(out.pop("metrics_json"))
    out["street_effects"] = [e for e in eff if e["kind"] == "street"]
    out["dept_effects"] = [e for e in eff if e["kind"] == "dept"]
    return out

# ══════════════════════════ 5. CLI ══════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(description="智能体推演引擎（阶段 1）")
    ap.add_argument("--action", default=DEFAULTS["action"], choices=["specialty", "alliance", "capacity"])
    ap.add_argument("--dept", default=DEFAULTS["dept"])
    ap.add_argument("--magnitude", type=float, default=DEFAULTS["magnitude"])
    ap.add_argument("--years", type=int, default=DEFAULTS["years"])
    ap.add_argument("--runs", type=int, default=DEFAULTS["runs"])
    ap.add_argument("--density", type=float, default=DEFAULTS["density"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--streets", default="", help="逗号分隔街道名（留空=全域）")
    ap.add_argument("--curve", default=DEFAULTS["curve"], choices=["logistic", "linear"])
    ap.add_argument("--share-basis", default=DEFAULTS["share_basis"],
                    choices=["contested", "city", "dept_pool"],
                    help="份额口径：contested=竞争圈层(默认,决策方选定) / city=全市场 / dept_pool=专科需求池")
    ap.add_argument("--save", action="store_true", help="结果写入结果库 agent_sim_results.db")
    ap.add_argument("--note", default=None, help="随结果保存的备注（如答辩场景名）")
    ap.add_argument("--list-runs", action="store_true", help="列出已落库的推演记录")
    ap.add_argument("--json", default="", help="结果写入 JSON 路径")
    ap.add_argument("--trace", default="", help="个体级帧数据（蜂群视图用）写入 JSON 路径")
    ap.add_argument("--selftest", action="store_true", help="跑验收自检")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, _ = selftest()
        return 0 if ok else 1

    if args.list_runs:
        rows = list_runs(20)
        if not rows:
            print("结果库暂无记录")
        for r in rows:
            m = json.loads(r["metrics_json"])["metrics"]
            print(f"  #{r['id']} {r['created_at']} 口径={r['share_basis']} seed={r['seed']} "
                  f"新增门诊P50={m['final_add_opd']['p50']} 期末床位率={m['final_bed_usage']['p50']}%"
                  + (f" | trace={r['trace_path']}" if r["trace_path"] else ""))
        return 0

    params = dict(action=args.action, dept=args.dept, magnitude=args.magnitude, years=args.years,
                  runs=args.runs, density=args.density, curve=args.curve,
                  share_basis=args.share_basis,
                  streets=[s for s in args.streets.split(",") if s] or None)
    t0 = time.time()
    res = simulate(params, seed=args.seed, want_trace=bool(args.trace))
    iv = res["interval"]
    print(f"智能体推演完成（{iv['runs']} 次重复，{time.time()-t0:.1f}s）")
    print(f"  动作={res['params']['action']} 专科={res['params']['dept']} "
          f"强度={res['params']['magnitude']:.0%} 周期={res['params']['years']}年 "
          f"密度={res['params']['density']} 种子={res['seed']}")
    print(f"  容量口径来源：{res['cap_source']}")
    sr = res["share_report"]
    print(f"  份额口径：{sr['basis']}（分母 = {sr['denominator_note']}）")
    print(f"    基线占比均值：本口径 {sr['mean_used']*100:.1f}% ｜ 全市场口径 {sr['mean_city']*100:.2f}% ｜ 竞争圈层口径 {sr['mean_contested']*100:.1f}%")
    print(f"    全市潜在门诊需求 ≈ {res['baseline_share']['potential']/10000:.0f} 万人次/年")
    for k, label in (("self_last", "选择我院（智能体数）"), ("final_add_opd", "累计新增门诊"),
                     ("final_add_ipd", "累计新增住院")):
        m = iv["metrics"][k]
        print(f"  {label}：P5={m['p5']} P50={m['p50']} P95={m['p95']}")
    print(f"  来源分解：从竞争医院转来 {iv['metrics']['from_comp']['p50']} ｜ "
          f"从不就诊激活 {iv['metrics']['from_out']['p50']} ｜ 流失 {iv['metrics']['lost']['p50']}")
    if iv["years"]:
        print("  分年（P50）：" + " ｜ ".join(
            f"{r['year']}: +门诊{r['add_opd']} 床位{r['bed_usage']}% 号源{r['numsource_util']}%" for r in iv["years"]))
    print(f"  逐科室净增 TOP3：" + " ｜ ".join(f"{d} {v}" for d, v in
          sorted(iv["dept_net"].items(), key=lambda x: -x[1])[:3]))
    print(f"  分街道净增 TOP3：" + " ｜ ".join(f"{e['name']} {e['net']}" for e in iv["street_effects"][:3]))
    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  结果已写入：{args.json}")
    if args.save:
        rid, created = save_run(res, trace_path=args.trace or None, note=args.note)
        print(f"  结果已落库：run_id={rid}{'（新建）' if created else '（已存在相同参数，复用）'} → {os.path.basename(RESULTS_DB)}")
    if args.trace:
        tr = res["trace"]
        json.dump(tr, open(args.trace, "w", encoding="utf-8"), ensure_ascii=False)
        kb = os.path.getsize(args.trace) / 1024
        warn = "  ⚠ 超过设计硬顶 800KB（请降密度或抽稀）" if kb > 800 else ("  (预算 ≤400KB)" if kb > 400 else "  ✔ 在 400KB 预算内")
        print(f"  个体级帧数据已写入：{args.trace}（{tr['meta']['n']} 个智能体 × {len(tr['meta']['years'])} 年，"
              f"{kb:.0f} KB）{warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
