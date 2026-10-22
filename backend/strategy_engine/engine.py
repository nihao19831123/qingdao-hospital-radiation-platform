#!/usr/bin/env python3
"""战略推演引擎: 街道级, 1-5年周期, 三类战略动作.
基线: 2025 三医街道患者分布(门诊55万/住院2.6万) + 真实科室画像(医生/号源).
"""
import json
import math
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "hospital.db")
HEATMAP = os.path.join(os.path.dirname(BASE), "heatmap_data.json")

HOSPITAL = (120.3884087, 36.1795722)  # 三医 WGS-84

# ---------- 数据加载 ----------
_cache = {}

def _data():
    if "data" not in _cache:
        d = json.load(open(HEATMAP, encoding="utf-8"))
        _cache["data"] = d
    return _cache["data"]

def _streets():
    """街道列表: name/district/pop2025/coords"""
    d = _data()
    pop25 = d["population"].get("2025", {})
    out = {}
    for f in d["street_geo"]["features"]:
        n = f["properties"]["name"]
        dist = f["properties"].get("district", "")
        c = d["coords"].get(n, {})
        out[n] = {"district": dist, "pop": pop25.get(n, 0),
                  "lng": c.get("lng"), "lat": c.get("lat")}
    return out

def _street_patients():
    """街道 门诊/住院 基线 (归一化到 55万门诊/2.6万住院)"""
    d = _data()
    dis25 = d["discharge"].get("2025", {})
    opd25 = d["opd"].get("2025", {})
    tot_dis = sum(dis25.values()) or 1
    tot_opd = sum(opd25.values()) or 1
    out = {}
    for n in dis25:
        out[n] = {
            "ipd": dis25[n] / tot_dis * 26000,
            "opd": opd25.get(n, 0) / tot_opd * 550000,
        }
    return out

def _dept_share():
    """科室出院占比: street -> {dept: 占比}"""
    d = _data()
    dep = d.get("departments", {})
    out = {}
    for n, dd in dep.items():
        s = sum(dd.values())
        if s > 0:
            out[n] = {k: v / s for k, v in dd.items()}
    return out

def dist_km(street):
    """街道到三医距离"""
    s = _streets().get(street)
    if not s or not s["lng"]:
        return 30.0
    dlat = (s["lat"] - HOSPITAL[1]) * 111
    dlng = (s["lng"] - HOSPITAL[0]) * 89.6
    return math.sqrt(dlat * dlat + dlng * dlng)

# ---------- HRI 医疗辐射指数 ----------
def _hri_parts(year="2025"):
    """HRI 基础量: (各街道出院, 各街道人口, 全院出院合计, 全市人口合计).
    口径与前端 metricValue('hri') / ai_detector 的 hri25 完全一致:
    以 street_geo 全部街道为分母集合, 出院(discharge)+人口(population)同年占比."""
    d = _data()
    dis = d["discharge"].get(year, {})
    pop = d["population"].get(year, {})
    names = [f["properties"]["name"] for f in d["street_geo"]["features"]]
    tot_v = sum(dis.get(n, 0) for n in names)
    tot_p = sum(pop.get(n, 0) for n in names)
    return dis, pop, tot_v, tot_p

def hri_street(street, year="2025"):
    """街道 HRI = (街道出院占全院出院比) / (街道人口占全市人口比).
    与前端地图 'HRI 辐射指数' 图层同口径 (2025 最高香港中路街道≈1.674)."""
    dis, pop, tot_v, tot_p = _hri_parts(year)
    v, p = dis.get(street, 0), pop.get(street, 0)
    if v <= 0 or p <= 0 or tot_v <= 0 or tot_p <= 0:
        return None
    return (v / tot_v) / (p / tot_p)

def hri_region(streets, year="2025"):
    """区域聚合 HRI = (区域出院占全院出院比) / (区域人口占全市人口比).
    街道聚合 = 分子分母各自求和, 与单街道口径自洽; 区域=全域时恒为 1."""
    dis, pop, tot_v, tot_p = _hri_parts(year)
    v = sum(dis.get(s, 0) for s in streets)
    p = sum(pop.get(s, 0) for s in streets)
    if v <= 0 or p <= 0 or tot_v <= 0 or tot_p <= 0:
        return None
    return (v / tot_v) / (p / tot_p)

def _db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- 医院容量 ----------
def capacity():
    """医院容量基线"""
    conn = _db()
    params = {r["param"]: r["value"] for r in conn.execute("SELECT param, value FROM hospital_params")}
    conn.close()
    beds = params.get("beds", 800)
    ipd_base = params.get("annual_ipd", 26000)
    usage = params.get("bed_usage_rate", 0.64)
    cap = {
        "opd_base": params.get("annual_opd", 550000),       # 年门诊
        "ipd_base": ipd_base,                               # 年住院
        "beds": beds,
        "bed_usage": usage,                                 # 床位使用率
        # 等效年住院容量: 按 800床/64%/26000住院 基线校准, 床位数变化时线性伸缩
        "ipd_capacity": (beds / 800.0) * ipd_base / max(usage, 0.1),
        "weekly_numsource": 13759,                          # 周号源
        "radius80_base": params.get("radius80_km", 15.1),
    }
    cap["opd_capacity"] = cap["weekly_numsource"] * 52      # 年门诊供给上限
    return cap

# ---------- 时间释放 ----------
def release(year_idx, years, magnitude):
    """第 year_idx 年 (0-based) 释放的幅度比例: 线性累进"""
    if years <= 1:
        return magnitude
    return magnitude * (year_idx + 1) / years

# ---------- 推演 ----------
def simulate(params):
    """
    params: {
      name: str,
      action: 'specialty'|'alliance'|'capacity',
      dept: '心内科' (specialty 用, 或 '全院'),
      streets: [街道名...] (空=全部),
      magnitude: 0.1-0.5,
      years: 1-5,
      extra: {bed_add: 100} (capacity 附加)
    }
    返回: {years: [...], metrics, street_effects}
    """
    action = params.get("action", "specialty")
    years = max(1, min(5, int(params.get("years", 3))))
    magnitude = max(0.05, min(0.5, float(params.get("magnitude", 0.2))))
    streets = params.get("streets") or list(_street_patients().keys())
    dept = params.get("dept", "全院")
    cap = capacity()

    streets_meta = _streets()
    patients = _street_patients()
    dept_share = _dept_share()

    # 街道级年度增量 (相对基线 2025)
    street_inc = {s: {"opd": 0.0, "ipd": 0.0} for s in patients}
    for s in streets:
        if s not in patients:
            continue
        d_km = dist_km(s)
        # 距离衰减: 15km 内 1.0, 更远递减
        decay = 1.0 if d_km <= 15 else max(0.25, 15.0 / d_km)
        share = dept_share.get(s, {})
        if action == "specialty":
            # 专科渗透: 目标科室的患者被吸引增加
            if dept == "全院":
                dep_w = 1.0
                opd_share = 1.0
            else:
                dep_w = share.get(dept, 0)
                if dep_w <= 0:
                    dep_w = 0.05  # 该科室在该街道无历史 -> 小基础
                opd_share = dep_w
            street_inc[s]["opd"] = patients[s]["opd"] * opd_share * magnitude * decay
            street_inc[s]["ipd"] = patients[s]["ipd"] * opd_share * magnitude * decay * 0.8
        elif action == "alliance":
            # 医联体: 激活低覆盖街道潜在需求 (人口就诊率缺口)
            if patients[s]["opd"] > 0 and streets_meta[s]["pop"] > 0:
                visit_rate = patients[s]["opd"] / streets_meta[s]["pop"]
                base_rate = 0.10  # 全市人均年门诊约 0.1 次/人到我院
                gap = max(0.0, base_rate - visit_rate * 0.6)
                street_inc[s]["opd"] = streets_meta[s]["pop"] * gap * magnitude * decay * 0.6
                street_inc[s]["ipd"] = street_inc[s]["opd"] * 0.05
        elif action == "capacity":
            # 资源扩容: 释放被号源/床位压制的需求
            util = cap["opd_base"] / max(cap["opd_capacity"], 1)
            suppressed = max(0.0, (util - 0.8) / 0.2)  # 0..1
            street_inc[s]["opd"] = patients[s]["opd"] * magnitude * (0.3 + 0.7 * suppressed) * decay
            street_inc[s]["ipd"] = patients[s]["ipd"] * magnitude * 0.5 * decay

    # ---- HRI 辐射指数 (口径与前端 metricValue('hri') 一致, 详见 hri_region 注释) ----
    # hri_before = 基线区域 HRI = (区域出院占全院比)/(区域人口占全市比);
    # 推演期末 hri_after = hri_before × (1 + 区域累计新增住院 / 区域基线住院).
    # 推导: 前端口径的分子是"区域出院占全院出院比". 推演只增加我院患者, 全市人口
    # 与其他街道患者均不变, 因此新增住院全部计入区域分子、全院出院分母按基期不变,
    # 区域患者占比的提升倍数恰为 (V_区域+Δ)/V_区域, HRI 等比放大. 该口径保证:
    # ① 基线值与前端地图 HRI 图层完全一致; ② 全域推演时也能体现全院辐射强度净提升
    # (而非被总量归一化抵消为 1).
    region_streets = [s for s in streets if s in patients]
    hri_before = hri_region(region_streets)
    v_region = sum(patients[s]["ipd"] for s in region_streets)  # 区域基线住院(归一化口径)

    # 分年结果 (累进释放: 每年增量 = 满额 x (i+1)/sum(1..years), 末年末年累计恰好=满额)
    years_out = []
    cum_opd = cum_ipd = 0.0
    tri = years * (years + 1) / 2  # 1+2+...+years
    for yi in range(years):
        share = (yi + 1) / tri
        add_opd = sum(v["opd"] for v in street_inc.values()) * share
        add_ipd = sum(v["ipd"] for v in street_inc.values()) * share
        cum_opd += add_opd
        cum_ipd += add_ipd
        new_usage = cap["bed_usage"] + cum_ipd / max(cap["ipd_capacity"], 1)
        new_util = (cap["opd_base"] + cum_opd) / max(cap["opd_capacity"], 1)
        # 80% 半径: 用街道患者分布增量加权重算 (近似: 半径向增量分布方向收缩)
        r80 = cap["radius80_base"] * (1 - 0.06 * min(1.0, cum_opd / cap["opd_base"] * 20))
        # 逐年区域 HRI: 按当年累计新增住院等比折算 (公式见上方推导)
        hri_y = round(hri_before * (1 + cum_ipd / v_region), 3) \
            if (hri_before is not None and v_region > 0) else None
        years_out.append({
            "year": 2025 + yi + 1,
            "add_opd": round(add_opd), "add_ipd": round(add_ipd),
            "cum_opd": round(cum_opd), "cum_ipd": round(cum_ipd),
            "bed_usage": round(new_usage * 100, 1),
            "numsource_util": round(new_util * 100, 1),
            "radius80": round(r80, 1),
            "hri": hri_y,
        })

    # 街道级最终效果 (第 years-1 年增量, 供地图)
    final_share = years / tri if years > 0 else 1.0
    # 专科患者增量的科室标注: 专科提升=目标科室(增量已按 dept_share 加权), 医联体/扩容=全院
    sp_dept = dept if action == "specialty" else "全院"
    street_final = []
    for s, inc in street_inc.items():
        if inc["opd"] + inc["ipd"] <= 0:
            continue
        street_final.append({
            "name": s, "district": streets_meta[s]["district"],
            "add_opd": round(inc["opd"] * final_share),
            "add_ipd": round(inc["ipd"] * final_share),
            "d_km": round(dist_km(s), 1),
            # 专科患者增量(住院口径): specialty 时 inc.ipd 已是目标科室 dept_share 加权增量
            "specialty_patients": round(inc["ipd"] * final_share),
            "specialty_dept": sp_dept,
        })
    street_final.sort(key=lambda x: -(x["add_opd"] + x["add_ipd"]))

    risk = []
    if years_out and years_out[-1]["bed_usage"] > 90:
        risk.append(f"床位使用率将达 {years_out[-1]['bed_usage']}%，超过90%警戒线，需同步扩容床位或优化住院流程")
    if years_out and years_out[-1]["numsource_util"] > 95:
        risk.append(f"号源利用率将达 {years_out[-1]['numsource_util']}%，门诊供给接近饱和")
    if action == "specialty" and dept != "全院":
        risk.append(f"「{dept}」能力提升需匹配学科建设投入（人才引进/设备），建议结合医院实际评估可行性")

    return {
        "params": params,
        "action_cn": {"specialty": "专科能力提升", "alliance": "医联体建设", "capacity": "资源扩容"}[action],
        "years": years_out,
        "street_effects": street_final,
        "top_streets": street_final[:10],
        "risks": risk,
        "summary": {
            "final_add_opd": years_out[-1]["cum_opd"] if years_out else 0,
            "final_add_ipd": years_out[-1]["cum_ipd"] if years_out else 0,
            "final_bed_usage": years_out[-1]["bed_usage"] if years_out else 0,
            "final_numsource_util": years_out[-1]["numsource_util"] if years_out else 0,
            "final_radius80": years_out[-1]["radius80"] if years_out else 0,
            # HRI 辐射指数: 基线区域 HRI → 期末区域 HRI (口径见 simulate 内推导注释)
            "hri_before": round(hri_before, 3) if hri_before is not None else None,
            "hri_after": years_out[-1]["hri"] if years_out else None,
            # 专科患者增量(住院口径): 专科提升=目标科室 dept_share 加权增量; 医联体/扩容=全院增量
            "specialty_patients": {
                "dept": sp_dept,
                "add_ipd": years_out[-1]["cum_ipd"] if years_out else 0,
            },
        },
    }

if __name__ == "__main__":
    # 测试
    r = simulate({"action": "specialty", "dept": "心内科", "streets": [], "magnitude": 0.2, "years": 3})
    print("动作:", r["action_cn"])
    for y in r["years"]:
        print(f"  {y['year']}: +门诊{y['add_opd']} +住院{y['add_ipd']} 床位率{y['bed_usage']}% 号源率{y['numsource_util']}% HRI {y['hri']}")
    print("HRI:", r["summary"]["hri_before"], "->", r["summary"]["hri_after"])
    print("专科患者增量:", r["summary"]["specialty_patients"])
    print("风险:", r["risks"])
    print("TOP街道:", [(s["name"], s["add_opd"]) for s in r["street_effects"][:5]])
