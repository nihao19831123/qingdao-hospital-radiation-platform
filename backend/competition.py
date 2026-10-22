#!/usr/bin/env python3
"""竞争压力指数 (Competition Pressure Index, CPI) — 补齐"医院竞争环境"分析短板.

数据源: backend/heatmap_data.json (只读, 不写)
  - hospitals: 25 家二级以上公立医院 {name, grade(三甲15·二甲9·三乙1), lat, lon, is_main=本院(青岛市第三人民医院, 三乙)}
  - coords: 134 街道中心点 {lng, lat}
  - street_geo.features[].properties: {name, district}

口径推导 (可解释/可答辩):
1. 竞争名单: 等级为"三甲"的医院 (本院为三乙, 核心截流竞争来自三甲; 本院不参与竞争名单)
   —— 对应《推演方案初版》优化5"医疗竞争影响因子: 医院等级/重点专科/距离"
2. 每个街道:
   d_self = 街道中心 → 本院 (120.3884087,36.1795722, 与 strategy_engine/engine.py 同口径) 距离 km
   d_comp = 街道中心 → 最近三甲竞争医院距离 km; nearest = 最近竞争医院名与等级
   截流判定: d_comp < d_self → 竞争医院比本院更近, 存在截流风险
3. CPI (0-100) = WEIGHT_DIST × 距离劣势分 + WEIGHT_DENSITY × 三甲密度分
   距离劣势分 = 100 × max(0, (d_self - d_comp) / d_self)
     —— "截流劣势比": 竞争医院比本院近的比例. 竞争医院比本院近一半 → 50 分; 本院更近 → 0 分.
        本院更近时距离维度压力归零, 因此 CPI≥50(高/极高)必然包含截流, 口径可辩护.
   三甲密度分 = min(100, 10km内三甲数×25 + (15km内三甲数-10km内三甲数)×5)
     —— 高水平医疗可替代性密度: 内圈每家 25 分, 外圈(10-15km)每家 5 分, 封顶 100.
   15km 内无三甲竞争医院 → CPI 封顶 NO_COMP_CAP(25): 远郊无有效截流竞争, 取低值.
4. 分级与管理含义:
   低 (<25)  : 竞争影响弱, 巩固基本盘即可
   中 (25-49): 竞争显现, 关注患者流失信号与口碑
   高 (50-74): 竞争截流明显, 需专科差异化 + 转诊对接
   极高 (≥75): 竞争主导区域, 全力防御 (医联体/绿色通道/专科品牌)
"""
import datetime
import json
import math
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "heatmap_data.json")
HOSPITAL = (120.3884087, 36.1795722)  # 本院(三医) WGS-84, 与 strategy_engine/engine.py 同口径

# ---- 可调常量 (权重答辩口径: 距离是截流的直接动因, 占主导 0.6; 密度反映可替代性, 0.4) ----
WEIGHT_DIST = 0.6         # 距离劣势权重
WEIGHT_DENSITY = 0.4      # 三甲密度权重
DENSITY_R1_KM = 10.0      # 密度内圈半径
DENSITY_R2_KM = 15.0      # 密度外圈半径
DENSITY_SCORE_R1 = 25     # 10km 内每家三甲得分
DENSITY_SCORE_R2 = 5      # 10-15km 每家三甲得分
NO_COMP_CAP = 25          # 15km 内无三甲时 CPI 封顶(低值)
COMP_GRADES = ("三甲", "三乙", "三级")   # 竞争医院等级档: 三级及以上(数据按官网核实后, 区市医院多已为三乙/三级)

_cache = {}


def _data():
    """heatmap_data.json 只读 + 模块级缓存 (大文件, 避免重复解析)"""
    if not _cache:
        _cache.update(json.load(open(DATA_FILE, encoding="utf-8")))
    return _cache


def load_hospitals():
    """医院全量列表 (只读)"""
    return _data().get("hospitals", [])


def _dist_km(lon1, lat1, lon2, lat2):
    """平面近似距离 km (1°lat≈111km, 1°lng≈89.6km@青岛纬度), 与 engine.dist_km 同口径"""
    dlat = (lat1 - lat2) * 111
    dlng = (lon1 - lon2) * 89.6
    return math.sqrt(dlat * dlat + dlng * dlng)


def _competition_hospitals():
    """竞争名单: 三甲医院 (本院不参与)"""
    return [h for h in load_hospitals()
            if h.get("grade") in COMP_GRADES and not h.get("is_main")]


def _districts():
    """街道 → 区 (来自 street_geo properties)"""
    out = {}
    for f in _data().get("street_geo", {}).get("features", []):
        out[f["properties"]["name"]] = f["properties"].get("district", "")
    return out


def nearest_competition(street):
    """街道 → 最近三甲竞争医院 {name, level, dist}; 无坐标/无竞争医院返回 None"""
    c = _data().get("coords", {}).get(street)
    if not c:
        return None
    best = None
    for h in _competition_hospitals():
        d = _dist_km(c["lng"], c["lat"], h["lon"], h["lat"])
        if not best or d < best["dist"]:
            best = {"name": h["name"], "level": h["grade"], "dist": d}
    return best


def cpi_level(cpi):
    """CPI → (等级, 管理含义)"""
    if cpi < 25:
        return "低", "竞争影响弱, 巩固基本盘"
    if cpi < 50:
        return "中", "竞争显现, 关注流失信号"
    if cpi < 75:
        return "高", "截流明显, 需差异化+转诊对接"
    return "极高", "竞争主导, 全力防御(医联体/绿色通道/专科品牌)"


def street_row(street):
    """单街道完整竞争画像行 (与 street_cpi() 表同结构); 街道无坐标返回 None"""
    d = _data()
    c = d.get("coords", {}).get(street)
    comp = _competition_hospitals()
    if not c or not comp:
        return None
    d_self = _dist_km(c["lng"], c["lat"], HOSPITAL[0], HOSPITAL[1])
    dists = sorted((_dist_km(c["lng"], c["lat"], h["lon"], h["lat"]), h) for h in comp)
    d_comp, nearest = dists[0]
    n1 = sum(1 for dd, _ in dists if dd <= DENSITY_R1_KM)
    n2 = sum(1 for dd, _ in dists if dd <= DENSITY_R2_KM)
    # 距离劣势分: 截流劣势比 (本院更近时为 0)
    score_dist = max(0.0, (d_self - d_comp) / d_self) * 100 if d_self > 0 else 0.0
    # 三甲密度分: 内圈每家 R1 分 + 外圈每家 R2 分, 封顶 100
    score_dens = min(100.0, n1 * DENSITY_SCORE_R1 + (n2 - n1) * DENSITY_SCORE_R2)
    cpi = WEIGHT_DIST * score_dist + WEIGHT_DENSITY * score_dens
    if n2 == 0:
        cpi = min(cpi, NO_COMP_CAP)  # 15km 内无三甲 → 低值
    cpi = round(cpi)
    lv, _ = cpi_level(cpi)
    return {
        "name": street,
        "district": _districts().get(street, ""),
        "d_self": round(d_self, 1),
        "d_comp": round(d_comp, 1),
        "nearest_name": nearest["name"],
        "nearest_level": nearest["grade"],
        "cpi": cpi,
        "level": lv,
        "comp_count_10km": n1,
        "comp_count_15km": n2,
    }


def street_cpi():
    """全 134 街道 CPI 表 (按 cpi 降序)"""
    rows = [street_row(n) for n in _data().get("coords", {}).keys()]
    return sorted([r for r in rows if r], key=lambda x: -x["cpi"])


def street_competition(street):
    """归因/解释用紧凑画像: {cpi, level, nearest_name, nearest_level, nearest_dist}"""
    r = street_row(street)
    if not r:
        return {"cpi": None, "level": None, "nearest_name": None,
                "nearest_level": None, "nearest_dist": None}
    return {"cpi": r["cpi"], "level": r["level"], "nearest_name": r["nearest_name"],
            "nearest_level": r["nearest_level"], "nearest_dist": r["d_comp"]}


def hospitals_context():
    """三级及以上医院清单 (三甲 + 本院三乙), 按距本院距离升序 — 竞争环境总览"""
    out = []
    for h in load_hospitals():
        if h.get("grade") not in COMP_GRADES:
            continue  # 只列三级及以上 (二甲不构成等级竞争)
        out.append({
            "name": h["name"], "level": h["grade"],
            "is_main": bool(h.get("is_main")),
            "lon": h["lon"], "lat": h["lat"],
            "opd": h.get("opd"), "ipd": h.get("ipd"),
            "dist_to_us": round(_dist_km(h["lon"], h["lat"], HOSPITAL[0], HOSPITAL[1]), 1),
        })
    out.sort(key=lambda x: x["dist_to_us"])
    return out


def competition_overview():
    """GET /api/competition 返回体"""
    streets = street_cpi()
    counts = {"低": 0, "中": 0, "高": 0, "极高": 0}
    for s in streets:
        counts[s["level"]] += 1
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "hospitals": hospitals_context(),
        "streets": streets,
        "summary": {
            "street_count": len(streets),
            "low_count": counts["低"], "mid_count": counts["中"],
            "high_count": counts["高"], "extreme_count": counts["极高"],
            # 极高/高竞争街道 TOP10 (streets 已按 cpi 降序)
            "top10": [{"name": s["name"], "district": s["district"], "cpi": s["cpi"],
                       "level": s["level"], "nearest_name": s["nearest_name"],
                       "nearest_level": s["nearest_level"], "d_comp": s["d_comp"]}
                      for s in streets[:10]],
        },
    }


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name:
        print(json.dumps(street_row(name), ensure_ascii=False, indent=1))
    else:
        rows = street_cpi()
        print("街道数:", len(rows))
        for s in rows[:15]:
            print(f"{s['cpi']:>3} {s['level']:<2} {s['name']}({s['district']}) "
                  f"d_self={s['d_self']} d_comp={s['d_comp']} {s['nearest_name']}")
        print("…")
        for s in rows[-5:]:
            print(f"{s['cpi']:>3} {s['level']:<2} {s['name']}({s['district']}) "
                  f"d_self={s['d_self']} d_comp={s['d_comp']} {s['nearest_name']}")
