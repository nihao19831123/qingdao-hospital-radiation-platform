#!/usr/bin/env python3
"""② 智能归因: 对异常事件计算科室贡献度 + 门诊/住院贡献 + 竞争压力(附加字段)."""
import json
import os

import competition

DATA_FILE = os.path.join(os.path.dirname(__file__), "heatmap_data.json")

def attribute(street):
    """返回 {dept_contrib: [...], visit_contrib: {...}, competition: {...}(附加字段)}"""
    data = json.load(open(DATA_FILE, encoding="utf-8"))
    depts = data.get("departments", {})
    dis24 = data["discharge"].get("2024", {})
    dis25 = data["discharge"].get("2025", {})
    opd24 = data["opd"].get("2024", {})
    opd25 = data["opd"].get("2025", {})

    # 竞争压力附加字段 (competition.street_competition; 旧字段全部原样保留, 仅新增 competition 键)
    comp = competition.street_competition(street)

    v24 = dis24.get(street, 0)
    v25 = dis25.get(street, 0)
    total_chg = v25 - v24
    if total_chg == 0:
        return {"dept_contrib": [], "visit_contrib": {}, "total_chg": 0,
                "competition": comp}

    # 科室贡献: 街道2025科室结构 按增量比例
    dept_contrib = []
    if street in depts:
        dd = depts[street]
        s = sum(dd.values())
        for k, v in sorted(dd.items(), key=lambda x: -x[1]):
            if s <= 0:
                continue
            contrib = v / s * total_chg
            dept_contrib.append({
                "dept": k, "change": round(contrib),
                "pct": round(v / s * 100, 1),
            })
    # 门诊/住院贡献
    o_chg = opd25.get(street, 0) - opd24.get(street, 0)
    # 门诊贡献按门诊增量折算(门诊人次与住院人次不同量级, 用变化方向加权)
    abs_o, abs_i = abs(o_chg), abs(total_chg)
    visit_contrib = {"opd_pct": 0, "ipd_pct": 0}
    if abs_o + abs_i > 0:
        visit_contrib["opd_pct"] = round(abs_o / (abs_o + abs_i) * 100, 1)
        visit_contrib["ipd_pct"] = round(abs_i / (abs_o + abs_i) * 100, 1)
    return {"dept_contrib": dept_contrib, "visit_contrib": visit_contrib,
            "total_chg": total_chg, "opd_chg": o_chg, "competition": comp}

if __name__ == "__main__":
    import sys
    street = sys.argv[1] if len(sys.argv) > 1 else "城阳街道"
    r = attribute(street)
    print(street, "总增量:", r["total_chg"])
    for d in r["dept_contrib"][:5]:
        print(" ", d["dept"], d["change"], f"({d['pct']}%)")
    print(" 门诊贡献:", r["visit_contrib"]["opd_pct"], "% 住院贡献:", r["visit_contrib"]["ipd_pct"], "%")
