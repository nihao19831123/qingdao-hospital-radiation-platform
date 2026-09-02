#!/usr/bin/env python3
"""① AI 异常检测器: 扫描街道, 发现4类异常, 计算异常分, 取TOP5事件."""
import json
import os
import sys

DATA_FILE = os.path.join(os.path.dirname(__file__), "heatmap_data.json")

def load_data():
    return json.load(open(DATA_FILE, encoding="utf-8"))

def detect():
    data = load_data()
    pop = data["population"]           # {"2020": {street: pop}, ...}
    dis = data["discharge"]            # {"2024": {...}, "2025": {...}}
    opd = data["opd"]
    depts = data.get("departments", {})
    district_of = {f["properties"]["name"]: f["properties"]["district"]
                   for f in data["street_geo"]["features"]}

    events = []
    for street in dis.get("2025", {}):
        v25 = dis["2025"].get(street, 0)
        v24 = dis["2024"].get(street, 0)
        p25 = pop.get("2025", {}).get(street, 0)
        p24 = pop.get("2024", {}).get(street, 0)
        p20 = pop.get("2020", {}).get(street, 0)
        o25 = opd.get("2025", {}).get(street, 0)
        o24 = opd.get("2024", {}).get(street, 0)
        if v24 <= 0 or v25 <= 0 or p24 <= 0 or p25 <= 0:
            continue

        pat_rate = (v25 - v24) / v24 * 100
        pat_chg = v25 - v24
        # 每万人就诊 (VPR)
        vpr24 = v24 / p24 * 10000
        vpr25 = v25 / p25 * 10000
        vpr_rate = (vpr25 - vpr24) / vpr24 * 100 if vpr24 > 0 else 0
        # HRI (患者占比/人口占比)
        tot_v24 = sum(dis["2024"].values()); tot_v25 = sum(dis["2025"].values())
        tot_p25 = sum(pop["2025"].values())
        hri24 = (v24 / tot_v24) / (p24 / sum(pop["2024"].values()))
        hri25 = (v25 / tot_v25) / (p25 / tot_p25)
        hri_rate = (hri25 - hri24) / hri24 * 100 if hri24 > 0 else 0
        # 人口增长 (2020->2025) vs 患者增长 (2024->2025)
        pop_g = (p25 / p20 - 1) * 100 if p20 > 0 else 0
        # 门诊 vs 住院贡献
        opd_rate = (o25 - o24) / o24 * 100 if o24 > 0 else 0

        # ---- 4 类异常 ----
        # 1) 🔴 异常增长: 患者同比明显增长且变化量显著
        if pat_rate >= 20 and pat_chg >= 15:
            events.append({
                "type": "异常增长", "level": "red", "street": street,
                "district": district_of.get(street, ""),
                "pat_rate": round(pat_rate, 1), "pat_chg": pat_chg,
                "pop_growth": round(pop_g, 1), "hri24": round(hri24, 2), "hri25": round(hri25, 2),
                "vpr_rate": round(vpr_rate, 1), "opd_rate": round(opd_rate, 1),
                "score": round(pat_rate * 1.0 + abs(hri_rate) * 0.5 + abs(vpr_rate) * 0.3, 1),
            })
        # 2) 🔴 异常下降
        if pat_rate <= -20 and pat_chg <= -15:
            events.append({
                "type": "异常下降", "level": "red", "street": street,
                "district": district_of.get(street, ""),
                "pat_rate": round(pat_rate, 1), "pat_chg": pat_chg,
                "pop_growth": round(pop_g, 1), "hri24": round(hri24, 2), "hri25": round(hri25, 2),
                "vpr_rate": round(vpr_rate, 1), "opd_rate": round(opd_rate, 1),
                "score": round(abs(pat_rate) * 1.0 + abs(hri_rate) * 0.5, 1),
            })
        # 3) 🟠 覆盖不足: 人口增长明显但患者增长远低于人口
        if pop_g >= 5 and (pat_rate - pop_g) <= -10:
            events.append({
                "type": "覆盖不足", "level": "orange", "street": street,
                "district": district_of.get(street, ""),
                "pat_rate": round(pat_rate, 1), "pat_chg": pat_chg,
                "pop_growth": round(pop_g, 1), "hri24": round(hri24, 2), "hri25": round(hri25, 2),
                "vpr_rate": round(vpr_rate, 1), "opd_rate": round(opd_rate, 1),
                "score": round((pop_g - pat_rate) * 1.2, 1),
            })
        # 4) 🟢 专科优势: 街道某科室占比显著高于全院
        if depts and street in depts:
            all_dept = {}
            for s, dd in depts.items():
                for k, v in dd.items():
                    all_dept[k] = all_dept.get(k, 0) + v
            tot_dept = sum(all_dept.values())
            for dept, v in depts[street].items():
                if v <= 0:
                    continue
                share = v / max(sum(depts[street].values()), 1) * 100
                all_share = all_dept.get(dept, 0) / tot_dept * 100 if tot_dept else 0
                if share - all_share >= 15 and v >= 15:
                    events.append({
                        "type": "专科优势", "level": "green", "street": street,
                        "district": district_of.get(street, ""),
                        "dept": dept, "dept_share": round(share, 1), "all_share": round(all_share, 1),
                        "dept_v": v,
                        "pat_rate": round(pat_rate, 1), "pop_growth": round(pop_g, 1),
                        "hri24": round(hri24, 2), "hri25": round(hri25, 2),
                        "score": round((share - all_share) * 0.8 + v * 0.1, 1),
                    })

    # 按异常分排序取 TOP5 (不同类型尽量保留)
    events.sort(key=lambda x: -x["score"])
    top = events[:5]
    # 补充: 不同类型去重后补足5个
    if len(set(e["type"] for e in top)) < 3 and len(events) > 5:
        seen = {e["street"] + e["type"] for e in top}
        for e in events[5:]:
            if len(top) >= 5:
                break
            if e["street"] + e["type"] not in seen:
                top.append(e)
                seen.add(e["street"] + e["type"])
    return top

if __name__ == "__main__":
    for e in detect():
        print(e["level"], e["type"], e["street"], e.get("district", ""), "score=", e["score"])
