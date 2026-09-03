#!/usr/bin/env python3
"""医院能力画像: 导入医生脱敏表 + 号源数据 + 基本信息 -> hospital.db"""
import json
import os
import sqlite3
import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
DB = os.path.join(BASE, "hospital.db")

DOCTOR_XLSX = os.path.join(ROOT, "2026033脱敏信息+(1).xlsx")
NUMSOURCE_JSON = "/root/hospital_numsource.json"

def build():
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # 科室能力画像
    cur.execute("""CREATE TABLE dept_capacity(
        dept_code TEXT PRIMARY KEY, dept_name TEXT, big_dept TEXT,
        doctors INTEGER DEFAULT 0, chief_doctors INTEGER DEFAULT 0,
        associate_chief INTEGER DEFAULT 0, attending INTEGER DEFAULT 0,
        weekly_numsource INTEGER DEFAULT 0, numsource_fees TEXT DEFAULT '',
        numsource_weekdays INTEGER DEFAULT 0, updated_at TEXT)""")

    # 医生按科室聚合 (脱敏表)
    wb = openpyxl.load_workbook(DOCTOR_XLSX, data_only=True)
    ws = wb["Sheet1"]
    dept_stats = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[1]:
            continue
        dept = str(row[1]).strip()
        title = str(row[3]).strip() if row[3] else ""
        s = dept_stats.setdefault(dept, {"doctors": 0, "chief": 0, "assoc": 0, "attend": 0})
        s["doctors"] += 1
        if "主任医师" in title:
            s["chief"] += 1
        elif "副主任医师" in title:
            s["assoc"] += 1
        elif "主治医师" in title:
            s["attend"] += 1

    # 号源数据 (科室名匹配医生表)
    ns = json.load(open(NUMSOURCE_JSON, encoding="utf-8")) if os.path.exists(NUMSOURCE_JSON) else {"depts": {}}
    ns_by_name = {}
    for code, dep in ns.get("depts", {}).items():
        key = dep["dept_name"].replace("门诊", "").replace("（", "(").replace("）", ")")
        ns_by_name[key] = dep

    import datetime
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    # 匹配: 号源科室 -> 医生表科室 (模糊: 号源科室名去掉'门诊'后匹配医生科室名的前几字)
    # 医生表科室 -> 号源科室 映射 (名称不一致的)
    MANUAL_MAP = [
        ("心内科门诊", ["心内一科", "心内二科", "心内科"]),
        ("小儿科门诊", ["儿科", "小儿科"]),
        ("中医科门诊", ["中医科", "中医康复科"]),
        ("皮肤科门诊", ["皮肤科"]),
        ("神经内科门诊", ["神经内科"]),
        ("肿瘤科门诊", ["肿瘤科"]),
        ("内分泌科门诊", ["内分泌科"]),
        ("肾病老年医学科门诊", ["肾病老年医学科"]),
        ("康复医学科门诊", ["康复医学科"]),
        ("妇科门诊", ["妇科"]),
        ("产科门诊", ["产科医疗", "产科"]),
        ("骨科（创伤、脊柱、手足）门诊", ["骨科", "关节外科"]),
        ("神经外科门诊", ["神经外科"]),
        ("胸心外科门诊", ["胸外科"]),
        ("肛肠门诊", ["肛肠外科"]),
        ("泌尿外科、结石门诊", ["泌尿外科"]),
        ("肝胆外科、结石门诊", ["肝胆外科"]),
        ("胃肠外一科门诊", ["胃肠外科"]),
        ("普外（甲状腺、乳腺）门诊", ["甲状腺乳腺外科"]),
        ("耳鼻喉科门诊", ["耳鼻咽喉头颈外科"]),
        ("感染性肠道门诊", ["感染性疾病科"]),
        ("麻醉门诊", ["麻醉科"]),
        ("疼痛门诊", []),
        ("病理科门诊", ["病理科"]),
        ("放射科咨询门诊", ["放射科"]),
        ("超声医学科门诊", ["超声医学科"]),
    ]
    used_doctor_depts = set()
    for code, dep in ns.get("depts", {}).items():
        name = dep["dept_name"]
        base_name = name.replace("门诊", "").replace("（", "(").replace("）", ")")
        matched_depts = []
        for ns_name, doc_kws in MANUAL_MAP:
            if name == ns_name or (ns_name and base_name in ns_name):
                for kw in doc_kws:
                    for dname in dept_stats:
                        if kw and kw in dname and dname not in matched_depts:
                            matched_depts.append(dname)
        # 补充: 模糊匹配
        if not matched_depts:
            for dname in dept_stats:
                if (len(base_name) >= 4 and base_name in dname) or (len(dname) >= 4 and dname in base_name):
                    matched_depts.append(dname)
        for dname in matched_depts:
            used_doctor_depts.add(dname)
        docs = sum(dept_stats[d]["doctors"] for d in matched_depts) if matched_depts else 0
        chiefs = sum(dept_stats[d]["chief"] for d in matched_depts)
        assocs = sum(dept_stats[d]["assoc"] for d in matched_depts)
        attends = sum(dept_stats[d]["attend"] for d in matched_depts)
        week_tot = sum(v["am"] + v["pm"] for v in dep["days"].values())
        fees = "/".join(sorted({str(f) for v in dep["days"].values() for f in v["fees"]}))
        rows.append((code, name, dep.get("big_dept", ""), docs, chiefs, assocs, attends,
                     week_tot, fees, len(dep["days"]), now))
    # 医生表有但号源没有的科室 (如手术/住院科室) 也入库
    for dname, s in dept_stats.items():
        if dname in used_doctor_depts:
            continue
        if any(x in dname for x in ["待用", "院领导", "医务科", "门诊部", "轮转"]):
            continue
        rows.append((f"DOC-{dname}", dname, "医生科室", s["doctors"], s["chief"], s["assoc"], s["attend"], 0, "", 0, now))

    cur.executemany("INSERT INTO dept_capacity VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    # 医院基础参数表
    cur.execute("""CREATE TABLE hospital_params(
        param TEXT PRIMARY KEY, value REAL, note TEXT)""")
    params = [
        ("beds", 900, "医院编制床位(可修改)"),
        ("bed_usage_rate", 0.64, "床位使用率基线"),
        ("annual_opd", 550000, "年门诊量基线"),
        ("annual_ipd", 26000, "年住院量基线"),
        ("surgery_per_day", 30, "日均手术量"),
        ("avg_wait_min", 5, "平均候诊分钟"),
        ("radius80_km", 15.1, "80%辐射半径基线km"),
    ]
    cur.executemany("INSERT INTO hospital_params VALUES (?,?,?)", params)

    # 推演表
    cur.execute("""CREATE TABLE strategy_config(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TEXT,
        params TEXT)""")
    cur.execute("""CREATE TABLE strategy_result(
        id INTEGER PRIMARY KEY AUTOINCREMENT, config_id INTEGER,
        scenario TEXT, result TEXT, ai_advice TEXT, created_at TEXT)""")

    conn.commit()
    print(f"数据库: {DB}")
    cur.execute("SELECT COUNT(*) FROM dept_capacity")
    print(f"科室画像行数: {cur.fetchone()[0]}")
    cur.execute("SELECT dept_name, doctors, weekly_numsource FROM dept_capacity WHERE weekly_numsource > 0 ORDER BY weekly_numsource DESC LIMIT 8")
    for r in cur.fetchall():
        print(" ", r)
    conn.close()

if __name__ == "__main__":
    build()
