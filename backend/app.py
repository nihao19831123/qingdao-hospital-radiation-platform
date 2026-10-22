#!/usr/bin/env python3
"""AI 医疗服务辐射智能巡检引擎 - FastAPI 服务.
启动: python app.py  →  http://localhost:8000
"""
import json
import os
import sys
import sqlite3
import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategy_engine"))
import ai_detector
import ai_attribution
import ai_explainer
import ai_advisor
import competition
import strategy_advisor
import engine as strategy_engine
import report_html
import agent_api            # 智能体推演 API（阶段1–3，独立路由文件，只增不减）

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # 项目根目录(HTML 所在)

app = FastAPI(title="AI医疗服务辐射智能巡检引擎")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(agent_api.router)   # /api/strategy/agent/* (新增端点, 现有端点与页面零改动)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "青岛街道人口热力图.html"))


@app.get("/echarts.min.js")
def echarts():
    return FileResponse(os.path.join(ROOT, "echarts.min.js"))


@app.get("/maplibre-gl.js")
def ml_js():
    return FileResponse(os.path.join(ROOT, "maplibre-gl.js"))


@app.get("/maplibre-gl.css")
def ml_css():
    return FileResponse(os.path.join(ROOT, "maplibre-gl.css"))


@app.get("/swarm")
def swarm_page():
    """个体级患者智能体蜂群视图（阶段 6；数据全部取自 /api/strategy/agent/*）"""
    return FileResponse(os.path.join(BASE, "agent_swarm.html"))


@app.get("/api/mapdata")
def mapdata():
    """MapLibre WGS-84 地图数据 (含收缩街道+边界线)"""
    p = os.path.join(BASE, "mapdata_wgs84.json")
    ip = os.path.join(BASE, "street_inset.json")
    if os.path.exists(p) and os.path.exists(ip):
        d = json.load(open(p, encoding="utf-8"))
        ins = json.load(open(ip, encoding="utf-8"))
        d["streets_inset"] = ins["streets_inset"]
        d["streets_border"] = ins["streets_border"]
        return JSONResponse(d)
    return JSONResponse({"error": "mapdata 未生成"}, status_code=404)


@app.get("/api/inspect")
def inspect():
    """AI 智能巡检: 检测 + 归因 + 大模型解释(并发)"""
    from concurrent.futures import ThreadPoolExecutor
    events = ai_detector.detect()
    results = [None] * len(events)

    def work(i):
        e = events[i]
        att = ai_attribution.attribute(e["street"])
        exp = ai_explainer.explain(e, att)
        results[i] = {"event": e, "attribution": att, "explain": exp}

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(work, range(len(events))))
    return {"count": len(results), "items": results}


@app.get("/api/inspect/{street}")
def inspect_street(street: str):
    e = next((x for x in ai_detector.detect() if x["street"] == street), None)
    if not e:
        return JSONResponse({"error": "该街道无异常事件"}, status_code=404)
    att = ai_attribution.attribute(street)
    exp = ai_explainer.explain(e, att)
    return {"event": e, "attribution": att, "explain": exp}


@app.get("/api/competition")
def competition_api():
    """医院竞争环境: 全街道竞争压力指数 CPI + 三级及以上医院清单 (纯只读计算)"""
    return competition.competition_overview()


@app.get("/api/inspect/{street}/advice")
def inspect_advice(street: str):
    """AI 战略助手: 巡检事件 → 战略动作建议 (DeepSeek, 规则兜底)"""
    e = next((x for x in ai_detector.detect() if x["street"] == street), None)
    if not e:
        return JSONResponse({"error": "该街道无异常事件"}, status_code=404)
    att = ai_attribution.attribute(street)
    r = ai_advisor.advise_street(e, att)
    return {"street": street, "judgement": r["judgement"], "actions": r["actions"],
            "source": r["source"]}


# ============ 战略推演 API ============
def _sdb():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "strategy_engine", "hospital.db"))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_advice_table():
    """幂等创建 AI 战略建议表 (只新增, 不改动任何现有表)"""
    conn = _sdb()
    conn.execute("""CREATE TABLE IF NOT EXISTS ai_strategy_advice(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER,
        scenario TEXT,
        advice_type TEXT,
        content TEXT,
        source TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()


_ensure_advice_table()


@app.get("/api/hospital/capacity")
def hospital_capacity():
    """医院能力画像"""
    return strategy_engine.capacity()


@app.get("/api/hospital/departments")
def hospital_departments():
    """医院资源容量画像: 全院参数 + 科室能力全表 (dept_capacity 106 行, 按周号源降序).
    科室级无床位/手术量/平均住院日数据 (真实数据源没有), 不返回不编造."""
    conn = _sdb()
    params = {r["param"]: r["value"] for r in
              conn.execute("SELECT param, value FROM hospital_params")}
    rows = conn.execute("""SELECT dept_name, big_dept, doctors, chief_doctors, associate_chief,
                           attending, weekly_numsource, numsource_fees, numsource_weekdays
                           FROM dept_capacity ORDER BY weekly_numsource DESC""").fetchall()
    conn.close()
    departments = [dict(r) for r in rows]
    total = {
        "doctors": sum(d["doctors"] or 0 for d in departments),
        # 高级职称 = 主任医师 + 副主任医师
        "chiefs": sum((d["chief_doctors"] or 0) + (d["associate_chief"] or 0) for d in departments),
        "depts": len(departments),
        "weekly_numsource": sum(d["weekly_numsource"] or 0 for d in departments),
    }
    return {"hospital_params": params, "departments": departments, "total": total}


@app.post("/api/strategy/run")
async def strategy_run(req: dict):
    """运行一次推演并保存"""
    params = req.get("params", req)
    result = strategy_engine.simulate(params)
    # AI 建议: 优先 DeepSeek, 无 key / 失败自动落规则文本
    advice = strategy_advisor.advise_single(result, fallback=_strategy_advice(result))
    # 入库
    conn = _sdb()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    name = params.get("name") or f"{result['action_cn']}-{datetime.date.today().isoformat()}"
    cur = conn.execute("INSERT INTO strategy_config(name, created_at, params) VALUES (?,?,?)",
                       (name, now, json.dumps(params, ensure_ascii=False)))
    cid = cur.lastrowid
    conn.execute("INSERT INTO strategy_result(config_id, scenario, result, ai_advice, created_at) VALUES (?,?,?,?,?)",
                 (cid, result["action_cn"], json.dumps(result, ensure_ascii=False), json.dumps(advice, ensure_ascii=False), now))
    # AI 建议独立入表 (需求: ai_strategy_advice)
    conn.execute("INSERT INTO ai_strategy_advice(config_id, scenario, advice_type, content, source, created_at) VALUES (?,?,?,?,?,?)",
                 (cid, result["action_cn"], "single", advice["text"], advice["source"], now))
    conn.commit()
    conn.close()
    result["config_id"] = cid
    result["ai_advice"] = advice
    return result


def _strategy_advice(result):
    """规则生成战略建议 (可后续接 DeepSeek)"""
    s = result["summary"]
    y = result["years"]
    lines = [f"【{result['action_cn']}】推演{y[-1]['year']}年："]
    lines.append(f"预计累计新增门诊 {s['final_add_opd']:,} 人次、住院 {s['final_add_ipd']:,} 人次；")
    lines.append(f"床位使用率 {s['final_bed_usage']}%，号源利用率 {s['final_numsource_util']}%，80%辐射半径 {s['final_radius80']}km。")
    # 效益判断
    gain = s["final_add_opd"] + s["final_add_ipd"] * 8
    action = result["action_cn"]
    if action == "专科能力提升":
        top = result["top_streets"][:3]
        lines.append(f"受益最大区域：{'、'.join(t['name'] for t in top)}（李沧及周边近域为主），专科品牌辐射增强。")
        if s["final_bed_usage"] < 85:
            lines.append("床位仍有缓冲空间（<85%），无需立即扩容，建议优先保障专科门诊承载。")
    elif action == "医联体建设":
        lines.append("远郊低覆盖街道需求被激活，HRI 洼地补强，跨区域辐射结构改善。")
    elif action == "资源扩容":
        if s["final_numsource_util"] > 90:
            lines.append("扩容后号源利用率仍偏高，建议扩容幅度加大或同步提高复诊效率。")
        else:
            lines.append("扩容释放了被压制的就诊需求，号源利用率回落到健康区间。")
    if result["risks"]:
        lines.append("风险：" + "；".join(result["risks"]))
    else:
        lines.append("综合评估：该方案资源压力可控，可实施。")
    return {"text": " ".join(lines), "source": "rule"}


@app.get("/api/strategy/list")
def strategy_list():
    conn = _sdb()
    rows = conn.execute("""SELECT c.id, c.name, c.created_at, c.params, r.result
                           FROM strategy_config c LEFT JOIN strategy_result r ON r.config_id = c.id
                           ORDER BY c.id DESC""").fetchall()
    conn.close()
    out = []
    for r in rows:
        res = json.loads(r["result"]) if r["result"] else {}
        out.append({"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                    "params": json.loads(r["params"]), "summary": res.get("summary", {})})
    return {"count": len(out), "items": out}


@app.get("/api/strategy/{sid}")
def strategy_detail(sid: int):
    conn = _sdb()
    r = conn.execute("SELECT * FROM strategy_config WHERE id=?", (sid,)).fetchone()
    res = conn.execute("SELECT * FROM strategy_result WHERE config_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
    conn.close()
    if not r:
        return JSONResponse({"error": "未找到"}, status_code=404)
    return {"config": dict(r), "result": json.loads(res["result"]) if res else None}


@app.delete("/api/strategy/{sid}")
def strategy_delete(sid: int):
    conn = _sdb()
    conn.execute("DELETE FROM strategy_result WHERE config_id=?", (sid,))
    conn.execute("DELETE FROM strategy_config WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/strategy/scenario")
async def strategy_scenario(req: dict):
    """情景/敏感性分析: 以 magnitude 为中性, 保守×0.75 / 中性×1.0 / 乐观×1.25 三档推演.
    纯本地计算 (不写库, 不调 DeepSeek), 秒回; 返回三档 summary+years 与规则区间结论."""
    params = req.get("params", req)
    try:
        m = float(params.get("magnitude", 0.2))
    except (TypeError, ValueError):
        m = 0.2
    scenarios = []
    for name, mult in (("保守", 0.75), ("中性", 1.0), ("乐观", 1.25)):
        p = dict(params)
        p["magnitude"] = round(max(0.05, min(0.5, m * mult)), 4)  # clamp 到引擎允许范围
        r = strategy_engine.simulate(p)
        scenarios.append({"name": name, "magnitude": p["magnitude"], "action_cn": r["action_cn"],
                          "summary": r["summary"], "years": r["years"]})
    return {"params": params, "scenarios": scenarios,
            "conclusion": {"text": _scenario_conclusion(params, scenarios), "source": "rule"}}


def _scenario_conclusion(params, scenarios):
    """三档结果 → 规则区间结论 (min=保守 / 中性 / max=乐观, 引擎对 magnitude 单调)"""
    sm = [sc["summary"] for sc in scenarios]
    opd = [s["final_add_opd"] for s in sm]
    ipd = [s["final_add_ipd"] for s in sm]
    bed = [s["final_bed_usage"] for s in sm]
    util = [s["final_numsource_util"] for s in sm]
    dept = params.get("dept") or "全院"
    action_cn = scenarios[1].get("action_cn", "战略动作")
    try:
        mag_pct = round(float(params.get("magnitude", 0.2)) * 100)
    except (TypeError, ValueError):
        mag_pct = "—"
    # 床位警戒判断: 三档均低于90 / 乐观突破 / 全部突破
    if bed[2] <= 90:
        bed_verdict = "三情景均低于 90% 警戒线，资源压力可控"
    elif bed[0] <= 90:
        bed_verdict = "乐观情景将突破 90% 警戒线，建议同步扩容床位或优化住院流程"
    else:
        bed_verdict = "三情景均将突破 90% 警戒线，床位扩容为方案落地前置条件"
    return (f"【{dept}·{action_cn} {mag_pct}% / {params.get('years', 3)}年】"
            f"期末累计新增门诊预计 {_wan(opd[0])}~{_wan(opd[2])} 人次（中性 {_wan(opd[1])}），"
            f"新增住院 {_wan(ipd[0])}~{_wan(ipd[2])} 人次（中性 {_wan(ipd[1])}）；"
            f"期末床位使用率区间 {bed[0]}%~{bed[2]}%（中性 {bed[1]}%），{bed_verdict}；"
            f"号源利用率区间 {util[0]}%~{util[2]}%。"
            f"（保守/中性/乐观 = 战略效果兑现程度 75%/100%/125% 假设）")


def _wan(n):
    """大数转 万 口径 (与 report_html._wan 一致, 避免循环依赖而本地实现)"""
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "0"
    return f"{n / 10000:.1f}万" if n >= 10000 else f"{n:,.0f}"


@app.post("/api/report/generate")
async def report_generate(req: dict):
    """决策报告一键导出: 纯本地 HTML 排版, 不调 DeepSeek (直接嵌入已有 ai_advice, 缺失时规则兜底).
    body: {kind:'single'|'compare', name?, params 或 result?, ai_advice?, payload?|items?+recommendation?+scenarios?}
    返回 {html}; 单方案传完整 result 可避免二次模拟."""
    kind = req.get("kind", "single")
    if kind == "compare":
        payload = req.get("payload") or {
            "items": req.get("items", []),
            "recommendation": req.get("recommendation"),
            "scenarios": req.get("scenarios", []),
        }
        return {"html": report_html.build_report("compare", payload, None)}
    # 单方案: 优先用前端传来的完整 result; 缺失时后端自行模拟一次补齐 (不写库)
    result = req.get("result")
    if not (isinstance(result, dict) and result.get("years")):
        result = strategy_engine.simulate(req.get("params", req))
    if req.get("name"):
        result = dict(result)
        result["name"] = req["name"]
    advice = req.get("ai_advice")
    if not (isinstance(advice, dict) and advice.get("text")):
        advice = _strategy_advice(result)  # 本地规则兜底, 瞬时完成
    if req.get("agent_sim"):               # 智能体推演一节(可选, 设计 §9.3); 不传则报告内容与原来完全一致
        result = dict(result)
        result["agent_sim"] = req["agent_sim"]
    return {"html": report_html.build_report("single", result, advice)}


@app.post("/api/strategy/compare")
async def strategy_compare(req: dict):
    """多方案对比: {scenarios: [{name, params}...]}
    返回: {count, items: [...含 ai_advice], recommendation: {winner, reasons, source}}"""
    from concurrent.futures import ThreadPoolExecutor
    scenarios = req.get("scenarios", [])
    runs = []
    for sc in scenarios:
        r = strategy_engine.simulate(sc.get("params", {}))
        runs.append({"name": sc.get("name", "方案"), "result": r})
    # 并发生成各方案 AI 建议 (DeepSeek 慢, 规则兜底本地瞬时完成)
    items = [None] * len(runs)

    def work(i):
        r = runs[i]["result"]
        adv = strategy_advisor.advise_single(r, fallback=_strategy_advice(r))
        items[i] = {"name": runs[i]["name"], "summary": r["summary"],
                    "years": r["years"], "action": r["action_cn"], "risks": r["risks"],
                    "ai_advice": adv}
    if runs:
        with ThreadPoolExecutor(max_workers=min(5, len(runs))) as pool:
            list(pool.map(work, range(len(runs))))
    # 综合推荐: 优先 DeepSeek, 失败回退规则 (资源压力可控且新增患者最多)
    recommendation = strategy_advisor.advise_compare(items)
    return {"count": len(items), "items": items, "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn
    print("启动 AI 医疗服务辐射智能巡检引擎: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
