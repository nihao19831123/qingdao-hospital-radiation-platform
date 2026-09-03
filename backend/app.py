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
import engine as strategy_engine

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


# ============ 战略推演 API ============
def _sdb():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "strategy_engine", "hospital.db"))
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/hospital/capacity")
def hospital_capacity():
    """医院能力画像"""
    return strategy_engine.capacity()


@app.post("/api/strategy/run")
async def strategy_run(req: dict):
    """运行一次推演并保存"""
    params = req.get("params", req)
    result = strategy_engine.simulate(params)
    # AI 建议
    advice = _strategy_advice(result)
    # 入库
    conn = _sdb()
    name = params.get("name") or f"{result['action_cn']}-{datetime.date.today().isoformat()}"
    cur = conn.execute("INSERT INTO strategy_config(name, created_at, params) VALUES (?,?,?)",
                       (name, datetime.datetime.now().isoformat(timespec="seconds"), json.dumps(params, ensure_ascii=False)))
    cid = cur.lastrowid
    conn.execute("INSERT INTO strategy_result(config_id, scenario, result, ai_advice, created_at) VALUES (?,?,?,?,?)",
                 (cid, result["action_cn"], json.dumps(result, ensure_ascii=False), json.dumps(advice, ensure_ascii=False),
                  datetime.datetime.now().isoformat(timespec="seconds")))
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


@app.post("/api/strategy/compare")
async def strategy_compare(req: dict):
    """多方案对比: {scenarios: [{name, params}...]}"""
    scenarios = req.get("scenarios", [])
    out = []
    for sc in scenarios:
        r = strategy_engine.simulate(sc.get("params", {}))
        out.append({"name": sc.get("name", "方案"), "summary": r["summary"],
                    "years": r["years"], "action": r["action_cn"], "risks": r["risks"]})
    return {"count": len(out), "items": out}


if __name__ == "__main__":
    import uvicorn
    print("启动 AI 医疗服务辐射智能巡检引擎: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
