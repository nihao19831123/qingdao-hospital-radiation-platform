#!/usr/bin/env python3
"""AI 医疗服务辐射智能巡检引擎 - FastAPI 服务.
启动: python app.py  →  http://localhost:8000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ai_detector
import ai_attribution
import ai_explainer

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


if __name__ == "__main__":
    import uvicorn
    print("启动 AI 医疗服务辐射智能巡检引擎: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
