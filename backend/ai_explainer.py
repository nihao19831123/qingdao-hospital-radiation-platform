#!/usr/bin/env python3
"""③ 大模型解释器: 把结构化检测结果交给 DeepSeek 生成管理结论.
无 API key 时回退到规则模板 (演示模式).
"""
import json
import os
import re
import urllib.request

def _load_key():
    # 从环境变量或 ~/.hermes/.env 读取
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

def _template(event, attribution):
    """规则模板生成 (离线可用)"""
    e = event
    d = attribution
    dept_top = d["dept_contrib"][:3]
    dept_line = "、".join(f"{x['dept']}({x['pct']}%)" for x in dept_top) or "无明显科室集中"
    opd_pct = d["visit_contrib"].get("opd_pct", 0)
    lines = [
        f"【{e.get('type','')}】{e['district']}「{e['street']}」",
    ]
    if e.get("pat_rate") is not None:
        lines.append(f"该街道本年度患者同比{e['pat_rate']:+.1f}%（变化{e['pat_chg']:+d}人次）")
    if e.get("pop_growth") is not None:
        lines.append(f"人口增速{e['pop_growth']:+.1f}%")
    if e.get("hri24") is not None:
        lines.append(f"HRI由{e['hri24']}变化至{e['hri25']}")
    if e.get("dept") is not None:
        lines.append(f"「{e['dept']}」占该街道出院{e['dept_share']}%（全院{e['all_share']}%），形成专科优势")
    lines.append(f"增量主要来源科室：{dept_line}；门诊贡献约{opd_pct}%")
    advice = {
        "异常增长": "建议进一步分析该区域患者增长来源、转诊渠道及复诊结构，判断是否形成稳定的辐射优势。",
        "异常下降": "建议核查该区域服务可及性、医保政策及转诊流向变化，评估是否存在患者流失。",
        "覆盖不足": "建议加强该区域专科宣传、医联体协作与转诊对接，抓住人口增长带来的服务机会。",
        "专科优势": "建议巩固该专科品牌，结合学科建设进一步扩大区域影响力。",
    }.get(e.get("type", ""), "建议结合数据持续监测。")
    lines.append(f"【建议关注】{advice}")
    return " ".join(lines)

def explain(event, attribution):
    """调用 DeepSeek 生成管理洞察; 失败则回退模板"""
    key = _load_key()
    if not key:
        return {"text": _template(event, attribution), "source": "template"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是医院管理决策助手。请根据给定的结构化医疗数据，生成一段面向医院领导的管理分析结论（150字内），包含：现象概述、原因归因、建议关注。语言专业、简洁、有管理视角。"},
            {"role": "user", "content": json.dumps({"event": event, "attribution": attribution}, ensure_ascii=False)},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            r = json.loads(resp.read().decode())
            text = r["choices"][0]["message"]["content"].strip()
            return {"text": text, "source": "deepseek"}
    except Exception:
        return {"text": _template(event, attribution), "source": "template"}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import ai_detector, ai_attribution
    events = ai_detector.detect()
    if events:
        e = events[0]
        att = ai_attribution.attribute(e["street"])
        print("事件:", e["type"], e["street"])
        r = explain(e, att)
        print("来源:", r["source"])
        print(r["text"])
