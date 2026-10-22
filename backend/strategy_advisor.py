#!/usr/bin/env python3
"""AI 战略建议器: 把战略推演结果交给 DeepSeek 生成面向医院领导的战略结论.
- advise_single(result): 单方案推演的管理结论
- advise_compare(items): 多方案对比的综合推荐
无 API key 或调用失败时回退规则文本 (演示模式), 返回结构 {"text"/"winner","source"}.
"""
import json
import os
import re
import urllib.request

def _load_key():
    # 从环境变量或 ~/.hermes/.env 读取 (与 ai_explainer 保持一致, 禁止写入代码)
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

def _chat(system_prompt, user_payload):
    """调用 DeepSeek 返回文本; 无 key 或失败返回 None"""
    key = _load_key()
    if not key:
        return None
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.4,
        "max_tokens": 600,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

# ---------- 单方案建议 ----------
def _single_payload(result):
    """压缩推演结果为喂给模型的载荷 (去掉街道明细长列表)"""
    params = dict(result.get("params", {}) or {})
    streets = params.pop("streets", None)
    params["streets_count"] = len(streets) if isinstance(streets, list) else 0
    return {
        "action": result.get("action_cn", ""),
        "params": params,
        "summary": result.get("summary", {}),
        "years": result.get("years", []),
        "top_streets": [{"name": t["name"], "add_opd": t["add_opd"], "add_ipd": t["add_ipd"]}
                        for t in result.get("top_streets", [])[:5]],
        "risks": result.get("risks", []),
    }

def _single_fallback_text(result):
    """最小规则文本 (调用方未提供兜底时使用)"""
    s = result.get("summary", {})
    return (f"【{result.get('action_cn', '')}】预计累计新增门诊 {s.get('final_add_opd', 0):,} 人次、"
            f"住院 {s.get('final_add_ipd', 0):,} 人次；床位使用率 {s.get('final_bed_usage', 0)}%、"
            f"号源利用率 {s.get('final_numsource_util', 0)}%。（规则判断）")

def advise_single(result, fallback=None):
    """单方案推演的 AI 战略建议; fallback 为 {"text","source"} 规则兜底"""
    text = _chat(
        "你是医院战略管理决策助手。请根据给定的战略推演结果，生成一段面向医院领导的战略推演管理结论（200字内），"
        "必须包含：①方案动作与预期收益量化概述；②资源压力判断（床位使用率/号源利用率是否逼近警戒线）；"
        "③风险提示；④落地建议。语言专业、简洁、有管理视角，直接输出结论。",
        _single_payload(result),
    )
    if text:
        return {"text": text, "source": "deepseek"}
    if isinstance(fallback, dict) and fallback.get("text"):
        return {"text": fallback["text"], "source": fallback.get("source", "rule")}
    return {"text": _single_fallback_text(result), "source": "rule"}

# ---------- 多方案对比推荐 ----------
def _compare_payload(items):
    """各方案摘要 (name/action/summary/risks)"""
    return [{"name": it.get("name", ""), "action": it.get("action", ""),
             "summary": it.get("summary", {}), "risks": it.get("risks", [])}
            for it in items]

def _rule_compare(items):
    """规则推荐: 资源压力可控 (床位≤90%、号源≤95%) 且新增患者最多的方案"""
    def gain(it):
        s = it.get("summary", {})
        return s.get("final_add_opd", 0) + s.get("final_add_ipd", 0) * 8
    def safe(it):
        s = it.get("summary", {})
        return s.get("final_bed_usage", 0) <= 90 and s.get("final_numsource_util", 0) <= 95
    if not items:
        return {"winner": "", "reasons": "无对比方案。", "source": "rule"}
    pool = [it for it in items if safe(it)] or items
    win = max(pool, key=gain)
    s = win.get("summary", {})
    pressure = "资源压力可控" if pool is not items else "各方案资源压力均接近警戒线，需配套扩容"
    reasons = (f"该方案预计累计新增门诊 {s.get('final_add_opd', 0):,} 人次、住院 {s.get('final_add_ipd', 0):,} 人次，"
               f"期末床位使用率 {s.get('final_bed_usage', 0)}%、号源利用率 {s.get('final_numsource_util', 0)}%，"
               f"{pressure}；按收益-风险综合权衡为最优。（规则判断）")
    return {"winner": win.get("name", ""), "reasons": reasons, "source": "rule"}

def advise_compare(items):
    """多方案对比的综合推荐: {"winner","reasons","source"}"""
    if not items:
        return {"winner": "", "reasons": "无对比方案。", "source": "rule"}
    text = _chat(
        "你是医院战略管理决策助手。请根据给定的多个战略方案推演摘要，生成中文综合比较结论（250字内）："
        "客观比较各方案的累计新增门诊/住院、期末资源压力（床位使用率/号源利用率）与风险，"
        "并给出“最优方案推荐”及理由（收益-风险权衡）。语言专业、简洁。"
        "最后必须单独一行，严格按格式输出：最优方案：<方案名>（方案名须与输入中的方案名称完全一致）。",
        _compare_payload(items),
    )
    if text:
        # 解析末行 “最优方案：xxx”
        winner = None
        body = text
        for line in text.splitlines():
            m = re.search(r"最优方案\s*[:：]\s*(.+)", line.strip())
            if m:
                cand = m.group(1).strip().strip('"').strip("「」“”")
                # 与输入方案名精确匹配 → 前缀/包含匹配
                names = [it.get("name", "") for it in items]
                if cand in names:
                    winner = cand
                else:
                    hits = [n for n in names if cand in n or n in cand]
                    winner = hits[0] if hits else None
                body = text.replace(line, "").strip()
                break
        if winner:
            return {"winner": winner, "reasons": body, "source": "deepseek"}
    return _rule_compare(items)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategy_engine"))
    import engine
    r = engine.simulate({"action": "specialty", "dept": "心内科", "streets": [],
                         "magnitude": 0.2, "years": 3})
    a = advise_single(r)
    print("单方案来源:", a["source"])
    print(a["text"])
    items = []
    for name, act in [("方案A 专科提升", "specialty"), ("方案B 医联体", "alliance"), ("方案C 资源扩容", "capacity")]:
        rr = engine.simulate({"action": act, "dept": "全院" if act != "specialty" else "心内科",
                              "streets": [], "magnitude": 0.2, "years": 3})
        items.append({"name": name, "action": rr["action_cn"], "summary": rr["summary"], "risks": rr["risks"]})
    c = advise_compare(items)
    print("\n对比推荐来源:", c["source"], "| 最优:", c["winner"])
    print(c["reasons"])
