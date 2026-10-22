#!/usr/bin/env python3
"""④ AI 战略助手: 对 AI 巡检事件生成战略动作建议 (感知 → 洞察 → 推演 的联动层).
- advise_street(event, attribution): 调 DeepSeek 输出 {"judgement", "actions", "source"}
- 无 key / 调用失败 / JSON 解析失败时回退规则模板 (按事件类型给默认动作)
动作 action 取值: specialty(专科能力提升) / alliance(医联体建设) / capacity(资源扩容).
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

def _chat(prompt_payload):
    """调用 DeepSeek 返回文本; 无 key 或失败返回 None (urllib 方式, 同 ai_explainer)"""
    key = _load_key()
    if not key:
        return None
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content":
                "你是医院战略助手。根据巡检事件与归因数据，给出该区域的医疗需求判断和战略动作建议。"
                "必须严格输出 JSON（不要 markdown 代码块）："
                '{"judgement": "该区域医疗需求判断(50字内)", '
                '"actions": [{"action": "specialty|alliance|capacity", "dept": "心内科或null", '
                '"note": "建议说明(40字内)"}]}，actions 1-3 条。'
                "action 含义: specialty=专科能力提升(可带dept), alliance=医联体建设(dept为null), "
                "capacity=资源扩容(dept为null)。"
                "若数据中competition竞争压力为高或极高，优先给出alliance(医联体/转诊对接)类动作，note可引用最近竞争医院。"},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
        "temperature": 0.4,
        "max_tokens": 400,
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

def _parse_actions_json(text):
    """解析 DeepSeek 返回的 JSON (容忍 markdown 代码块包裹); 失败返回 None"""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"\{.*\}", t, re.S)  # 剥离 ```json 包裹或前后多余文字
    if m:
        t = m.group(0)
    try:
        obj = json.loads(t)
    except Exception:
        return None
    judgement = obj.get("judgement")
    actions = obj.get("actions")
    if not isinstance(actions, list) or not actions:
        return None
    out_acts = []
    for a in actions[:3]:
        if not isinstance(a, dict):
            continue
        act = a.get("action")
        if act not in ("specialty", "alliance", "capacity"):
            continue
        dept = a.get("dept")
        if act != "specialty":
            dept = None  # 医联体/扩容为全院动作
        out_acts.append({"action": act, "dept": dept, "note": str(a.get("note", ""))[:60]})
    if not out_acts:
        return None
    return {"judgement": str(judgement or "")[:120], "actions": out_acts}

# ---------- 规则兜底 ----------
def _top_dept(event, attribution):
    """默认目标科室: 优先事件自带科室(专科优势), 其次归因贡献第一的科室"""
    if event.get("dept"):
        return event["dept"]
    if attribution:
        contrib = attribution.get("dept_contrib") or []
        if contrib:
            return contrib[0].get("dept")
    return None

def _rule_advice(event, attribution):
    """规则模板: 按事件类型给默认动作 (异常增长→专科巩固, 覆盖不足→医联体激活, 专科优势→专科扩影响).
    高/极高竞争压力街道 (attribution.competition) 优先插入 alliance(医联体/转诊对接) 动作."""
    t = event.get("type", "")
    dept = _top_dept(event, attribution)
    comp = (attribution or {}).get("competition") or {} if isinstance(attribution, dict) else {}
    if t == "异常增长":
        judgement = f"{event['street']}患者同比+{event.get('pat_rate', 0)}%，医疗需求正在快速释放，建议巩固承接能力。"
        actions = [{"action": "specialty", "dept": dept,
                    "note": f"需求快速释放，建议{'「' + dept + '」' if dept else '优势专科'}能力提升承接增量"}]
    elif t == "异常下降":
        judgement = f"{event['street']}患者同比{event.get('pat_rate', 0)}%，存在患者流失风险，需核查可及性。"
        actions = [{"action": "alliance", "dept": None,
                    "note": "核查流失原因，建议医联体/转诊对接挽回患者"}]
    elif t == "覆盖不足":
        judgement = f"{street_pop(event)}人口增长+{event.get('pop_growth', 0)}%但患者增长不足，服务覆盖存在缺口。"
        actions = [{"action": "alliance", "dept": None,
                    "note": "人口红利未转化，建议医联体建设激活低覆盖需求"},
                   {"action": "specialty", "dept": dept,
                    "note": "同步提升专科渗透，承接人口增长带来的就诊需求"}]
    elif t == "专科优势":
        dept = event.get("dept") or dept
        judgement = f"{event['street']}「{dept}」占出院{event.get('dept_share', 0)}%（全院{event.get('all_share', 0)}%），专科辐射优势明显。"
        actions = [{"action": "specialty", "dept": dept,
                    "note": f"建议「{dept}」能力提升，扩大区域专科影响力"}]
    else:
        judgement = f"{event.get('street', '')}出现{t}事件，建议结合数据持续监测。"
        actions = [{"action": "capacity", "dept": None, "note": "建议资源扩容保障服务承接"}]
    # 竞争压力附加: 高/极高竞争街道优先给出 alliance 动作 (note 引用最近竞争医院)
    if comp.get("level") in ("高", "极高"):
        note = (f"距{comp['nearest_name']}({comp['nearest_level']})仅{comp['nearest_dist']}km且更近，"
                f"优先医联体/转诊对接防截流")
        if actions and actions[0]["action"] == "alliance":
            actions[0]["note"] = note  # 首动作已是医联体 → 改写 note 引用最近竞争医院
        else:
            actions.insert(0, {"action": "alliance", "dept": None, "note": note})
            actions = actions[:3]  # 维持 1-3 条约束
    return {"judgement": judgement, "actions": actions, "source": "rule"}

def street_pop(event):
    return event.get("street", "该区域")

def advise_street(event, attribution=None):
    """巡检事件 → 战略动作建议.
    返回 {"judgement": str, "actions": [{action, dept, note}], "source": "deepseek"|"rule"}"""
    payload = {"event": event, "attribution": attribution,
               "competition": (attribution or {}).get("competition") if isinstance(attribution, dict) else None}
    text = _chat(payload)
    parsed = _parse_actions_json(text)
    if parsed:
        return {"judgement": parsed["judgement"], "actions": parsed["actions"], "source": "deepseek"}
    return _rule_advice(event, attribution)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import ai_detector, ai_attribution
    events = ai_detector.detect()
    if events:
        e = events[0]
        att = ai_attribution.attribute(e["street"])
        r = advise_street(e, att)
        print("事件:", e["type"], e["street"], "| 来源:", r["source"])
        print("判断:", r["judgement"])
        for a in r["actions"]:
            print(" 动作:", a)
