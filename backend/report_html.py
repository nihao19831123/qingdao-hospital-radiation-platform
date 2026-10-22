#!/usr/bin/env python3
"""决策报告 HTML 生成器 (纯标准库, 无外部依赖).
build_report(kind, payload, advice) → 返回完整自包含 HTML 字符串 (内嵌 CSS, A4 打印友好):
- kind='single' : payload = engine.simulate 完整输出 (可附带 name), advice = {"text","source"}
- kind='compare': payload = /api/strategy/compare 响应 (items/recommendation, 可含 scenarios 参数表)
生成过程不发起任何网络请求; AI 建议直接嵌入调用方已产出的 advice (DeepSeek/规则), 保证秒出.
"""
import datetime
from html import escape


def _e(v):
    """html 转义 (数字/None 自动转字符串, None → '—')"""
    if v is None:
        return "—"
    return escape(str(v), quote=True)

# ---------- 数据口径与模型说明 (固定段落) ----------
_CALIBRATION = [
    ("人口基数", "第七次全国人口普查街道人口 × WorldPop 栅格分布 × 区级总量校准"),
    ("就诊分布", "街道级就诊（门诊/住院）辐射模型模拟（2020—2026）"),
    ("医院容量", "院内真实运行参数：编制床位 800 张 / 年门诊 55 万人次 / 年住院 2.6 万人次 / 周号源 13,759 个"),
    ("推演方法", "战略参数 × 距离衰减 × 容量约束 的街道级仿真，按年度累进释放效果"),
    ("数据声明", "模拟数据仅用于决策实验演示，不作为真实运营数据"),
]

_CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", "SimHei", sans-serif;
       color: #1f2937; background: #fff; font-size: 12px; line-height: 1.75; margin: 0; padding: 28px 32px; }
.cover { text-align: center; border-bottom: 3px double #15803d; padding: 40px 0 22px; margin-bottom: 20px; }
.cover h1 { font-size: 22px; color: #14532d; letter-spacing: 2px; margin: 0 0 14px; }
.cover .sub { font-size: 14px; color: #374151; margin: 4px 0; }
.cover .meta { font-size: 11.5px; color: #6b7280; margin-top: 14px; }
h2.sec { font-size: 14.5px; color: #14532d; border-left: 4px solid #16a34a; padding-left: 9px; margin: 20px 0 6px; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 10px; }
th, td { border: 1px solid #9ca3af; padding: 5px 8px; font-size: 11.5px; text-align: left; vertical-align: top; }
th { background: #ecfdf5; color: #14532d; white-space: nowrap; }
tbody tr:nth-child(even) td { background: #f8fafc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.kv td:first-child { width: 128px; background: #f0fdf4; color: #374151; white-space: nowrap; }
.risk { border-left: 3px solid #d97706; background: #fffbeb; padding: 6px 10px; margin: 5px 0; }
.risk-none { border-left: 3px solid #16a34a; background: #f0fdf4; padding: 6px 10px; margin: 5px 0; }
.advice { border: 1px solid #bbf7d0; background: #f0fdf4; padding: 10px 12px; border-radius: 4px; }
.src-tag { display: inline-block; font-size: 10px; color: #fff; background: #16a34a; border-radius: 3px; padding: 0 6px; margin-left: 6px; vertical-align: 1px; }
.src-tag.rule { background: #64748b; }
.note { color: #6b7280; font-size: 10.5px; }
.footer { margin-top: 30px; padding-top: 8px; border-top: 1px solid #d1d5db;
          text-align: center; color: #6b7280; font-size: 10.5px; }
@media print { body { padding: 0; } .cover { padding-top: 8px; } }
"""


# ---------- 格式化 ----------
def _num(n):
    try:
        return f"{float(n):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _wan(n):
    """大数转 万 口径 (结论/摘要用)"""
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "0"
    return f"{n / 10000:.1f}万" if n >= 10000 else f"{n:,.0f}"


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def _pct(v):
    return "—" if v is None else f"{v}%"


def _hri(s):
    if s.get("hri_before") is None or s.get("hri_after") is None:
        return "—"
    return f"{s['hri_before']} → {s['hri_after']}"


def _table(headers, rows):
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td{c[1]}>{c[0]}</td>" if isinstance(c, tuple) else f"<td>{c}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def _kv(pairs):
    trs = "".join(f"<tr><td>{_e(kv[0])}</td><td>{kv[1]}</td></tr>" for kv in pairs if kv)
    return f'<table class="kv"><tbody>{trs}</tbody></table>'


def _advice_html(advice, title="AI 战略建议"):
    """advice = {"text","source"}; 缺失时给出占位说明"""
    if not (isinstance(advice, dict) and advice.get("text")):
        return f"<h2 class=\"sec\">{title}</h2><div class=\"advice note\">（本次未生成 AI 建议，可先运行推演获取 DeepSeek/规则建议后导出）</div>"
    src = advice.get("source", "rule")
    tag = "DeepSeek 大模型生成" if src == "deepseek" else "规则引擎生成"
    cls = "" if src == "deepseek" else " rule"
    return (f"<h2 class=\"sec\">{title}</h2>"
            f"<div class=\"advice\">{_e(advice['text'])}<span class=\"src-tag{cls}\">{tag}</span></div>")


def _calibration_html(prefix=""):
    rows = [(_e(k), _e(v)) for k, v in _CALIBRATION]
    return f"<h2 class=\"sec\">{prefix}数据口径与模型说明</h2>" + _kv(rows) + \
           '<div class="note">本报告全部指标由战略推演引擎本地计算生成；「AI 建议」部分来源已在文中标注。</div>'


# ---------- 智能体推演一节（可选，设计 §9.3 报告集成）----------
def _sub(t):
    """小节标题（不新增 CSS 类，避免影响既有排版）"""
    return f'<div style="font-weight:600;color:#14532d;margin:11px 0 3px">{_e(t)}</div>'


def _agent_sim_html(a, prefix=""):
    """a = {'params','seed','algo_version','data_version','precision','metrics','years',
             'street_effects', 可选 'sensitivity','backtest','note'}
    必须带：参数快照、种子、算法版本、数据版本、精度档（可复算）。"""
    if not isinstance(a, dict):
        return ""
    p = a.get("params") or {}
    streets = p.get("streets") or []
    act_cn = {"specialty": "专科能力提升", "alliance": "医联体建设", "capacity": "资源扩容"}.get(p.get("action"), p.get("action", "—"))
    mag = p.get("magnitude")
    parts = [f'<h2 class="sec">{prefix}智能体推演（个体级 Logit 仿真）</h2>']

    # 1 可复算信息
    parts.append(_sub("① 可复算信息（同参数 + 同种子可逐位复算）"))
    parts.append(_kv([
        ("战略动作", f"{_e(act_cn)}" + ("" if not p.get("dept") else f" · {_e(p['dept'])}")),
        ("战略强度 / 周期", f"{round(float(mag) * 100)}% / {_e(p.get('years', '—'))} 年" if mag is not None else "—"),
        ("影响区域", _e(f"{len(streets)} 个指定街道" if streets else "全域（全部街道）")),
        ("份额口径", _e((a.get("share_report") or {}).get("basis", p.get("share_basis", "—")))
                     + "（" + _e((a.get("share_report") or {}).get("denominator_note", "—")) + "）"),
        ("随机种子", _e(a.get("seed"))),
        ("算法版本 / 数据版本", f"{_e(a.get('algo_version'))} / {_e(a.get('data_version'))}"),
        ("精度档", _e(a.get("precision") or "—")),
        ("智能体规模", _e(a.get("agent_count") or (a.get("metrics") or {}).get("agents") or "—")),
    ]))

    # 2 结果区间（P5/P50/P95）
    m = a.get("metrics") or {}
    def rng(k, cn, unit="", signed=False):
        v = m.get(k)
        if not isinstance(v, dict):
            return None
        f = (lambda x: ("+" if signed and x > 0 else "") + _num(x))
        return (cn, f"{f(v.get('p5'))} ~ {f(v.get('p95'))}（中性 {f(v.get('p50'))}{_e(unit)}）")
    rows = [x for x in [
        rng("final_add_opd", "累计新增门诊（人次）", signed=True),
        rng("final_add_ipd", "累计新增住院（人次）", signed=True),
        rng("final_bed_usage", "期末床位使用率", "%"),
        rng("final_numsource_util", "期末号源利用率", "%"),
        rng("self_last", "期末选择我院的智能体（个）"),
        rng("from_comp", "其中：从竞争医院转来（个）", signed=True),
        rng("from_out", "其中：从不就诊被激活（个）", signed=True),
        rng("lost", "流失（个）"),
    ] if x]
    if rows:
        parts.append(_sub("② 推演结果区间（蒙特卡洛 P5–P95，非单点值）"))
        parts.append(_kv(rows))

    # 3 分年结果
    yrs = a.get("years") or []
    if yrs:
        rows = [[
            _e(y.get("year")), ("+" + _num(y.get("add_opd")), ' class="num"'),
            (f"{_num(y.get('add_opd_p5'))} ~ {_num(y.get('add_opd_p95'))}", ' class="num"'),
            (_num(y.get("cum_opd")), ' class="num"'), (_num(y.get("cum_ipd")), ' class="num"'),
            _pct(y.get("bed_usage")), _pct(y.get("numsource_util")),
            (_num(y.get("spill_opd")), ' class="num"'),
        ] for y in yrs]
        parts.append(_sub("③ 分年推演结果（容量溢出滚入次年）"))
        parts.append(_table(["年份", "新增门诊", "新增门诊 P5~P95", "累计门诊", "累计住院",
                            "床位使用率", "号源利用率", "溢出滚存(门诊)"], rows))

    # 4 重点街道
    se = (a.get("street_effects") or [])[:8]
    if se:
        rows = [[
            _e(e.get("name")), _e(e.get("district", "—")), (_e(e.get("d_km", "—")), ' class="num"'),
            (_e(e.get("agents", "—")), ' class="num"'),
            f"{_e(e.get('self_pre', '—'))} → {_e(e.get('self_cur', '—'))}", (_e(e.get("net")), ' class="num"'),
            (_e(e.get("from_comp")), ' class="num"'), (_e(e.get("from_out")), ' class="num"'),
            (_e(e.get("lost")), ' class="num"'),
        ] for e in se]
        parts.append(_sub("④ 重点影响街道 TOP8（净增 = 转来 + 激活 − 流失）"))
        parts.append(_table(["街道", "行政区", "距离(km)", "个体数", "我院(期初→期末)", "净增",
                            "转来", "激活", "流失"], rows))

    # 5 敏感性
    sens = a.get("sensitivity")
    if isinstance(sens, dict) and sens.get("rows"):
        base_opd = (sens.get("base") or {}).get("add_opd_p50")
        rows = []
        for r in sorted(sens["rows"], key=lambda x: x.get("rank", 99)):
            rows.append([
                _e(r.get("rank")), _e(r.get("param_cn", r.get("param"))),
                (_num((r.get("down") or {}).get("add_opd_p50")), ' class="num"'),
                (_num(base_opd), ' class="num"'),
                (_num((r.get("up") or {}).get("add_opd_p50")), ' class="num"'),
                (_e(r.get("elasticity")), ' class="num"'),
            ])
        parts.append(_sub(f"⑤ 敏感性分析（各参数 ±{round(float(sens.get('delta', 0.2)) * 100)}% 扰动，主指标=累计新增门诊 P50）"))
        parts.append(_table(["排名", "参数", "−20%", "基准", "+20%", "弹性"], rows))
        parts.append(f'<div class="note">方向自检：{_e(sens.get("passed"))}/{_e(sens.get("total"))} 通过；'
                     f'{_e(sens.get("verdict", ""))}。{_e(sens.get("note", ""))}</div>')
        for fl in (sens.get("flags") or []):
            parts.append(f'<div class="risk">⚠ {_e(fl.get("title"))}<br/>{_e(fl.get("detail"))}</div>')

    # 6 回测
    bt = a.get("backtest")
    if isinstance(bt, dict) and bt:
        variants = []
        if bt.get("folds") or bt.get("mape_avg") is not None:
            variants.append(("回测", bt))
        for k, cn in (("variant1_distance_decay", "变体1 距离衰减单因子"),
                      ("variant3_loglin_hri", "变体3 对数线性 + HRI 历史行为")):
            if isinstance(bt.get(k), dict):
                variants.append((cn, bt[k]))
        rows = []
        for cn, v in variants:
            for f in (v.get("folds") or []):
                rows.append([_e(cn), _e(f.get("year")), (_e(f.get("mape")), ' class="num"'),
                             (_e(f.get("spearman")), ' class="num"'), (_e(f.get("streets", "—")), ' class="num"'),
                             ("达标" if (f.get("mape", 99) <= 25 and f.get("spearman", 0) >= 0.7) else "见判据")])
            if v.get("mape_avg") is not None:
                rows.append([_e(cn), "综合", (_e(v.get("mape_avg")), ' class="num"'),
                             (_e(v.get("spearman_avg")), ' class="num"'), "—", _e(v.get("verdict", ""))])
        if rows:
            parts.append(_sub("⑥ 回测误差表（拟合 2020–2022 / 外推验证 2023–2024，训练与验证严格分离）"))
            parts.append(_table(["模型变体", "验证年份", "MAPE(%)", "秩相关", "参与街道", "判定"], rows))
            parts.append(f'<div class="note">判据：{_e(bt.get("criterion", "MAPE ≤ 25% 且 秩相关 ≥ 0.7"))}；'
                         '误差表全部展示，不做只报好看指标的裁剪。</div>')

    # 7 口径声明
    parts.append(_sub("⑦ 智能体推演口径声明（答辩须照此表述）"))
    for line in [
        "智能体为<b>合成患者智能体</b>（街道级人口 × 科室结构的抽样代表，非真实患者、不含任何个人信息）；",
        "推演是基于个体离散选择（多选项 Logit）的<b>模型模拟，仅供决策实验</b>，不作为医院真实业务数据；",
        a.get("note") or "基线量级由真实数据锚定（街道真实门诊量反解 p0），距离等机制只负责战略增量的分配；",
        "与平台原有『经验式战略推演』引擎的<b>绝对量级不可并列展示</b>（机制不同），两引擎以方向与空间格局互证；",
        "全部数字可由『参数 → 公式 → 结果』复核，大模型（DeepSeek）不参与任何计算。",
    ]:
        parts.append(f'<div class="note">· {line}</div>')
    return "".join(parts)


def _page(body):
    return ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>战略推演决策报告</title><style>' + _CSS + '</style></head><body>'
            + body +
            '<div class="footer">本报告由平台自动生成 · 可在浏览器中按 Ctrl+P 打印或保存为 PDF</div>'
            '</body></html>')


# ---------- 单方案报告 ----------
def _single(payload, advice):
    params = payload.get("params", {}) or {}
    s = payload.get("summary", {}) or {}
    name = payload.get("name") or params.get("name") or "未命名方案"
    action_cn = payload.get("action_cn", "战略推演")
    dept = params.get("dept") or "全院"
    mag = params.get("magnitude")
    mag_cn = f"{round(float(mag) * 100)}%" if mag is not None else "—"
    streets = params.get("streets") or []
    region = f"{len(streets)} 个指定街道" if streets else "全域（全部街道）"

    # 封面
    cover = ('<div class="cover"><h1>医院医疗服务战略数字孪生平台<br/>战略推演决策报告</h1>'
             f'<div class="sub">方案名称：{_e(name)}</div>'
             f'<div class="sub">战略动作：{_e(action_cn)}{"" if dept == "全院" else " · " + _e(dept)}</div>'
             f'<div class="meta">生成时间：{_now()}　|　报告类型：单方案推演决策报告</div></div>')

    # 一、方案参数
    sec1 = "<h2 class=\"sec\">一、方案参数</h2>" + _kv([
        ("方案名称", _e(name)),
        ("战略动作", _e(action_cn)),
        ("目标科室", _e(dept)),
        ("提升幅度", _e(mag_cn)),
        ("推演年限", _e(f"{params.get('years', '—')} 年")),
        ("影响区域", _e(region)),
    ])

    # 二、推演结果摘要
    hri_line = ""
    if s.get("hri_before") is not None and s.get("hri_after") is not None:
        hri_line = ("区域 HRI 辐射指数", f"<b>{_e(s['hri_before'])} → {_e(s['hri_after'])}</b>（出院/人口占比口径）")
    sp = s.get("specialty_patients") or {}
    sp_line = ("「%s」患者增量（住院口径）" % _e(sp.get("dept") or dept),
               "+" + _num(sp.get("add_ipd"))) if sp.get("add_ipd") is not None else None
    sec2 = "<h2 class=\"sec\">二、推演结果摘要</h2>" + _kv([
        ("累计新增门诊", f"<b>{_num(s.get('final_add_opd'))}</b> 人次"),
        ("累计新增住院", f"<b>{_num(s.get('final_add_ipd'))}</b> 人次"),
        ("期末床位使用率", _pct(s.get("final_bed_usage"))),
        ("期末号源利用率", _pct(s.get("final_numsource_util"))),
        ("80% 辐射半径", f"{_e(s.get('final_radius80', '—'))} km"),
        hri_line,
        sp_line,
    ])

    # 三、分年结果表
    yrows = [[
        _e(y.get("year")), ("+" + _num(y.get("add_opd")), ' class="num"'),
        ("+" + _num(y.get("add_ipd")), ' class="num"'),
        (_e(y.get("cum_opd")), ' class="num"'), (_e(y.get("cum_ipd")), ' class="num"'),
        _pct(y.get("bed_usage")), _pct(y.get("numsource_util")),
        f"{_e(y.get('radius80', '—'))} km",
        _e(y.get("hri")) if y.get("hri") is not None else "—",
    ] for y in payload.get("years", [])]
    sec3 = ("<h2 class=\"sec\">三、分年推演结果</h2>" +
            _table(["年份", "新增门诊", "新增住院", "累计门诊", "累计住院", "床位使用率", "号源利用率", "80%半径", "区域HRI"], yrows))

    # 四、重点影响街道 TOP8
    tops = (payload.get("top_streets") or payload.get("street_effects") or [])[:8]
    trows = [[_e(t.get("name")), _e(t.get("district", "—")), (_e(t.get("d_km", "—")), ' class="num"'),
              ("+" + _num(t.get("add_opd")), ' class="num"'), ("+" + _num(t.get("add_ipd")), ' class="num"')]
             for t in tops]
    sec4 = ("<h2 class=\"sec\">四、重点影响街道 TOP8</h2>" +
            _table(["街道", "行政区", "距离(km)", "新增门诊", "新增住院"], trows))

    # 五、风险提示
    risks = payload.get("risks") or []
    if risks:
        sec5 = "<h2 class=\"sec\">五、风险提示</h2>" + "".join(
            f'<div class="risk">{_e(r)}</div>' for r in risks)
    else:
        sec5 = '<h2 class="sec">五、风险提示</h2><div class="risk-none">推演期末床位使用率与号源利用率均未触及警戒线，资源压力可控。</div>'

    # 六、智能体推演（可选一节，设计 §9.3；不传 agent_sim 时与原来完全一致）
    if payload.get("agent_sim"):
        sec6 = _agent_sim_html(payload["agent_sim"], "六、")
        sec7 = _advice_html(advice, title="七、AI 战略建议")
        sec8 = _calibration_html("八、")
        return _page(cover + sec1 + sec2 + sec3 + sec4 + sec5 + sec6 + sec7 + sec8)
    # 六、AI 战略建议
    sec6 = _advice_html(advice, title="六、AI 战略建议")
    # 七、口径
    sec7 = _calibration_html("七、")
    return _page(cover + sec1 + sec2 + sec3 + sec4 + sec5 + sec6 + sec7)


# ---------- 对比报告 ----------
def _compare(payload, advice=None):
    items = payload.get("items", []) or []
    rec = payload.get("recommendation") or {}
    # scenarios (可选): [{name, params}] 前端对比请求的原始参数, 按方案名对应
    sc_params = {sc.get("name"): (sc.get("params") or {}) for sc in payload.get("scenarios", []) or []}

    cover = ('<div class="cover"><h1>医院医疗服务战略数字孪生平台<br/>战略推演决策报告</h1>'
             '<div class="sub">报告类型：多方案对比决策报告</div>'
             f'<div class="sub">对比方案数：{len(items)} 个</div>'
             f'<div class="meta">生成时间：{_now()}</div></div>')

    # 一、对比方案列表
    def _p_of(it):
        p = sc_params.get(it.get("name"), {})
        return p or (it.get("params") or {})
    sec1rows = []
    for it in items:
        p = _p_of(it)
        streets = p.get("streets") or []
        sec1rows.append([
            _e(it.get("name")), _e(it.get("action", p.get("action", "—"))), _e(p.get("dept", "全院")),
            (f"{round(float(p['magnitude']) * 100)}%" if p.get("magnitude") is not None else "—"),
            _e(f"{p.get('years', it.get('years', [{}])[-1].get('year', '—') if it.get('years') else '—')}"),
            _e(f"{len(streets)} 个街道" if streets else "全域"),
        ])
    sec1 = ("<h2 class=\"sec\">一、对比方案列表</h2>" +
            _table(["方案名称", "战略动作", "目标科室", "提升幅度", "推演年限", "影响区域"], sec1rows))

    # 二、结果对比表
    sec2rows = []
    for it in items:
        s = it.get("summary", {}) or {}
        risks = it.get("risks") or []
        sec2rows.append([
            _e(it.get("name")), _e(it.get("action", "—")),
            ("+" + _num(s.get("final_add_opd")), ' class="num"'),
            ("+" + _num(s.get("final_add_ipd")), ' class="num"'),
            _pct(s.get("final_bed_usage")), _pct(s.get("final_numsource_util")),
            f"{_e(s.get('final_radius80', '—'))} km", _hri(s),
            _e(risks[0] if risks else "资源压力可控"),
        ])
    sec2 = ("<h2 class=\"sec\">二、结果对比</h2>" +
            _table(["方案名称", "动作", "累计新增门诊", "累计新增住院", "期末床位率", "期末号源率", "80%半径", "HRI变化", "风险摘要"], sec2rows))

    # 三、AI 综合推荐
    if rec.get("winner"):
        src = rec.get("source", "rule")
        tag = "DeepSeek 大模型生成" if src == "deepseek" else "规则引擎生成"
        cls = "" if src == "deepseek" else " rule"
        sec3 = ("<h2 class=\"sec\">三、AI 综合推荐</h2>"
                f"<div class=\"advice\"><b>最优方案：{_e(rec['winner'])}</b>"
                f'<span class="src-tag{cls}">{tag}</span>'
                f"<div style=\"margin-top:4px\">{_e(rec.get('reasons', ''))}</div></div>")
    else:
        sec3 = '<h2 class="sec">三、AI 综合推荐</h2><div class="advice note">（本次未生成综合推荐）</div>'

    # 四、各方案 AI 建议
    parts = ["<h2 class=\"sec\">四、各方案 AI 战略建议</h2>"]
    for it in items:
        adv = it.get("ai_advice") or {}
        if adv.get("text"):
            src = adv.get("source", "rule")
            tag = "DeepSeek 大模型生成" if src == "deepseek" else "规则引擎生成"
            cls = "" if src == "deepseek" else " rule"
            parts.append(f'<div style="margin:6px 0"><b>{_e(it.get("name"))}</b>'
                         f'<div class="advice">{_e(adv["text"])}<span class="src-tag{cls}">{tag}</span></div></div>')
        else:
            parts.append(f'<div style="margin:6px 0"><b>{_e(it.get("name"))}</b><div class="note">（未生成建议）</div></div>')
    sec4 = "".join(parts)

    sec5 = _calibration_html("五、")
    return _page(cover + sec1 + sec2 + sec3 + sec4 + sec5)


def build_report(kind, payload, advice=None):
    """kind='single'|'compare'; 见模块 docstring"""
    payload = payload or {}
    if kind == "compare":
        return _compare(payload, advice)
    return _single(payload, advice or {})
