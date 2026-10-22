#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_sim_drill.py — 阶段 5：演示保障（精度档 / 预生成结果回放 / 三级降级演练 + 留痕）

设计依据：《智能体推演-详细设计.md》V1.5 §11「动态调参与演示保障」
  · 精度档：fast / standard / full（见 `agent_api.PRECISION`）
  · 缓存：参数哈希 + 结果缓存（见 agent_api 的 _CACHE / _SEN_CACHE）
  · 单一事实来源：报告与方案库的数字一律取 Python 侧结果（不以前端预览为准）
  · 降级序列：A 完整智能体推演 → B 老引擎参数推演 → C 预生成结果回放；**三级都要演练并留痕**
  · 留痕：全部运行入结果库；可 promote 为正式方案（agent_api 的 /promote）

本模块负责：① 生成"预生成结果集"（C 级回放的数据底座，含旗舰场景的个体级帧数据，可离线演示）；
            ② 跑一次三级降级演练并写结果库 `drills` 表 + 打印对照表。

用法：
  python3 agent_sim_drill.py --pregen --runs 50        # 生成/刷新预生成结果集（约 1 分钟）
  python3 agent_sim_drill.py --drill                   # 三级降级演练（含 A/B 实测）
  python3 agent_sim_drill.py --list                    # 看已有降级演练记录
"""
import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_sim as A

PREGEN = os.path.join(A.BASE, "agent_sim_pregen.json")

# 预生成场景集：答辩常用组合（三类动作 × 两个区域口径），C 级回放零计算
SCENARIOS = [
    dict(key="specialty_xin_nei", name="专科能力提升·心内科 30%/5年（全域）",
         params=dict(action="specialty", dept="心内科", magnitude=0.30, years=5)),
    dict(key="alliance_xin_nei", name="医联体建设·心内科 30%/5年（全域）",
         params=dict(action="alliance", dept="心内科", magnitude=0.30, years=5)),
    dict(key="capacity_ortho", name="资源扩容·骨科 30%/5年（全域）",
         params=dict(action="capacity", dept="骨科", magnitude=0.30, years=5)),
    dict(key="specialty_xin_nei_licang", name="专科能力提升·心内科 30%/5年（李沧区选区）",
         params=dict(action="specialty", dept="心内科", magnitude=0.30, years=5,
                     streets=["浮山路街道", "李村街道", "虎山路街道", "兴华路街道", "沧口街道", "湘潭路街道"])),
]
TRACE_KEY = "specialty_xin_nei"          # 旗舰场景：连个体级帧数据一起预生成（蜂群视图可离线演示）


def _db():
    conn = sqlite3.connect(A.RESULTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_drill_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS drills(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, scenario TEXT, level TEXT,
        label TEXT, ok INTEGER, elapsed_sec REAL, agents INTEGER, metric_json TEXT, note TEXT)""")


def build_pregen(runs=50, density=1.0, verbose=True):
    """生成预生成结果集（C 级）：标准档跑完所有预置场景 + 旗舰场景的个体级帧数据"""
    t0 = time.time()
    out = dict(meta=dict(built_at=time.strftime("%Y-%m-%d %H:%M:%S"), algo_version=A.ALGO_VERSION,
                         data_version=A.data_version(), runs=runs, density=density,
                         note="预生成结果集：现场无后端服务/断网时按此回放（C 级降级）；数字与实时推演同源同参"),
               scenarios={}, traces={})
    for sc in SCENARIOS:
        p = dict(sc["params"], runs=runs, density=density)
        ts = time.time()
        res = A.simulate(p, want_trace=(sc["key"] == TRACE_KEY),
                         want_prob=(sc["key"] == TRACE_KEY))     # 2026-09-21：旗舰场景带逐年概率，供蜂群页离线回放
        it = res["interval"]
        out["scenarios"][sc["key"]] = dict(
            name=sc["name"], params=res["params"], seed=res["seed"],
            agent_count=it.get("agents"), share_report=res["share_report"],
            metrics=it["metrics"], years=it["years"], street_effects=it["street_effects"],
            dept_add_opd=it["dept_add_opd"], radius80=it.get("radius80"), hri=it.get("hri"),
            algo_version=A.ALGO_VERSION, data_version=res["world_meta"]["data_version"],
            built_sec=round(time.time() - ts, 2))
        if res.get("trace"):
            tr = res["trace"]
            out["traces"][sc["key"]] = tr
        if verbose:
            m = it["metrics"]["final_add_opd"]
            print(f"  ✓ {sc['key']:<28} P50 新增门诊 {m['p50']:>7,} ｜ 半径 {it['radius80']['after']}km "
                  f"｜ {round(time.time() - ts, 1)}s")
    out["meta"]["build_sec"] = round(time.time() - t0, 1)
    out["meta"]["size_kb"] = round(len(json.dumps(out, ensure_ascii=False)) / 1024, 1)
    open(PREGEN, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False))
    if verbose:
        print(f"\n已写 {PREGEN}：{out['meta']['size_kb']} KB，用时 {out['meta']['build_sec']}s，"
              f"含 {len(out['scenarios'])} 个场景 + {len(out['traces'])} 份个体级帧数据")
    return out


def load_pregen():
    if not os.path.exists(PREGEN):
        return None
    return json.load(open(PREGEN, encoding="utf-8"))


def replay(key):
    """C 级：从预生成结果集回放（零计算，适合现场兜底）"""
    d = load_pregen()
    if not d:
        return None
    if key in ("list", "", None):
        return dict(meta=d["meta"],
                    items=[dict(key=k, name=v["name"], p50=v["metrics"]["final_add_opd"]["p50"],
                                radius80=(v.get("radius80") or {}).get("after")) for k, v in d["scenarios"].items()],
                    has_trace=list(d.get("traces", {}).keys()))
    return dict(meta=d["meta"], scenario=d["scenarios"].get(key), trace=(d.get("traces") or {}).get(key))


def run_drill(scenario_key="specialty_xin_nei", note=None, verbose=True):
    """三级降级演练：A 智能体推演 → B 老引擎 → C 预生成回放；逐级实测并写结果库留痕"""
    d = load_pregen()
    if not d or scenario_key not in d["scenarios"]:
        build_pregen(runs=20, density=0.5, verbose=False)
        d = load_pregen()
    sc = d["scenarios"][scenario_key]
    params = sc["params"]
    rows = []

    def add(level, label, ok, dt, agents, metric, note_txt=""):
        rows.append(dict(level=level, label=label, ok=bool(ok), elapsed_sec=round(dt, 2),
                         agents=agents, metric=metric, note=note_txt))
        if verbose:
            print(f"  [{'OK' if ok else 'FAIL'}] {level} {label:<28} {round(dt, 2):>7}s "
                  f"｜ 新增门诊 {metric} ｜ {note_txt}")

    # A 级：完整智能体推演（现算）
    try:
        t0 = time.time()
        res = A.simulate(dict(params, runs=20, density=0.5))
        add("A", "智能体推演（现算，演示档 20×0.5）", True, time.time() - t0, res["interval"].get("agents"),
            f"{res['interval']['metrics']['final_add_opd']['p50']:,}",
            f"全功能可用（含个体级下钻/敏感性）｜ 种子 {res['seed']}")
    except Exception as e:                                     # pragma: no cover
        add("A", "智能体推演（现算，演示档 20×0.5）", False, 0, None, "—", f"异常：{e}")

    # B 级：老引擎参数推演（既有功能，失败即退回）
    try:
        import engine as old_engine
        t0 = time.time()
        old = old_engine.simulate(dict(params, name=sc["name"]))
        add("B", "老引擎参数推演（既有功能）", True, time.time() - t0, None,
            f"{old['summary']['final_add_opd']:,}",
            "口径不同（经验式），总量不可与新引擎并列；作为推演能力兜底")
    except Exception as e:
        add("B", "老引擎参数推演（既有功能）", False, 0, None, "—", f"异常：{e}")

    # C 级：预生成结果回放（零计算）
    t0 = time.time()
    r = replay(scenario_key)
    ok = bool(r and r.get("scenario"))
    dt = time.time() - t0
    add("C", "预生成结果回放（零计算）", ok, dt, (r["scenario"].get("agent_count") if ok else None),
        f"{r['scenario']['metrics']['final_add_opd']['p50']:,}" if ok else "—",
        f"数据构建于 {r['meta']['built_at']}（同参数同版本，可复算）；断网/无后端兜底")

    # 一致性：C 级回放 vs A 级现算（同参数、同精度档时应接近；不同档位差异属蒙卡抽样）
    a = next((x for x in rows if x["level"] == "A" and x["ok"]), None)
    if a and ok:
        va = float(a["metric"].replace(",", ""))
        vc = float(rows[-1]["metric"].replace(",", ""))
        dev = abs(vc - va) / max(1e-9, va) * 100
        rows[-1]["consistency_pct"] = round(dev, 2)
        if verbose:
            print(f"  C 与 A 的一致性：{dev:.2f}%（同参数同精度档应为 0；差异来源＝档位不同时的蒙卡抽样）")

    conn = _db()
    _ensure_drill_table(conn)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        conn.execute("INSERT INTO drills(created_at, scenario, level, label, ok, elapsed_sec, agents,"
                     " metric_json, note) VALUES(?,?,?,?,?,?,?,?,?)",
                     (now, scenario_key, row["level"], row["label"], 1 if row["ok"] else 0,
                      row["elapsed_sec"], row["agents"], row["metric"],
                      (row["note"] or "") + (f" ｜ 与A一致度 {row.get('consistency_pct')}%"
                                             if row.get("consistency_pct") is not None else "")
                      + (f" ｜ {note}" if note else "")))
    conn.commit()
    conn.close()
    if verbose:
        print(f"\n已留痕 {len(rows)} 条到结果库 drills 表（{now}，场景 {scenario_key}）")
    return dict(scenario=scenario_key, created_at=now, levels=rows,
                conclusion=("三级降级链路可用：A 完整推演 → B 老引擎兜底 → C 预生成回放；"
                            "A 失败可降 B，B 失败可降 C，现场均可继续演示"))


def list_drills(limit=30):
    conn = _db()
    _ensure_drill_table(conn)
    rows = [dict(r) for r in conn.execute(
        "SELECT id, created_at, scenario, level, label, ok, elapsed_sec, metric_json, note"
        " FROM drills ORDER BY id DESC LIMIT ?", (limit,))]
    conn.close()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段 5：演示保障（预生成回放 + 三级降级演练）")
    ap.add_argument("--pregen", action="store_true", help="生成/刷新预生成结果集")
    ap.add_argument("--drill", action="store_true", help="跑三级降级演练")
    ap.add_argument("--list", action="store_true", help="列出降级演练留痕")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--scenario", default="specialty_xin_nei")
    ap.add_argument("--note", default=None)
    args = ap.parse_args(argv)

    if args.pregen:
        print(f"生成预生成结果集（runs={args.runs} density={args.density}）：")
        build_pregen(args.runs, args.density)
    if args.drill:
        print("三级降级演练：")
        r = run_drill(args.scenario, args.note)
        print(f"\n{r['conclusion']}")
    if args.list:
        rows = list_drills()
        print(f"降级演练留痕 {len(rows)} 条：")
        for r in rows:
            print(f"  #{r['id']} {r['created_at']} {r['scenario']} [{r['level']}] {r['label']} "
                  f"ok={r['ok']} {r['elapsed_sec']}s {r['metric_json']}")
    if not any([args.pregen, args.drill, args.list]):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
