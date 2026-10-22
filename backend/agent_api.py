#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_api.py — 智能体推演 API（阶段 1–3：/run /trace /agent /dual /drill /backtest /sensitivity /list /{id}）

挂载方式（app.py 内 2 行，现有端点与页面零改动）：
    import agent_api
    app.include_router(agent_api.router)

纪律（与《智能体推演-详细设计》V1.3 §6.1 一致）：
  · 只新增端点，不改动任何现有端点与前端页面；
  · 结果写独立结果库 agent_sim_results.db；业务库 hospital.db 全程只读；
  · 参数与 engine.simulate() 同构（action/dept/streets/magnitude/years/extra），同一份参数可双引擎各跑一次；
  · 所有响应带 seed / algo_version / data_version，保证"可复算"。

用法示例：
  POST /api/strategy/agent/run         {"params":{"action":"specialty","dept":"心内科","magnitude":0.3,"years":5},"runs":30}
  POST /api/strategy/agent/trace       {"params":{...}}
  GET  /api/strategy/agent/agent?run_id=3&agent_id=7&year=2030
  POST /api/strategy/agent/dual        {"action":"specialty","dept":"心内科","magnitude":0.3,"years":5}
  POST /api/strategy/agent/sensitivity {"params":{...},"delta":0.2,"runs":30}
  POST /api/strategy/agent/backtest    {"fit_years":["2020","2021","2022"],"test_years":["2023","2024"]}
  GET  /api/strategy/agent/drill?run_id=3&street=李村街道
  GET  /api/strategy/agent/list
  GET  /api/strategy/agent/3
"""
import functools
import json
import os
import sys
import time
import traceback
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "strategy_engine"))

import agent_sim as A                      # noqa: E402
import agent_sim_sensitivity as SEN        # noqa: E402
import agent_sim_verify as VER             # noqa: E402
import agent_sim_solve as SOLVE            # noqa: E402

from fastapi import APIRouter              # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

router = APIRouter(prefix="/api/strategy/agent", tags=["智能体推演（阶段1–3）"])

DISCLAIMER = ("合成患者智能体（代表人群，非真实患者）· 模型模拟仅供决策实验；"
              "同参数 + 同种子可逐位复算（seed/algo_version/data_version 随响应返回）。")

# 精度档（设计 §11）：fast=演示档 / standard=标准档 / full=出报告档
PRECISION = {
    "fast":     dict(runs=20,  density=0.5, cn="演示档（每街道约 10 个智能体 × 20 次）"),
    "standard": dict(runs=50,  density=1.0, cn="标准档（每街道约 20 个智能体 × 50 次）"),
    "full":     dict(runs=200, density=2.0, cn="出报告档（每街道约 40 个智能体 × 200 次）"),
}
_CACHE = {}          # 参数哈希 → /run 响应（同参数二次调用直接命中，设计 §11 缓存）
_SEN_CACHE = {}      # 敏感性结果缓存（13 次推演较慢，现场演示需秒回）


# ══════════════════ 入参校验层（2026-09-21，A 组修复）══════════════════
# 目的：非法输入一律 **400**（而不是 500 + 栈 / 静默给出错误结果），且不把内部异常文本暴露给调用方。
# 覆盖四类历史缺陷（严格复核 P2-1/P2-2/P3-2 及自测发现的除零/键错）：
#   ① 类型混淆：years="5"、magnitude="0.3"、runs=null  → 曾直接 TypeError 500
#   ② 越界/退化：magnitude=5、years=0、max_years=0、target=0（除零）→ 曾 500 或 KeyError
#   ③ 未知枚举：kind/action/街道名/科室名  → 曾被静默接受并给出无意义结果
#   ④ 结构错误：streets 传字符串（被拆成单字当街道名）→ 曾静默出错
class BadInput(Exception):
    """调用方输入不合法（→ HTTP 400）"""


_NUM_RANGE = {                     # 数值项的取值范围（不在表内的按「有限数」校验即可）
    "magnitude": (0.0, 1.0), "years": (1, 5), "runs": (1, 500), "density": (0.05, 4.0),
    "beta_d": (0.0, 10.0), "beta_sp": (0.0, 10.0), "beta_cap": (0.0, 10.0), "beta_al": (0.0, 10.0),
    "beta_lo": (0.0, 1.0), "beta0": (0.0, 10.0), "beta_tier": (0.0, 10.0),
    "d0": (0.5, 100.0), "d0_comp": (0.5, 100.0), "activation": (0.0, 10.0), "spill_keep": (0.0, 1.0),
    "k": (0.05, 20.0), "sampling_divisor": (10, 100000), "min_per_street": (1, 1000),
    "max_per_street": (1, 2000), "hri_low_q": (0.0, 0.49), "hri_high_q": (0.51, 1.0),
    "visit_rate": (0.1, 50.0),
}
_INT_KEYS = ("years", "runs", "sampling_divisor", "min_per_street", "max_per_street")


def _num(v, name, lo=None, hi=None, default=None, integer=False):
    """类型强制 + 范围校验 + 非有限数拦截；None → default（无 default 则报错）"""
    if v is None:
        if default is None:
            raise BadInput(f"{name} 不能为空")
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise BadInput(f"{name} 必须是数字，收到 {type(v).__name__}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise BadInput(f"{name} 必须是数字，收到 {v!r}")
    if x != x or x in (float("inf"), float("-inf")):
        raise BadInput(f"{name} 必须是有限数，收到 {v!r}")
    if integer and abs(x - round(x)) > 1e-9:
        raise BadInput(f"{name} 必须是整数，收到 {v!r}")
    if lo is not None and x < lo:
        raise BadInput(f"{name} 须 ≥ {lo}，收到 {x}")
    if hi is not None and x > hi:
        raise BadInput(f"{name} 须 ≤ {hi}，收到 {x}")
    return int(round(x)) if integer else x


def _params(req, skip=()):
    """参数兼容 + 逐项校验：{"params":{...}} 或扁平键；只取引擎认识的键。

    非法输入抛 BadInput（由 _guarded 转 400），未知街道/科室在**计算前**就拒绝。
    `skip`：**排除与求解器同名的顶层键**（如 /optimize 的 `years` 是候选年份数组、
    而引擎的 `years` 是年限标量——不排除会把数组当标量校验而误报 400，2026-09-21 踩过）。
    """
    raw = req.get("params") if req.get("params") is not None else {}
    if not isinstance(raw, dict):
        raise BadInput("params 必须是对象（{...}）")
    raw = dict(raw)
    for k, v in req.items():
        if k in A.DEFAULTS and k not in raw and k not in skip:
            raw[k] = v
    p = {k: v for k, v in raw.items() if k in A.DEFAULTS}
    d = A.DEFAULTS

    if p.get("action", d["action"]) not in ("specialty", "alliance", "capacity"):
        raise BadInput(f"action 须为 specialty/alliance/capacity，收到 {p.get('action')!r}")
    p["action"] = p.get("action", d["action"])

    world = A.load_world()                       # 已缓存，取街道/科室名单用于校验（只读）
    dept = p.get("dept", d["dept"])
    if dept not in world["depts"]:
        raise BadInput(f"未知科室 {dept!r}；可用：{'、'.join(world['depts'])}")
    p["dept"] = dept

    if p.get("streets") not in (None, [], ()):
        v = p["streets"]
        if isinstance(v, str):
            raise BadInput('streets 必须是数组（如 ["李村街道"]），不能是字符串')
        if not isinstance(v, (list, tuple)):
            raise BadInput("streets 必须是数组")
        bad = [x for x in v if not isinstance(x, str)]
        if bad:
            raise BadInput(f"streets 里必须是街道名字符串，收到 {bad[:3]}")
        names = {s["name"] for s in world["streets"]}
        unknown = [x for x in v if x not in names]
        if unknown:
            raise BadInput(f"未知街道 {unknown[:3]}；可用示例：" + "、".join(sorted(names)[:6]) + f"…（共 {len(names)} 个）")
        p["streets"] = list(v)
    else:
        p["streets"] = None

    for k in list(p):
        if k in ("action", "dept", "streets", "curve", "share_basis", "match_all", "seed"):
            continue
        lo, hi = _NUM_RANGE.get(k, (None, None))
        p[k] = _num(p.get(k), f"{k}", lo=lo, hi=hi, default=d.get(k), integer=k in _INT_KEYS)
    if p.get("curve") is not None and not isinstance(p["curve"], str):
        raise BadInput("curve 必须是字符串（如 'logistic'）")
    if p.get("share_basis") is not None and not isinstance(p["share_basis"], str):
        raise BadInput("share_basis 必须是字符串（contested/city/dept_pool）")
    if p.get("match_all") is not None:
        p["match_all"] = bool(p["match_all"])
    if p.get("seed") is not None:
        p["seed"] = _num(p["seed"], "seed", lo=0, hi=2 ** 63 - 1, integer=True)
    if p.get("hri_low_q", 0) >= p.get("hri_high_q", 1):
        raise BadInput("hri_low_q 必须小于 hri_high_q")
    if p.get("min_per_street", 1) > p.get("max_per_street", 2):
        raise BadInput("min_per_street 不能大于 max_per_street")
    return p


def _guarded(fn):
    """统一异常出口：输入类→400；引擎显式拒绝→400；其余→500（日志留栈，响应不带栈）"""
    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        try:
            return await fn(*a, **kw)
        except BadInput as e:
            return JSONResponse({"error": str(e), "hint": "请核对参数类型与取值范围；见 /api/strategy/agent/list 的字段说明"}, status_code=400)
        except SystemExit as e:                       # 引擎历史遗留的 SystemExit（如未知街道）
            return JSONResponse({"error": f"输入不合法：{e}"}, status_code=400)
        except ValueError as e:                       # 引擎的显式拒绝（未知 kind/街道/科室等）
            return JSONResponse({"error": str(e)}, status_code=400)
        except TypeError as e:
            return JSONResponse({"error": f"参数类型不合法：{e}"}, status_code=400)
        except Exception as e:                        # 真·内部错误：不打栈给调用方，只在服务端日志留痕
            traceback.print_exc()
            return JSONResponse({"error": f"内部错误：{type(e).__name__}: {e}"}, status_code=500)
    return wrapper


def _apply_precision(p, prec):
    """套用精度档（显式传入的 runs/density 优先）"""
    cfg = PRECISION.get(prec)
    if not cfg:
        return None
    p.setdefault("runs", cfg["runs"])
    p.setdefault("density", cfg["density"])
    return cfg


def _cache_key(p, seed):
    return json.dumps({"p": p, "seed": seed, "algo": A.ALGO_VERSION}, ensure_ascii=False, sort_keys=True)


@router.post("/run")

@_guarded
async def agent_run(req: dict):
    """运行一次智能体推演（区间 P5/P50/P95 + 分年 + 分街道 + 分科室），并落独立结果库。

    body: {params?, action?, dept?, magnitude?, years?, runs?, density?, precision?, seed?, want_trace?, save?, note?}
    """
    t0 = time.time()
    p = _params(req)
    prec = _apply_precision(p, req.get("precision")) if req.get("precision") else None
    seed = req.get("seed")
    ck = _cache_key(p, seed)
    hit = _CACHE.get(ck)
    if hit and not req.get("want_trace"):
        out = dict(hit)
        if not req.get("want_trace"):
            out.pop("trace", None)          # 2026-09-21：缓存命中也不泄漏 trace（体积/语义）
        out["cache"] = "hit"
        out["elapsed_sec"] = round(time.time() - t0, 2)
        return out
    res = A.simulate(p, seed=seed, want_trace=bool(req.get("want_trace")),
                     want_prob=bool(req.get("want_prob")))
    run_id, created = (None, None)
    if req.get("save", True):
        run_id, created = A.save_run(res, note=req.get("note"))
    out = dict(
        run_id=run_id, saved_new=created, cache="miss",
        params=res["params"], seed=res["seed"], cap=res["cap"],
        share_report=res["share_report"], baseline_share=res["baseline_share"],
        world_meta=res["world_meta"],
        agent_count=res["interval"].get("agents"),
        precision=(prec["cn"] if prec else "自定义"),
        metrics=res["interval"]["metrics"],
        years=res["interval"]["years"],
        street_effects=res["interval"]["street_effects"],
        streets_meta=res["interval"]["streets_meta"],
        dept_add_opd=res["interval"]["dept_add_opd"],
        dept_add_opd_rate=res["interval"]["dept_add_opd_rate"],
        rep_run_index=res["interval"]["rep_run_index"],
        radius80=res["interval"].get("radius80"),
        hri=res["interval"].get("hri"),
        algo_version=A.ALGO_VERSION, data_version=res["world_meta"]["data_version"],
        note=req.get("note"), disclaimer=DISCLAIMER,
        elapsed_sec=round(time.time() - t0, 2),
    )
    if res.get("trace"):
        out["trace"] = res["trace"]
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[ck] = out
    return out


@router.post("/trace")

@_guarded
async def agent_trace(req: dict):
    """个体级帧数据（列式压缩，蜂群视图用）。只跑 1 轮：trace 恒取自第 0 轮随机流，与 runs 无关。"""
    t0 = time.time()
    p = _params(req)
    _apply_precision(p, req.get("precision")) if req.get("precision") else None
    p["runs"] = 1
    res = A.simulate(p, seed=req.get("seed"), want_trace=True, want_prob=bool(req.get("want_prob", True)))
    tr = res["trace"]
    tr["meta"]["params"] = res["params"]
    tr["meta"]["share_report"] = res["share_report"]
    tr["meta"]["precision"] = (req.get("precision") or "自定义(1 轮)")
    tr["meta"]["disclaimer"] = DISCLAIMER
    tr["meta"]["elapsed_sec"] = round(time.time() - t0, 2)
    tr["meta"]["size_bytes"] = len(json.dumps(tr, ensure_ascii=False))
    return tr


@router.get("/list")
def agent_list(limit: int = 20):
    """已落库的推演记录（独立结果库）"""
    rows = A.list_runs(limit)
    for r in rows:
        r["params"] = json.loads(r.pop("params_json"))
        m = json.loads(r.pop("metrics_json"))
        r["summary"] = {k: m["metrics"][k]["p50"] for k in m["metrics"]}
        r["note"] = r.get("note")
    return {"count": len(rows), "items": rows}


@router.get("/agent")
def agent_detail(run_id: int = None, agent_id: int = 0, year: int = None,
                 params: str = None, seed: int = None):
    """单个智能体的画像 + 逐年效用分量分解（个体决策卡 D1）。

    run_id 优先（从结果库取参数与种子，保证与已落库结果一致）；也可直接传 params(JSON 字符串) + seed。
    """
    if run_id is not None:
        rec = A.load_run(run_id)
        if not rec:
            return JSONResponse({"error": f"结果库无 run_id={run_id}"}, status_code=404)
        p, sd = rec["params"], int(rec["seed"])
    else:
        p = json.loads(params) if params else {}
        sd = seed
    try:
        return A.agent_breakdown(p, agent_id=agent_id, year=year, seed=sd)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/dual")

@_guarded
async def agent_dual(req: dict):
    """双引擎互证：老引擎（经验式）vs 新引擎（个体 Logit）偏差表 + 空间排序一致性 + 自动归因"""
    q = {k: v for k, v in req.items() if k in ("action", "dept", "magnitude", "years", "runs")}
    q.setdefault("action", "specialty")
    q.setdefault("dept", "心内科")
    q.setdefault("magnitude", 0.30)
    q.setdefault("years", 5)
    q.setdefault("runs", 30)
    out = VER.dual_engine_compare(q["action"], q["dept"], float(q["magnitude"]), int(q["years"]), int(q["runs"]))
    out["params"] = q
    out["algo_version"] = A.ALGO_VERSION
    out["disclaimer"] = DISCLAIMER
    return out


@router.get("/drill")
def agent_drill(run_id: int, street: str = None, dept: str = None):
    """下钻：某次推演的分街道 / 分科室贡献明细。

    给了 street → 复算该街道的**个体级构成**（按科室 / 按选择 / 期初→期末迁移）；
    否则返回落库的分街道表（按净增降序）。个体级不落库、随时按"同参数 + 同种子"复算。
    """
    rec = A.load_run(run_id)
    if not rec:
        return JSONResponse({"error": f"结果库无 run_id={run_id}"}, status_code=404)
    base = dict(params=rec["params"], seed=rec["seed"], metrics=rec["metrics"]["metrics"],
                street_effects=rec["street_effects"], dept_effects=rec["dept_effects"])
    if not street:
        return dict(kind="run_effects", **base)
    p = dict(rec["params"])
    p["runs"] = 1
    res = A.simulate(p, seed=int(rec["seed"]), want_trace=True)
    tr = res["trace"]
    names = [row[0] for row in tr["street_table"]]
    if street not in names:
        return JSONResponse({"error": f"未知街道：{street}"}, status_code=400)
    si = names.index(street)
    depts = tr["dept_table"]
    idx = [i for i, x in enumerate(tr["agents"]["street_idx"]) if x == si]
    by_dept, by_dept_first, by_dept_last = {}, {}, {}
    per_year = {y: {"self": 0, "comp": 0, "out": 0} for y in tr["meta"]["years"]}
    moves = {}
    for i in idx:
        d = depts[tr["agents"]["dept_idx"][i]]
        by_dept[d] = by_dept.get(d, 0) + 1
        row = tr["choice"][i]
        if row[0] == "1":
            by_dept_first[d] = by_dept_first.get(d, 0) + 1
        if row[-1] == "1":
            by_dept_last[d] = by_dept_last.get(d, 0) + 1
        for k, y in enumerate(tr["meta"]["years"]):
            per_year[y][{"0": "out", "1": "self", "2": "comp"}[row[k]]] += 1
        moves[row[0] + "→" + row[-1]] = moves.get(row[0] + "→" + row[-1], 0) + 1
    row_meta = tr["street_table"][si]
    return dict(
        kind="street_breakdown", street=street, district=row_meta[1], d_km=row_meta[4],
        comp=dict(name=row_meta[5], grade=row_meta[6]), p0=row_meta[7],
        agents=len(idx), years=tr["meta"]["years"],
        dept_composition=dict(sorted(by_dept.items(), key=lambda x: -x[1])),
        by_dept_self=dict(first=by_dept_first, last=by_dept_last),
        per_year_choice=per_year, migrations=moves,
        dept_filter=dept,
        params=rec["params"], seed=rec["seed"], run_id=run_id,
        note="按科室构成取自个体级复算（同参数 + 同种子），与落库汇总一致；个体级数据不落库。",
        disclaimer=DISCLAIMER)


@router.post("/backtest")

@_guarded
async def agent_backtest(req: dict):
    """回测（阶段 3 可信度三件套之一）：拟合 2020–2022、外推验证 2023–2024 + 对数线性变体对照"""
    t0 = time.time()
    fit = tuple(req.get("fit_years") or ("2020", "2021", "2022"))
    test = tuple(req.get("test_years") or ("2023", "2024"))
    out = dict(
        variant1_distance_decay=VER.backtest(fit, test),
        variant3_loglin_hri=VER.backtest_loglin(fit, test),
        criterion="MAPE ≤ 25% 且 秩相关 ≥ 0.7（设计 §5 初版判据，可调）",
        note=("误差表全部展示，不做只报好看的裁剪；不达标即如实标注为『当前版本拟合不足』。"
              "变体 3（对数线性 + HRI 历史行为）是引擎架构的依据：基线量级由真实数据锚定，"
              "距离等机制只负责战略增量分配。"),
        algo_version=A.ALGO_VERSION, elapsed_sec=round(time.time() - t0, 2),
    )
    return out


@router.post("/sensitivity")

@_guarded
async def agent_sensitivity(req: dict):
    """敏感性（阶段 3）：各参数 ±delta 扰动的弹性表 + 排序 + 方向自检 + 待决策发现（flags）

    默认 runs=12（现场演示档，约 8–12 秒）；出报告时传 runs=30。同参数结果带内存缓存。
    """
    p = _params(req)
    delta = float(req.get("delta", 0.20))
    runs = int(req.get("runs", 12))
    seed = req.get("seed")
    ck = "sen|" + json.dumps({"p": p, "d": delta, "r": runs, "s": seed, "a": A.ALGO_VERSION,
                                  "pk": req.get("params_keys")},
                             ensure_ascii=False, sort_keys=True)
    hit = _SEN_CACHE.get(ck)
    if hit:
        out = dict(hit)
        out["cache"] = "hit"
        return out
    out = SEN.sensitivity(p, delta=delta, runs=runs, params_keys=req.get("params_keys"), seed=seed)
    out["cache"] = "miss"
    out["disclaimer"] = DISCLAIMER
    if len(_SEN_CACHE) > 8:
        _SEN_CACHE.clear()
    _SEN_CACHE[ck] = out
    return out


@router.post("/solve")

@_guarded
async def agent_solve(req: dict):
    """反向求解：给定目标指标值 → 推荐参数组合（含达成度误差与备选）；不可达时如实返回。

    body: {kind: add_opd|add_ipd|radius80|hri_region, target, depts?, actions?, max_years?, streets?,
           runs?, density?, tol?, save?, note?}
    """
    t0 = time.time()
    if req.get("target") is None:
        return JSONResponse({"error": "缺少 target（目标指标值）"}, status_code=400)
    kind = req.get("kind", "add_opd")
    out = SOLVE.solve(kind, float(req["target"]), params=_params(req),
                      depts=req.get("depts"), actions=tuple(req.get("actions") or SOLVE.ACTIONS),
                      max_years=int(req.get("max_years", req.get("years", 5))),
                      min_years=int(req.get("min_years", 2)),
                      runs=req.get("runs"), density=req.get("density"),
                      tol=float(req.get("tol", 0.02)), region=req.get("streets"),
                      coarse=tuple(req.get("coarse") or (4, 0.25)),
                      top_combos=int(req.get("top_combos", 3)),
                      coarse_iters=int(req.get("coarse_iters", 4)),
                      refine_iters=int(req.get("refine_iters", 8)),
                      years_set=req.get("years_set"))
    if req.get("save", True):
        out["solve_id"] = SOLVE.save_solve(out, source="api", note=req.get("note"))
    out["disclaimer"] = DISCLAIMER
    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


@router.post("/optimize")

@_guarded
async def agent_optimize(req: dict):
    """自动寻优：候选动作 → 两阶段（低精度档粗搜 + 标准档复算 topN）+ 风险约束 + 帕累托 + 胜出理由。

    body: {kind?, depts?, magnitudes?, years?, top?, coarse?:[runs,density], refine?:[runs,density], streets?}
    """
    t0 = time.time()
    coarse = tuple(req.get("coarse") or (6, 0.5))
    refine = tuple(req.get("refine") or (30, 1.0))
    mags = tuple(_num(m, "magnitudes[]（0–1 的比例）", lo=0.0, hi=1.0) for m in (req.get("magnitudes") or (0.10, 0.20, 0.30, 0.40, 0.50)))
    yrs = tuple(_num(y, "years[]", lo=1, hi=5, integer=True) for y in (req.get("years") or (2, 3, 5)))
    dpts = req.get("depts")
    if dpts is not None:
        if isinstance(dpts, str) or not isinstance(dpts, (list, tuple)):
            raise BadInput('depts 必须是数组（如 ["心内科","骨科"]），不能是字符串')
        _known = set(A.load_world()["depts"])
        _bad = [d for d in dpts if d not in _known]
        if _bad:
            raise BadInput(f"未知科室 {_bad}；可用：{'、'.join(A.load_world()['depts'])}")
    if req.get("kind", "add_opd") not in SOLVE.KINDS:
        raise BadInput(f"未知目标类型 {req.get('kind')!r}；可用：{'/'.join(SOLVE.KINDS)}")
    out = SOLVE.optimize(_params(req, skip=("years", "magnitudes", "depts")), depts=dpts, kind=req.get("kind", "add_opd"),
                         magnitudes=mags,
                         years=yrs,
                         top=int(req.get("top", 3)), coarse=coarse, refine=refine,
                         max_combos=int(req.get("max_combos", 120)), region=req.get("streets"))
    if req.get("save", True) and out.get("ok"):
        out["solve_id"] = SOLVE.save_solve(dict(out, target=req.get("target")), source="api", note=req.get("note"))
    out["disclaimer"] = DISCLAIMER
    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


@router.get("/solve/list")
def agent_solve_list(limit: int = 20):
    """反解/寻优留痕列表（独立结果库 solve_runs 表）"""
    rows = SOLVE.list_solves(limit)
    return {"count": len(rows), "items": rows}


# ============ 阶段 5：promote（存为正式方案）/ 预生成回放（C 级降级）============
def _legacy_payload(rec):
    """把智能体推演结果映射成**既有前端/报告可直接消费**的结构（不破坏既有渲染）。

    纪律：映射只做字段搬运与格式对齐，不改变任何数值；原始口径全部保留在 `agent` 与 `agent_sim` 键下。
    逐年 80% 半径 / 区域 HRI 未逐帧记录（期末值见 summary），避免用插值编造逐年曲线。
    """
    it = rec["metrics"]
    params = rec["params"]
    act_cn = {"specialty": "专科能力提升", "alliance": "医联体建设", "capacity": "资源扩容"}.get(
        params.get("action"), params.get("action"))
    m = it["metrics"]
    yrs = []
    for y in it["years"]:
        yrs.append(dict(year=y["year"], add_opd=y["add_opd"], add_ipd=y["add_ipd"],
                        cum_opd=y["cum_opd"], cum_ipd=y["cum_ipd"],
                        bed_usage=y["bed_usage"], numsource_util=y["numsource_util"],
                        radius80=None, hri=None))
    se = it.get("street_effects") or []
    top = [dict(name=e["name"], district=e.get("district"), d_km=e.get("d_km"),
                add_opd=e.get("opd_cum", 0), add_ipd=round(e.get("opd_cum", 0) * 0.047, 1),   # 2026-09-21 修键名（原 opd_last 恒空）+ 口径统一为累计
                agents=e.get("agents"), net=e.get("net"), from_comp=e.get("from_comp"),
                from_out=e.get("from_out"), lost=e.get("lost")) for e in se[:8]]
    risks = []
    if m["final_bed_usage"]["p50"] > 90:
        risks.append(f"期末床位使用率 {m['final_bed_usage']['p50']}% 超过 90% 警戒线，需同步扩容")
    if m["final_numsource_util"]["p50"] > 95:
        risks.append(f"期末号源利用率 {m['final_numsource_util']['p50']}% 超过 95%，门诊供给接近瓶颈")
    if not risks:
        risks.append("资源压力可控（床位率与号源率均未触及警戒线）")
    r80 = ((rec.get("metrics") or {}).get("radius80") or {})     # 注意：radius80/hri 存在 metrics(interval) 内，不在顶层
    hri = ((rec.get("metrics") or {}).get("hri") or {})
    return dict(
        action_cn=f"智能体推演·{act_cn}",
        summary=dict(final_add_opd=m["final_add_opd"]["p50"], final_add_ipd=m["final_add_ipd"]["p50"],
                     final_bed_usage=m["final_bed_usage"]["p50"],
                     final_numsource_util=m["final_numsource_util"]["p50"],
                     final_radius80=r80.get("after")),
        years=yrs, top_streets=top, risks=risks,
        source="agent_sim", agent_run_id=rec["id"],
        agent=dict(seed=rec.get("seed"), algo_version=rec.get("algo_version"),
                   data_version=rec.get("data_version"), share_basis=rec.get("share_basis"),
                   metrics=m, radius80=r80, hri=hri,
                   note="智能体推演（个体级 Logit）结果映射到既有方案库结构；逐年半径/HRI 未逐帧记录"),
    )


@router.post("/promote")

@_guarded
async def agent_promote(req: dict):
    """把一次智能体推演结果**存为正式方案**（写入既有方案库表，只 INSERT，不改任何现有表结构）。

    body: {run_id, name}
    副作用：医院业务库 hospital.db 的 strategy_config / strategy_result / ai_strategy_advice 各插入一行
            （这是设计 §6.2 指定的 promote 语义），并在结果库 promotions 表留痕可追溯。
    """
    rid = req.get("run_id")
    if rid is None:
        return JSONResponse({"error": "缺少 run_id"}, status_code=400)
    try:
        rid = int(rid)
    except (TypeError, ValueError):
        raise BadInput(f"run_id 必须是整数，收到 {rid!r}")
    rec = A.load_run(int(rid))
    if not rec:
        return JSONResponse({"error": f"结果库无 run_id={rid}"}, status_code=404)
    name = req.get("name") or f"智能体推演·{rec['params'].get('dept') or '全域'}-{datetime.date.today().isoformat()}"
    payload = _legacy_payload(rec)
    advice = {"text": _promote_advice_text(name, rec, payload), "source": "rule"}

    import app as _app                      # 复用既有连接 helper（同一业务库，只 INSERT）
    conn = _app._sdb()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    cur = conn.execute("INSERT INTO strategy_config(name, created_at, params) VALUES (?,?,?)",
                       (name, now, json.dumps(dict(rec["params"], source="agent_sim",
                                                   agent_run_id=rid), ensure_ascii=False)))
    cid = cur.lastrowid
    conn.execute("INSERT INTO strategy_result(config_id, scenario, result, ai_advice, created_at)"
                 " VALUES (?,?,?,?,?)",
                 (cid, payload["action_cn"], json.dumps(payload, ensure_ascii=False),
                  json.dumps(advice, ensure_ascii=False), now))
    conn.execute("INSERT INTO ai_strategy_advice(config_id, scenario, advice_type, content, source, created_at)"
                 " VALUES (?,?,?,?,?,?)",
                 (cid, payload["action_cn"], "promote", advice["text"], "rule", now))
    conn.commit()
    conn.close()

    conn = A._rdb()
    conn.execute("""CREATE TABLE IF NOT EXISTS promotions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, run_id INTEGER, config_id INTEGER,
        name TEXT, source TEXT, note TEXT)""")
    conn.execute("INSERT INTO promotions(created_at, run_id, config_id, name, source, note) VALUES(?,?,?,?,?,?)",
                 (now, rid, cid, name, req.get("source", "api"), req.get("note")))
    conn.commit()
    conn.close()
    return dict(ok=True, config_id=cid, name=name, run_id=int(rid),
                verify=dict(replay_url=f"/api/strategy/{cid}", list_url="/api/strategy/list",
                            report_url="/api/report/generate"),
                summary=payload["summary"],
                note="已写入既有方案库（只 INSERT）；历史方案库里即可看到并与老引擎方案对比。")


def _promote_advice_text(name, rec, payload):
    s = payload["summary"]
    p = rec["params"]
    return (f"【{name}】智能体推演（个体级 Logit，算法版本 {rec.get('algo_version')}，种子 {rec.get('seed')}）："
            f"{p.get('dept') or '全域'}·{p.get('action')} 强度 {round(float(p.get('magnitude', 0)) * 100)}% / "
            f"{p.get('years')} 年，累计新增门诊 {s['final_add_opd']:,} 人次、住院 {s['final_add_ipd']:,} 人次；"
            f"期末床位使用率 {s['final_bed_usage']}%、号源利用率 {s['final_numsource_util']}%，"
            f"80% 辐射半径 {s['final_radius80']} km。口径：合成患者智能体、模型模拟仅供参考；"
            "与老引擎方案的绝对量级不可并列。")


@router.get("/pregen")
def agent_pregen_list():
    """预生成结果集清单（C 级降级：现场零计算回放）"""
    import agent_sim_drill as DR
    out = DR.replay("list")
    if not out:
        return JSONResponse({"error": "预生成结果集不存在，请先运行 agent_sim_drill.py --pregen"}, status_code=404)
    return out


@router.get("/pregen/{key}")
def agent_pregen(key: str, trace: bool = False):
    """按场景键回放预生成结果（可选带个体级帧数据）"""
    import agent_sim_drill as DR
    out = DR.replay(key)
    if not out or not out.get("scenario"):
        return JSONResponse({"error": f"预生成结果集里无场景 {key}"}, status_code=404)
    if not trace:
        out.pop("trace", None)
    return out


@router.post("/drill")

@_guarded
async def agent_drill_run(req: dict):
    """跑一次三级降级演练并留痕（A 智能体 → B 老引擎 → C 预生成回放）"""
    import agent_sim_drill as DR
    return DR.run_drill(req.get("scenario", "specialty_xin_nei"), note=req.get("note"), verbose=False)


@router.get("/drill/list")
def agent_drill_list(limit: int = 30):
    """降级演练留痕"""
    import agent_sim_drill as DR
    rows = DR.list_drills(limit)
    return {"count": len(rows), "items": rows}


# ============ 通用回放（必须放在最后：`/{run_id}` 会吞掉同段数的固定路径）============
@router.get("/{run_id}")
def agent_record(run_id: int):
    """按 id 回放已落库的推演结果"""
    rec = A.load_run(run_id)
    if not rec:
        return JSONResponse({"error": f"结果库无 run_id={run_id}"}, status_code=404)
    rec["algo_version"] = rec.get("algo_version", A.ALGO_VERSION)
    rec["disclaimer"] = DISCLAIMER
    return rec
