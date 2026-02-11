# -*- coding: utf-8 -*-
"""
Gomtang Index (Korea Fear & Greed) - Genspark-like daily report for GitHub Pages

Inputs (repo local):
- data/korea_fear_greed_v2_ks200_last1y_rescaled_with_forward.csv
- data/fg_forward_summary_last1y_rescaled_pretty.csv
- data/summary_last1y_rescaled.json (optional)

Outputs (docs/):
- docs/곰탕지수_1Y리스케일_YYYY-MM-DD_embedded.html
- docs/곰탕지수_1Y리스케일_latest_embedded.html
- docs/index.html  (Genspark-like shell + latest content)
- docs/report_index.json  (최근 10개 날짜 목록)
- docs/assets/*.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

FG_CSV = DATA_DIR / "korea_fear_greed_v2_ks200_last1y_rescaled_with_forward.csv"
FWD_SUMMARY_CSV = DATA_DIR / "fg_forward_summary_last1y_rescaled_pretty.csv"
SUMMARY_JSON = DATA_DIR / "summary_last1y_rescaled.json"

REPORT_INDEX_JSON = DOCS_DIR / "report_index.json"


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _read_fg() -> pd.DataFrame:
    if not FG_CSV.exists():
        raise FileNotFoundError(f"Missing input: {FG_CSV}")
    df = pd.read_csv(FG_CSV)

    # Raw header confirms '날짜' column exists [Source: raw CSV]
    if "날짜" in df.columns:
        df["date"] = pd.to_datetime(df["날짜"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    else:
        raise ValueError(f"Cannot find date column. Columns={list(df.columns)[:30]}...")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x*100:.2f}%"


def _fmt_num(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.{nd}f}"


def _bucket_order(x: str) -> int:
    order = {
        "Extreme Fear": 0,
        "Fear": 1,
        "Neutral": 2,
        "Greed": 3,
        "Extreme Greed": 4,
    }
    return order.get(str(x), 99)


def _trend_bin(value: float) -> str:
    """
    2축 히트맵의 X축(추세) bin을 만든다.
    스크린샷의 '3D/5D' 느낌을 내기 위해 (최근변화) 기준으로 5개 구간으로 쪼갬.
    """
    if pd.isna(value):
        return "NA"
    if value <= -0.05:
        return "↓강"
    if value <= -0.02:
        return "↓"
    if value < 0.02:
        return "→"
    if value < 0.05:
        return "↑"
    return "↑강"


def _read_forward_summary_table_html() -> str:
    if not FWD_SUMMARY_CSV.exists():
        return "<p><b>forward 요약표 파일 없음</b></p>"
    df = pd.read_csv(FWD_SUMMARY_CSV)
    return df.to_html(index=False, escape=False)


def _read_summary_json() -> dict:
    if not SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


# -----------------------------
# Charts
# -----------------------------
def _fig_fg_line(df: pd.DataFrame, out_png: Path) -> None:
    y_col = "fear_greed_1y_rescaled" if _safe_col(df, "fear_greed_1y_rescaled") else "fear_greed"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df[y_col], linewidth=2, color="#4aa3ff")
    ax.set_title("곰탕지수(FG) 라인차트", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("FG (0~100)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _fig_components_grid(df: pd.DataFrame, out_png: Path) -> None:
    comps = ["momentum", "strength", "breadth", "putcall", "volatility", "safe_haven", "junk_bond", "margin_loan_ratio"]
    have = [c for c in comps if _safe_col(df, c)]
    if not have:
        return

    last = df.tail(200).copy()
    n = len(have)
    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i, c in enumerate(have[: rows * cols]):
        ax = axes[i]
        ax.plot(last["date"], last[c], linewidth=1.6, color="#86b7ff")
        ax.set_title(c, fontsize=10)
        ax.grid(True, alpha=0.2)

    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    fig.suptitle("구성요소(컴포넌트) 라인차트", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _make_heatmap_tables(df: pd.DataFrame):
    """
    히트맵 2개:
    - 히트맵1: 평균(20D) 수익률
    - 히트맵2: 승률(20D)
    2축: Y=fg_bucket_1y, X=trend_bin(최근 5D 수익률 기반)
    """
    req = ["fg_bucket_1y", "fwd_ret_20d_1y", "index_close"]
    if not all(_safe_col(df, c) for c in req):
        return None, None, None

    d = df.copy()
    d["ret_5d"] = d["index_close"].pct_change(5)
    d["trend_bin"] = d["ret_5d"].apply(_trend_bin)

    d = d.dropna(subset=["fg_bucket_1y", "trend_bin", "fwd_ret_20d_1y"])
    d["bucket"] = d["fg_bucket_1y"].astype(str)
    d["trend_bin"] = d["trend_bin"].astype(str)

    bucket_order = sorted(d["bucket"].unique(), key=_bucket_order)
    trend_order = ["↓강", "↓", "→", "↑", "↑강"]

    pivot_mean = (
        d.groupby(["bucket", "trend_bin"])["fwd_ret_20d_1y"]
        .mean()
        .unstack("trend_bin")
        .reindex(index=bucket_order, columns=trend_order)
    )
    pivot_win = (
        d.groupby(["bucket", "trend_bin"])["fwd_ret_20d_1y"]
        .apply(lambda s: float((s > 0).mean()))
        .unstack("trend_bin")
        .reindex(index=bucket_order, columns=trend_order)
    )

    # 최신 셀 위치
    latest = d.sort_values("date").iloc[-1]
    latest_bucket = str(latest["bucket"])
    latest_trend = str(latest["trend_bin"])

    return pivot_mean, pivot_win, (latest_bucket, latest_trend)


def _fig_heatmap(pivot: pd.DataFrame, out_png: Path, title: str, fmt: str, center: float | None, latest_cell=None) -> None:
    if pivot is None or pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = "RdYlGn" if center is not None else "YlGn"
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.6,
        linecolor="rgba(255,255,255,0.10)",
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("FG Bucket")

    # 최신 셀 박스 표시
    if latest_cell is not None:
        rlab, clab = latest_cell
        if rlab in pivot.index and clab in pivot.columns:
            r = list(pivot.index).index(rlab)
            c = list(pivot.columns).index(clab)
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=2.5))

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -----------------------------
# Strategy A/B rules (B안: 규칙 재현)
# -----------------------------
def _strategy_rules(df: pd.DataFrame) -> dict:
    """
    스크린샷의 'A안(40/60)' / 'B안(3D 추세)' 같은 느낌을 내기 위한 규칙 기반 추천 생성.
    데이터에 존재하는:
    - fear_greed_1y_rescaled (0~100)
    - fg_bucket_1y (Extreme Fear ... Extreme Greed)
    - index_close 기반 3D/5D 수익률
    를 이용해 A/B를 산출한다.
    """
    last = df.iloc[-1].copy()

    fg = float(last["fear_greed_1y_rescaled"]) if _safe_col(df, "fear_greed_1y_rescaled") else float(last["fear_greed"])
    bucket = str(last["fg_bucket_1y"]) if _safe_col(df, "fg_bucket_1y") else "NA"

    # 3D/5D ret
    ret3 = float(df["index_close"].pct_change(3).iloc[-1]) if _safe_col(df, "index_close") else np.nan
    ret5 = float(df["index_close"].pct_change(5).iloc[-1]) if _safe_col(df, "index_close") else np.nan

    # A: FG 레벨 기반(보수적)
    # - Extreme Fear/Fear: 적극(매수)
    # - Neutral: 중립
    # - Greed/Extreme Greed: 보수(부분 축소)
    if bucket in ["Extreme Fear", "Fear"]:
        a_action = "확대"
        a_text = "A안(레벨): 공포 구간 → 분할매수/비중확대"
    elif bucket == "Neutral":
        a_action = "중립"
        a_text = "A안(레벨): 중립 구간 → 유지/분할 접근"
    else:
        a_action = "축소"
        a_text = "A안(레벨): 탐욕 구간 → 분할익절/리스크 축소"

    # B: 단기 추세 기반(공격적)
    # - 3D↑ & 5D↑: 추세추종(확대)
    # - 3D↓ & 5D↓: 방어(축소)
    # - 그 외: 중립
    if (not pd.isna(ret3)) and (not pd.isna(ret5)) and ret3 > 0 and ret5 > 0:
        b_action = "확대"
        b_text = "B안(추세): 3D/5D 동반 상승 → 추세추종(비중확대)"
    elif (not pd.isna(ret3)) and (not pd.isna(ret5)) and ret3 < 0 and ret5 < 0:
        b_action = "축소"
        b_text = "B안(추세): 3D/5D 동반 하락 → 방어(비중축소)"
    else:
        b_action = "중립"
        b_text = "B안(추세): 혼조 → 중립/대기"

    # 3단계 최종 의견: A와 B를 합성
    score = 0
    score += {"확대": 1, "중립": 0, "축소": -1}.get(a_action, 0)
    score += {"확대": 1, "중립": 0, "축소": -1}.get(b_action, 0)
    if score >= 1:
        final = "매수 우위"
    elif score <= -1:
        final = "방어 우위"
    else:
        final = "중립"

    return {
        "fg": fg,
        "bucket": bucket,
        "ret3": ret3,
        "ret5": ret5,
        "A": {"action": a_action, "text": a_text},
        "B": {"action": b_action, "text": b_text},
        "final": final,
    }


# -----------------------------
# Report index (최근 10개)
# -----------------------------
def _update_report_index(new_dated_file: str, asof: str) -> dict:
    """
    docs/report_index.json 생성/갱신:
    - 최근 10개 날짜 파일 목록
    - 최신 기준일
    """
    items = []
    if REPORT_INDEX_JSON.exists():
        try:
            items = json.loads(REPORT_INDEX_JSON.read_text(encoding="utf-8")).get("items", [])
        except Exception:
            items = []

    # 중복 제거 후 맨 앞에 추가
    items = [x for x in items if x.get("file") != new_dated_file]
    items.insert(0, {"file": new_dated_file, "asof": asof})
    items = items[:10]

    payload = {
        "updated_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_asof": asof,
        "items": items,
    }
    REPORT_INDEX_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# -----------------------------
# HTML builder (Genspark-like)
# -----------------------------
def _build_html(asof: str, kpi: dict, strategy: dict, img: dict, fwd_table_html: str, report_index_payload: dict) -> str:
    # KPI cards
    kpi_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">곰탕 지수값(1Y)</div>
        <div class="kpi-value">{_fmt_num(kpi["fg"], 2)}</div>
        <div class="kpi-sub">0~100</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">구간(정체)</div>
        <div class="kpi-value">{kpi["bucket_range"]}</div>
        <div class="kpi-sub">{kpi["bucket_name"]}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">추세(+5p)</div>
        <div class="kpi-pills">
          <span class="pill">3D: {_fmt_pct(kpi["ret3"])}</span>
          <span class="pill">5D: {_fmt_pct(kpi["ret5"])}</span>
        </div>
        <div class="kpi-sub">* index_close 기반</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">KOSPI200</div>
        <div class="kpi-pills">
          <span class="pill">종가: {_fmt_num(kpi["index_close"], 2)}</span>
          <span class="pill">3D: {_fmt_pct(kpi["ret3"])}</span>
          <span class="pill">5D: {_fmt_pct(kpi["ret5"])}</span>
        </div>
        <div class="kpi-sub">단기 변화율</div>
      </div>
    </div>
    """

    # Toolbar + dropdown from report_index.json
    # (Genspark 느낌: 우측 상단 드롭다운 + 최신 버튼)
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>곰탕지수 리포트</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #0f1a30;
      --card: #111f3a;
      --line: rgba(255,255,255,0.10);
      --text: #e8eefc;
      --muted: rgba(232,238,252,0.72);
      --accent: #66a6ff;
      --good: #4dd4ac;
      --warn: #ffcc66;
    }}
    body {{
      margin:0; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif;
    }}
    .topbar {{
      display:flex; justify-content:space-between; align-items:center;
      padding: 10px 14px; border-bottom: 1px solid var(--line);
      background: rgba(15,26,48,0.85);
      position: sticky; top: 0; backdrop-filter: blur(10px); z-index: 10;
    }}
    .brand {{ font-weight: 700; font-size: 14px; color: var(--text); opacity:0.95; }}
    .toolbar {{ display:flex; gap:8px; align-items:center; }}
    select, button {{
      background: #0e1a33; color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px; padding: 8px 10px; font-size: 12px;
    }}
    button {{ cursor:pointer; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 14px; }}
    .hero {{
      background: radial-gradient(1200px 400px at 30% 0%, rgba(102,166,255,0.20), transparent 60%),
                  radial-gradient(900px 300px at 80% 30%, rgba(77,212,172,0.15), transparent 65%);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin-top: 10px;
    }}
    .title-row {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
    h1 {{ margin:0; font-size: 22px; }}
    .badge {{
      display:inline-block; padding: 4px 10px; border-radius: 999px;
      border: 1px solid var(--line); background: rgba(255,255,255,0.04);
      font-size: 12px; color: var(--text);
    }}

    .kpi-grid {{
      margin-top: 14px;
      display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    }}
    @media (max-width: 980px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 520px) {{
      .kpi-grid {{ grid-template-columns: 1fr; }}
    }}
    .kpi-card {{
      background: rgba(17,31,58,0.85);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      min-height: 84px;
    }}
    .kpi-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .kpi-value {{ font-size: 20px; font-weight: 800; }}
    .kpi-sub {{ margin-top: 6px; font-size: 12px; color: var(--muted); }}
    .kpi-pills {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .pill {{
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
    }}

    .section {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(15,26,48,0.55);
      padding: 12px;
    }}
    .section h2 {{ margin: 0 0 10px 0; font-size: 15px; }}
    .grid2 {{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 12px; }}
    @media (max-width: 980px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}

    .card {{
      border: 1px solid var(--line);
      background: rgba(17,31,58,0.75);
      border-radius: 14px;
      padding: 12px;
    }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: rgba(232,238,252,0.95); }}

    img {{ width: 100%; height: auto; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); }}

    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid rgba(255,255,255,0.10); padding: 7px 9px; font-size: 12px; }}
    th {{ text-align:left; color: rgba(232,238,252,0.92); background: rgba(255,255,255,0.03); }}

    .muted {{ color: var(--muted); font-size: 12px; line-height: 1.6; }}
    .ab-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    @media (max-width: 980px) {{ .ab-grid {{ grid-template-columns: 1fr; }} }}
    .ab-box {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.03);
      border-radius: 14px;
      padding: 12px;
    }}
    .ab-title {{ font-weight: 800; margin-bottom: 8px; }}
    .ab-final {{
      margin-top: 8px;
      display:inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(102,166,255,0.15);
      border: 1px solid rgba(102,166,255,0.35);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">곰탕지수 리포트</div>
    <div class="toolbar">
      <select id="dateSelect" title="날짜 선택(최근 10개)"></select>
      <button onclick="window.location.href='report_index.json'">날짜폴더(보기)</button>
      <button onclick="window.location.href='index.html'">최신</button>
    </div>
  </div>

  <div class="container">
    <div class="hero">
      <div class="title-row">
        <h1>한국곰탕지수(1Y 리스케일) 일일 리포트</h1>
        <span class="badge">{asof}</span>
      </div>
      <div class="muted" style="margin-top:8px;">
        * GitHub Actions가 레포 <code>data/</code>를 읽어서 매일 자동 생성합니다. (UTC 생성시각: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")})
      </div>

      {kpi_html}
    </div>

    <div class="section">
      <h2>1) 3단계 투자전략(A/B 동시 표기)</h2>
      <div class="ab-grid">
        <div class="ab-box">
          <div class="ab-title">A안(레벨) · {strategy["A"]["action"]}</div>
          <div class="muted">{strategy["A"]["text"]}</div>
        </div>
        <div class="ab-box">
          <div class="ab-title">B안(추세) · {strategy["B"]["action"]}</div>
          <div class="muted">{strategy["B"]["text"]}</div>
        </div>
      </div>
      <div class="ab-final">최종 의견: <b>{strategy["final"]}</b></div>
    </div>

    <div class="section">
      <h2>2) 곰탕지수 라인차트</h2>
      <div class="card">
        <img src="{img["fg_line"]}" alt="FG line" />
      </div>
    </div>

    <div class="section">
      <h2>3) 구성요소(컴포넌트) 라인차트</h2>
      <div class="card">
        <img src="{img["components"]}" alt="components" />
      </div>
    </div>

    <div class="section">
      <h2>4) 히트맵(최신 셀 위치 표시)</h2>
      <div class="grid2">
        <div class="card">
          <h3>히트맵 1 — 평균 수익(20D)</h3>
          <img src="{img["heat_mean"]}" alt="heat mean" />
        </div>
        <div class="card">
          <h3>히트맵 2 — 승률(20D)</h3>
          <img src="{img["heat_win"]}" alt="heat win" />
        </div>
      </div>

      <div class="card" style="margin-top:12px;">
        <h3>Forward 요약표</h3>
        {fwd_table_html}
      </div>
    </div>
  </div>

<script>
async function initDates() {{
  const res = await fetch('report_index.json?ts=' + Date.now());
  const data = await res.json();
  const sel = document.getElementById('dateSelect');
  sel.innerHTML = '';
  (data.items || []).forEach((it, idx) => {{
    const opt = document.createElement('option');
    opt.value = it.file;
    opt.textContent = it.asof + (idx===0 ? ' (최신)' : '');
    sel.appendChild(opt);
  }});
  sel.onchange = () => {{
    window.location.href = sel.value;
  }};
}}
initDates();
</script>
</body>
</html>"""
    return html


def main():
    _ensure_dirs()

    df = _read_fg()

    # as-of date
    asof_dt = df["date"].iloc[-1]
    asof = asof_dt.strftime("%Y-%m-%d")

    # KPI 계산 (FG + bucket range + index close + 3D/5D)
    fg_val = float(df["fear_greed_1y_rescaled"].iloc[-1]) if _safe_col(df, "fear_greed_1y_rescaled") else float(df["fear_greed"].iloc[-1])
    bucket_name = str(df["fg_bucket_1y"].iloc[-1]) if _safe_col(df, "fg_bucket_1y") else "NA"

    # bucket_range(예: 55~60) 느낌: 5단위로 내림/올림
    lo = int(np.floor(fg_val / 5) * 5)
    hi = int(lo + 5)
    bucket_range = f"{lo}~{hi}"

    index_close = float(df["index_close"].iloc[-1]) if _safe_col(df, "index_close") else np.nan
    ret3 = float(df["index_close"].pct_change(3).iloc[-1]) if _safe_col(df, "index_close") else np.nan
    ret5 = float(df["index_close"].pct_change(5).iloc[-1]) if _safe_col(df, "index_close") else np.nan

    kpi = {
        "fg": fg_val,
        "bucket_name": bucket_name,
        "bucket_range": bucket_range,
        "index_close": index_close,
        "ret3": ret3,
        "ret5": ret5,
    }

    # 투자전략(A/B 룰 B안)
    strategy = _strategy_rules(df)

    # Charts
    fg_line_png = ASSETS_DIR / "fg_line.png"
    comps_png = ASSETS_DIR / "components_grid.png"
    heat_mean_png = ASSETS_DIR / "heat_mean_20d.png"
    heat_win_png = ASSETS_DIR / "heat_win_20d.png"

    _fig_fg_line(df, fg_line_png)
    _fig_components_grid(df, comps_png)

    pivot_mean, pivot_win, latest_cell = _make_heatmap_tables(df)
    _fig_heatmap(pivot_mean, heat_mean_png, "평균 수익(20D) — Bucket x 추세(5D)", fmt=".3f", center=0.0, latest_cell=latest_cell)
    _fig_heatmap(pivot_win, heat_win_png, "승률(20D) — Bucket x 추세(5D)", fmt=".2f", center=None, latest_cell=latest_cell)

    img = {
        "fg_line": f"assets/{fg_line_png.name}",
        "components": f"assets/{comps_png.name}",
        "heat_mean": f"assets/{heat_mean_png.name}",
        "heat_win": f"assets/{heat_win_png.name}",
    }

    # forward summary table
    fwd_table_html = _read_forward_summary_table_html()

    # 파일 생성
    dated_file = f"곰탕지수_1Y리스케일_{asof}_embedded.html"
    latest_file = "곰탕지수_1Y리스케일_latest_embedded.html"

    payload = _update_report_index(dated_file, asof)

    html = _build_html(
        asof=asof,
        kpi=kpi,
        strategy=strategy,
        img=img,
        fwd_table_html=fwd_table_html,
        report_index_payload=payload,
    )

    (DOCS_DIR / dated_file).write_text(html, encoding="utf-8")
    (DOCS_DIR / latest_file).write_text(html, encoding="utf-8")
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")

    print("[OK] wrote reports:")
    print(" -", DOCS_DIR / dated_file)
    print(" -", DOCS_DIR / latest_file)
    print(" -", DOCS_DIR / "index.html")
    print("[OK] wrote index json:")
    print(" -", REPORT_INDEX_JSON)


if __name__ == "__main__":
    main()
