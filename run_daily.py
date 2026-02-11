# -*- coding: utf-8 -*-
"""
Gomtang Index (Korea Fear & Greed) - Daily HTML report generator for GitHub Actions + Pages

Inputs (repo local):
- data/korea_fear_greed_v2_ks200_last1y_rescaled_with_forward.csv
- data/fg_forward_summary_last1y_rescaled_pretty.csv
- data/summary_last1y_rescaled.json (optional)

Outputs (docs/):
- docs/곰탕지수_1Y리스케일_YYYY-MM-DD_embedded.html
- docs/곰탕지수_1Y리스케일_latest_embedded.html
- docs/index.html  (copy of latest)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

FG_CSV = DATA_DIR / "korea_fear_greed_v2_ks200_last1y_rescaled_with_forward.csv"
FWD_SUMMARY_CSV = DATA_DIR / "fg_forward_summary_last1y_rescaled_pretty.csv"
SUMMARY_JSON = DATA_DIR / "summary_last1y_rescaled.json"


def _ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _read_fg() -> pd.DataFrame:
    if not FG_CSV.exists():
        raise FileNotFoundError(f"Missing input: {FG_CSV}")

    df = pd.read_csv(FG_CSV)
    # Column names observed in raw header include '날짜' [Source: raw file]
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


def _fig_line_fg(df: pd.DataFrame, out_png: Path) -> None:
    # prefer 1y rescaled series if exists
    y_col = None
    for c in ["fear_greed_1y_rescaled", "fear_greed"]:
        if _safe_col(df, c):
            y_col = c
            break
    if y_col is None:
        raise ValueError("No FG series found (fear_greed_1y_rescaled / fear_greed).")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df[y_col], linewidth=2, color="#1f77b4")
    ax.set_title(f"곰탕지수(FG) 추이 — {y_col}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("FG")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _fig_components(df: pd.DataFrame, out_png: Path) -> None:
    comps = ["momentum", "strength", "breadth", "putcall", "volatility", "safe_haven", "junk_bond", "margin_loan_ratio"]
    have = [c for c in comps if _safe_col(df, c)]
    if len(have) == 0:
        return

    last = df.tail(180).copy()  # 최근 약 6~9개월 정도만 보기 좋게
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in have:
        ax.plot(last["date"], last[c], linewidth=1.8, label=c)
    ax.set_title("컴포넌트 추이(최근 구간)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _fig_heatmap_forward(df: pd.DataFrame, out_png: Path) -> None:
    # Make heatmap of fg_bucket_1y vs (mean fwd_ret_20d_1y, winrate)
    req = ["fg_bucket_1y", "fwd_ret_20d_1y", "fwd_ret_60d_1y"]
    if not all(_safe_col(df, c) for c in req):
        return

    d = df.dropna(subset=req).copy()

    def _bucket_order(x: str) -> int:
        # Expect strings like 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        order = {
            "Extreme Fear": 0,
            "Fear": 1,
            "Neutral": 2,
            "Greed": 3,
            "Extreme Greed": 4,
        }
        return order.get(str(x), 99)

    buckets = sorted(d["fg_bucket_1y"].astype(str).unique(), key=_bucket_order)

    stats = []
    for b in buckets:
        sub = d[d["fg_bucket_1y"].astype(str) == str(b)]
        mean20 = float(sub["fwd_ret_20d_1y"].mean())
        win20 = float((sub["fwd_ret_20d_1y"] > 0).mean())
        mean60 = float(sub["fwd_ret_60d_1y"].mean())
        win60 = float((sub["fwd_ret_60d_1y"] > 0).mean())
        stats.append([b, mean20, win20, mean60, win60])

    s = pd.DataFrame(stats, columns=["bucket", "mean20", "win20", "mean60", "win60"]).set_index("bucket")

    # 2x2 heatmap
    mat = np.array([
        [s.loc[b, "mean20"] for b in buckets],
        [s.loc[b, "win20"] for b in buckets],
        [s.loc[b, "mean60"] for b in buckets],
        [s.loc[b, "win60"] for b in buckets],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.0,
        cbar=True,
        ax=ax,
        xticklabels=buckets,
        yticklabels=["mean_20d", "winrate_20d", "mean_60d", "winrate_60d"],
    )
    ax.set_title("버킷별 Forward 성과(평균/승률)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _read_forward_summary_table() -> str:
    if not FWD_SUMMARY_CSV.exists():
        return "<p><b>forward 요약표 파일 없음</b></p>"

    df = pd.read_csv(FWD_SUMMARY_CSV)
    # Render as HTML table
    return df.to_html(index=False, escape=False)


def _read_summary_json() -> dict:
    if not SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_html(title: str, asof: str, img_paths: dict, summary_json: dict, fwd_table_html: str) -> str:
    # Basic, fully self-contained HTML (embedded images as relative paths under docs/)
    def _img_tag(rel: str, alt: str) -> str:
        if not rel:
            return ""
        return f'<div class="card"><h3>{alt}</h3><img src="{rel}" alt="{alt}" /></div>'

    summary_block = ""
    if summary_json:
        # Pretty key-value list
        items = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in summary_json.items()
            if isinstance(k, str)
        )
        summary_block = f"""
        <div class="card">
          <h3>요약</h3>
          <table class="kv">{items}</table>
        </div>
        """

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: #111b2e;
      --text: #e8eefc;
      --muted: #a9b6d3;
      --line: rgba(255,255,255,0.12);
      --accent: #6ea8fe;
    }}
    body {{
      margin: 0; padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 18px; }}
    .title {{
      padding: 14px 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(110,168,254,0.20), rgba(17,27,46,0.2));
      border-radius: 14px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0; font-size: 22px; }}
    .meta {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    @media (min-width: 900px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 14px;
      padding: 12px;
      overflow: hidden;
    }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 16px; color: #dbe6ff; }}
    img {{ width: 100%; height: auto; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      font-size: 13px;
    }}
    th {{ color: #dbe6ff; text-align: left; background: rgba(255,255,255,0.03); }}
    .kv td:first-child {{ width: 42%; color: var(--muted); }}
    a {{ color: var(--accent); text-decoration: none; }}
    .footer {{ color: var(--muted); font-size: 12px; margin-top: 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">
      <h1>{title}</h1>
      <div class="meta">기준일: <b>{asof}</b> · 생성시각(UTC): {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <div class="grid">
      {summary_block}
      <div class="card">
        <h3>Forward 요약표</h3>
        {fwd_table_html}
      </div>
    </div>

    <div class="grid" style="margin-top:14px;">
      {_img_tag(img_paths.get("fg_line",""), "라인차트")}
      {_img_tag(img_paths.get("components",""), "컴포넌트 라인차트")}
    </div>

    <div class="grid" style="margin-top:14px;">
      {_img_tag(img_paths.get("heatmap",""), "히트맵(Forward 평균/승률)")}
      <div class="card">
        <h3>설명</h3>
        <p style="color:var(--muted); line-height:1.6; margin:0;">
          이 페이지는 GitHub Actions에서 레포 내부 데이터(<code>data/</code>)를 읽어 매일 자동 생성됩니다.
          GitHub Pages 루트 주소는 항상 <code>docs/index.html</code>(최신 리포트 복사본)을 보여줍니다.
        </p>
      </div>
    </div>

    <div class="footer">
      <div>Repo: <a href="https://github.com/gomtang-dori/gomtang-report" target="_blank">gomtang-dori/gomtang-report</a></div>
    </div>
  </div>
</body>
</html>"""
    return html


def main():
    _ensure_dirs()

    df = _read_fg()

    # asof date: last row date
    asof_dt = df["date"].iloc[-1]
    asof = asof_dt.strftime("%Y-%m-%d")

    # generate images into docs/assets
    fg_line_png = ASSETS_DIR / "fg_line.png"
    comps_png = ASSETS_DIR / "components_line.png"
    heat_png = ASSETS_DIR / "forward_heatmap.png"

    _fig_line_fg(df, fg_line_png)
    _fig_components(df, comps_png)
    _fig_heatmap_forward(df, heat_png)

    img_paths = {
        "fg_line": f"assets/{fg_line_png.name}" if fg_line_png.exists() else "",
        "components": f"assets/{comps_png.name}" if comps_png.exists() else "",
        "heatmap": f"assets/{heat_png.name}" if heat_png.exists() else "",
    }

    fwd_table_html = _read_forward_summary_table()
    summary_json = _read_summary_json()

    title = "곰탕지수 (1Y 리스케일) — Daily Report"
    html = _build_html(title=title, asof=asof, img_paths=img_paths, summary_json=summary_json, fwd_table_html=fwd_table_html)

    dated_name = f"곰탕지수_1Y리스케일_{asof}_embedded.html"
    latest_name = "곰탕지수_1Y리스케일_latest_embedded.html"

    dated_path = DOCS_DIR / dated_name
    latest_path = DOCS_DIR / latest_name
    index_path = DOCS_DIR / "index.html"

    dated_path.write_text(html, encoding="utf-8")
    latest_path.write_text(html, encoding="utf-8")
    index_path.write_text(html, encoding="utf-8")

    print(f"[OK] wrote: {dated_path}")
    print(f"[OK] wrote: {latest_path}")
    print(f"[OK] wrote: {index_path}")


if __name__ == "__main__":
    main()
