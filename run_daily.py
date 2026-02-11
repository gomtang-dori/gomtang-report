# run_daily.py
# 목적: GitHub Actions에서 곰탕지수 리포트를 "직접" 생성해서 docs/에 저장
# 산출물:
# - docs/index.html (최신)
# - docs/곰탕지수_1Y리스케일_YYYY-MM-DD_embedded.html (날짜별)
# - docs/곰탕지수_1Y리스케일_YYYY-MM-DD.html (가벼운 버전: 이미지 외부참조)
#
# 입력은 현재 Genspark에서 만든 CSV/JSON을 다운로드해서 사용(토큰 URL 만료 가능)
# 안정화: 다음 단계에서 이 파일들을 레포에 vendor/로 커밋 권장

import base64
import io
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_LAST1Y_WITH_FWD_URL = "https://www.genspark.ai/api/files/s/0vTYPWpI?token=Z0FBQUFBQnBqSnZGQTNuUVRPcTdyVkE1TkJnQlNON0xhMzhhb2tJVUNrUkNHbEpfRzhDVW1tNmVKbXhNRW1rT1o5WWw4UWlxTU83MkdDdlNtOG9aNGtpRVU1SUN3aU9BYkhlUVVaVWZYRDRVdWpRVG1MMjNUUEtKNnZXWk82SFJZMHFsUkxua0pTTmowN1E2cGM1d0QxREc4R2NUY1NQQ0lNUUVCN2Q1ZWlVTzIzdmg0OEc5QXM2S2lwUGRzWDFtWmlDa3BzWUpHUlRRZ3R5aFpHeHRiM0ZwUXAtVEgyQWQwR2VZa2pFcDBmamN6YkpmNXUwSURlNEJEZGdWTkxyc1pBYUkzamNrVGdjWnM2MWY5TFp4SjRkdXFVb0NTcFZYZHc9PQ"
FWD_BUCKET_SUMMARY_URL = "https://www.genspark.ai/api/files/s/zF8mcR6Z?token=Z0FBQUFBQnBqSnZGX3lWZi14eEVsd2FIMnhJajEtMEI4aWQtS09OV0dfODVCSGFnaWtqMHM1OVRXTG5YaWI5RXlNV1pVQTFMakFJSkt4VnRoRGdSMkRNWTBCbUUwXzZESTVBLWlZMXd5a3lyMjBkMzFLOE1RUFJYTk5lMVhYOW45eXA1UmhBbjBZLTJrdVdZN0plTW80V2JzeVpGaVZ3U3dGX25MMXNnYjJjcThZbWVpVWRXNFB4NVZvVFhfZVJBelFRazNUY2RRTFJuSDNaY2F0bEVPS0VraXlUbHh6WWhTbXFNdXE4MENTN3pXUzZNaV8tUDBBUHJDQml1XzJMYzVYWGtRMXBTYl81eWpnWC1qVHBpdDduWUhJU2tHZUU2NGc9PQ"
SUMMARY_JSON_URL = "https://www.genspark.ai/api/files/s/iVE4ooxf?token=Z0FBQUFBQnBqSnZGVjZUUDliNkhQUURsa09zaUhpa0xfRmc3RFRiLTJVY3JUd19mY0dRVWx6N3lqdy1GdGtyNFc0dnFKNlRLZUlETUtMS1ExR3JpX1l5cDhYbUwzMVN6TXA0Rk56VFV4Q3JxLVFXRkJsUGsyVXByc25CRFJzc2xDZHZmNEJCNElkOVNIUGQ2bzB3UUg2ZUdFN0tfczBmQVpmdlVCSHVGZFhyN1BsVG1NV3E2d1lDcnhLMXRMbTQ2RGVfb1Utd3NCT1MtS3dSX20tY0hyWGgzRUFDMjNKQmhhTEIxSXAyYjEzNHZvZDdmZXhHSUI4YkgwcVN1dGxlT3hjVkVWVEdtR085Smd5T0ljUXhTZU9NQ0gyY1psWnlRQVE9PQ"

OUT_DOCS = Path("docs")
OUT_DOCS.mkdir(parents=True, exist_ok=True)

def read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def read_json(url: str) -> dict:
    return json.loads(pd.read_json(url, typ="series").to_json())

def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ensure_columns(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing columns. Need one of {candidates}. Have: {list(df.columns)[:30]} ...")

def make_line_chart(df: pd.DataFrame, date_col: str, score_col: str):
    x = pd.to_datetime(df[date_col])
    y = df[score_col].astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(x, y, lw=1.6)
    ax.set_title("곰탕지수(1Y 리스케일) 라인차트")
    ax.grid(True, alpha=0.3)
    return fig

def make_component_chart(df: pd.DataFrame, date_col: str, cols: list[str]):
    x = pd.to_datetime(df[date_col])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c in cols:
        if c in df.columns:
            ax.plot(x, df[c].astype(float), lw=1.2, label=c)
    ax.set_title("컴포넌트(가능한 컬럼만 표시)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    return fig

def bucket_forward_block(fwd: pd.DataFrame, bucket: str) -> str:
    out = []
    out.append(f"<h3>버킷별 Forward 승률(B안 참고)</h3>")
    out.append(f"<p><b>현재 버킷</b>: {bucket}</p>")
    for h in [20, 60]:
        sub = fwd[(fwd["bucket"] == bucket) & (fwd["horizon_trading_days"] == h)]
        if len(sub) == 0:
            out.append(f"<p>{h}D: 데이터 없음</p>")
            continue
        r = sub.iloc[0]
        out.append(
            f"<p><b>{h}거래일</b> 승률(>0): {r['win_rate_pct']:.1f}% (n={int(r['count'])}), 평균 {r['mean_pct']:.2f}%</p>"
        )
    return "\n".join(out)

def main():
    # 1) 입력 데이터 다운로드
    df = read_csv(DATA_LAST1Y_WITH_FWD_URL)
    fwd = read_csv(FWD_BUCKET_SUMMARY_URL)

    # 2) 컬럼 추정(데이터 구조가 바뀌어도 최대한 견고하게)
    date_col = ensure_columns(df, ["date", "Date", "base_date", "dt"])
    score_col = ensure_columns(df, ["fg_rescaled", "gomtang_rescaled", "fear_greed_rescaled", "score_rescaled", "fg_score_rescaled", "score"])
    bucket_col = ensure_columns(df, ["bucket", "regime", "fg_bucket"])

    # 3) 최신 날짜/버킷
    df_sorted = df.sort_values(date_col)
    latest = df_sorted.iloc[-1]
    date_str = str(latest[date_col])[:10]
    bucket = str(latest[bucket_col])

    # 4) 차트 생성(embedded는 base64로 삽입)
    fig1 = make_line_chart(df_sorted, date_col, score_col)
    b64_line = fig_to_base64_png(fig1)

    # 컴포넌트 후보 컬럼 (있으면 표시)
    comp_candidates = ["mom_20d", "mom_60d", "vol_20d", "vol_60d", "breadth", "putcall", "skew"]
    fig2 = make_component_chart(df_sorted, date_col, comp_candidates)
    b64_comp = fig_to_base64_png(fig2)

    # 5) 통계(간단 버전): 현재 점수/버킷 + forward
    fwd_block = bucket_forward_block(fwd, bucket)

    # 6) HTML 생성 (embedded)
    html = f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>곰탕지수 최신 리포트</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',Arial,sans-serif; margin: 16px; }}
.nav {{ position: sticky; top:0; background:#111; color:#fff; padding:10px 12px; border-radius:10px; }}
.nav a {{ color:#fff; text-decoration:none; margin-right:12px; }}
.card {{ border:1px solid #e5e5e5; border-radius:12px; padding:12px; margin:12px 0; }}
.small {{ color:#666; font-size: 13px; }}
img {{ max-width: 100%; border-radius: 10px; }}
</style>
</head>
<body>
<div class="nav">
  <b>곰탕지수 리포트</b>
  <span class="small" style="color:#bbb; margin-left:8px;">기준일 {date_str}</span>
</div>

<div class="card">
  <h2>요약</h2>
  <p><b>현재 버킷</b>: {bucket}</p>
  <p><b>현재 점수</b>: {float(latest[score_col]):.2f}</p>
</div>

<div class="card">
  <h2>라인차트</h2>
  <img src="data:image/png;base64,{b64_line}" />
</div>

<div class="card">
  <h2>컴포넌트 차트</h2>
  <img src="data:image/png;base64,{b64_comp}" />
</div>

<div class="card" id="stats">
  {fwd_block}
</div>

<div class="card">
  <h2>파일</h2>
  <p class="small">이 페이지는 GitHub Actions가 자동 생성합니다.</p>
</div>
</body>
</html>
""".strip()

    # 7) 저장 (docs/)
    dated_embedded = OUT_DOCS / f"곰탕지수_1Y리스케일_{date_str}_embedded.html"
    latest_embedded = OUT_DOCS / "곰탕지수_1Y리스케일_latest_embedded.html"
    index_html = OUT_DOCS / "index.html"

    dated_embedded.write_text(html, encoding="utf-8")
    latest_embedded.write_text(html, encoding="utf-8")
    index_html.write_text(html, encoding="utf-8")

    # 가벼운 버전(plain)은 우선 embedded와 동일 파일로 저장(추후 이미지 외부 분리 가능)
    (OUT_DOCS / f"곰탕지수_1Y리스케일_{date_str}.html").write_text(html, encoding="utf-8")

    print("OK:", dated_embedded, latest_embedded, index_html)

if __name__ == "__main__":
    main()
