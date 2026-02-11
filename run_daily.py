import sys, subprocess
from pathlib import Path

SCRIPT_URL = "https://www.genspark.ai/api/files/s/hFgCOO21?token=Z0FBQUFBQnBqSVBIbjF5R1I2a2NDczVYSWNMV0RnbkExRS03bWFqQk9qVnBFeFNoYl9LT0ItOWk1SWtBZE00YkZMVmNhQXNPNUJ6Vi1Sb3JMT3BvUWc2XzIybVF1Z2tVNlR1dm5ITWVETXF3VFBCM001dkdXdG92cUo2VjJ0a3d4R2dNNWVVZ1ZHWmlNRF83ay1ETFVVcEloTWR5ZktLd3pfRGo0N2ItRXBic2RDSWtGZ2xxWGJLUWNya1JWX0VzTWxBeGpFTHlRdnRORkxHbDVrWlRDTzlxd3JPeWJSYTdvc0lOWXhFVXBkMTZrMTlTUHh1MjlVNDZXQUYzUVJ1TlJMeEdES0FpN2VoVzRSd1hpdTFybktENV82S1h3X2wyNUE9PQ"

OUTDIR = Path("docs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def sh(cmd):
    subprocess.check_call(cmd)

def main():
    # 1) 원본 스크립트 다운로드
    sh(["curl", "-L", SCRIPT_URL, "-o", "gomtang_source.py"])

    # 2) GitHub Actions에서 쓰기 가능한 경로로 강제 패치
    #    (원본은 /mnt/user-data/outputs/... 같은 경로를 쓰려 해서 권한 에러 발생) [Source](https://www.genspark.ai/api/files/s/DH1E8DDE)
    src = Path("gomtang_source.py").read_text(encoding="utf-8", errors="ignore")

    replacements = [
        ("/mnt/user-data/outputs", "./outputs"),
        ("/mnt/user-data", "./outputs"),
        ("OUT_DIR = ", "OUT_DIR = "),  # no-op, 자리 유지
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    # 혹시 mkdir 대상이 고정 문자열이면 한번 더 안전장치: outputs 디렉터리 선생성
    Path("outputs").mkdir(parents=True, exist_ok=True)

    Path("gomtang_source.py").write_text(src, encoding="utf-8")

    # 3) 실행
    sh([sys.executable, "gomtang_source.py"])

    # 4) 산출물 복사 (Pages 공개용 docs/)
    for f in Path(".").glob("곰탕지수_1Y리스케일_*.html"):
        (OUTDIR / f.name).write_bytes(f.read_bytes())

    # 5) index.html = 최신파일(접속 편의)
    latest = OUTDIR / "곰탕지수_1Y리스케일_latest_embedded.html"
    if latest.exists():
        (OUTDIR / "index.html").write_bytes(latest.read_bytes())

if __name__ == "__main__":
    main()
