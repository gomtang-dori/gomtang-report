import sys, subprocess
from pathlib import Path

SCRIPT_URL = "https://www.genspark.ai/api/files/s/hFgCOO21?token=Z0FBQUFBQnBqSVBIbjF5R1I2a2NDczVYSWNMV0RnbkExRS03bWFqQk9qVnBFeFNoYl9LT0ItOWk1SWtBZE00YkZMVmNhQXNPNUJ6Vi1Sb3JMT3BvUWc2XzIybVF1Z2tVNlR1dm5ITWVETXF3VFBCM001dkdXdG92cUo2VjJ0a3d4R2dNNWVVZ1ZHWmlNRF83ay1ETFVVcEloTWR5ZktLd3pfRGo0N2ItRXBic2RDSWtGZ2xxWGJLUWNya1JWX0VzTWxBeGpFTHlRdnRORkxHbDVrWlRDTzlxd3JPeWJSYTdvc0lOWXhFVXBkMTZrMTlTUHh1MjlVNDZXQUYzUVJ1TlJMeEdES0FpN2VoVzRSd1hpdTFybktENV82S1h3X2wyNUE9PQ"

OUTDIR = Path("docs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def sh(cmd):
    subprocess.check_call(cmd)

def main():
    sh(["curl", "-L", SCRIPT_URL, "-o", "gomtang_source.py"])
    sh([sys.executable, "gomtang_source.py"])

    for f in Path(".").glob("곰탕지수_1Y리스케일_*.html"):
        (OUTDIR / f.name).write_bytes(f.read_bytes())

    latest = OUTDIR / "곰탕지수_1Y리스케일_latest_embedded.html"
    if latest.exists():
        (OUTDIR / "index.html").write_bytes(latest.read_bytes())

if __name__ == "__main__":
    main()
