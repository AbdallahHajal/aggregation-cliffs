"""Poll a ChemAGG result CSV until it is complete, then save it.

The server writes the file asynchronously and recycles temp ids, so a naive fetch can
return an empty or truncated file. This polls until the parsed row count matches the
expected number of analogues, and refuses to save a short file silently.
"""
import io, ssl, sys, time, urllib.request
import pandas as pd

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
BASE = "https://admet.scbdd.com/static/uploadfile/"


def fetch(name, expect, out, tries=20, wait=15):
    url = BASE + name
    for i in range(1, tries + 1):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=300, context=CTX).read()
            d = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in ("Identifier", "Category", "Probability"))
            if len(d) >= expect:
                open(out, "wb").write(raw)
                print(f"OK  {len(d)} rows (expected {expect}) -> {out}")
                return True
            print(f"  try {i}: {len(d)}/{expect} rows, waiting {wait}s")
        except Exception as e:
            print(f"  try {i}: {type(e).__name__} {str(e)[:60]}, waiting {wait}s")
        time.sleep(wait)
    print(f"FAILED after {tries} tries")
    return False


if __name__ == "__main__":
    ok = fetch(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    sys.exit(0 if ok else 1)
