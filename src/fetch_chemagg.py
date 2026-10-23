"""Poll a ChemAGG result CSV until it is complete, then save it.

The server writes the file asynchronously and recycles temp ids, so a naive fetch can
return an empty or truncated file. This polls until the parsed row count matches the
expected number of analogues, and refuses to save a short file silently.
"""
import hashlib, io, os, ssl, sys, time, urllib.error, urllib.request
import pandas as pd

BASE = "https://admet.scbdd.com/static/uploadfile/"
INSECURE_ENV = "CHEMAGG_ALLOW_INSECURE_TLS"


def _context():
    """Verify the certificate. Fall back only on an explicit opt-in.

    As of 2026-09-29 admet.scbdd.com served an EXPIRED certificate, and the results used
    in this work were downloaded with verification disabled. That is recorded here rather
    than hidden: what was sent is public PubChem structures and what came back is a
    prediction table, so no credential was exposed, but an unverified channel cannot
    attest that the table was not altered in transit. Each saved file is therefore
    hashed below so a re-download can be compared against it.

    Verification is now on by default and the script fails closed. Set
    CHEMAGG_ALLOW_INSECURE_TLS=1 to reproduce the original unverified download.
    """
    ctx = ssl.create_default_context()
    try:
        urllib.request.urlopen(
            urllib.request.Request(BASE, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60, context=ctx).read(64)
        return ctx
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        if os.environ.get(INSECURE_ENV) != "1":
            raise SystemExit(
                f"{BASE} certificate does not verify ({str(e.reason)[:70]}).\n"
                f"Refusing to download over an unverified channel. To reproduce the\n"
                f"original download anyway, set {INSECURE_ENV}=1 and re-run.")
        print(f"WARNING: {BASE} certificate does not verify; proceeding unverified "
              f"because {INSECURE_ENV}=1. Public structures only, no credentials sent.")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


CTX = _context()


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
                print(f"    source {url}")
                print(f"    sha256 {hashlib.sha256(raw).hexdigest()}")
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
