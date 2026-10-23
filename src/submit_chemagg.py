#!/usr/bin/env python
"""Submit an SDF to the ChemAGG web server and return the result CSV name.

The service has no API. Its index page carries a Django form that POSTs a multipart
upload to /ChemAGG/calc_cf_file/ and answers with a results page containing a link to
a generated CSV under /static/uploadfile/. This reproduces that exact request so the
filter can be scored from a script rather than by hand, which is what the Methods
section claims.

The server returns 413 above roughly 1 MB, so files are submitted in chunks.
Requests are throttled; this is somebody else's academic server.

Usage: python submit_chemagg.py <chunk.sdf> [<chunk.sdf> ...]
"""
import http.cookiejar
import os
import pathlib
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = "https://admet.scbdd.com"
INDEX = BASE + "/ChemAGG/index/"
POST = BASE + "/ChemAGG/calc_cf_file/"
THROTTLE_S = 5.0
INSECURE_ENV = "CHEMAGG_ALLOW_INSECURE_TLS"


def _context():
    """Verify the certificate. Fall back only on an explicit opt-in.

    As of 2026-09-29 admet.scbdd.com served an EXPIRED certificate, and the submissions
    behind this work were made with verification disabled. That is recorded rather than
    hidden: what was sent is public PubChem structures and what came back is a prediction
    table, so no credential was exposed, but an unverified channel cannot attest that the
    table was not altered in transit.

    Verification is now on by default and this fails closed. Set
    CHEMAGG_ALLOW_INSECURE_TLS=1 to reproduce the original unverified submission.
    """
    ctx = ssl.create_default_context()
    try:
        urllib.request.urlopen(
            urllib.request.Request(INDEX, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60, context=ctx).read(64)
        return ctx
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        if os.environ.get(INSECURE_ENV) != "1":
            raise SystemExit(
                f"{BASE} certificate does not verify ({str(e.reason)[:70]}).\n"
                f"Refusing to submit over an unverified channel. To reproduce the\n"
                f"original submission anyway, set {INSECURE_ENV}=1 and re-run.")
        print(f"WARNING: {BASE} certificate does not verify; proceeding unverified "
              f"because {INSECURE_ENV}=1. Public structures only, no credentials sent.")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


CTX = _context()

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=CTX),
    urllib.request.HTTPCookieProcessor(jar),
)
opener.addheaders = [("User-Agent", "Mozilla/5.0 (academic research; filter benchmark)")]


def csrf_token() -> str:
    html = opener.open(INDEX, timeout=120).read().decode("utf-8", "replace")
    m = re.search(r"name=['\"]csrfmiddlewaretoken['\"]\s+value=['\"]([^'\"]+)", html)
    if m:
        return m.group(1)
    for c in jar:
        if c.name in ("csrftoken", "csrf"):
            return c.value
    raise RuntimeError("no CSRF token found on the index page")


def multipart(fields: dict, filename: str, filedata: bytes, field_name: str):
    boundary = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    out.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: chemical/x-mdl-sdfile\r\n\r\n".encode()
        + filedata + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def submit(path: pathlib.Path, field_name="molefile") -> str | None:
    token = csrf_token()
    data = path.read_bytes()
    body, ctype = multipart(
        {"csrfmiddlewaretoken": token, "SelectDataSource": "file", "smiles": ""},
        path.name, data, field_name)
    req = urllib.request.Request(POST, data=body, headers={
        "Content-Type": ctype, "Referer": INDEX,
        "Cookie": "; ".join(f"{c.name}={c.value}" for c in jar),
    })
    try:
        html = opener.open(req, timeout=600).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  {path.name}: {type(e).__name__} {str(e)[:90]}")
        return None
    m = re.search(r"/static/uploadfile/(temp_file-\d+\.sdf_predicted_with_s\.csv)", html)
    if m:
        return m.group(1)
    print(f"  {path.name}: submitted but no result link found "
          f"({len(html)} bytes returned)")
    return None


if __name__ == "__main__":
    for p in sys.argv[1:]:
        path = pathlib.Path(p)
        name = submit(path)
        print(f"{path.name} -> {name}")
        time.sleep(THROTTLE_S)
