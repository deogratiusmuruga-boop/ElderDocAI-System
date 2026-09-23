#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Stage 3: full-text download from PMC Cloud (S3, HTTPS).

For each pending PMCid: list the S3 prefix (PMC{id}.) to find the versioned
folder, then download the JATS XML and the JSON metadata object into raw/.
Resumable and capped (--limit per run). Integrity fields recorded in
manifest/download_status.json. No chunking / no RAG.
"""
import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
MAN = BASE / "manifest"
RAW = BASE / "raw"
UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research; dev corpus stage3)"}
SLEEP = 0.15
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
STATUS_FILE = MAN / "download_status.json"


def fetch(url, timeout=180, attempts=4):
    req = urllib.request.Request(url, headers=UA)
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last = e
            time.sleep(1.5 + 1.5 * i)
    raise last


def find_version(pmc):
    url = BUCKET + "/?list-type=2&delimiter=%2F&prefix=" + pmc + "."
    xml = fetch(url, timeout=120).decode("utf-8", "replace")
    m = re.search(r"<Prefix>(" + pmc + r"\.\d+)/</Prefix>", xml)
    return m.group(1) if m else None


def load_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {}


def save_status(s):
    STATUS_FILE.write_text(json.dumps(s, indent=2, sort_keys=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()
    sel = json.loads((MAN / "selected.json").read_text(encoding="utf-8"))
    pending = sel["pending"]
    status = load_status()
    done = 0
    for raw_id in pending:
        if raw_id in status and status[raw_id].get("ok"):
            continue
        if done >= args.limit:
            break
        pmc = raw_id if raw_id.startswith("PMC") else "PMC" + raw_id
        try:
            ver = find_version(pmc)
            if not ver:
                status[raw_id] = {"ok": False, "error": "version_not_found", "ts": None}
                done += 1
                time.sleep(SLEEP)
                continue
            xurl = f"{BUCKET}/{ver}/{ver}.xml"
            jurl = f"{BUCKET}/{ver}/{ver}.json"
            xmlb = fetch(xurl)
            jb = fetch(jurl)
            folder = RAW / ver
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (ver + ".xml")).write_bytes(xmlb)
            (folder / (ver + ".json")).write_bytes(jb)
            txt = xmlb.decode("utf-8", "replace")
            ok = bool(xmlb) and "<article" in txt and bool(re.search(r"PMC\s*\d+", txt))
            status[raw_id] = {"ok": bool(ok), "version": ver,
                              "bytes": len(xmlb), "error": None if ok else "integrity",
                              "ts": datetime.now(timezone.utc).isoformat()}
            if not ok:
                status[raw_id]["note"] = "xml non-empty + <article> + PMCID present"
        except Exception as e:
            status[raw_id] = {"ok": False, "error": str(e)[:160],
                              "ts": datetime.now(timezone.utc).isoformat()}
        done += 1
        time.sleep(SLEEP)
    save_status(status)
    okc = sum(1 for v in status.values() if v.get("ok"))
    print(json.dumps({"limit": args.limit, "attempted_this_run": done,
                      "total_status": len(status), "ok_total": okc}, indent=2))


if __name__ == "__main__":
    main()