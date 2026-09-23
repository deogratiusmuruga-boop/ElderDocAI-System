import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research; debug)"}


def get(url):
    req = urllib.request.Request(url, headers=UA, method="GET")
    return urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")


for pid in ["PMC10002240", "PMC13573782", "PMC13901"]:
    u = ("https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&delimiter=%2F&prefix="
         + pid + ".")
    try:
        d = get(u)
        print("==", pid, "len", len(d))
        print(d[:900])
        m = re.search(r"<Prefix>(" + re.escape(pid) + r"\.\d+)/</Prefix>", d)
        print("match:", m.group(1) if m else None)
    except Exception as e:
        print(pid, "ERR", str(e)[:200])
for key in ["PMC13901.1/PMC13901.1.xml", "PMC13901.1/PMC13901.1.json"]:
    u = "https://pmc-oa-opendata.s3.amazonaws.com/" + key
    try:
        req = urllib.request.Request(u, headers=UA, method="HEAD")
        r = urllib.request.urlopen(req, timeout=120)
        print(key, "status", r.status, "len", r.headers.get("Content-Length"))
    except Exception as e:
        print(key, "ERR", str(e)[:200])