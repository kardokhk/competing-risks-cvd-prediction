"""Download NHANES XPT component files and NCHS Linked Mortality .dat files.

All HTTP goes through urllib.request with a Mozilla User-Agent (system curl/wget
are broken in this environment). A file is only accepted if it looks like a real
SAS xport (magic 'HEADER RECORD') / non-empty .dat; HTML soft-404 pages are
rejected. Every attempt is recorded in the manifest, with the UTC retrieval
time, the HTTP status, the byte count and the SHA-256 of the accepted file, so
that a data access date can be recovered from the repository later.
"""
from __future__ import annotations
import csv
import hashlib
import time
import urllib.error
import urllib.request as u
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CYCLES, COMPONENT_CANDIDATES, NHANES_XPT_URL, MORT_BASE, MORT_FILE,
    RAW_NHANES, RAW_MORT, MANIFEST, USER_AGENT, suffix_letter,
)

XPT_MAGIC = b"HEADER RECORD"

MANIFEST_FIELDS = ["cycle", "component", "filename", "url", "status",
                   "retrieved_utc", "http_status", "bytes", "sha256"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, timeout: int = 180) -> tuple[bytes | None, str]:
    """Fetch url. Returns (body or None, HTTP status / error name as text)."""
    req = u.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with u.urlopen(req, timeout=timeout) as r:
            return r.read(), str(r.status)
    except urllib.error.HTTPError as e:
        return None, str(e.code)
    except Exception as e:
        return None, type(e).__name__


def _is_xpt(data: bytes | None) -> bool:
    return bool(data) and data[:len(XPT_MAGIC)] == XPT_MAGIC


def _candidate_names(component: str, suffix: str) -> list[str]:
    """Materialise candidate filenames for a component in one cycle."""
    letter = suffix_letter(suffix)
    out = []
    for tmpl in COMPONENT_CANDIDATES[component]:
        if "{SUFX}" in tmpl and letter == "":
            # L13_{SUFX} is meaningless for 1999-2000; skip (bare LAB## handles it)
            continue
        out.append(tmpl.replace("{SUF}", suffix).replace("{SUFX}", letter))
    # de-dup preserving order
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq


def download_nhanes(rows: list[dict]) -> None:
    for cycle, (seg, suffix, startyr) in CYCLES.items():
        for component in COMPONENT_CANDIDATES:
            dest = RAW_NHANES / f"{component}_{cycle}.XPT"
            status, used_url, used_name = "missing", "", ""
            http, ts, nbytes, digest = "", "", "", ""
            if dest.exists() and dest.stat().st_size > 1000:
                status = "cached"
                rows.append(dict(cycle=cycle, component=component, filename=dest.name,
                                 url="(cached)", status=status, retrieved_utc="",
                                 http_status="", bytes=str(dest.stat().st_size),
                                 sha256=_sha256_file(dest)))
                continue
            for name in _candidate_names(component, suffix):
                url = NHANES_XPT_URL.format(startyr=startyr, fname=name)
                data, http = _get(url)
                ts = _utcnow()
                if _is_xpt(data):
                    dest.write_bytes(data)
                    status, used_url, used_name = "ok", url, name
                    nbytes, digest = str(len(data)), _sha256(data)
                    break
                used_url = url
                time.sleep(0.2)
            rows.append(dict(cycle=cycle, component=component,
                             filename=used_name or "(none)",
                             url=used_url if status == "ok" else "",
                             status=status, retrieved_utc=ts, http_status=http,
                             bytes=nbytes, sha256=digest))
            print(f"[NHANES] {cycle:10s} {component:8s} -> {status} {used_name}")


def download_mortality(rows: list[dict]) -> None:
    for cycle, (seg, suffix, startyr) in CYCLES.items():
        s, e = startyr, startyr + 1
        fname = MORT_FILE.format(s=s, e=e)
        url = MORT_BASE + fname
        dest = RAW_MORT / fname
        if dest.exists() and dest.stat().st_size > 1000:
            rows.append(dict(cycle=cycle, component="MORT", filename=fname,
                             url="(cached)", status="cached", retrieved_utc="",
                             http_status="", bytes=str(dest.stat().st_size),
                             sha256=_sha256_file(dest)))
            print(f"[MORT]   {cycle:10s} MORT     -> cached")
            continue
        data, http = _get(url)
        ts = _utcnow()
        nbytes, digest = "", ""
        if data and len(data) > 1000 and not data.lstrip().startswith(b"<"):
            dest.write_bytes(data)
            status = "ok"
            nbytes, digest = str(len(data)), _sha256(data)
        else:
            status = "missing"
        rows.append(dict(cycle=cycle, component="MORT", filename=fname,
                         url=url, status=status, retrieved_utc=ts,
                         http_status=http, bytes=nbytes, sha256=digest))
        print(f"[MORT]   {cycle:10s} MORT     -> {status}")


def main() -> None:
    rows: list[dict] = []
    download_nhanes(rows)
    download_mortality(rows)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] in ("ok", "cached") for r in rows)
    print(f"\nManifest -> {MANIFEST}  ({ok}/{len(rows)} files present)")


if __name__ == "__main__":
    main()
