#!/usr/bin/env python3
"""Extract per-model TÜV Mängelquoten from the AUTO BILD TÜV-Report 2026 PDF.

The report has a real text layer (zero image cost). Each MODEL page carries a summary
block titled "Laufleistung in km / ohne Mängel / geringe Mängel / erhebliche Mängel /
verkehrsunsicher" with one row per age band:

    2-3 Jahre  <km_typ> <km_avg>  <ohne_typ ohne_avg>  <gering_typ gering_avg>
               <erhebl_typ erhebl_avg>  <unsicher_typ unsicher_avg>

`erhebliche Mängel` (significant defects) is the Mängelquote carcache scores on. km use a
space thousands-separator; every rate is a comma-decimal, so the rates are exactly the
comma-numbers on the line, in the fixed order above — the 5th/6th are model/class-average
erhebliche Mängel.

Usage:
    extract_tuv.py <report.pdf> --json tuv_extracted.json      # all model pages
    extract_tuv.py <report.pdf>                                # human-readable dump
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pypdf

AGE_BANDS = ["2-3", "4-5", "6-7", "8-9", "10-11", "12-13"]
_AGE_ROW = re.compile(r"^\s*(\d+-\d+)\s+Jahre\b")
_COMMA_NUM = re.compile(r"\d+,\d")
# The PDF text layer sometimes splits a two-digit rate at the space: "10,1" -> "1 0,1",
# "26,1" -> "2 6,1", "53,1" -> "5 3,1", and "4,1" -> "4 ,1". Re-join a lone space-bounded
# digit onto the decimal that follows. Safe for this table: every data cell is a comma
# decimal or a space-thousands km value ("70 000"), so no genuine standalone integer can be
# swallowed (a km group is never followed by a comma, and the lookbehind requires the digit
# to be space-bounded, so "9,9 6,5" — two real decimals — is never merged).
_SPLIT_A = re.compile(r"(?<=\s)(\d) (\d,\d)")   # "1 0,1" -> "10,1" (lone digit, space-bounded)
_SPLIT_B = re.compile(r"(\d) ,(\d)")            # "74 ,1" / "4 ,1" -> "74,1" / "4,1"


def _normalize(line: str) -> str:
    line = _SPLIT_B.sub(r"\1,\2", line)
    line = _SPLIT_A.sub(r"\1\2", line)
    return line
# "Opel Astra (K)" / "Bauzeit 2015 bis 2021" style header lines on a model page.
_BAUZEIT = re.compile(r"Bauzeit\s+(\d{4})\s+bis\s+(\d{4}|heute)", re.I)


def _rate(tok: str) -> float:
    return float(tok.replace(",", "."))


def parse_page(text: str) -> dict | None:
    """Return {'erheblich': {band: (model_pct, class_avg)}, 'bauzeiten': [...], ...} for a
    model page, or None if the page has no Mängeleinteilung summary block."""
    if "erhebliche Mängel" not in text and "MÄNGELEINTEILUNG" not in text:
        return None
    erheblich: dict[str, tuple[float, float]] = {}
    ohne: dict[str, tuple[float, float]] = {}
    for raw in text.splitlines():
        m = _AGE_ROW.match(raw)
        if not m:
            continue
        line = _normalize(raw)
        nums = _COMMA_NUM.findall(line)
        # 4 categories x (model, avg) = 8 comma-decimals. Skip the component-detail table
        # further down the page (its age rows carry the model name, not 8 rate numbers).
        if len(nums) < 8:
            continue
        band = m.group(1)
        ohne.setdefault(band, (_rate(nums[0]), _rate(nums[1])))
        erheblich.setdefault(band, (_rate(nums[4]), _rate(nums[5])))
    if not erheblich:
        return None
    bauzeiten = [(int(a), (2026 if b.lower() == "heute" else int(b)))
                 for a, b in _BAUZEIT.findall(text)]
    return {"erheblich": erheblich, "ohne": ohne, "bauzeiten": bauzeiten}


def _title_guess(text: str) -> str:
    """Best-effort model name: the line just before 'Bauzeit …', else the AUTOBILD footer
    context. The report prints e.g. 'Opel Astra' as a section title near the page bottom."""
    # The model name appears as a standalone header like "Opel Astra" right before the
    # "Bauzeit" spec lines; grab the token sequence immediately preceding the first Bauzeit.
    idx = text.find("Bauzeit")
    if idx != -1:
        head = text[:idx].splitlines()
        for line in reversed(head):
            s = line.strip()
            # a plausible "Make Model" line: 2-4 capitalised words, no digits
            if 1 < len(s) < 40 and re.match(r"^[A-Za-zÄÖÜäöüß.\- ]+$", s) and " " in s:
                return s
    return ""


def extract(pdf_path: Path) -> list[dict]:
    reader = pypdf.PdfReader(str(pdf_path))
    if reader.is_encrypted:
        reader.decrypt("")
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        parsed = parse_page(text)
        if not parsed:
            continue
        out.append({
            "page": i,                       # 0-based reader index
            "title_guess": _title_guess(text),
            "bauzeiten": parsed["bauzeiten"],
            "erhebliche_maengel": {b: {"model": parsed["erheblich"][b][0],
                                        "class_avg": parsed["erheblich"][b][1]}
                                    for b in AGE_BANDS if b in parsed["erheblich"]},
            "text": text,                    # keep the verbatim page for later use
        })
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--json", type=Path, help="write full structured JSON here")
    args = ap.parse_args(argv)

    records = extract(args.pdf)
    print(f"found {len(records)} model summary pages", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        for r in records:
            bz = ", ".join(f"{a}-{b}" for a, b in r["bauzeiten"]) or "?"
            print(f"\np{r['page']:>3}  {r['title_guess'] or '?':<28} Bauzeit {bz}")
            for band, v in r["erhebliche_maengel"].items():
                print(f"       {band:>6}: {v['model']:5.1f}%  (Ø {v['class_avg']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
