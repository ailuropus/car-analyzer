#!/usr/bin/env python3
"""extract.py — turn a mobile.de listing PDF into a carcache record skeleton using the
PDF's TEXT layer. No vision model, no image budget.

This is an OPTIONAL helper (the carcache.py core stays stdlib-only). It needs pypdf:

    pip install pypdf

Usage:
    python3 extract.py inbox/455281247.pdf                 # print text + skeleton JSON
    python3 extract.py inbox/455281247.pdf --json out.json # also write the skeleton

Everything mobile.de states in a LABELLED ROW is parsed here: the Technische Daten table, the
page header (make/model), the dealer block, and the Ausstattung list. No field is left for a
human to fill — the record is a pure function of the PDF, so the same PDF always yields the
same record.

What this file deliberately does NOT parse is the seller's free-text description, which has no
structure to grep. That is review.py's job (an LLM call), and it writes only carcache.LLM_FIELDS.

    python3 extract.py inbox/<id>.pdf --json out.json
    python3 carcache.py add <id> --file out.json --pdf inbox/<id>.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import carcache
from carcache import SCHEMA_FIELDS, extract_ids_from_bytes


def read_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("extract.py needs pypdf:  pip install pypdf")
    reader = PdfReader(str(path))
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    # The text layer carries typographic LIGATURES — the font's single glyph for 'fi' comes back
    # as U+FB01, so 'Sindelfingen' arrives as 'Sindelﬁngen' and 'käuflich' as 'käuﬂich'. Stored
    # unfolded, that town then fails to geocode and the dealer name silently differs from the
    # same dealer parsed elsewhere. NFKC maps each ligature to its plain letters and leaves
    # umlauts alone (unlike NFKD, which would decompose 'ü' into 'u' + a combining mark).
    return unicodedata.normalize("NFKC", raw)


# The listing-id URL as TEXT. `extract_ids_from_bytes` scans raw bytes (free, stdlib) and catches
# mobile.de's own "Print" export, whose URL sits in the byte stream as literal ASCII. A browser
# "Print to PDF" of the same page instead glyph-encodes every character, so 'id=461739858' never
# appears as bytes — but pypdf reconstructs it from the font's ToUnicode map. Same pattern as
# carcache._ID_RE, matched against the extracted text as a fallback.
_ID_TEXT_RE = re.compile(r"(?:details\.html\?|printView\.html\?|[?&])id=(\d{6,})")


def resolve_id(pdf: Path, text_reader=read_text) -> str | None:
    """The mobile.de listing id for a PDF: raw-byte scan first (free), then the text layer as a
    fallback for browser-print PDFs whose URL is glyph-encoded rather than literal ASCII."""
    ids = extract_ids_from_bytes(Path(pdf).read_bytes())
    if ids:
        return ids[0]
    import collections
    counts = collections.Counter(_ID_TEXT_RE.findall(text_reader(pdf)))
    return counts.most_common(1)[0][0] if counts else None


def _num(text: str) -> int | None:
    return int(text.replace(".", "").replace(" ", "")) if text else None


def _search(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(group).strip() if m else None


def _fuel(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    if "diesel" in low:
        return "Diesel"
    if "hybrid" in low or "elektro" in low:
        return "PHEV" if "plug" in low else "Hybrid-Benzin"
    if "benzin" in low:
        return "Benzin"
    return raw


def _has(text: str, *needles: str) -> bool:
    """Whether any needle appears, tolerating the PDF's whitespace.

    The text layer runs equipment items together without separators, so a listing that reads
    'Laderaumboden variabel' on mobile.de arrives here as 'Laderaumbodenvariabel'. Matching a
    multi-word needle literally therefore silently missed real options. Whitespace in a needle
    means 'any amount, including none'."""
    low = text.lower()
    return any(re.search(re.escape(n.lower()).replace(r"\ ", r"\s*"), low) for n in needles)


def _price_rating(text: str) -> str | None:
    """mobile.de's own five-bucket price verdict. Order matters: 'sehr guter preis' contains
    'guter preis', so the more specific phrase must be tested first."""
    for phrase, val in (("sehr guter preis", "sehr_gut"), ("guter preis", "gut"),
                        ("fairer preis", "fair"), ("erhöhter preis", "erhoeht"),
                        ("hoher preis", "hoch")):
        if phrase in text.lower():
            return val
    return None


def _price(text: str) -> int | None:
    # the listing price recurs; take the most common "X.XXX €" amount
    amounts = re.findall(r"(\d{1,3}(?:\.\d{3})+)\s*€", text)
    if not amounts:
        return None
    from collections import Counter
    return _num(Counter(amounts).most_common(1)[0][0])


def _relevant(text: str) -> str:
    """Drop mobile.de's trailing 'similar vehicles' / recommendation blocks — their other-
    car feature and price mentions would otherwise leak into this listing's parse."""
    cut = len(text)
    for marker in ("Ähnliche Fahrzeuge", "Ähnliche Angebote", "Das könnte Sie auch",
                   "Weitere Angebote des Händlers", "Kunden interessierten sich"):
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


# Anhängerkupplung wording -> schema value. mobile.de prints exactly one of these in the
# Ausstattung list. `abnehmbar` (detachable) and `klappbar` (folding) both mean a hitch that
# can be got out of the way, which is what `fitted_swivel` represents for scoring.
_TOWBAR_WORDING = [
    (r"Anhängerkupplung\s+schwenkbar", "fitted_swivel"),
    (r"Anhängerkupplung\s+abnehmbar", "fitted_swivel"),
    (r"Anhängerkupplung\s+klappbar", "fitted_swivel"),
    (r"Anhängerkupplung\s+fest", "fitted_fixed"),
    (r"Anhängerkupplung\s+Vorbereitung", "prep"),
]


def _make_model(text: str) -> tuple[str | None, str | None]:
    """(make, model) from the print header — `7/14/26, 10:21 AM Ford Focus für 14.950 €`.

    mobile.de repeats this on every page, and it holds the CANONICAL make+model with no trim
    noise, which is what `model_reliability_basis` is keyed on ('ford focus'). The Anzeige title
    line is the marketing string ('Ford FocusTurnier ST-Line*NAVI*LED*KAM*ACC') and is useless
    for identity — it goes to `variant` instead."""
    m = re.search(r"\d+/\d+/\d+,.*?[AP]M\s+(.+?)\s+für\s+[\d.]+\s*€", text)
    if not m:
        return None, None
    parts = m.group(1).strip().split(None, 1)
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


def _unfallfrei_declared(text: str) -> bool | None:
    """The seller's Fahrzeugzustand tick-box, verbatim — NOT a judgement about the car.

    True  = the seller ticked 'Unfallfrei'.
    False = the seller declared damage ('Unfallschaden' / 'Reparierter Unfallschaden').
    None  = the seller stated nothing.

    This is only ever the DECLARATION. Whether the description contradicts it is a separate
    fact, found by review.py and carried as a finding; keeping the two apart is what lets the
    report show a seller who ticked the box and then described a dent."""
    zu = _search(r"Fahrzeugzustand\s+([^\n]+)", text)
    if not zu:
        return None
    if re.search(r"Unfallschaden|Unfallwagen", zu):
        return False
    if "Unfallfrei" in zu:
        return True
    return None


# The seller block appears MORE THAN ONCE per PDF, in variants that differ in ways no single
# anchor survives:
#
#   …Finanzierung berechnen     …Impressum & weitere Angaben     …
#   SMR-Auto seit 2001          SMR-Auto seit 2001Paindorf…35    Dirk Wiebeck      <- own line
#   DE-85293 Reichertshausen    DE-85293 Reichertshausen         DE-14482 Potsdam
#   Tel.: +49 …                 Zur Homepage des Händlers        Tel.: +49 …
#
# The postcode is the only reliable anchor. What follows it is not: some listings show no phone
# number at all. What precedes it is not either: the name may share the postcode's line or sit
# on the one above, and in the Impressum variant the street is glued on with no separator
# ('Auto JajehNeckarstr. 3' — no regex can find that seam).
#
# So: collect EVERY candidate, then pick the SHORTEST name. The bare name is always a prefix of
# the name+street variant, which makes "shortest" exactly "the one without an address" — and it
# is deterministic, which matters because `dealer` is part of the duplicate key and the same
# dealer must parse identically on every listing.
_SELLER = re.compile(r"([^\n]*)\n([^\n]*?)DE-\d{5}\s+([^\n]+?)\s*\n")

# The dealer's clean name in the page header, followed by its rating count and an "Über diesen
# Händler" link ('Anhängercenter Leonberg(34)Über diesen Händler'). This is the one place the name
# appears WITHOUT the street glued to it, and it survives the layout that defeats the postcode
# heuristic below — in a browser "Print to PDF" the seller block collapses to a single line with
# the address run onto the name ('Anhängercenter LeonbergBahnhofstraße 87DE-71229 Leonberg'), which
# has no seam a regex can find. Preferred when present; the postcode scan stays as the fallback and
# still supplies the town.
_SELLER_HEADER = re.compile(r"([^\n]+?)\(\d+\)Über diesen Händler")

# a line that is page furniture rather than a dealer name
_NOT_A_NAME = re.compile(r"^\s*$|^\d+\s*/|\d{1,2}/\d{1,2}/\d{2,4},|^https?:|^Tel\.|berechnen$|"
                         r"^Impressum|^Händler$|^Alle |anzeigen$")


def _seller(text: str) -> tuple[str | None, str | None]:
    """(dealer, town) — the dealer name and their town."""
    best: tuple[int, str] | None = None
    town = None
    for prev, same, raw_town in _SELLER.findall(text):
        # the name shares the postcode's line, or failing that sits on the line above
        name = same.strip() or ("" if _NOT_A_NAME.search(prev) else prev.strip())
        # mobile.de elides long names ('Autowelt Winkler + Schreib…'); the elision is
        # deterministic for a dealer, so it stays consistent across that dealer's listings.
        # Both spellings must go: read_text() normalises NFKC, which expands U+2026 to '...'.
        name = re.sub(r"\s*seit\s+\d{4}", "", name)
        name = re.sub(r"\s*(?:\.\.\.|…)\s*$", "", name).strip(" ,-")
        if not name or _NOT_A_NAME.search(name):
            continue
        if best is None or len(name) < best[0]:
            best = (len(name), name)
            # a town may be qualified ('Asperg/Ludwigsburg bei Stuttgart') — keep the primary
            town = re.split(r"\s*/\s*|\s+bei\s+", raw_town.strip())[0] or None
    # The header link carries the name street-free (see _SELLER_HEADER); prefer it when present,
    # but keep the postcode scan's town — the header has no town.
    header = _SELLER_HEADER.search(text)
    name = header.group(1).strip() if header else (best[1] if best else None)
    return name, town


def parse_listing(text: str) -> dict:
    """Parse every labelled row of a mobile.de listing text layer into a schema record."""
    text = _relevant(text)
    rec: dict = {f: None for f in SCHEMA_FIELDS}
    rec["notable_options"] = []

    rec["make"], rec["model"] = _make_model(text)
    # The trim line, verbatim from its own labelled row. NOT the marketing headline
    # ('Ford FocusTurnier ST-Line*NAVI*LED*KAM*ACC…'), which only some captures even contain.
    # Case is as the dealer typed it ('vignale', 'ST-LINE') — normalising would mangle 'L&K 4×4'.
    rec["variant"] = _search(r"Ausstattungslinie\s+([^\n]+)", text)
    rec["unfallfrei_declared"] = _unfallfrei_declared(text)
    rec["body"] = _search(r"Kategorie\s+(\w+)", text)
    rec["km"] = _num(_search(r"Kilometerstand\s+([\d\.]+)\s*km", text) or "")
    rec["power_ps"] = _num(_search(r"\((\d+)\s*PS\)", text) or "")
    rec["fuel"] = _fuel(_search(r"Kraftstoffart\s+([^\n]+)", text))
    rec["gearbox"] = _search(r"Getriebe\s+(\w+)", text)
    rec["displacement_cm3"] = _num(_search(r"Hubraum\s+([\d\.]+)\s*cm", text) or "")
    rec["first_reg"] = _search(r"Erstzulassung\s+(\d{2}/\d{4})", text)
    rec["owners"] = _num(_search(r"Fahrzeughalter\s+(\d+)", text) or "")
    rec["emission_class"] = (_search(r"Schadstoffklasse\s+(Euro\s?\S+)", text) or "").replace(" ", "") or None
    rec["tuv"] = _search(r"\bHU\s+(\d{2}/\d{4})", text) or ("Neu" if _has(text, "TÜV neu", "HU neu") else None)
    rec["climate"] = _search(r"Klimatisierung\s+([^\n]+)", text)
    rec["parking"] = _search(r"Einparkhilfe\s+([^\n]+)", text)

    comb = _search(r"Energieverbrauch\s*\(komb\.\)\s*([\d,]+|--)", text)
    rec["consumption_l100"] = float(comb.replace(",", ".")) if comb and comb != "--" else None
    co2 = _search(r"Emissionen\s*\(komb\.\)\s*([\d,]+|--)", text)
    rec["co2_gkm"] = _num(co2) if co2 and co2 != "--" else None

    rec["price_eur"] = _price(text)
    rec["price_rating"] = _price_rating(text)
    dr = _search(r"(\d,\d)\s*\(\d+\s*Bewertung", text)
    rec["dealer_rating"] = float(dr.replace(",", ".")) if dr else None

    rec["service_history"] = True if _has(text, "Scheckheftgepflegt", "Scheckheft") else None

    # equipment booleans (present => True, else left None = "not listed")
    rec["carplay"] = True if _has(text, "Apple CarPlay", "Android Auto") else None
    rec["navi"] = True if _has(text, "Navigationssystem", "Navi ") else None
    rec["heated_seats"] = True if _has(text, "Sitzheizung") else None
    rec["heated_wheel"] = True if _has(text, "Lenkradheizung", "beheiztes Lenkrad",
                                       "beheizbares Lenkrad", "Lenkrad heizbar") else None
    rec["standheizung"] = True if _has(text, "Standheizung") else None
    rec["variable_floor"] = True if _has(text, "Laderaumboden variabel", "variabler Ladeboden",
                                         "Laderaumboden verstellbar", "Laderaumboden doppelt",
                                         "doppelter Ladeboden") else None
    # ACC must be word-bounded (a bare "acc" substring matches unrelated words); a plain
    # "Abstandswarner" is a distance warner, NOT adaptive cruise.
    rec["adaptive_cruise"] = True if (_has(text, "Abstandstempomat", "Abstandsregeltempomat",
                                           "adaptiver Tempomat", "adaptive Geschwindig")
                                      or re.search(r"\bACC\b", text)) else None
    rec["cruise"] = True if _has(text, "Tempomat", "Geschwindigkeitsregel") else None
    rec["roof_window"] = True if _has(text, "Panorama-Dach", "Panoramadach", "Schiebedach",
                                      "Glasdach", "Glasschiebedach", "Panoramaglasdach",
                                      "Ausstelldach", "Panorama-Glasdach", "Glas-Schiebedach") else None
    # Blind-spot monitor. "Totwinkel" covers Totwinkel-Assistent/-Warner; the VW/Seat family
    # brands it "Side Assist"; Ford calls it BLIS / Blind Spot Information System.
    rec["blind_spot"] = True if (_has(text, "Totwinkel", "Toter Winkel", "Toter-Winkel",
                                      "Blind Spot", "Blind-Spot", "Side Assist", "Spurwechselassistent")
                                 or re.search(r"\bBLIS\b", text)) else None
    # Reverse/park camera. mobile.de puts it in the Einparkhilfe row ("Kamera") and dealers
    # write R-Cam/Rückfahrkamera/360°. A bare "Einparkhilfe" (sensors only) must NOT count.
    rec["rear_camera"] = True if (_has(text, "Rückfahrkamera", "Rückfahrcamera", "Rueckfahrkamera",
                                       "Kamera", "360°", "360 Grad", "Rundumsicht")
                                  or re.search(r"\bR[- ]?CAM\b", text, re.I)) else None
    rec["keyless"] = True if (_has(text, "Keyless", "Schlüssellose Zentralverriegelung",
                                   "schlüsselloser Zugang", "Komfortzugang", "Smart Entry")
                              or re.search(r"\bKESSY\b", text, re.I)) else None

    # Towbar: the wording states the type outright, so there is nothing to judge. A hitch that
    # is absent from the Ausstattung list stays None ("not listed"), never False.
    for pattern, value in _TOWBAR_WORDING:
        if re.search(pattern, text):
            rec["towbar"] = value
            break
    else:
        # `Anhängevorrichtung` is a hitch by another name. `Anhängelast` is NOT — it is the
        # towing-capacity spec row, present on cars with no hitch at all.
        if _has(text, "Anhängerkupplung", "Anhängevorrichtung") or re.search(r"\bAHK\b", text):
            rec["towbar"] = "capable"   # fitted, type unstated

    rec["towing_braked_kg"] = _num(_search(r"Anhängelast\s+gebremst\s+([\d\.]+)\s*kg", text) or "")

    # `Garantie` is a bare Ausstattung tick-box — mobile.de states no duration or terms, so this
    # is a boolean, not a string. The old string values ("36 Monate Garantie") were read out of
    # the seller's prose by a vision model; terms live in the description, which is review.py's
    # half of the PDF and reaches the record as a `warranty` finding.
    rec["warranty"] = True if re.search(r"\n\s*Garantie\s*\n", text) else None

    rec["dealer"], rec["location"] = _seller(text)
    return rec


def batch_inbox(store: dict, inbox_dir, out_dir, force: bool = False, text_reader=read_text):
    """Write a skeleton <id>.json for every NEW PDF in inbox/ (cached ids skipped).
    Returns (made, skipped, unknown)."""
    made, skipped, unknown = [], [], []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for pdf in sorted(Path(inbox_dir).glob("*.pdf")):
        cid = resolve_id(pdf, text_reader)
        if not cid:
            unknown.append(pdf)
            continue
        if cid in store and not force:
            skipped.append(cid)
            continue
        skeleton = parse_listing(text_reader(pdf))
        out_path = Path(out_dir) / f"{cid}.json"
        out_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
        made.append((cid, pdf, out_path))
    return made, skipped, unknown


def reprocess_store(store: dict, processed_dir, ids=None, do_all: bool = False, text_reader=read_text):
    """Rebuild PARSED_FIELDS from each car's archived PDF (LLM_FIELDS preserved).
    Selection: given ids, else every extract-stale car, else (--all) every car with a PDF.
    Returns [(id, status, diffs)]; the caller saves the store."""
    if ids:
        targets = [str(i) for i in ids]
    elif do_all:
        targets = list(store.keys())
    else:
        targets = [cid for cid, car in store.items() if carcache.is_extract_stale(car)]

    results = []
    for cid in targets:
        if cid not in store:
            results.append((cid, "not-in-store", []))
            continue
        pdf = Path(processed_dir) / f"{cid}.pdf"
        if not pdf.exists():
            results.append((cid, "missing-pdf", []))
            continue
        parsed = parse_listing(text_reader(pdf))
        _, diffs = carcache.merge_reprocessed(store[cid], parsed)
        results.append((cid, "reprocessed", diffs))
    return results


def verify_store(store: dict, processed_dir, ids=None, text_reader=read_text):
    """Independent audit: for each car with an archived PDF, (a) re-parse and compare stored
    PARSED_FIELDS to the fresh parse, and (b) check core fields (km / power / first-reg /
    declared accident status) against SEPARATE regexes — logic independent of parse_listing.
    Returns [(id, status, issues)] with issues = [(field, stored, source)]."""
    targets = [str(i) for i in ids] if ids else list(store.keys())
    results = []
    for cid in targets:
        if cid not in store:
            results.append((cid, "not-in-store", []))
            continue
        pdf = Path(processed_dir) / f"{cid}.pdf"
        if not pdf.exists():
            results.append((cid, "missing-pdf", []))
            continue
        rec = store[cid]
        text = text_reader(pdf)
        rel = _relevant(text)
        parsed = parse_listing(text)
        issues = []
        # (a) stored parsed fields should equal a fresh parse
        for f in carcache.PARSED_FIELDS:
            pv = parsed.get(f)
            if pv is not None and rec.get(f) != pv:
                issues.append((f, rec.get(f), pv))
        # (b) independent ground-truth regexes (different code path than parse_listing)
        def gt(pat, cast=str):
            m = re.search(pat, rel)
            return cast(m.group(1)) if m else None
        for f, val in {
            "km": gt(r"Kilometerstand\s+([\d.]+)\s*km", lambda s: int(s.replace(".", ""))),
            "power_ps": gt(r"\((\d+)\s*PS\)", int),
            "first_reg": gt(r"Erstzulassung\s+(\d{2}/\d{4})"),
        }.items():
            if val is not None and rec.get(f) != val:
                issues.append((f + " [indep]", rec.get(f), val))
        # (c) cross-check the DECLARATION via a different code path than _unfallfrei_declared().
        # Scoped to the Fahrzeugzustand row on purpose: searching the whole document for
        # 'unfallfrei' would also hit the seller's prose, conflating the tick-box with what the
        # description says about the car — two separate facts this schema keeps apart.
        gt_unf = (False if re.search(r"Fahrzeugzustand[^\n]*Unfall(?:schaden|wagen)", rel)
                  else (True if re.search(r"Fahrzeugzustand[^\n]*Unfallfrei", rel) else None))
        if rec.get("unfallfrei_declared") != gt_unf:
            issues.append(("unfallfrei_declared [indep]", rec.get("unfallfrei_declared"), gt_unf))
        results.append((cid, "ok" if not issues else "MISMATCH", issues))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract carcache record skeletons from mobile.de PDFs (text layer).")
    ap.add_argument("pdf", nargs="?", help="a single PDF to extract")
    ap.add_argument("--inbox", action="store_true", help="extract every NEW PDF in inbox/ to <id>.json")
    ap.add_argument("--reprocess", action="store_true", help="rebuild stored records from processed/ PDFs")
    ap.add_argument("--verify", action="store_true", help="audit stored records against their archived PDFs")
    ap.add_argument("--all", action="store_true", help="with --reprocess: every car, not just extract-stale")
    ap.add_argument("--out-dir", default=str(carcache.SKELETON_DIR),
                    help="where --inbox writes skeletons (default: skeletons/, gitignored scratch)")
    ap.add_argument("--json", help="single-PDF mode: write the skeleton JSON here")
    ap.add_argument("--quiet", action="store_true", help="don't print the raw text")
    ap.add_argument("--force", action="store_true", help="with --inbox: include already-cached ids")
    ap.add_argument("--ids", nargs="*", help="with --reprocess: restrict to these ids")
    ap.add_argument("--store", default=str(carcache.DEFAULT_STORE), help="path to cars.json")
    args = ap.parse_args(argv)

    if args.verify:
        store = carcache.load_store(args.store)
        results = verify_store(store, carcache.PROCESSED_DIR, ids=args.ids)
        ok = mism = nopdf = 0
        for cid, status, issues in results:
            if status == "ok":
                ok += 1
            elif status == "missing-pdf":
                nopdf += 1
            else:
                mism += 1
                print(f"MISMATCH {cid}  {store[cid].get('model', '')[:40]}")
                for field, stored, src in issues:
                    print(f"    {field}: stored={stored!r}  source={src!r}")
        print(f"\n{ok} ok · {mism} mismatch · {nopdf} without PDF  (of {len(results)} checked)")
        return 1 if mism else 0

    if args.reprocess:
        store = carcache.load_store(args.store)
        results = reprocess_store(store, carcache.PROCESSED_DIR, ids=args.ids, do_all=args.all)
        changed = 0
        for cid, status, diffs in results:
            if status != "reprocessed":
                print(f"{cid}: {status}", file=sys.stderr)
            elif diffs:
                changed += 1
                print(f"{cid}: {len(diffs)} field(s) updated")
                for field, old, new in diffs:
                    print(f"    {field}: {old!r} -> {new!r}")
            else:
                print(f"{cid}: no change (re-stamped v{carcache.EXTRACT_VERSION})")
        carcache.save_store(store, args.store)
        done = sum(1 for _, s, _ in results if s == "reprocessed")
        print(f"\nreprocessed {done} car(s), {changed} with changes, at extract v{carcache.EXTRACT_VERSION}")
        return 0

    if args.inbox:
        store = carcache.load_store(args.store)
        made, skipped, unknown = batch_inbox(store, carcache.INBOX_DIR, args.out_dir, force=args.force)
        for cid, pdf, out_path in made:
            print(f"NEW    {cid}  {pdf.name}  ->  {out_path}")
        for cid in skipped:
            print(f"cached {cid}  already stored — skipped (--force to re-extract)")
        for pdf in unknown:
            print(f"?      {pdf.name}  no id found")
        print(f"\n{len(made)} skeleton(s) written · {len(skipped)} cached · {len(unknown)} unknown")
        if made:
            print("\nAdd them, then run the LLM review to read the sellers' descriptions:")
            for cid, pdf, out_path in made:
                print(f"  python3 carcache.py add {cid} --file {out_path} --pdf {pdf}")
            print("  .venv/bin/python review.py --store")
        return 0

    if not args.pdf:
        ap.error("give a PDF path, or use --inbox / --reprocess")

    path = Path(args.pdf)
    text = read_text(path)
    cid = resolve_id(path, lambda _p: text)
    skeleton = parse_listing(text)
    if not args.quiet:
        print("=" * 72)
        print(text)
        print("=" * 72)
    print(f"\n# mobile.de id: {cid or 'NOT FOUND'}")
    print(json.dumps(skeleton, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote skeleton -> {args.json}  (review it, then `carcache.py add`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
