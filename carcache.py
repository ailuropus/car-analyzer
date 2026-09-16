#!/usr/bin/env python3
"""carcache — a tiny local cache + ranking tool for used-car listings from mobile.de.

Reading a listing PDF with a vision model is expensive, but the extracted facts are
tiny and stable. So we extract each listing ONCE, store it in cars.json keyed by the
mobile.de listing id, and run every comparison / ranking / filter off that JSON.

    python3 carcache.py add 459081695 --file listing.json
    python3 carcache.py list
    python3 carcache.py filter --towbar fitted_fixed --max-km 110000 --adaptive_cruise true
    python3 carcache.py rank
    python3 carcache.py report

Python 3.11+, standard library only. See README.md for the full extraction workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------------
# Paths & schema
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STORE = BASE_DIR / "cars.json"
DEFAULT_PROFILE = BASE_DIR / "profile.json"
DEFAULT_TEXTS = BASE_DIR / "report_text.json"
DEFAULT_REPORT = BASE_DIR / "report.md"
INBOX_DIR = BASE_DIR / "inbox"        # drop new listing PDFs here
PROCESSED_DIR = BASE_DIR / "processed"  # PDFs are moved here once handled
SKELETON_DIR = BASE_DIR / "skeletons"  # scratch <id>.json extraction skeletons (gitignored);
                                       # `add` reads them into cars.json, then they're disposable

# mobile.de listing URL + a browser-ish header set for best-effort availability checks.
LISTING_URL_TMPL = "https://suchen.mobile.de/fahrzeuge/details.html?id={id}"
HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml",
}
# Text that confidently means the listing is gone (mobile.de German phrasings).
REMOVAL_MARKERS = (
    "nicht mehr verfügbar", "nicht mehr verfuegbar", "angebot nicht gefunden",
    "fahrzeug wurde verkauft", "inserat wurde entfernt", "no longer available",
    "not found", "wurde bereits verkauft",
)
# Accepted words for a manual / browser availability decision.
AVAILABILITY_TRUTH: dict[str, bool | None] = {
    "true": True, "active": True, "live": True, "yes": True,
    "false": False, "inactive": False, "sold": False, "no": False,
    "unknown": None, "?": None,
}

# Bump when the extraction logic or record SCHEMA changes -> `reprocess` rebuilds the
# stored fields from the archived PDFs. Bump SCORING_VERSION when the derivation / scoring /
# TCO logic changes -> `recompute` refreshes the stored score snapshots.
EXTRACT_VERSION = 4
# v11: cost's third component is the CASH running cost (no depreciation — the asking price
# already has its own third, and depreciation is derived from it) on an absolute curve
# (cost.running) instead of min-max across the ranked set. Every criterion is now
# candidate-set independent.
# v12: the fuel component prefers MEASURED real-world consumption (profile.model_consumption,
# from spritmonitor.de) over the listing's WLTP figure, which is optimistic by a
# model-dependent amount rather than a constant.
# v13: is_phev() also reads the `variant` field (plug-in badges like GTE / e-Hybrid live
# there, not in `model`) and the keyword list gained gte/e-tron/recharge — fixes plug-ins
# that were mis-scored as regular hybrids (VW Passat GTE, Seat Leon FR e-Hybrid).
SCORING_VERSION = 13

# Record fields (the id is the dict key, so it is not stored inside the record).
SCHEMA_FIELDS: list[str] = [
    "model", "make", "variant", "body", "price_eur", "price_rating", "first_reg", "km", "owners",
    "power_ps", "fuel", "gearbox", "displacement_cm3", "consumption_l100", "co2_gkm",
    "emission_class", "unfallfrei_declared", "tuv", "service_history", "climate",
    "adaptive_cruise", "cruise", "carplay", "navi", "heated_seats", "heated_wheel",
    "standheizung", "variable_floor", "roof_window", "blind_spot", "rear_camera", "keyless",
    "towbar", "towing_braked_kg", "parking", "warranty",
    "dealer", "dealer_rating", "location", "travel_minutes", "travel_km", "travel_mode",
    "notable_options", "options_from_prose", "findings", "review_summary",
    "reviewed_by", "reviewed_at", "reviewed_source", "review_prompt_sha",
    "source_pdf", "superseded_by",
    "fetched_date", "listing_url", "active", "last_checked", "extract_version",
]
LIST_FIELDS = {"notable_options", "options_from_prose", "findings"}

# Field ownership, split by WHERE THE FACT LIVES in the PDF rather than by what it is about.
#
# PARSED_FIELDS come from labelled rows in the text layer (the Technische Daten table, the page
# header, the dealer block, the Ausstattung list). A regex owns them: deterministic, free, and
# reproducible. `reprocess` rebuilds them from the archived PDF.
#
# LLM_FIELDS come from the seller's free-text description, which has no structure to grep. Only
# review.py writes them, and `reprocess` preserves them.
#
# The two sets are disjoint and exhaustive over the fact-bearing schema; everything else is
# managed meta (source_pdf, availability, versions, the `computed` snapshot).
PARSED_FIELDS = [
    "make", "model", "variant", "body", "price_eur", "price_rating", "first_reg", "km", "owners",
    "power_ps", "fuel", "gearbox", "displacement_cm3", "consumption_l100", "co2_gkm",
    "emission_class", "tuv", "unfallfrei_declared", "climate", "dealer", "dealer_rating",
    "location", "service_history", "warranty", "towbar", "towing_braked_kg",
    "parking", "adaptive_cruise", "cruise", "carplay", "navi", "heated_seats", "heated_wheel",
    "standheizung", "variable_floor", "roof_window", "blind_spot", "rear_camera", "keyless",
]
LLM_FIELDS = [
    "findings", "review_summary", "notable_options", "options_from_prose",
    "reviewed_by", "reviewed_at", "reviewed_source", "review_prompt_sha",
]

# Human-friendly criterion labels, in weighting order.
CRITERIA: list[str] = [
    "reliability", "condition", "equipment_fit",
    "autobahn", "cost", "mileage", "location",
]
CRIT_ABBR = {
    "reliability": "rel", "condition": "cnd",
    "equipment_fit": "eqp", "autobahn": "aut", "cost": "cst", "mileage": "mil",
    "location": "loc",
}
# Full-name headers used in the reports (the terminal tables keep the short abbreviations).
CRIT_LABEL = {
    "reliability": "Reliability", "condition": "Condition",
    "equipment_fit": "Equipment", "autobahn": "Power",
    "cost": "Cost", "mileage": "Mileage", "location": "Location",
}
# Wishlist options shown in the equipment-detail matrix, with display labels.
OPTION_LABELS = {
    "adaptive_cruise": "Adaptive cruise", "towbar": "Towbar", "standheizung": "Standheizung",
    "variable_floor": "Variable floor", "roof_window": "Roof / pano", "carplay": "CarPlay",
    "heated_seats": "Heated seats", "navi": "Navi", "heated_wheel": "Heated wheel",
    "blind_spot": "Blind spot", "rear_camera": "Rear camera", "keyless": "Keyless",
}


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------

def load_store(path: Path | str = DEFAULT_STORE) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_store(store: dict[str, dict], path: Path | str = DEFAULT_STORE) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_profile(path: Path | str = DEFAULT_PROFILE) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_texts(path: Path | str = DEFAULT_TEXTS) -> dict[str, Any]:
    """All static report prose, so it can be edited without touching Python. A missing file
    is not fatal — every lookup falls back to a built-in default."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def text_at(texts: dict, dotted: str, default: str = "") -> str:
    """Fetch a dotted key ('sections.ranking.heading') from the texts config, else default."""
    cur: Any = texts
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if isinstance(cur, str) else default


def fill_text(template: str, ctx: dict) -> str:
    """Substitute $placeholders from ctx. Unknown ones are left alone rather than raising,
    so a typo in the text config degrades gracefully instead of breaking the report."""
    from string import Template
    return Template(str(template)).safe_substitute(ctx)


def fmt_weight(w: float) -> str:
    """Weights are shown as percentages everywhere on the page: 0.2 -> '20%'.

    Rounded to one decimal, trailing zeros stripped: a derived share like 1/3 must read '33.3%',
    not '33.3333%'."""
    return f"{round(float(w) * 100, 1):g}%"


# --------------------------------------------------------------------------------------
# Versioning: detect when stored data / scores drift from the current logic + profile
# --------------------------------------------------------------------------------------

def profile_fingerprint(profile: dict) -> str:
    """Short hash of the scoring-relevant profile. Changes whenever you edit the search
    parameters or scoring config, so stored score snapshots can be flagged stale."""
    import hashlib

    keys = ["weights", "wishlist", "equipment", "mileage", "location",
            "reliability_model", "model_reliability_basis", "condition",
            "autobahn", "cost", "tco", "model_consumption", "defaults"]
    payload = {k: profile.get(k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def is_extract_stale(car: dict) -> bool:
    """True when the record's stored fields predate the current extraction logic/schema."""
    ev = car.get("extract_version")
    return ev is None or ev < EXTRACT_VERSION


def is_score_stale(car: dict, profile: dict) -> bool:
    """True when the stored score snapshot was computed under different logic or profile."""
    comp = car.get("computed") or {}
    return (not comp
            or comp.get("scoring_version") != SCORING_VERSION
            or comp.get("profile_fingerprint") != profile_fingerprint(profile))


def merge_reprocessed(record: dict, parsed: dict) -> tuple[dict, list[tuple[str, Any, Any]]]:
    """Overwrite PARSED_FIELDS from a fresh PDF parse, preserve LLM + meta fields, stamp the
    current EXTRACT_VERSION. Returns the record and a list of (field, old, new) changes.

    The parser is AUTHORITATIVE over its own fields, including when it yields None. A None from
    the parser is a real answer — 'the seller did not list this' — not a gap to be backfilled
    from whatever the record happened to hold. Keeping a stale non-null value here would break
    the guarantee the whole split exists to provide: that a record is a pure function of its
    PDF, so the same PDF always produces the same record."""
    diffs: list[tuple[str, Any, Any]] = []
    for field in PARSED_FIELDS:
        new = parsed.get(field)
        if record.get(field) != new:
            diffs.append((field, record.get(field), new))
            record[field] = new
    record["extract_version"] = EXTRACT_VERSION
    return record, diffs


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reg_year(first_reg: str | None) -> int | None:
    """'MM/YYYY' -> 2018."""
    if not first_reg:
        return None
    try:
        return int(str(first_reg).split("/")[-1])
    except ValueError:
        return None


def days_old(fetched_date: str | None, today: date | None = None) -> int | None:
    if not fetched_date:
        return None
    today = today or date.today()
    try:
        d = datetime.strptime(str(fetched_date), "%Y-%m-%d").date()
    except ValueError:
        return None
    return (today - d).days


def _fold(s: str) -> str:
    """Lowercase and strip diacritics, so 'Mégane' and 'Megane' are the same key.

    Every map in profile.json (reliability basis, autobahn, TCO) resolves by substring, and a
    miss is SILENT — the car just takes the fallback. mobile.de's page header writes 'Megane'
    while the basis was keyed 'renault mégane', so a car with real TÜV/ADAC data scored the 5.0
    fallback and nothing said so."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower())
                   if not unicodedata.combining(c))


def _hay(car: dict) -> str:
    return _fold(f"{car.get('make', '')} {car.get('model', '')}")


def lookup_value(car: dict, table: dict[str, Any], default: float) -> float:
    """Resolve a per make/model value. The most specific key (by length) whose every
    token appears in 'make model' wins, so 'opel insignia' beats a bare 'opel'."""
    hay = _hay(car)
    best_val: Any = None
    best_len = -1
    for key, val in table.items():
        k = _fold(key)
        if all(tok in hay for tok in k.split()) and len(k) > best_len:
            best_val, best_len = val, len(k)
    return float(best_val) if best_val is not None else float(default)


# --------------------------------------------------------------------------------------
# Total cost of ownership (annual) — ADAC-style component model:
#   fuel + Kfz-Steuer (computed) + maintenance + insurance + depreciation + HU
# --------------------------------------------------------------------------------------

def car_age_years(car: dict, today: date | None = None) -> int | None:
    yr = reg_year(car.get("first_reg"))
    return None if yr is None else max(0, (today or date.today()).year - yr)


def car_segment(car: dict, profile: dict) -> str:
    """'mid' for larger estates (Insignia/Mondeo/Optima/Passat/Superb), else 'compact'."""
    hay = _hay(car)
    for kw in profile.get("segments", {}).get("mid_keywords", []):
        if str(kw).lower() in hay:
            return "mid"
    return "compact"


def real_consumption(car: dict, profile: dict) -> dict | None:
    """The crowd-sourced REAL-WORLD consumption entry for this car's model+engine from
    profile['model_consumption'], or None when nothing matches.

    Why this outranks the listing's own figure: the stated number is WLTP, which is
    systematically optimistic AND optimistic by a model-dependent amount (median +26% here,
    range +2%..+49%). A constant bias would cancel out in a ranking; a varying one does not —
    it made thirsty engines look closer to frugal ones than they are.

    Matching: every token of `match` must appear in the car's make+model, `fuel` must equal the
    record's fuel, the DRIVETRAIN must agree (an entry's `phev` flag vs `is_phev(car)`), and the
    power must be within `ps_tolerance`. Closest power wins, then the largest sample.

    The drivetrain check is not pedantry: "Hybrid-Benzin" covers both a 48V mild hybrid and a
    plug-in, whose real consumptions differ by ~2x (a Leon e-Hybrid logs ~3.5 l/100km, a mild
    hybrid ~6.2). Matching on fuel alone once assigned the mild-hybrid figure to plug-in cars."""
    cfg = profile.get("model_consumption", {})
    entries = cfg.get("entries") or []
    tol = float(cfg.get("ps_tolerance", 8))
    hay = _hay(car)
    ps = car.get("power_ps")
    fuel = (car.get("fuel") or "").lower()
    car_phev = is_phev(car, profile)
    best: tuple[float, int, dict] | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        key = _fold(str(e.get("match", "")))
        if not key or not all(tok in hay for tok in key.split()):
            continue
        efuel = str(e.get("fuel", "")).lower()
        if efuel and efuel != fuel:
            continue
        if bool(e.get("phev", False)) != car_phev:
            continue
        eps = e.get("ps")
        if ps is None or eps is None:
            dist = float(tol)
        else:
            dist = abs(float(ps) - float(eps))
            if dist > tol:
                continue
        cand = (dist, -int(e.get("n", 0)), e)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[2] if best else None


def estimate_consumption(car: dict, tco_cfg: dict) -> tuple[float, bool]:
    """Return (l/100km, estimated?). Prefer the stated figure, fall back to CO2, then default.
    Real-world data, when configured, is applied earlier by energy_cost()."""
    cons = car.get("consumption_l100")
    if cons is not None:
        return float(cons), False
    co2 = car.get("co2_gkm")
    if co2 is not None:
        return round(float(co2) / 23.2, 2), True  # ~2320 g CO2 per litre petrol
    return float(tco_cfg.get("default_consumption_l100", 6.5)), True


def is_phev(car: dict, profile: dict) -> bool:
    """True for a PLUG-IN hybrid. Its WLTP l/100km is a blended figure that assumes mains
    charging and excludes the electricity, so it must not be used as the running cost.
    Full/mild hybrids (no plug) are excluded — their stated figure is real petrol burn."""
    cfg = profile.get("tco", {}).get("phev", {})
    if "hybrid" not in (car.get("fuel") or "").lower():
        return False
    cons = car.get("consumption_l100")
    if cons is not None and float(cons) <= float(cfg.get("detect_max_wltp_l100", 3.0)):
        return True
    # The plug-in badge usually lives in `variant` (GTE, e-Hybrid, PHEV), not `model`.
    hay = f"{car.get('model', '')} {car.get('variant', '')} " \
          f"{' '.join(car.get('notable_options') or [])}".lower()
    return any(str(k).lower() in hay for k in cfg.get("keywords", []))


def energy_cost(car: dict, profile: dict) -> tuple[float, float, bool]:
    """Annual (petrol EUR, electricity EUR, estimated?). A plug-in hybrid runs electric_share
    of its km on grid electricity and the rest on the engine; everything else burns petrol at
    its real-world / stated / estimated consumption.

    Preference order for the litres: measured real-world (`model_consumption`) > the listing's
    stated WLTP > CO2-derived > default. `estimated` stays False for a real-world figure — it
    is a measurement, just not this listing's own."""
    tco_cfg = profile.get("tco", {})
    km = float(tco_cfg.get("annual_km", 6000))
    price_l = float(tco_cfg.get("fuel_price_eur_l", 1.85))
    real = real_consumption(car, profile)

    if is_phev(car, profile):
        cfg = tco_cfg.get("phev", {})
        share = clamp(float(cfg.get("electric_share", 0.6)), 0.0, 1.0)
        elec = (km * share / 100.0) * float(cfg.get("kwh_per_100km", 18.0)) \
            * float(cfg.get("electricity_price_eur_kwh", 0.35))
        if real is not None:
            # Drivers log every fill-up, so a PHEV's figure is its REAL petrol burn across all
            # km (electric ones included) — it replaces the modelled engine consumption whole,
            # and must NOT be scaled by (1 - share) again. Electricity is still modelled.
            petrol = float(real["real_l100"]) / 100.0 * km * price_l
            return petrol, elec, False
        petrol = (km * (1 - share) / 100.0) * float(cfg.get("petrol_l100_engine", 6.8)) * price_l
        return petrol, elec, True  # the engine-only consumption is a model, not a stated figure

    if real is not None:
        return float(real["real_l100"]) / 100.0 * km * price_l, 0.0, False

    cons, estimated = estimate_consumption(car, tco_cfg)
    return cons / 100.0 * km * price_l, 0.0, estimated


def _co2_tax_tiered(co2: float) -> float:
    """CO2 tax component for cars first registered from 2021 (staggered rates, §8 KraftStG)."""
    if co2 <= 95:
        return 0.0
    if co2 <= 115:
        return (co2 - 95) * 2.00
    if co2 <= 135:
        return 40 + (co2 - 115) * 2.20
    if co2 <= 155:
        return 84 + (co2 - 135) * 2.50
    if co2 <= 175:
        return 134 + (co2 - 155) * 2.90
    if co2 <= 195:
        return 192 + (co2 - 175) * 3.40
    return 260 + (co2 - 195) * 4.00


def kfz_steuer(car: dict) -> float:
    """German annual vehicle tax, computed from displacement + CO2 + fuel + registration year."""
    import math
    disp = car.get("displacement_cm3") or 0
    is_diesel = "diesel" in (car.get("fuel") or "").lower()
    disp_tax = (math.ceil(disp / 100) if disp else 0) * (9.5 if is_diesel else 2.0)

    co2 = car.get("co2_gkm")
    if co2 is None:  # estimate from consumption when CO2 isn't stated
        cons = car.get("consumption_l100")
        co2 = cons * (26.5 if is_diesel else 23.3) if cons is not None else 0.0
    co2 = float(co2)

    year = reg_year(car.get("first_reg")) or 2015
    if year >= 2021:
        co2_tax = _co2_tax_tiered(co2)
    else:
        free = 95 if year >= 2014 else (110 if year >= 2012 else 120)
        co2_tax = max(0.0, co2 - free) * 2.0
    return round(disp_tax + co2_tax, 2)


def maintenance_cost(car: dict, profile: dict) -> float:
    cfg = profile["tco"]["maintenance"]
    base = float(cfg.get(car_segment(car, profile), cfg.get("compact", 600)))
    age = car_age_years(car) or 0
    uplift = min(float(cfg.get("age_uplift_cap", 0.5)), age * float(cfg.get("age_uplift_per_year", 0.05)))
    return round(base * (1 + uplift), 2)


def insurance_cost(car: dict, profile: dict) -> float:
    cfg = profile["tco"]["insurance"]
    base = float(cfg.get(car_segment(car, profile), cfg.get("compact", 550)))
    ps = car.get("power_ps") or 0
    if ps > 220:
        base += float(cfg.get("ps_over_220", 350))
    elif ps > 180:
        base += float(cfg.get("ps_over_180", 150))
    return round(base, 2)


def depreciation_cost(car: dict, profile: dict) -> float:
    """Estimated annual value loss over the hold period, from the current price. Older and
    stronger-resale cars retain more; the biggest TCO component for most cars."""
    cfg = profile["tco"]["depreciation"]
    price = car.get("price_eur")
    if not price:
        return 0.0
    hold = int(profile["tco"].get("hold_years", 5))
    age = car_age_years(car) or 5
    retention = float(cfg.get("base_retention", 0.86)) + age * float(cfg.get("age_bonus_per_year", 0.004))
    retention += lookup_value(car, cfg.get("brand_retention_adj", {}), 0.0)
    retention = min(float(cfg.get("retention_cap", 0.90)), max(0.5, retention))
    return round((price - price * (retention ** hold)) / hold, 2)


def annual_tco(car: dict, profile: dict) -> tuple[float, dict, bool]:
    """Annual TCO + a component breakdown. estimated=True when the energy cost had to be
    modelled (no stated consumption, or a plug-in hybrid's engine-only figure)."""
    tco_cfg = profile.get("tco", {})
    petrol, elec, estimated = energy_cost(car, profile)
    parts = {
        "fuel": round(petrol),
        "tax": round(kfz_steuer(car)),
        "maintenance": round(maintenance_cost(car, profile)),
        "insurance": round(insurance_cost(car, profile)),
        "depreciation": round(depreciation_cost(car, profile)),
        "hu": round(float(tco_cfg.get("hu_per_year", 65))),
    }
    if elec > 0:
        parts["electricity"] = round(elec)
    return round(float(sum(parts.values())), 2), parts, estimated


# --------------------------------------------------------------------------------------
# Data-derived criteria
# --------------------------------------------------------------------------------------

# Severities review.py may emit. `note` is context, never a penalty — see review_prompt.md.
CONDITION_SEVERITIES = ("major", "minor", "note")

# Finding categories that describe the physical state of THIS car, and so belong in `condition`.
# `inconsistency` is deliberately absent: it reports a contradiction in the ADVERT (the text says
# 204 PS, the data table says 150), which is a data-quality observation about the seller's
# paperwork. A typo does not make the car worse, so an inconsistency finding is surfaced in the
# report and scored zero.
CONDITION_CATEGORIES = ("accident", "mechanical", "corrosion", "wear",
                        "service_history", "usage", "warranty", "other")


def count_findings(car: dict) -> tuple[int, int]:
    """(majors, minors) among `findings` that the condition penalty should charge for.

    Two kinds of finding are skipped:

    * anything outside CONDITION_CATEGORIES (i.e. `inconsistency`) — not about the car's state.
    * `accident` findings on a car whose seller DECLARED the accident. The declaration already
      carries the full `accident_confirmed` penalty, which is the heavier of the two, so also
      charging the finding would bill the same crash twice. When the seller declared the car
      accident-free, prose damage is NOT suppressed — it is graded on its own merits by the
      reviewer (cosmetic -> `minor`, serious -> `major`), which is the only way such damage
      can reach the score at all.

    `note` never counts. There is no keyword fallback: severity is stated, never guessed."""
    declared_accident = car.get("unfallfrei_declared") is False
    majors = minors = 0
    for f in car.get("findings") or []:
        if not isinstance(f, dict):
            continue
        if f.get("category") not in CONDITION_CATEGORIES:
            continue
        if f.get("category") == "accident" and declared_accident:
            continue
        if f.get("severity") == "major":
            majors += 1
        elif f.get("severity") == "minor":
            minors += 1
    return majors, minors


def condition(car: dict, profile: dict) -> float:
    """Single 0..10 score from current reported condition + history. Starts at 10 and
    subtracts for accident status, extra owners, missing service/TÜV, and reported defects:
    a major disclosed mechanical defect (leak, head gasket, rust-through, engine/gearbox
    damage) is penalised heavily so e.g. a car with reported coolant leakage cannot score
    high; minor cautions are small and capped; spec/neutral notes don't penalise.

    Accidents are charged exactly once — see count_findings(). A seller who declares the
    accident pays `accident_confirmed`; a seller who declares the car accident-free pays only
    for what the reviewer actually finds in the description, at the severity it deserves."""
    cfg = profile.get("condition", {})
    pen = cfg.get("penalties", {})
    score = 10.0

    declared = car.get("unfallfrei_declared")
    if declared is False:
        score -= float(pen.get("accident_confirmed", 4.0))
    elif declared is None:
        score -= float(pen.get("accident_unknown", 1.0))

    owners = car.get("owners")
    if owners:
        score -= max(0, int(owners) - 1) * float(pen.get("owner_step", 0.6))

    if car.get("service_history") is not True:
        score -= float(pen.get("no_service_history", 1.5))
    if not car.get("tuv"):
        score -= float(pen.get("no_tuv", 1.0))

    majors, minors = count_findings(car)
    score -= majors * float(pen.get("major_defect", 3.5))
    score -= min(float(pen.get("minor_issue_cap", 2.0)), minors * float(pen.get("minor_issue", 0.4)))

    return round(clamp(score, 0.0, 10.0), 2)


def condition_detail(car: dict, profile: dict) -> tuple[list[tuple[str, str, float]], float]:
    """([(badge, explanation, delta), ...], final score) — every deduction condition() applied,
    so the report can show exactly what was held against a car. Deltas are negative."""
    cfg = profile.get("condition", {})
    pen = cfg.get("penalties", {})
    items: list[tuple[str, str, float]] = []

    declared = car.get("unfallfrei_declared")
    if declared is False:
        items.append(("accident", "seller declares accident damage",
                      -float(pen.get("accident_confirmed", 4.0))))
    elif declared is None:
        items.append(("acc?", "accident-free not stated by the seller",
                      -float(pen.get("accident_unknown", 1.0))))

    owners = car.get("owners")
    if owners and int(owners) > 1:
        items.append((f"{owners} own", f"{owners} previous owners",
                      -(int(owners) - 1) * float(pen.get("owner_step", 0.6))))

    if car.get("service_history") is not True:
        items.append(("no svc", "no service history (Scheckheft)",
                      -float(pen.get("no_service_history", 1.5))))
    if not car.get("tuv"):
        items.append(("no TÜV", "no TÜV/HU stated", -float(pen.get("no_tuv", 1.0))))

    majors, minors = count_findings(car)
    if majors:
        items.append(("defect", f"{majors} disclosed major defect(s)",
                      -majors * float(pen.get("major_defect", 3.5))))
    if minors:
        items.append(("minor", f"{minors} minor caution(s) (capped)",
                      -min(float(pen.get("minor_issue_cap", 2.0)),
                           minors * float(pen.get("minor_issue", 0.4)))))
    return items, condition(car, profile)


def autobahn_detail(car: dict, profile: dict) -> dict:
    """The parts that make up autobahn_score(), for the report's Power breakdown."""
    acfg = profile.get("autobahn", {})
    hay = _hay(car)
    for key, val in acfg.get("map", {}).items():
        if _fold(key) in hay:
            return {"override": round(clamp(float(val), 0.0, 10.0), 2), "ps": car.get("power_ps"),
                    "base": None, "power_bonus": None, "segment_bonus": 0.0,
                    "segment": car_segment(car, profile), "score": autobahn_score(car, profile)}
    ps = car.get("power_ps") or 0
    base = float(acfg.get("base", 5.0))
    power_bonus = max(0.0, ps - 130) * float(acfg.get("per_ps_over_130", 0.06))
    bonus = 0.0
    for key, val in acfg.get("segment_bonus", {}).items():
        if _fold(key) in hay:
            bonus = max(bonus, float(val))
    if (car.get("displacement_cm3") or 0) >= acfg.get("big_engine_cm3", 1_000_000):
        bonus = max(bonus, float(acfg.get("big_engine_bonus", 0.0)))
    return {"override": None, "ps": ps, "base": base, "power_bonus": round(power_bonus, 2),
            "segment_bonus": round(bonus, 2), "segment": car_segment(car, profile),
            "score": autobahn_score(car, profile)}


def _has_option(car: dict, option: str) -> bool:
    """Whether a wishlist option is present. Towbar counts only when actually fitted.

    The union of two sources, taken at SCORING time rather than at write time:

      * the parsed Ausstattung booleans (the seller's structured equipment list), and
      * `options_from_prose` — options the reviewer found only in the free-text description,
        because sellers routinely describe kit they forgot to tick.

    Unioning here instead of writing the LLM's answer onto the boolean keeps the parsed field a
    pure function of the PDF, so `reprocess` stays safe and provenance stays legible: the record
    still says where each fact came from."""
    if option in (car.get("options_from_prose") or []):
        return True
    if option == "towbar":
        return car.get("towbar") in ("fitted_fixed", "fitted_swivel")
    return car.get(option) is True


def wishlist_share(car: dict, profile: dict) -> float:
    """0..10 = weighted share of the wishlist present. `wishlist` is {option: weight}
    (a plain list is treated as all weight 1)."""
    wishlist = profile.get("wishlist", {})
    if isinstance(wishlist, list):
        wishlist = {opt: 1 for opt in wishlist}
    total = sum(float(w) for w in wishlist.values())
    if total <= 0:
        return 0.0
    got = sum(float(w) for opt, w in wishlist.items() if _has_option(car, opt))
    return round(10.0 * got / total, 2)


def year_score(car: dict, profile: dict) -> float | None:
    """0..10 recency from the registration year (year_worst -> 0, year_best -> 10, clamped).
    None when the year is unknown or no equipment.year config is present."""
    eqcfg = profile.get("equipment", {})
    yr = reg_year(car.get("first_reg"))
    if yr is None:
        return None
    best = float(eqcfg.get("year_best", 2025))
    worst = float(eqcfg.get("year_worst", 2016))
    if best == worst:
        return 10.0
    return round(clamp(10.0 * (yr - worst) / (best - worst), 0.0, 10.0), 2)


def equipment_fit(car: dict, profile: dict) -> float:
    """0..10 blending the weighted wishlist share with a registration-year recency score.
    Controlled by profile['equipment'] {wishlist_weight, year_weight, year_best, year_worst}.
    With year_weight 0 (or no config, or unknown year) it is the pure wishlist share."""
    base = wishlist_share(car, profile)
    eqcfg = profile.get("equipment", {})
    yw = float(eqcfg.get("year_weight", 0.0))
    if yw <= 0:
        return base
    ys = year_score(car, profile)
    if ys is None:
        return base
    ww = float(eqcfg.get("wishlist_weight", 1.0 - yw))
    total = ww + yw
    return round((ww * base + yw * ys) / (total if total else 1.0), 2)


def model_map_lookup(car: dict, table: dict) -> dict:
    """Resolve a per-model config map for this car: the longest key whose tokens all appear
    in the folded 'make model variant' wins (so 'opel insignia' beats 'opel'). Keys starting
    with '_' are comments. Empty dict when nothing matches."""
    hay = _hay(car)
    best_val: dict = {}
    best_len = -1
    for key, val in table.items():
        if str(key).startswith("_"):
            continue
        k = _fold(key)
        if all(tok in hay for tok in k.split()) and len(k) > best_len and isinstance(val, dict):
            best_val, best_len = val, len(k)
    return best_val


def reliability_basis(car: dict, profile: dict) -> dict:
    """The TÜV/ADAC figures behind the reliability score for this car's make/model, resolved
    with the same longest-key-match rule as the score. Empty dict when none is configured."""
    return model_map_lookup(car, profile.get("model_reliability_basis", {}))


def safety_basis(car: dict, profile: dict) -> dict:
    """This car's Euro NCAP result (profile `model_safety`) — INFORMATIONAL, never scored:
    every catalogue model is 5-star and NCAP percentages are not comparable across protocol
    years, so a score would rank test vintage, not safety. The generation is picked by
    registration year: the test with the largest from_year <= reg year (a 2017 Leon gets the
    2012 test, a 2021 Leon the 2020 one). Empty dict when the model has no entry."""
    entry = model_map_lookup(car, profile.get("model_safety", {}))
    tests = entry.get("tests") if entry else None
    if not tests:
        return {}
    year = reg_year(car.get("first_reg"))
    eligible = [t for t in tests
                if year is None or int(t.get("from_year", 0)) <= year]
    if not eligible:
        eligible = [min(tests, key=lambda t: int(t.get("from_year", 0)))]
    return max(eligible, key=lambda t: int(t.get("from_year", 0)))


# TÜV age bands (years). A listing's age selects the band; nearest band is used if a model's
# table lacks the exact one. ADAC is keyed by registration year directly (see _lookup_adac).
_TUV_BANDS = [(0, 3, "2-3"), (4, 5, "4-5"), (6, 7, "6-7"), (8, 9, "8-9"), (10, 999, "10-11")]
_BAND_MID = {"2-3": 2.5, "4-5": 4.5, "6-7": 6.5, "8-9": 8.5, "10-11": 10.5}


def _lookup_tuv(tuv_map: dict | None, age: int | None) -> tuple[float | None, str | None]:
    """Pick the TÜV Mängelquote cell for a listing's age from {band: pct}. Exact band if
    present, else the nearest available band (by band midpoint)."""
    if not tuv_map or age is None:
        return None, None
    band = next((k for lo, hi, k in _TUV_BANDS if lo <= age <= hi), "2-3")
    if band in tuv_map:
        return float(tuv_map[band]), band
    nearest = min(tuv_map.keys(), key=lambda k: abs(_BAND_MID.get(k, 99) - age))
    return float(tuv_map[nearest]), nearest


def _lookup_adac(adac_map: dict | None, year: int | None) -> tuple[float | None, str | None]:
    """Pick the ADAC breakdowns-per-1000 cell for a listing's registration year from
    {year: rate}. Exact year if present, else the nearest available year (newest if unknown)."""
    if not adac_map:
        return None, None
    yrs = [(int(k), k) for k in adac_map if str(k).lstrip("-").isdigit()]
    if not yrs:
        return None, None
    if year is not None and str(year) in adac_map:
        return float(adac_map[str(year)]), str(year)
    target = year if year is not None else max(y for y, _ in yrs)
    yv, yk = min(yrs, key=lambda t: abs(t[0] - target))
    return float(adac_map[yk]), yk


def reliability_detail(car: dict, profile: dict, today: date | None = None) -> dict:
    """Derive the 0..10 reliability score from the objective figures in model_reliability_basis,
    matched to THIS listing: the TÜV Mängelquote for the car's age band and the ADAC breakdown
    rate for its registration year are looked up directly (real per-cell data, no projection),
    each mapped to 0..10 against a fixed market reference (mean->5, +/-2*sd->10/0) and combined
    by metric_weights. Older listing => older band / year => higher rate => lower score.

    Returns {'score','fallback','age','year','components': {metric: {'raw','key','sub'}}}.
    'fallback' is True when no cell resolved for either metric (then score = fallback_score)."""
    cfg = profile.get("reliability_model", {})
    basis = reliability_basis(car, profile)
    age = car_age_years(car, today)
    year = reg_year(car.get("first_reg"))
    gain = float(cfg.get("gain", 2.5))
    midpoint = float(cfg.get("midpoint", 5.0))
    weights = cfg.get("metric_weights", {"tuv": 0.5, "adac": 0.5})
    refs = cfg.get("reference", {})

    lookups = {
        "tuv": _lookup_tuv(basis.get("tuv_by_age") if basis else None, age),
        "adac": _lookup_adac(basis.get("adac_by_year") if basis else None, year),
    }
    components: dict = {}
    num = den = 0.0
    for metric, (raw, key) in lookups.items():
        w = float(weights.get(metric, 0))
        if raw is None or w <= 0:
            continue
        ref = refs.get(metric, {})
        mean, sd = float(ref.get("mean", raw)), float(ref.get("sd", 1.0)) or 1.0
        sub = round(clamp(midpoint + gain * (mean - raw) / sd, 0.0, 10.0), 2)
        components[metric] = {"raw": raw, "key": key, "sub": sub}
        num += w * sub
        den += w

    if den <= 0:
        return {"score": round(float(cfg.get("fallback_score", 5.0)), 2),
                "fallback": True, "age": age, "year": year, "components": {}}
    return {"score": round(num / den, 2), "fallback": False, "age": age, "year": year,
            "components": components}


def reliability_score(car: dict, profile: dict, today: date | None = None) -> float:
    """The 0..10 reliability ranking input (see reliability_detail for the derivation)."""
    return reliability_detail(car, profile, today)["score"]


def travel_minutes_for(car: dict, profile: dict) -> tuple[float | None, str | None]:
    """(door-to-door minutes from the profile origin, how we know it). Uses the stored value
    when `carcache.py travel` has filled it (mode public > car), else derives it from a stored
    distance at avg_kmh, else gives up. Never touches the network."""
    cfg = profile.get("location", {})
    mins = car.get("travel_minutes")
    if mins is not None:
        return float(mins), car.get("travel_mode") or "stored"
    km = car.get("travel_km")
    if km is not None:
        kmh = float(cfg.get("avg_kmh", 95)) or 95.0
        return float(km) / kmh * 60.0, car.get("travel_mode") or "car_km"
    return None, None


def location_score(car: dict, profile: dict) -> float:
    """0..10 from travel time to the dealer on an ABSOLUTE curve (candidate-set independent):
    <= best_minutes -> 10, >= worst_minutes -> 0, linear and clamped. With best_minutes 90,
    Berlin and Brandenburg both land on 10. Unknown travel -> fallback_score."""
    cfg = profile.get("location", {})
    mins, _mode = travel_minutes_for(car, profile)
    if mins is None:
        return round(float(cfg.get("fallback_score", 5.0)), 2)
    best = float(cfg.get("best_minutes", 90))
    worst = float(cfg.get("worst_minutes", 480))
    if worst == best:
        return 10.0
    t = (mins - best) / (worst - best)
    return round(clamp(10.0 * (1.0 - t), 0.0, 10.0), 2)


def mileage_score(km: float | None, profile: dict) -> float:
    """0..10 from the odometer on an ABSOLUTE curve (independent of the candidate set):
    best_km -> 10, worst_km -> 0, linear and clamped. Configured in profile['mileage']."""
    cfg = profile.get("mileage", {})
    best = float(cfg.get("best_km", 30000))
    worst = float(cfg.get("worst_km", 180000))
    if worst == best:
        return 10.0
    t = (float(km or 0) - best) / (worst - best)
    return round(clamp(10.0 * (1.0 - t), 0.0, 10.0), 2)


# --------------------------------------------------------------------------------------
# Cost — a composite of what you pay up front, what the market thinks of that price, and what
# the car costs to keep. Three separate questions that a single €/yr figure cannot answer:
# a cheap car can be poor value, and a well-priced car can be expensive to run.
#
# The scored "running" third is the CASH running cost — the TCO components minus depreciation.
# Depreciation is estimated FROM the asking price, and the asking price already has its own
# third: scoring depreciation too charged the same euros twice (a €15k car lost on the price
# curve AND was billed ~€300/yr more depreciation than a €10k car). It also imported the
# hand-set brand_retention_adj priors into the score. Depreciation stays in the displayed
# TCO/yr figure — it is a real cost — it just no longer votes.
# --------------------------------------------------------------------------------------

def cash_running(tco_parts: dict) -> float:
    """The €/yr a car actually costs to keep on the road: every TCO component except
    depreciation (fuel, electricity, maintenance, insurance, tax, HU)."""
    return float(sum(v for k, v in tco_parts.items() if k != "depreciation"))

def price_score(price: float | None, profile: dict) -> float:
    """0..10 from the asking price on an ABSOLUTE curve (independent of the candidate set):
    best_eur -> 10, worst_eur -> 0, linear and clamped. Configured in profile['cost']['price']."""
    cfg = profile.get("cost", {}).get("price", {})
    best = float(cfg.get("best_eur", 10000))
    worst = float(cfg.get("worst_eur", 15000))
    if worst == best:
        return 10.0
    t = (float(price or worst) - best) / (worst - best)
    return round(clamp(10.0 * (1.0 - t), 0.0, 10.0), 2)


def price_rating_score(car: dict, profile: dict) -> float:
    """0..10 from mobile.de's own price verdict — the market's read on whether this price is
    fair for this car, which the absolute price curve cannot express. Five buckets from
    'Sehr guter Preis' (10) to 'Hoher Preis' (0); an unrated listing scores neutral rather than
    0, since silence is not a bad price."""
    cfg = profile.get("cost", {})
    mapping = cfg.get("price_rating_map", {})
    rating = car.get("price_rating")
    if rating is None:
        return float(cfg.get("price_rating_unknown", 5))
    return float(mapping.get(rating, cfg.get("price_rating_unknown", 5)))


def running_cost_score(running_eur: float | None, profile: dict) -> float:
    """0..10 from the cash running cost on an ABSOLUTE curve (independent of the candidate
    set, like price and mileage): best_eur_yr -> 10, worst_eur_yr -> 0, linear and clamped.
    Configured in profile['cost']['running']. This replaced a min-max normalisation across
    the ranked set — the last candidate-set-relative term in the model — so ranking a subset
    now re-scores nothing."""
    cfg = profile.get("cost", {}).get("running", {})
    best = float(cfg.get("best_eur_yr", 1800))
    worst = float(cfg.get("worst_eur_yr", 3600))
    if worst == best:
        return 10.0
    t = (float(running_eur or worst) - best) / (worst - best)
    return round(clamp(10.0 * (1.0 - t), 0.0, 10.0), 2)


def cost_detail(car: dict, profile: dict,
                running_eur: float) -> tuple[list[tuple[str, str, float, float]], float]:
    """([(key, label, component score, weight share), ...], composite 0..10).

    `weights` in profile['cost'] are relative and normalised here, so the default 1/1/1 means
    equal thirds and can be retuned without touching code. `running_eur` is the CASH running
    cost (cash_running of the TCO parts) — depreciation is deliberately not scored, see the
    section comment above."""
    cfg = profile.get("cost", {})
    weights = cfg.get("weights", {}) or {"price": 1, "price_rating": 1, "running": 1}
    parts = {
        "price": price_score(car.get("price_eur"), profile),
        "price_rating": price_rating_score(car, profile),
        "running": running_cost_score(running_eur, profile),
    }
    total_w = sum(float(w) for w in weights.values()) or 1.0
    items, composite = [], 0.0
    for key, label in (("price", "Asking price"), ("price_rating", "Price rating"),
                       ("running", "Running cost (cash)")):
        share = float(weights.get(key, 0)) / total_w
        items.append((key, label, parts[key], share))
        composite += parts[key] * share
    return items, round(clamp(composite, 0.0, 10.0), 2)


def cost_score(car: dict, profile: dict, running_eur: float) -> float:
    """0..10 composite money score — see cost_detail()."""
    return cost_detail(car, profile, running_eur)[1]


def autobahn_score(car: dict, profile: dict) -> float:
    """0..10 from power (more PS = better) plus a light segment guess. A per-model
    override map in profile['autobahn']['map'] wins outright when a key matches."""
    acfg = profile.get("autobahn", {})
    hay = _hay(car)

    for key, val in acfg.get("map", {}).items():
        if _fold(key) in hay:
            return round(clamp(float(val), 0.0, 10.0), 2)

    ps = car.get("power_ps") or 0
    base = float(acfg.get("base", 5.0)) + max(0.0, ps - 130) * float(acfg.get("per_ps_over_130", 0.06))

    bonus = 0.0
    for key, val in acfg.get("segment_bonus", {}).items():
        if _fold(key) in hay:
            bonus = max(bonus, float(val))
    if (car.get("displacement_cm3") or 0) >= acfg.get("big_engine_cm3", 1_000_000):
        bonus = max(bonus, float(acfg.get("big_engine_bonus", 0.0)))

    return round(clamp(base + bonus, 0.0, 10.0), 2)


# --------------------------------------------------------------------------------------
# Duplicate detection + ranking
# --------------------------------------------------------------------------------------

# What makes two listings the same physical car. Price is deliberately absent: dealers
# routinely re-list at a new price, and (dealer, make, model, first_reg, km) is already a very
# strong fingerprint — an exact odometer reading alone is close to unique.
DEDUPE_KEY = ("make", "model", "dealer", "first_reg", "km")


def dedupe_key(car: dict) -> tuple | None:
    """The identity tuple, or None when the car is missing a component (never dedupe on
    partial information — a null would collide with every other null)."""
    vals = []
    for f in DEDUPE_KEY:
        v = car.get(f)
        if v is None or v == "":
            return None
        vals.append(str(v).strip().casefold() if isinstance(v, str) else v)
    return tuple(vals)


def duplicate_groups(store: dict[str, dict]) -> list[tuple[tuple, list[str]]]:
    """[(key, [ids…])] for every group of >1 listing of the same car, newest id last.

    mobile.de ids increase over time, so the numerically highest id is the current listing;
    the rest are the dealer's earlier ads for the same vehicle."""
    groups: dict[tuple, list[str]] = {}
    for cid, car in store.items():
        key = dedupe_key(car)
        if key is not None:
            groups.setdefault(key, []).append(cid)
    return [(k, sorted(v, key=lambda c: (len(c), c))) for k, v in sorted(groups.items())
            if len(v) > 1]


def apply_dedupe(store: dict[str, dict]) -> list[tuple[str, str]]:
    """Mark every superseded listing with `superseded_by: <winner id>`. Returns [(loser,
    winner)]. The losing records are KEPT — nothing is deleted — they are simply excluded
    from ranking and the report."""
    marked = []
    for _key, ids in duplicate_groups(store):
        winner = ids[-1]
        for loser in ids[:-1]:
            if store[loser].get("superseded_by") != winner:
                store[loser]["superseded_by"] = winner
                marked.append((loser, winner))
        store[winner]["superseded_by"] = None
    return marked


@dataclass
class RankResult:
    cid: str
    car: dict
    scores: dict[str, float]
    total: float
    tco: float
    tco_estimated: bool
    stale: bool = False
    tco_breakdown: dict = field(default_factory=dict)
    running: float = 0.0  # cash running €/yr (TCO minus depreciation) — the scored cost input


def compute_ranking(
    store: dict[str, dict],
    profile: dict,
    ids: Iterable[str] | None = None,
    stale_days: int | None = None,
    today: date | None = None,
) -> list[RankResult]:
    """Score every candidate. Deterministic: sorted by total desc, then TCO asc, then id asc.

    Every criterion is an absolute curve or a per-listing lookup, so a car's scores and total
    are independent of which other cars are ranked with it — ranking a subset re-scores
    nothing (a test guards this).

    Superseded listings (the dealer's earlier ad for a car that is listed again under a newer
    id) are excluded — they are the same vehicle, and ranking both would double-count it.
    Explicitly naming ids overrides that, so `show`/`rank <id>` can still reach one."""
    selected = [str(i) for i in ids] if ids else [
        cid for cid, car in store.items() if not car.get("superseded_by")]
    survivors = [(i, store[i]) for i in selected if i in store]

    tco_by_id = {cid: annual_tco(car, profile) for cid, car in survivors}

    weights = profile["weights"]
    results: list[RankResult] = []
    for cid, car in survivors:
        tco_val, tco_parts, estimated = tco_by_id[cid]
        running = cash_running(tco_parts)
        scores = {
            "reliability": reliability_score(car, profile, today),
            "condition": condition(car, profile),
            "equipment_fit": equipment_fit(car, profile),
            "autobahn": autobahn_score(car, profile),
            "cost": cost_score(car, profile, running),
            "mileage": mileage_score(car.get("km"), profile),
            "location": location_score(car, profile),
        }
        scores = {k: round(v, 2) for k, v in scores.items()}
        total = round(sum(weights[k] * scores[k] for k in weights), 2)
        stale = stale_days is not None and _is_stale(car, stale_days, today)
        results.append(RankResult(cid, car, scores, total, tco_val, estimated, stale, tco_parts,
                                  running=running))

    results.sort(key=lambda r: (-r.total, r.tco, r.cid))
    return results


def criteria_influence(results: list[RankResult], profile: dict) -> list[dict]:
    """How much each criterion ACTUALLY moves the ranking, as weight x observed spread.

    A criterion's nominal weight only matters where its scores differ between cars: a
    criterion on which every car scores 8-10 contributes almost nothing to the ORDER no
    matter its weight. `influence` = weight * sd(scores across the candidates); `share` is
    that as a fraction of the summed influence. Diagnosis only — the ranking itself still
    uses the nominal weights, and this table is how you tune them with open eyes."""
    weights = profile["weights"]
    rows: list[dict] = []
    for c in CRITERIA:
        vals = [r.scores[c] for r in results]
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        w = float(weights.get(c, 0.0))
        rows.append({"criterion": c, "weight": w, "mean": round(mean, 2),
                     "sd": round(sd, 2), "influence": round(w * sd, 3)})
    total = sum(r["influence"] for r in rows) or 1.0
    for r in rows:
        r["share"] = round(r["influence"] / total, 3)
    return rows


# Fixed seed => `rank --sensitivity` is deterministic run-to-run, like everything else in rank.
SENSITIVITY_SEED = 20260717


def weight_sensitivity(results: list[RankResult], profile: dict, n: int = 500,
                       spread: float = 0.30) -> dict:
    """How robust the ranking order is to the top-level weights themselves.

    Perturbs every weight independently by a uniform factor in [1-spread, 1+spread] and
    re-sorts, n times: {'n', 'spread', 'top1': {cid: count}, 'top3': {cid: count}}. A car
    with top1 == n is #1 under every plausible weighting — the decision does not hinge on
    the exact weights. The perturbed weights are not re-normalised: a common scale factor
    cannot change an ordering, only the RELATIVE weights move, which is the point.

    Only the combination weights move; the per-criterion scores are reused as computed (they
    do not depend on `weights`), so this is exact, costs microseconds, and stays offline."""
    import random
    rng = random.Random(SENSITIVITY_SEED)
    weights = {k: float(v) for k, v in profile["weights"].items()}
    top1: dict[str, int] = {}
    top3: dict[str, int] = {}
    for _ in range(n):
        w = {c: v * rng.uniform(1.0 - spread, 1.0 + spread) for c, v in weights.items()}
        order = sorted(results,
                       key=lambda r: (-sum(w[c] * r.scores[c] for c in w), r.tco, r.cid))
        top1[order[0].cid] = top1.get(order[0].cid, 0) + 1
        for r in order[:3]:
            top3[r.cid] = top3.get(r.cid, 0) + 1
    return {"n": n, "spread": spread, "top1": top1, "top3": top3}


def _is_stale(car: dict, stale_days: int, today: date | None = None) -> bool:
    age = days_old(car.get("fetched_date"), today)
    return age is not None and age > stale_days


# --------------------------------------------------------------------------------------
# Record add / normalise
# --------------------------------------------------------------------------------------

def normalize_record(record: dict) -> dict:
    """Coerce an incoming record to the schema: drop any stray 'id', fill missing
    fields with null (lists with []), default fetched_date to today."""
    incoming = dict(record)
    incoming.pop("id", None)
    out: dict[str, Any] = {}
    for f in SCHEMA_FIELDS:
        if f in incoming:
            out[f] = incoming[f]
        elif f in LIST_FIELDS:
            out[f] = []
        elif f == "fetched_date":
            out[f] = date.today().isoformat()
        else:
            out[f] = None
    return out


def add_listing(store: dict, cid: str, record: dict, force: bool = False) -> tuple[bool, str]:
    """Upsert one listing. Returns (changed, message). This is the guard that stops a
    PDF from ever being re-processed: an existing id is skipped unless force=True."""
    cid = str(cid)
    if cid in store and not force:
        return False, f"{cid}: already cached — skipping (use --force to overwrite)"
    existed = cid in store
    store[cid] = normalize_record(record)
    store[cid]["extract_version"] = EXTRACT_VERSION
    return True, f"{cid}: {'updated (forced overwrite)' if existed else 'added'}"


# --------------------------------------------------------------------------------------
# Filtering (generic --field value)
# --------------------------------------------------------------------------------------

RANGE_OPS: dict[str, tuple[str, str]] = {
    "max-km": ("km", "<="), "min-km": ("km", ">="),
    "max-price": ("price_eur", "<="), "min-price": ("price_eur", ">="),
    "max-year": ("__year__", "<="), "min-year": ("__year__", ">="),
    "max-ps": ("power_ps", "<="), "min-ps": ("power_ps", ">="),
    "max-owners": ("owners", "<="), "min-owners": ("owners", ">="),
    "max-km-yr": ("__km_per_year__", "<="),
}


def coerce(text: str) -> Any:
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_filter_conditions(tokens: list[str]) -> list[tuple[str, str]]:
    """Turn ['--towbar', 'fitted_fixed', '--adaptive_cruise', 'true'] into pairs.
    A flag with no following value (or followed by another --flag) means 'true'."""
    conds: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"expected --field, got {tok!r}")
        key = tok[2:]
        if "=" in key:  # support --field=value
            key, val = key.split("=", 1)
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            val = tokens[i + 1]
            i += 2
        else:
            val = "true"
            i += 1
        conds.append((key, val))
    return conds


def _field_value(car: dict, field_name: str) -> Any:
    if field_name == "__year__":
        return reg_year(car.get("first_reg"))
    if field_name == "__km_per_year__":
        year = reg_year(car.get("first_reg"))
        km = car.get("km")
        if year is None or km is None:
            return None
        age = max(1, date.today().year - year)
        return km / age
    return car.get(field_name)


def matches_condition(car: dict, key: str, raw_value: str) -> bool:
    if key in RANGE_OPS:
        field_name, op = RANGE_OPS[key]
        actual = _field_value(car, field_name)
        if actual is None:
            return False
        target = float(raw_value)
        return actual <= target if op == "<=" else actual >= target

    # equality on a schema field (accept both hyphen and underscore spellings)
    field_name = key if key in SCHEMA_FIELDS else key.replace("-", "_")
    if field_name not in SCHEMA_FIELDS:
        raise ValueError(f"unknown filter field: --{key}")
    actual = car.get(field_name)
    target = coerce(raw_value)
    if isinstance(actual, str) and isinstance(target, str):
        return actual.lower() == target.lower()
    return actual == target


def apply_conditions(store: dict, conditions: list[tuple[str, str]]) -> list[str]:
    out = []
    for cid, car in store.items():
        if all(matches_condition(car, k, v) for k, v in conditions):
            out.append(cid)
    return out


# --------------------------------------------------------------------------------------
# Text-table rendering
# --------------------------------------------------------------------------------------

def render_table(headers: list[str], rows: list[list[Any]], aligns: str | None = None) -> str:
    n = len(headers)
    aligns = aligns or "l" * n
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in cells:
        for j, c in enumerate(row):
            widths[j] = max(widths[j], len(c))

    def fmt(row: list[str]) -> str:
        parts = []
        for j, c in enumerate(row):
            parts.append(c.rjust(widths[j]) if aligns[j] == "r" else c.ljust(widths[j]))
        return "  ".join(parts).rstrip()

    lines = [fmt([str(h) for h in headers]), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in cells)
    return "\n".join(lines)


def trunc(text: Any, width: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= width else s[: width - 1] + "…"


def _display_name(car: dict) -> str:
    """'Ford Focus ST-Line' — what a human reads in a table.

    `make`/`model` are the canonical identity parsed from the page header ('Ford', 'Focus'),
    which is what the reliability tables are keyed on; `variant` is the trim line. Joining them
    for display keeps the stored fields clean and comparable while the report still shows the
    trim that tells two otherwise-identical listings apart."""
    parts = [car.get("make"), car.get("model"), car.get("variant")]
    return " ".join(str(p) for p in parts if p) or "—"


def fmt_int(x: Any) -> str:
    return "-" if x is None else f"{int(x):,}"


def towbar_short(value: Any) -> str:
    return {
        "fitted_fixed": "fixed", "fitted_swivel": "swivel",
        "capable": "capable", "prep": "prep",
    }.get(value, "-" if value is None else str(value))


# --------------------------------------------------------------------------------------
# Listing URL + PDF id extraction (stdlib only; no image cost)
# --------------------------------------------------------------------------------------

def listing_url(cid: str, car: dict | None = None) -> str:
    """The mobile.de detail URL for a listing: stored value if present, else derived."""
    if car and car.get("listing_url"):
        return str(car["listing_url"])
    return LISTING_URL_TMPL.format(id=cid)


_ID_RE = re.compile(rb"(?:details\.html\?|printView\.html\?|[?&])id=(\d{6,})")


def extract_ids_from_bytes(data: bytes) -> list[str]:
    """Find mobile.de listing ids in raw bytes and in any FlateDecode streams.
    Returns ids ordered by how often they appear (most frequent first) — the listing's
    own id dominates a mobile.de PDF, while dealer/finance links appear less."""
    import collections
    import zlib

    counts: collections.Counter[str] = collections.Counter()

    def scan(blob: bytes) -> None:
        for m in _ID_RE.findall(blob):
            counts[m.decode()] += 1

    scan(data)
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        chunk = data[start:end]
        for attempt in (chunk, chunk.strip(b"\r\n")):
            try:
                scan(zlib.decompress(attempt))
                break
            except Exception:
                continue
    return [cid for cid, _ in counts.most_common()]


def pdf_id(path: Path | str) -> str | None:
    """Best-effort mobile.de id for a listing PDF, or None if none is found."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    ids = extract_ids_from_bytes(data)
    return ids[0] if ids else None


# --------------------------------------------------------------------------------------
# Availability (best-effort — mobile.de bot-blocks datacenter requests, so a 403/999
# is reported as 'unknown' and never silently flips a car to sold)
# --------------------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 15.0) -> tuple[int, str, str]:
    """(status, final_url, body_prefix). Raises on network error."""
    import urllib.request

    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(8000).decode("utf-8", "ignore")
        return resp.status, resp.geturl(), body


def check_availability(url: str, fetch=_http_get) -> tuple[str, str]:
    """Classify a listing URL as 'active' | 'inactive' | 'unknown' with a short detail.
    Only a confident removal signal (404/410 or removal text) yields 'inactive'."""
    import urllib.error

    try:
        status, final_url, body = fetch(url)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return "inactive", f"HTTP {exc.code}"
        return "unknown", f"blocked/HTTP {exc.code}"
    except Exception as exc:  # timeouts, DNS, connection resets, …
        return "unknown", f"{type(exc).__name__}"

    low = body.lower()
    if any(marker in low for marker in REMOVAL_MARKERS):
        return "inactive", "removal text on page"
    if status == 200:
        return "active", "reachable"
    return "unknown", f"HTTP {status}"


def refresh_store(
    store: dict,
    ids: Iterable[str] | None = None,
    fetch=_http_get,
    today: date | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """Ping each listing and update its `active` / `last_checked` fields.
    Returns (rows, newly_inactive) where rows = (id, previous, new_state, detail)."""
    today = (today or date.today()).isoformat()
    selected = [str(i) for i in ids] if ids else list(store.keys())
    rows: list[tuple[str, str, str, str]] = []
    newly_inactive: list[str] = []
    for cid in selected:
        if cid not in store:
            continue
        car = store[cid]
        prev = car.get("active")
        state, detail = check_availability(listing_url(cid, car), fetch=fetch)
        car["last_checked"] = today
        if state == "active":
            car["active"] = True
        elif state == "inactive":
            car["active"] = False
            if prev is not False:
                newly_inactive.append(cid)
        # 'unknown' leaves the previous value untouched
        prev_str = {True: "active", False: "inactive", None: "?"}.get(prev, str(prev))
        rows.append((cid, prev_str, state, detail))
    return rows, newly_inactive


def active_marker(car: dict) -> str:
    return {True: "live", False: "SOLD", None: "?"}.get(car.get("active"), "?")


# --------------------------------------------------------------------------------------
# Inbox intake (dedupe PDFs by id before any vision extraction)
# --------------------------------------------------------------------------------------

@dataclass
class IntakeItem:
    path: Path
    cid: str | None
    status: str          # 'cached' | 'new' | 'unknown-id'
    archived_to: Path | None = None


def scan_inbox(store: dict, inbox: Path, processed: Path, force: bool = False) -> list[IntakeItem]:
    """Look at every PDF in `inbox`, extract its id, and sort into cached / new / unknown.
    Recognised PDFs are renamed to <id>.pdf; cached ones are moved to `processed`.
    New ones stay in `inbox` (as <id>.pdf) awaiting vision extraction + `add`."""
    items: list[IntakeItem] = []
    if not inbox.exists():
        return items
    for path in sorted(inbox.glob("*.pdf")):
        cid = pdf_id(path)
        if cid is None:
            items.append(IntakeItem(path, None, "unknown-id"))
            continue
        # normalise to <id>.pdf so downstream paths are ASCII and self-documenting
        staged = path if path.name == f"{cid}.pdf" else _rename_safe(path, inbox / f"{cid}.pdf")
        if cid in store and not force:
            processed.mkdir(parents=True, exist_ok=True)
            dest = _rename_safe(staged, processed / f"{cid}.pdf")
            items.append(IntakeItem(dest, cid, "cached", archived_to=dest))
        else:
            items.append(IntakeItem(staged, cid, "new"))
    return items


def _rename_safe(src: Path, dest: Path) -> Path:
    """Move src -> dest, replacing dest if it already exists. Returns dest."""
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists():
        dest.unlink()
    src.replace(dest)
    return dest


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    if args.file:
        record = json.loads(Path(args.file).read_text(encoding="utf-8"))
    elif args.json:
        record = json.loads(args.json)
    else:
        print("error: provide --file <path> or --json '<...>'", file=sys.stderr)
        return 2

    cid = str(args.id)
    store = load_store(args.store)
    changed, message = add_listing(store, cid, record, force=args.force)
    print(message)
    if not changed:
        return 0

    car = store[cid]
    if not car.get("listing_url"):
        car["listing_url"] = listing_url(cid)
    if car.get("active") is None:
        car["active"] = True  # a listing you just saved is presumed live
    if not car.get("last_checked"):
        car["last_checked"] = car.get("fetched_date") or date.today().isoformat()

    if args.pdf:
        src = Path(args.pdf)
        if src.exists():
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            dest = _rename_safe(src, PROCESSED_DIR / f"{cid}.pdf")
            car["source_pdf"] = str(dest.relative_to(BASE_DIR)) if dest.is_relative_to(BASE_DIR) else str(dest)
            print(f"  archived PDF -> {car['source_pdf']}")
        else:
            print(f"  warning: --pdf {src} not found; source_pdf left unchanged", file=sys.stderr)

    save_store(store, args.store)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    cid = str(args.id)
    if cid not in store:
        print(f"{cid}: not in cache", file=sys.stderr)
        return 1
    car = store[cid]
    print(f"id: {cid}")
    for f in SCHEMA_FIELDS:
        val = car.get(f)
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val) if val else "-"
        elif val is None:
            val = "-"
        print(f"  {f:<18} {val}")
    return 0


def _car_table_rows(items: list[tuple[str, dict]], stale_days: int | None) -> tuple[list[str], list[list[Any]], str]:
    headers = ["id", "model", "€price", "year", "km", "PS", "towbar", "findings"]
    aligns = "llrrrrlr"
    if stale_days is not None:
        headers.append("fresh")
        aligns += "l"
    rows: list[list[Any]] = []
    for cid, car in items:
        row = [
            cid,
            trunc(_display_name(car), 34),
            fmt_int(car.get("price_eur")),
            reg_year(car.get("first_reg")) or "-",
            fmt_int(car.get("km")),
            car.get("power_ps") or "-",
            towbar_short(car.get("towbar")),
            len(car.get("findings") or []),
        ]
        if stale_days is not None:
            row.append("STALE" if _is_stale(car, stale_days) else "ok")
        rows.append(row)
    return headers, rows, aligns


def cmd_list(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    if not store:
        print("cache is empty")
        return 0
    items = sorted(store.items(), key=lambda kv: kv[0])
    headers, rows, aligns = _car_table_rows(items, args.stale_days)
    print(render_table(headers, rows, aligns))
    print(f"\n{len(items)} cars cached", end="")
    if args.stale_days is not None:
        n_stale = sum(1 for _, c in items if _is_stale(c, args.stale_days))
        print(f" · {n_stale} stale (fetched > {args.stale_days}d ago)", end="")
    print()
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    try:
        conditions = parse_filter_conditions(args.conditions)
        ids = apply_conditions(store, conditions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not conditions:
        print("no conditions given, e.g. filter --towbar fitted_fixed --max-km 110000")
        return 2
    items = [(cid, store[cid]) for cid in sorted(ids)]
    if not items:
        print("no cars match")
        return 0
    headers, rows, aligns = _car_table_rows(items, None)
    print(render_table(headers, rows, aligns))
    pretty = " ".join(f"--{k} {v}" for k, v in conditions)
    print(f"\n{len(items)}/{len(store)} match: {pretty}")
    return 0


def cmd_pdf_id(args: argparse.Namespace) -> int:
    cid = pdf_id(args.file)
    if cid is None:
        print("no mobile.de id found in PDF", file=sys.stderr)
        return 1
    print(cid)
    return 0


def cmd_intake(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    items = scan_inbox(store, Path(args.inbox), Path(args.processed), force=args.force)
    if not items:
        print(f"inbox is empty: {args.inbox}")
        return 0

    new = [it for it in items if it.status == "new"]
    cached = [it for it in items if it.status == "cached"]
    unknown = [it for it in items if it.status == "unknown-id"]

    for it in cached:
        print(f"cached   {it.cid}  already stored — archived to {it.archived_to.name}")
    for it in unknown:
        print(f"unknown  {it.path.name}  — no id found, read it manually")
    for it in new:
        print(f"NEW      {it.cid}  {it.path.name}  — needs extraction")

    print(f"\n{len(new)} new · {len(cached)} cached (skipped) · {len(unknown)} unknown")
    if new:
        rel = Path(args.inbox).name
        print("\nExtract each NEW listing, then add it, e.g.:")
        for it in new:
            print(f"  read {rel}/{it.path.name}  →  "
                  f"python3 carcache.py add {it.cid} --file {it.cid}.json --pdf {rel}/{it.path.name}")
    return 0


def _run_refresh(store: dict, ids: Iterable[str] | None, store_path: str) -> list[str]:
    """Shared by the `refresh` command and `--refresh`. Pings, saves, prints a summary."""
    rows, newly_inactive = refresh_store(store, ids)
    n_active = sum(1 for _, _, s, _ in rows if s == "active")
    n_inactive = sum(1 for _, _, s, _ in rows if s == "inactive")
    n_unknown = sum(1 for _, _, s, _ in rows if s == "unknown")
    save_store(store, store_path)
    print(f"refreshed {len(rows)}: {n_active} live · {n_inactive} sold · "
          f"{n_unknown} unverified")
    if n_unknown and not n_active and not n_inactive:
        print("  (mobile.de returned 403/blocked for every request — availability could not "
              "be verified from here; use `set-active` to record it manually.)")
    if newly_inactive:
        print("⚠ newly unavailable:")
        for cid in newly_inactive:
            print(f"    {cid} {trunc(_display_name(store[cid]), 40)}")
    return newly_inactive


def cmd_refresh(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    ids = args.ids or None
    rows, _ = refresh_store(store, ids)
    save_store(store, args.store)
    print(render_table(
        ["id", "was", "now", "detail", "model"],
        [[cid, prev, now, detail, trunc(_display_name(store[cid]), 32)] for cid, prev, now, detail in rows],
    ))
    n_unknown = sum(1 for _, _, s, _ in rows if s == "unknown")
    if n_unknown:
        print(f"\n{n_unknown} unverified — mobile.de commonly blocks automated checks (403); "
              "record truth with:  carcache.py set-active <id> sold")
    return 0


def cmd_set_active(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    cid = str(args.id)
    if cid not in store:
        print(f"{cid}: not in cache", file=sys.stderr)
        return 1
    state = args.state.lower()
    if state not in AVAILABILITY_TRUTH:
        print(f"error: state must be one of live/sold/unknown (got {args.state!r})", file=sys.stderr)
        return 2
    store[cid]["active"] = AVAILABILITY_TRUTH[state]
    store[cid]["last_checked"] = date.today().isoformat()
    save_store(store, args.store)
    print(f"{cid}: active = {store[cid]['active']} (checked {store[cid]['last_checked']})")
    return 0


def _geocode_variants(place: str) -> list[str]:
    """The query, then progressively simpler fallbacks — a miss costs the car its real travel
    time and silently drops it to the neutral fallback score, so it is worth a second try.

    * `Handewitt OT Weding` — 'OT' is Ortsteil (a district); Nominatim knows the municipality,
      not the abbreviation.
    * `Barchfeld - Immelborn` — a merged municipality; the first half resolves alone.
    * ligatures (`Sindelﬁngen`) are folded by _fold_text() at parse time, but old records may
      still carry them.
    """
    seen, out = set(), []
    for cand in (place,
                 re.split(r"\s+(?:OT|Ortsteil)\s+", place)[0],
                 re.split(r"\s*[-/]\s*", place)[0],
                 unicodedata.normalize("NFKC", place)):
        cand = cand.strip()
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _geocode(place: str, timeout: float = 20.0) -> tuple[float, float] | None:
    """Nominatim (OpenStreetMap) place -> (lat, lon). Network; used only by `travel`."""
    for cand in _geocode_variants(place):
        hit = _geocode_once(cand, timeout)
        if hit:
            return hit
    return None


def _geocode_once(place: str, timeout: float = 20.0) -> tuple[float, float] | None:
    import urllib.parse
    import urllib.request
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": f"{place}, Germany", "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "carcache/1.0 (personal car search)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    try:
        data = json.loads(raw)
    except ValueError:
        return None          # rate-limit / error page: not JSON, and not fatal
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def _osrm_route(origin: tuple[float, float], dest: tuple[float, float],
                timeout: float = 20.0) -> tuple[float, float] | None:
    """OSRM driving route -> (minutes, km). Network; used only by `travel`."""
    import urllib.request
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}?overview=false")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    route = data["routes"][0]
    return route["duration"] / 60.0, route["distance"] / 1000.0


def cmd_travel(args: argparse.Namespace) -> int:
    """Fill travel_minutes / travel_km / travel_mode for each listing from the profile origin.
    Runs ONCE per listing and stores the result, so rank/report stay network-free. Public
    transport would be preferred but the free DB endpoint is unavailable, so this uses OSRM
    driving time (Nominatim for geocoding, rate-limited to 1 req/s per their policy)."""
    import time
    store = load_store(args.store)
    profile = load_profile(args.profile)
    cfg = profile.get("location", {})
    o = cfg.get("origin", {})
    origin = (float(o.get("lat", 52.52)), float(o.get("lon", 13.405)))
    ids = [str(i) for i in (args.ids or list(store))]

    geo: dict[str, tuple[float, float] | None] = {}
    updated = skipped = failed = 0
    for cid in ids:
        car = store.get(cid)
        if car is None:
            continue
        if car.get("travel_minutes") is not None and not args.force:
            skipped += 1
            continue
        place = car.get("location")
        if not place:
            print(f"  {cid}: no location on record")
            failed += 1
            continue
        if place not in geo:
            try:
                geo[place] = _geocode(place)
            except Exception as exc:  # network/parse — treat as unknown
                print(f"  geocode {place}: {exc}")
                geo[place] = None
            time.sleep(1.1)  # Nominatim: max 1 request/second
        dest = geo[place]
        if dest is None:
            print(f"  {cid} {place}: geocode failed")
            failed += 1
            continue
        try:
            route = _osrm_route(origin, dest)
        except Exception as exc:
            print(f"  route {place}: {exc}")
            route = None
        if route is None:
            print(f"  {cid} {place}: route failed")
            failed += 1
            continue
        mins, km = route
        car["travel_minutes"] = round(mins)
        car["travel_km"] = round(km)
        car["travel_mode"] = "car"
        updated += 1
        print(f"  {cid}  {trunc(place, 24):24} {round(mins):>4} min  {round(km):>4} km")

    save_store(store, args.store)
    print(f"travel from {o.get('name', 'origin')}: {updated} updated · {skipped} already had data "
          f"· {failed} failed  (mode=car / OSRM driving time)")
    return 0


def cmd_check_urls(args: argparse.Namespace) -> int:
    """Emit the worklist of listing URLs to verify in a real browser (past the 403).
    Feed the results back with `apply-availability` or `set-active`."""
    store = load_store(args.store)
    ids = [str(i) for i in args.ids] if args.ids else list(store.keys())
    rows: list[list[Any]] = []
    for cid in ids:
        car = store.get(cid)
        if car is None:
            continue
        if args.unchecked and car.get("active") is not None:
            continue
        if args.stale_days is not None:
            age = days_old(car.get("last_checked"))
            if age is not None and age <= args.stale_days:
                continue
        rows.append([cid, active_marker(car), car.get("last_checked") or "never", listing_url(cid, car)])

    if args.json:
        print(json.dumps([{"id": r[0], "url": r[3]} for r in rows], indent=2))
        return 0
    if not rows:
        print("nothing to check")
        return 0
    print(render_table(["id", "act", "last_checked", "url"], rows))
    print(f"\n{len(rows)} to verify. Open each url in a browser, then record results with:\n"
          "  carcache.py set-active <id> live|sold        (one car)\n"
          "  carcache.py apply-availability --file r.json  (batch: {\"<id>\": \"live|sold\"})")
    return 0


def cmd_apply_availability(args: argparse.Namespace) -> int:
    """Apply a {id: live|sold|unknown} JSON mapping (e.g. produced from a browser pass)."""
    store = load_store(args.store)
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    today = date.today().isoformat()
    applied = 0
    newly_inactive: list[str] = []
    for cid, state in data.items():
        cid = str(cid)
        if cid not in store:
            print(f"  skip {cid}: not in cache", file=sys.stderr)
            continue
        key = str(state).lower()
        if key not in AVAILABILITY_TRUTH:
            print(f"  skip {cid}: bad state {state!r}", file=sys.stderr)
            continue
        prev = store[cid].get("active")
        store[cid]["active"] = AVAILABILITY_TRUTH[key]
        store[cid]["last_checked"] = today
        applied += 1
        if AVAILABILITY_TRUTH[key] is False and prev is not False:
            newly_inactive.append(cid)
    save_store(store, args.store)
    print(f"applied {applied} availability update(s)")
    if newly_inactive:
        print("⚠ newly unavailable:")
        for cid in newly_inactive:
            print(f"    {cid} {trunc(_display_name(store[cid]), 40)}")
    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    """Find listings that are the same physical car and keep only the newest.

    Dealers re-list a car under a new id when the old ad expires, so the store ends up holding
    the same vehicle twice — and it gets ranked twice. The older records are kept (nothing is
    deleted) but marked `superseded_by`, which takes them out of `rank` and the report."""
    store = load_store(args.store)
    groups = duplicate_groups(store)
    if not groups:
        print("no duplicates found")
        return 0
    for key, ids in groups:
        winner = ids[-1]
        print(f"\n{' '.join(str(k) for k in key[:2])} · {key[2]} · {key[3]} · {key[4]:,} km")
        for cid in ids:
            mark = "KEEP " if cid == winner else "super"
            car = store[cid]
            print(f"  {mark} {cid}  €{car.get('price_eur') or 0:,}  "
                  f"{active_marker(car)}  {trunc(_display_name(car), 34)}")
    if args.dry_run:
        print(f"\n{len(groups)} duplicate group(s) — nothing written (drop --dry-run to apply)")
        return 0
    marked = apply_dedupe(store)
    save_store(store, args.store)
    print(f"\n{len(marked)} listing(s) marked superseded; they stay in the store but are "
          f"excluded from rank/report")
    return 0


def cmd_recompute(args: argparse.Namespace) -> int:
    """Recompute score + TCO snapshots for the whole set and store them, stamped with the
    current scoring version + profile fingerprint. Ranking itself is always live; this just
    refreshes the persisted snapshot that `status` compares against."""
    store = load_store(args.store)
    profile = load_profile(args.profile)
    results = compute_ranking(store, profile)  # full set = canonical scores
    fp = profile_fingerprint(profile)
    today = date.today().isoformat()
    for r in results:
        store[r.cid]["computed"] = {
            "total": r.total, "tco": r.tco, "tco_estimated": r.tco_estimated,
            "tco_breakdown": r.tco_breakdown, "scores": r.scores,
            "scoring_version": SCORING_VERSION, "profile_fingerprint": fp, "computed_at": today,
        }
    save_store(store, args.store)
    print(f"recomputed {len(results)} ranked  (scoring v{SCORING_VERSION}, profile {fp})")
    return 0


def apply_parkplatz(store: dict, entries: list, today: str) -> dict:
    """Update price + availability from a mobile.de parkplatz scrape. `entries` is a list of
    {id, price_eur?, price_rating?, available?}. Returns a summary of what changed."""
    price_changes: list[tuple] = []
    newly_inactive: list[str] = []
    unknown: list[tuple] = []
    seen: set[str] = set()
    for e in entries:
        cid = str(e.get("id"))
        if not cid or cid == "None":
            continue
        seen.add(cid)
        if cid not in store:
            unknown.append((cid, e.get("price_eur")))
            continue
        car = store[cid]
        new_price = e.get("price_eur")
        if new_price is not None and car.get("price_eur") != new_price:
            price_changes.append((cid, car.get("price_eur"), new_price))
            car["price_eur"] = new_price
        if e.get("price_rating"):
            car["price_rating"] = e["price_rating"]
        avail = e.get("available")
        if avail is not None:
            prev = car.get("active")
            car["active"] = bool(avail)
            if not avail and prev is not False:
                newly_inactive.append(cid)
        car["last_checked"] = today
    return {
        "price_changes": price_changes,
        "newly_inactive": newly_inactive,
        "unknown": unknown,
        "not_in_parkplatz": [cid for cid in store if cid not in seen],
    }


def cmd_apply_parkplatz(args: argparse.Namespace) -> int:
    """Ingest a parkplatz scrape (list of {id, price_eur, available, price_rating}) and update
    each matched car's price + availability. Reports price moves, sold cars, and new ids."""
    store = load_store(args.store)
    entries = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if isinstance(entries, dict):  # allow {id: {...}} too
        entries = [{"id": k, **v} for k, v in entries.items()]
    r = apply_parkplatz(store, entries, date.today().isoformat())
    save_store(store, args.store)

    print(f"parkplatz: {len(entries)} listing(s) · {len(r['price_changes'])} price change(s) · "
          f"{len(r['newly_inactive'])} newly sold · {len(r['unknown'])} not yet in store")
    for cid, old, new in r["price_changes"]:
        arrow = "↓" if (old or 0) > new else "↑"
        print(f"  price {arrow} {cid} {trunc(_display_name(store[cid]), 34)}: €{old:,} → €{new:,}")
    if r["newly_inactive"]:
        print("⚠ newly unavailable:")
        for cid in r["newly_inactive"]:
            print(f"    {cid} {trunc(_display_name(store[cid]), 40)}")
    if r["unknown"]:
        print("new in parkplatz (not yet extracted — add via a detail-page/PDF pass):")
        for cid, price in r["unknown"]:
            print(f"    {cid}  €{price:,}" if price else f"    {cid}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show which stored cars have drifted from the current extraction / scoring logic."""
    store = load_store(args.store)
    profile = load_profile(args.profile)
    fp = profile_fingerprint(profile)
    print(f"extraction logic: v{EXTRACT_VERSION}   scoring logic: v{SCORING_VERSION}   "
          f"profile fingerprint: {fp}\n")

    rows: list[list[Any]] = []
    need_reprocess = need_recompute = missing_pdf = 0
    for cid, car in sorted(store.items()):
        ex_stale = is_extract_stale(car)
        # A superseded listing is never ranked, so `recompute` never refreshes its snapshot —
        # flagging it score-stale would nag about drift that can never be resolved. Its
        # extraction can still go stale, and that is worth reporting: it keeps its PDF.
        sc_stale = is_score_stale(car, profile) and not car.get("superseded_by")
        has_pdf = (PROCESSED_DIR / f"{cid}.pdf").exists()
        if ex_stale:
            need_reprocess += has_pdf
            missing_pdf += not has_pdf
        need_recompute += sc_stale
        rows.append([
            cid, trunc(_display_name(car), 30),
            car.get("extract_version") if car.get("extract_version") is not None else "-",
            "STALE" if ex_stale else "ok",
            "yes" if has_pdf else "NO",
            "STALE" if sc_stale else "ok",
        ])
    print(render_table(["id", "model", "extract_v", "extract", "pdf", "scores"], rows,
                       "llrlll"))

    print()
    if need_reprocess:
        print(f"• {need_reprocess} car(s) extract-STALE with an archived PDF → rebuild them:")
        print("    python3 extract.py --reprocess          (stale only; --all for every car)")
    if missing_pdf:
        print(f"• {missing_pdf} car(s) extract-STALE but the PDF is missing from processed/ "
              "→ drop the PDF in inbox/ and re-add.")
    if need_recompute:
        print(f"• {need_recompute} car(s) score-STALE (scoring logic or profile changed) → "
              "refresh snapshots:")
        print("    python3 carcache.py recompute")
    if not (need_reprocess or missing_pdf or need_recompute):
        print("everything is up to date.")
    print("\nnote: rank/report always compute live with the CURRENT logic + profile; the "
          "stored snapshot only drives this status view.")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    profile = load_profile(args.profile)
    if getattr(args, "refresh", False):
        _run_refresh(store, args.ids or None, args.store)
        print()
    results = compute_ranking(
        store, profile, ids=args.ids or None, stale_days=args.stale_days
    )

    if not results:
        print("no cars to rank")
    else:
        show_stale = args.stale_days is not None
        headers = ["#", "id", "model", "€price", "year", "km", "PS", "SCORE", "TCO/yr", "act"]
        aligns = "rllrrrrrrl"
        if show_stale:
            headers.append("fresh")
            aligns += "l"
        rows: list[list[Any]] = []
        for rank, r in enumerate(results, 1):
            row = [
                rank, r.cid, trunc(_display_name(r.car), 32),
                fmt_int(r.car.get("price_eur")),
                reg_year(r.car.get("first_reg")) or "-",
                fmt_int(r.car.get("km")),
                r.car.get("power_ps") or "-",
                f"{r.total:.2f}",
                f"{r.tco:,.0f}{'*' if r.tco_estimated else ''}",
                active_marker(r.car),
            ]
            if show_stale:
                row.append("STALE" if r.stale else "ok")
            rows.append(row)
        print(render_table(headers, rows, aligns))

        # Per-criterion breakdown
        print("\nPer-criterion breakdown (0–10, weighted total = SCORE):")
        bh = ["#", "id"] + [CRIT_ABBR[c] for c in CRITERIA] + ["SCORE"]
        ba = "rl" + "r" * len(CRITERIA) + "r"
        brows: list[list[Any]] = []
        for rank, r in enumerate(results, 1):
            brows.append([rank, r.cid] + [f"{r.scores[c]:.1f}" for c in CRITERIA] + [f"{r.total:.2f}"])
        print(render_table(bh, brows, ba))
        legend = "  ".join(f"{CRIT_ABBR[c]}={c}({profile['weights'][c]:.2f})" for c in CRITERIA)
        print(f"\nlegend: {legend}")
        print("TCO/yr in €; * = consumption estimated (no stated figure).")

        # What each criterion actually contributes to the ORDER (weight x observed spread) —
        # a criterion everyone scores the same on cannot rank anyone, whatever its weight.
        if len(results) > 1:
            print("\nCriterion influence on this ranking (weight × spread of scores):")
            inf = criteria_influence(results, profile)
            irows = [[d["criterion"], f"{d['weight']:.2f}", f"{d['mean']:.2f}", f"{d['sd']:.2f}",
                      f"{d['influence']:.3f}", f"{d['share']:.0%}"] for d in inf]
            print(render_table(["criterion", "weight", "mean", "sd", "w×sd", "share"],
                               irows, "lrrrrr"))
            print("share = fraction of the ranking's total spread this criterion drives; "
                  "compare it to the nominal weight when tuning.")

        n_sens = getattr(args, "sensitivity", None)
        if n_sens and len(results) > 1:
            sens = weight_sensitivity(results, profile, n=n_sens)
            print(f"\nWeight sensitivity: {sens['n']} rankings with every weight independently "
                  f"perturbed ×[{1 - sens['spread']:.2f}, {1 + sens['spread']:.2f}]:")
            name = {r.cid: trunc(_display_name(r.car), 34) for r in results}
            srows = [[cid, name[cid],
                      f"{sens['top1'].get(cid, 0) / sens['n']:.0%}",
                      f"{cnt / sens['n']:.0%}"]
                     for cid, cnt in sorted(sens["top3"].items(),
                                            key=lambda kv: (-sens["top1"].get(kv[0], 0), -kv[1]))]
            print(render_table(["id", "model", "P(#1)", "P(top 3)"], srows, "llrr"))
            print("100% P(#1) = the winner does not depend on the exact weights.")

        inactive = [r for r in results if r.car.get("active") is False]
        if inactive:
            print("\n⚠ no longer available (marked SOLD):")
            for r in inactive:
                print(f"  {r.cid} {trunc(_display_name(r.car), 32)} — last checked "
                      f"{r.car.get('last_checked') or '?'}")
        print("act = live/SOLD/? · run `rank --refresh` to re-check availability "
              "(mobile.de may block automated checks).")

    return 0


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def viewing_checklist(car: dict, profile: dict) -> list[str]:
    """What to ask or inspect BEFORE/AT a viewing, derived from this record only: every
    scored finding, plus each fact the seller did not state (a None is 'the seller did not
    say' — an answer for scoring, but a question for a buyer). Empty list = nothing beyond
    the standard used-car look-over."""
    items: list[str] = []

    if car.get("unfallfrei_declared") is None:
        items.append("Accident-free was NOT declared — ask explicitly, and get it in writing.")
    if car.get("service_history") is not True:
        stated = "seller states there is none" if car.get("service_history") is False \
            else "not stated in the listing"
        items.append(f"Service history ({stated}) — ask for the Scheckheft / invoices.")
    if not car.get("tuv"):
        items.append("HU/TÜV date not stated — ask when it expires.")
    if car.get("owners") is None:
        items.append("Number of previous owners not stated — ask (and check the Fahrzeugbrief).")
    if car.get("price_rating") is None:
        items.append("mobile.de gave this price no rating — compare against similar listings "
                     "yourself before negotiating.")
    wishlist = profile.get("wishlist", {})
    if "towbar" in wishlist and car.get("towbar") in ("capable", "prep"):
        items.append("Towbar is only prepared/possible, not fitted — ask what a retrofit costs.")

    sev_rank = {"major": 0, "minor": 1, "note": 2}
    findings = [f for f in (car.get("findings") or []) if isinstance(f, dict)]
    for f in sorted(findings, key=lambda x: sev_rank.get(x.get("severity"), 3)):
        sev, cat = f.get("severity"), f.get("category")
        summary = str(f.get("summary") or "").rstrip(".")
        if not summary:
            continue
        if cat == "inconsistency":
            items.append(f"Clarify the contradiction: {summary}.")
        elif sev in ("major", "minor"):
            items.append(f"Inspect ({sev}): {summary}.")
    return items


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def build_report(store: dict, profile: dict, ids: Iterable[str] | None = None) -> str:
    results = compute_ranking(store, profile, ids=ids)
    lines: list[str] = []

    lines.append("# carcache report")
    lines.append("")
    lines.append(f"_Generated {date.today().isoformat()} · {len(results)} candidates ranked._")
    lines.append("")
    inactive = [r for r in results if r.car.get("active") is False]
    if inactive:
        listed = ", ".join(f"{_display_name(r.car)} ({r.cid})" for r in inactive)
        lines.append(f"> ⚠ **{len(inactive)} listing(s) no longer available:** {listed}")
        lines.append("")
    weight_str = " · ".join(f"{CRIT_LABEL[c]} {fmt_weight(profile['weights'][c])}" for c in CRITERIA)
    lines.append(f"**Weights:** {weight_str}")
    lines.append("")

    # Ranking table
    lines.append("## Ranking")
    lines.append("")
    rank_rows = []
    for rank, r in enumerate(results, 1):
        rank_rows.append([
            rank, r.cid, _display_name(r.car),
            f"€{r.car.get('price_eur'):,}" if r.car.get("price_eur") is not None else "-",
            reg_year(r.car.get("first_reg")) or "-",
            f"{r.car.get('km'):,}" if r.car.get("km") is not None else "-",
            r.car.get("power_ps") or "-",
            f"**{r.total:.2f}**",
            f"€{r.tco:,.0f}{'*' if r.tco_estimated else ''}",
        ])
    lines.append(_md_table(
        ["#", "id", "model", "price", "year", "km", "PS", "score", "TCO/yr"], rank_rows))
    lines.append("")
    lines.append("_Score is weighted 0–10; TCO is shown separately — only its cash part "
                 "(minus depreciation) feeds the Cost criterion. * = consumption estimated._")
    lines.append("")

    # How stable is this ranking? (influence + weight sensitivity)
    lines.append("## How stable is this ranking?")
    lines.append("")
    if len(results) > 1:
        lines.append("_**Influence** = weight × spread of scores: what each criterion actually "
                     "contributes to the *order*. A criterion every car scores the same on "
                     "cannot rank anyone, whatever its weight — compare `share` to the nominal "
                     "weight when tuning._")
        lines.append("")
        inf = criteria_influence(results, profile)
        lines.append(_md_table(
            ["criterion", "weight", "mean", "sd", "w×sd", "share of ranking"],
            [[d["criterion"], f"{d['weight']:.2f}", f"{d['mean']:.2f}", f"{d['sd']:.2f}",
              f"{d['influence']:.3f}", f"{d['share']:.0%}"] for d in inf]))
        lines.append("")
        sens = weight_sensitivity(results, profile)
        name = {r.cid: _display_name(r.car) for r in results}
        lines.append(f"_**Weight sensitivity**: {sens['n']} re-rankings with every weight "
                     f"independently perturbed ×[{1 - sens['spread']:.2f}, "
                     f"{1 + sens['spread']:.2f}] (deterministic seed). A car at 100% P(#1) wins "
                     f"under every plausible weighting — the decision does not hinge on the "
                     f"exact weights._")
        lines.append("")
        srows = [[cid, trunc(name[cid], 40),
                  f"{sens['top1'].get(cid, 0) / sens['n']:.0%}", f"{cnt / sens['n']:.0%}"]
                 for cid, cnt in sorted(sens["top3"].items(),
                                        key=lambda kv: (-sens["top1"].get(kv[0], 0), -kv[1]))]
        lines.append(_md_table(["id", "model", "P(#1)", "P(top 3)"], srows))
        lines.append("")

    # Shortlist head-to-head: where the top cars actually differ
    top_n = results[:5]
    if len(top_n) >= 2:
        lines.append("## Shortlist head-to-head (top 5)")
        lines.append("")
        differing = [c for c in CRITERIA
                     if max(r.scores[c] for r in top_n) - min(r.scores[c] for r in top_n) > 1.0]
        if differing:
            lines.append("_Only the criteria on which the top 5 meaningfully differ (spread "
                         "> 1.0) — everything else is a wash between them and explains nothing._")
            lines.append("")
            hrows = []
            for r in top_n:
                hrows.append([trunc(_display_name(r.car), 30)]
                             + [f"**{r.scores[c]:.1f}**"
                                if r.scores[c] == max(x.scores[c] for x in top_n)
                                else f"{r.scores[c]:.1f}" for c in differing]
                             + [f"{r.total:.2f}"])
            lines.append(_md_table(["model"] + [CRIT_LABEL[c] for c in differing] + ["SCORE"],
                                   hrows))
            lines.append("")
            for c in differing:
                best = max(top_n, key=lambda r: r.scores[c])
                worst = min(top_n, key=lambda r: r.scores[c])
                lines.append(f"- **{CRIT_LABEL[c]}**: {_display_name(best.car)} "
                             f"({best.scores[c]:.1f}) vs {_display_name(worst.car)} "
                             f"({worst.scores[c]:.1f})")
            lines.append("")
        else:
            lines.append("_The top 5 score within 1.0 of each other on every criterion — the "
                         "order is driven by small differences everywhere, not one axis._")
            lines.append("")

    # Annual cost breakdown (ADAC-style components)
    lines.append("## Annual cost breakdown (€/yr)")
    lines.append("")
    lines.append(f"_ADAC-style model over a {profile['tco'].get('hold_years', 5)}-year hold at "
                 f"{profile['tco'].get('annual_km', 6000):,} km/yr. Depreciation is estimated value "
                 f"loss from the current price; Kfz-Steuer is computed exactly; maintenance, insurance "
                 f"and depreciation are estimates (see profile `_sources`). **cash/yr** (everything "
                 f"except depreciation) is what the Cost score uses — depreciation is derived from "
                 f"the asking price, which already has its own third of Cost, so scoring it too "
                 f"would charge the same euros twice._")
    lines.append("")
    cost_rows = []
    for rank, r in enumerate(results, 1):
        b = r.tco_breakdown
        cost_rows.append([
            rank, trunc(_display_name(r.car), 34),
            f"{b.get('depreciation', 0):,}", f"{b.get('fuel', 0):,}",
            f"{b.get('electricity', 0):,}" if b.get("electricity") else "–",
            f"{b.get('maintenance', 0):,}",
            f"{b.get('insurance', 0):,}", f"{b.get('tax', 0):,}", f"{b.get('hu', 0):,}",
            f"**{r.running:,.0f}**", f"{r.tco:,.0f}",
        ])
    lines.append(_md_table(
        ["#", "model", "deprec.", "petrol", "electr.", "mainten.", "insur.", "tax", "HU",
         "cash/yr", "TOTAL"],
        cost_rows))
    lines.append("")

    # Model × criteria matrix
    lines.append("## Model × criteria matrix")
    lines.append("")
    matrix_rows = []
    for r in results:
        matrix_rows.append(
            [trunc(_display_name(r.car), 40)]
            + [f"{r.scores[c]:.1f}" for c in CRITERIA]
            + [f"{r.total:.2f}"]
        )
    lines.append(_md_table(["model"] + [CRIT_LABEL[c] for c in CRITERIA] + ["SCORE"], matrix_rows))
    lines.append("")

    # Reliability derivation (objective per-cell TÜV/ADAC figures matched to each listing)
    lines.append("## Reliability — how each score is derived")
    lines.append("")
    lines.append("_Computed from objective per-cell figures matched to each listing: **TÜV** = the "
                 "model's Mängelquote for the car's age band; **ADAC** = its breakdowns-per-1000 for the "
                 "car's registration year (both lower = better). Each cell maps to 0–10 vs a fixed market "
                 "reference (**s**), then averaged 50/50. Older band / year ⇒ higher rate ⇒ lower score. "
                 "ADAC cells are official (ADAC Pannenstatistik 2025); TÜV cells are modelled from the "
                 "official age-band averages (per-model table is print-only). `–` = no cell (scored on the "
                 "other metric); `n/a` = neither → neutral fallback._")
    lines.append("")
    rb_rows = []
    for rank, r in enumerate(results, 1):
        d = reliability_detail(r.car, profile)
        comp = d["components"]
        t, a = comp.get("tuv"), comp.get("adac")
        rb_rows.append([
            rank, trunc(_display_name(r.car), 40),
            f"{t['key']}y" if t else "–", f"{_fmt_num(t['raw'])}%" if t else "–",
            f"{t['sub']:.1f}" if t else "–",
            a["key"] if a else "–", _fmt_num(a["raw"]) if a else "–",
            f"{a['sub']:.1f}" if a else "–",
            f"**{r.scores['reliability']:.1f}**" + (" `n/a`" if d["fallback"] else ""),
        ])
    lines.append(_md_table(
        ["#", "model", "TÜV band", "TÜV %", "s·TÜV", "reg yr", "ADAC /1k", "s·ADAC", "Reliability"],
        rb_rows))
    lines.append("")

    # Safety (Euro NCAP) — informational, deliberately unscored
    safety_rows, seen = [], set()
    for r in results:
        s = safety_basis(r.car, profile)
        if not s:
            continue
        key = (r.car.get("make"), r.car.get("model"), s.get("test_year"))
        if key in seen:
            continue
        seen.add(key)
        safety_rows.append([f"{r.car.get('make')} {r.car.get('model')}",
                            "★" * int(s.get("stars", 0)), f"{s.get('adult_pct')}%",
                            s.get("test_year"), s.get("note") or ""])
    if safety_rows:
        lines.append("## Safety (Euro NCAP) — informational, not scored")
        lines.append("")
        lines.append("_Every model here is **5-star**, and NCAP percentages are **not comparable "
                     "across test years** (the protocol gets stricter every few years — 85% under "
                     "the 2018 protocol is not worse than 94% under the 2012 one). Scoring this "
                     "would rank test vintage, not safety, so it is shown for reading, with the "
                     "test year, and deliberately kept out of the ranking._")
        lines.append("")
        safety_rows.sort(key=lambda row: str(row[0]))
        lines.append(_md_table(["model", "stars", "adult occupant", "test year", "note"],
                               safety_rows))
        lines.append("")

    # Equipment detail (which wishlist items each listing carries)
    lines.append("## Equipment detail")
    lines.append("")
    if float(profile.get("equipment", {}).get("year_weight", 0)) > 0:
        lines.append("_✓ = present. The Equipment score blends this weighted wishlist with "
                     "registration-year recency (`year` column)._")
    else:
        lines.append("_✓ = present. The Equipment score is the weighted share of this "
                     "wishlist (the `year` column is informational)._")
    lines.append("")
    wl = profile.get("wishlist", {})
    if isinstance(wl, list):
        wl = {o: 1 for o in wl}
    opts = sorted(wl.keys(), key=lambda o: (-float(wl[o]), o))
    eq_headers = (["#", "model"]
                  + [f"{OPTION_LABELS.get(o, o)} (×{float(wl[o]):g})" for o in opts]
                  + ["year", "Equipment"])
    eq_rows = []
    for rank, r in enumerate(results, 1):
        car = r.car
        eq_rows.append(
            [rank, trunc(_display_name(car), 34)]
            + ["✓" if _has_option(car, o) else "·" for o in opts]
            + [reg_year(car.get("first_reg")) or "–", f"{r.scores['equipment_fit']:.1f}"]
        )
    lines.append(_md_table(eq_headers, eq_rows))
    lines.append("")

    # Per-car notes
    lines.append("## Per-car notes")
    lines.append("")
    for rank, r in enumerate(results, 1):
        car = r.car
        top = sorted(CRITERIA, key=lambda c: profile["weights"][c] * r.scores[c], reverse=True)[:3]
        why = ", ".join(f"{c} {r.scores[c]:.1f}" for c in top)
        standout_crit = max(CRITERIA, key=lambda c: r.scores[c])
        options = ", ".join(car.get("notable_options") or []) or "—"
        findings = [f for f in (car.get("findings") or []) if isinstance(f, dict)]

        lines.append(f"### {rank}. {_display_name(car)}  ·  score {r.total:.2f}  ·  id {r.cid}")
        lines.append("")
        b = r.tco_breakdown
        brk = (f" — deprec €{b.get('depreciation', 0):,} · fuel €{b.get('fuel', 0):,} · "
               f"maint €{b.get('maintenance', 0):,} · ins €{b.get('insurance', 0):,} · "
               f"tax €{b.get('tax', 0):,} · HU €{b.get('hu', 0):,}") if b else ""
        lines.append(f"- **Price / TCO:** €{car.get('price_eur'):,} · ~€{r.tco:,.0f}/yr"
                     f"{' est.' if r.tco_estimated else ''}{brk}")
        lines.append(f"- **Why it fits:** strongest on {why}")
        lines.append(f"- **Standout:** {standout_crit} = {r.scores[standout_crit]:.1f}/10 · "
                     f"{car.get('km'):,} km · {car.get('owners')} owner(s) · "
                     f"{car.get('power_ps')} PS {car.get('fuel')}")
        lines.append(f"- **Notable options:** {options}")
        if findings:
            lines.append(f"- **Findings:** {len(findings)}")
            for f in sorted(findings, key=lambda x: {"major": 0, "minor": 1, "note": 2}
                            .get(x.get("severity"), 3)):
                lines.append(f"    - **{f.get('severity')}** ({f.get('category')}) "
                             f"{f.get('summary')} — _{f.get('evidence')}_")
        else:
            lines.append("- **Findings:** none")
        checklist = viewing_checklist(car, profile)
        if checklist:
            lines.append("- **Before viewing, ask / inspect:**")
            for item in checklist:
                lines.append(f"    - {item}")
        lines.append(f"- **Availability:** {active_marker(car)} (checked "
                     f"{car.get('last_checked') or 'never'}) · [listing]({listing_url(r.cid, car)})")
        lines.append(f"- **Provenance:** {car.get('source_pdf') or '—'} · fetched {car.get('fetched_date') or '—'}")
        lines.append("")

    return "\n".join(lines)


def _hsl_good(v: float) -> str:
    """Pastel background: 0 = red (poor), 10 = green (good)."""
    return f"hsl({clamp(v / 10.0, 0.0, 1.0) * 130:.0f} 60% 82%)"


def _hsl_cheap(value: float, lo: float, hi: float) -> str:
    """Pastel background where LOW is good (green) — for cost columns."""
    t = 0.5 if hi == lo else clamp((value - lo) / (hi - lo), 0.0, 1.0)
    return f"hsl({(1 - t) * 130:.0f} 60% 82%)"


_REPORT_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f7f7f8;color:#1c1c1e;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:min(1800px,96vw);margin:0 auto;padding:30px 20px 64px}
.rk-tbl,.mx,.eq{width:100%;table-layout:auto}
.rk-tbl td.l,.mx td.l,.eq td.l{white-space:normal;min-width:190px}
h1{font-size:25px;margin:0 0 2px}
h2{font-size:17px;margin:32px 0 8px;padding-bottom:6px;border-bottom:1px solid #e3e3e6}
.sub{color:#6b6b70;font-size:13px;margin:0 0 10px}
.filters{font-size:12px;color:#57575c;background:#ededf0;padding:8px 11px;border-radius:8px}
.toggle{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#3a3a3e;margin:12px 0 2px;cursor:pointer;user-select:none}
.badge{font-size:10px;font-weight:700;padding:1px 6px;border-radius:20px;vertical-align:1px}
.live{background:#d8f0dc;color:#1c6b2c}.sold{background:#f6d6d6;color:#a11}.unk{background:#e7e7ea;color:#88888e}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.rk-tbl{font-size:13px}
.rk-tbl th,.rk-tbl td{padding:5px 7px;text-align:center;border:2px solid #f7f7f8;white-space:nowrap}
.rk-tbl th{background:#ededf0;color:#54545a;font-weight:600;font-size:11px;letter-spacing:.02em}
.rk-tbl td.l,.rk-tbl th.l{text-align:left}
.rk-tbl td.rk{color:#9a9a9f;font-weight:700}
.rk-tbl td.score{font-weight:700}
.mname{font-weight:600;font-size:13px}.msub{font-size:11px;color:#8a8a8f}
a{color:inherit;text-decoration:none}
.mname a:hover,.mx td.l a:hover,.nt a:hover{text-decoration:underline;color:#1a5fb4}
.ext{color:#adadb3;font-size:11px;margin-left:2px}.mname a:hover .ext{color:#1a5fb4}
.warn{background:#f3cfcf;color:#a11;font-size:10px;font-weight:700;padding:0 4px;border-radius:3px;margin-left:3px}
.tag{background:#e2e6ec;color:#556;font-size:10px;padding:0 4px;border-radius:3px;margin-left:3px}
.mx{font-size:13px}
.mx th,.mx td{padding:5px 8px;text-align:center;border:2px solid #f7f7f8}
.mx th{background:#ededf0;color:#4b4b50;font-weight:600}
.mx td.l,.mx th.l{text-align:left;white-space:nowrap}
.mx td.score{font-weight:700}
.mx td.pt{font-weight:600;font-variant-numeric:tabular-nums}
h3{font-size:14px;font-weight:600;margin:22px 0 2px}
.rv{font-weight:600}
.rb{font-size:9.5px;color:#5c5c62;margin-top:1px;white-space:nowrap;font-variant-numeric:tabular-nums}
.rel2{font-size:11px;line-height:1.4;white-space:nowrap;font-variant-numeric:tabular-nums}
.cst1{font-size:13px;font-weight:600;line-height:1.3;font-variant-numeric:tabular-nums}
.rk-tbl td.cond{white-space:normal}
.cbadges{display:flex;flex-wrap:wrap;gap:2px 3px;justify-content:center;max-width:132px;margin:0 auto}
.cbadges .warn,.cbadges .tag{margin-left:0;white-space:nowrap}
.intro{font-size:13.5px;color:#3a3a3e;line-height:1.6;margin:6px 0 12px;max-width:88ch}
details.mm{margin:4px 0 0;font-size:12px;color:#5a5a5f}
details.mm summary{cursor:pointer;color:#1a5fb4;user-select:none}
details.mm div{padding:5px 0 2px;line-height:1.55}
.meth{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px 22px}
.meth section{border-top:2px solid #e3e3e6;padding-top:7px}
.meth h3{font-size:13px;margin:0 0 3px}
.meth h3 a{color:#1a5fb4}
.meth h3 .w{color:#9a9a9f;font-weight:400;font-size:11px}
.meth p{margin:0;font-size:12.5px;color:#4b4b50;line-height:1.5}
code{background:#ededf0;padding:0 3px;border-radius:3px;font-size:11.5px}
.eq{font-size:12px}
.eq th,.eq td{padding:4px 7px;text-align:center;border:2px solid #f7f7f8}
.eq th{background:#ededf0;color:#4b4b50;font-weight:600;font-size:10.5px}
.eq th.l,.eq td.l{text-align:left;white-space:nowrap}
.eq th.rot{font-size:10px;line-height:1.15}
.eq .yes{color:#1c6b2c;font-weight:700}
.eq .no{color:#c9c9ce}
.eq .wt{font-size:9px;color:#8a8a8f;font-weight:400}
.eq td.yr,.eq td.eqs{font-weight:600}
.legend{font-size:12px;color:#6b6b70;margin:8px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
.cr{display:flex;align-items:center;gap:10px;padding:4px 0}
.cr-l{width:230px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#3a3a3e}
.stack{flex:1;display:flex;height:20px;border-radius:5px;overflow:hidden;background:#e9e9ec}
.seg{height:100%;cursor:default}
.cr-t{width:60px;text-align:right;font-size:12px;color:#4b4b50}
.nt{margin:11px 0}.nt b{font-size:13px}.nt ul{margin:3px 0 0;padding-left:0;list-style:none}
.nt li{font-size:12.5px;color:#5a5a5f;margin:3px 0;padding-left:9px;border-left:2px solid #e4e4e8}
.nt li.major{border-left-color:#b0181b}.nt li.minor{border-left-color:#d98a1f}
.rsum{font-size:12.5px;color:#4b4b50;margin:2px 0 0;font-style:italic}
.sev{display:inline-block;min-width:38px;font-size:10px;font-weight:700;text-transform:uppercase;
letter-spacing:.03em;color:#8a8a90}
.nt li.major .sev{color:#b0181b}.nt li.minor .sev{color:#d98a1f}
.cat{font-size:10.5px;color:#9a9a9f;margin-right:5px}
.ev{display:block;font-size:11px;color:#9a9a9f;font-style:italic;margin-top:1px}
.foot{margin-top:40px;font-size:11px;color:#9a9a9f}
.tip{position:fixed;display:none;background:#26262a;color:#fff;font-size:12px;padding:3px 8px;border-radius:5px;pointer-events:none;z-index:20;white-space:nowrap}
body.liveonly [data-active="false"],body.liveonly [data-active="unknown"]{display:none}
@media(max-width:560px){.cr-l{width:120px}}
"""

_REPORT_JS = """
(function(){
 var tip=document.createElement('div');tip.className='tip';document.body.appendChild(tip);
 document.querySelectorAll('.seg').forEach(function(s){
  s.addEventListener('mousemove',function(e){tip.textContent=s.dataset.label+': €'+s.dataset.val;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+14)+'px';});
  s.addEventListener('mouseleave',function(){tip.style.display='none';});
 });
 var cb=document.getElementById('liveonly');
 if(cb){document.body.classList.toggle('liveonly',cb.checked);
  cb.addEventListener('change',function(){document.body.classList.toggle('liveonly',cb.checked);});}
})();
"""

_COST_KEYS = ["depreciation", "fuel", "electricity", "maintenance", "insurance", "tax", "hu"]
_COST_COLORS = {"depreciation": "#4e79a7", "fuel": "#f28e2b", "electricity": "#4bb3c4",
                "maintenance": "#59a14f", "insurance": "#e6c229", "tax": "#b07aa1", "hu": "#9c755f"}
_COST_LABELS = {"depreciation": "depreciation", "fuel": "petrol", "electricity": "electricity",
                "maintenance": "maintenance", "insurance": "insurance", "tax": "road tax", "hu": "HU"}


def _active_attr(car: dict) -> str:
    a = car.get("active")
    return "true" if a is True else ("false" if a is False else "unknown")


def _avail_badge(car: dict) -> str:
    return {True: '<span class="badge live">live</span>',
            False: '<span class="badge sold">SOLD</span>'}.get(
        car.get("active"), '<span class="badge unk">?</span>')


# Deduction badges that mean "damaged/defective" get the red treatment; softer paperwork
# deductions get a neutral tag.
_COND_SEVERE = {"accident", "defect"}


def _condition_cell(car: dict, profile: dict) -> tuple[str, str]:
    """(cell html, hover title): every deduction that was applied, shown as badges."""
    items, score = condition_detail(car, profile)
    if not items:
        return '<div class="rv">clean</div>', f"condition {score:.1f}/10 \u2014 no deductions"
    badges = "".join(
        f'<span class="{"warn" if b in _COND_SEVERE else "tag"}">{b}</span>' for b, _t, _d in items)
    title = f"condition {score:.1f}/10 \u00b7 " + " \u00b7 ".join(f"{t} ({d:+.1f})" for _b, t, d in items)
    return f'<div class="cbadges">{badges}</div>', title


def _autobahn_cell(car: dict, profile: dict) -> tuple[str, str]:
    ps = car.get("power_ps") or "?"
    seg = car_segment(car, profile)
    text = f"{ps} PS" + ('<span class="tag">mid</span>' if seg == "mid" else "")
    return text, f"{ps} PS · {seg} segment"


def _fmt_num(x: Any) -> str:
    """Trim a trailing '.0' so 5.0 shows as 5 but 5.7 stays 5.7."""
    if x is None:
        return "–"
    f = float(x)
    return str(int(f)) if f == int(f) else f"{f:g}"


PRICE_RATING_LABEL = {"sehr_gut": "sehr guter Preis", "gut": "guter Preis", "fair": "fairer Preis",
                      "erhoeht": "erhöhter Preis", "hoch": "hoher Preis"}


def _fuel_source_label(car: dict, profile: dict) -> str:
    """Where this car's litres/100km came from — measured, the listing's WLTP, or a model."""
    rc = real_consumption(car, profile)
    if rc:
        return (f"real-world {rc['real_l100']:g} l/100km from {rc['n']} logged vehicles "
                f"(spritmonitor.de)")
    stated = car.get("consumption_l100")
    if stated is not None:
        return f"listing's stated {stated} l/100km (WLTP — typically optimistic)"
    return "consumption estimated (no stated figure)"


def _cost_cell(car: dict, profile: dict, r: "RankResult") -> tuple[str, str]:
    """(cell html, hover title) for the composite Cost column: the score, with the €/yr it used
    to be underneath so the table still shows a real number. The hover names all three parts —
    the score alone would hide whether a car is cheap, well-priced, or cheap to run."""
    items, composite = cost_detail(car, profile, r.running)
    est = "*" if r.tco_estimated else ""
    inner = (f'<div class="cst1">{composite:.1f}</div>'
             f'<div class="rel2">€{r.tco:,.0f}/yr{est}</div>')
    rc = real_consumption(car, profile)
    fuel_src = (f"real {rc['real_l100']:g} l/100km (n={rc['n']})" if rc
                else f"stated {car.get('consumption_l100')} l/100km WLTP"
                if car.get("consumption_l100") else "consumption estimated")
    raw = {"price": f"€{car.get('price_eur') or 0:,}",
           "price_rating": PRICE_RATING_LABEL.get(car.get("price_rating"), "not rated"),
           "running": f"€{r.running:,.0f}/yr cash{est} · {fuel_src} "
                      f"(TCO €{r.tco:,.0f} incl. deprec.)"}
    parts = [f"{label} {raw[key]} → {score:.1f} ({share:.0%})"
             for key, label, score, share in items]
    return inner, " · ".join(parts) + f"  ⇒ {composite:.2f}"


def _reliability_cell(car: dict, profile: dict, score: float) -> tuple[str, str]:
    """(cell html, hover title): the two objective figures only (no score number) \u2014 the TÜV
    Mängelquote for this car's age band and the ADAC rate for its registration year. A metric
    with no cell shows an `n/a` badge. The cell background still carries the 0-10 score."""
    d = reliability_detail(car, profile)
    comp = d["components"]
    lines, parts = [], []
    for metric, label, unit in (("tuv", "TÜV", "%"), ("adac", "ADAC", "")):
        c = comp.get(metric)
        if c is None:
            lines.append(f'<div class="rel2">{label} <span class="warn">n/a</span></div>')
            parts.append(f"{label} no cell for this listing")
        else:
            lines.append(f'<div class="rel2">{label} {_fmt_num(c["raw"])}{unit}</div>')
            key = c["key"] + "y" if metric == "tuv" else c["key"]
            parts.append(f"{label} {key}: {_fmt_num(c['raw'])}{unit} \u21d2 {c['sub']:.1f}/10")
    head = "no TÜV/ADAC cell \u2014 neutral fallback" if d["fallback"] else f"reliability {score:.1f}/10"
    return "".join(lines), head + " \u00b7 " + " \u00b7 ".join(parts)


def _fmt_hm(minutes: float) -> str:
    """135 -> '2h15', 45 -> '45 min'."""
    m = int(round(minutes))
    h, r = divmod(m, 60)
    return f"{h}h{r:02d}" if h else f"{r} min"


def _location_cell(car: dict, profile: dict) -> tuple[str, str]:
    """(cell inner html, hover title): travel time from the origin, with the dealer town."""
    place = car.get("location") or "?"
    mins, mode = travel_minutes_for(car, profile)
    if mins is None:
        return (f'<span class="warn">n/a</span><div class="rb">{place}</div>',
                f"{place} — no travel data (run `carcache.py travel`)")
    origin = profile.get("location", {}).get("origin", {}).get("name", "origin")
    title = f"{place} · {_fmt_hm(mins)} from {origin}"
    km = car.get("travel_km")
    if km:
        title += f" · {km:,} km"
    title += f" · by {mode}"
    return f'<div class="rv">{_fmt_hm(mins)}</div><div class="rb">{place}</div>', title


def _equipment_cell(car: dict, profile: dict, score: float) -> tuple[str, str]:
    """(cell html, hover title): the equipment score. The registration year feeds it but is
    not repeated here \u2014 it is already in the model column."""
    yr = reg_year(car.get("first_reg"))
    ys = year_score(car, profile)
    share = wishlist_share(car, profile)
    title = f"equipment {score:.1f}/10 \u00b7 wishlist {share:.1f}/10"
    if ys is not None:
        title += f" \u00b7 year {yr} ({ys:.1f}/10)"
    return f'<div class="rv">{score:.1f}</div>', title


def _report_ctx(profile: dict, esc) -> dict:
    """Every $placeholder the text config can use, derived from the live profile so the
    prose can never drift from the actual scoring."""
    rm = profile.get("reliability_model", {})
    ref = rm.get("reference", {})
    pen = profile.get("condition", {}).get("penalties", {})
    eq = profile.get("equipment", {})
    ml = profile.get("mileage", {})
    lo = profile.get("location", {})
    ab = profile.get("autobahn", {})
    tco = profile.get("tco", {})
    phev = tco.get("phev", {})
    cost_cfg = profile.get("cost", {})
    _cw = cost_cfg.get("weights", {}) or {}
    _tw = sum(float(v) for v in _cw.values()) or 1.0
    cost_shares = {k: float(_cw.get(k, 0)) / _tw for k in ("price", "price_rating", "running")}
    running_cfg = cost_cfg.get("running", {})
    wl = profile.get("wishlist", {})
    if isinstance(wl, list):
        wl = {o: 1 for o in wl}
    return {
        "rel_midpoint": rm.get("midpoint", 5.0), "rel_gain": rm.get("gain", 2.5),
        "rel_metric_weights": ", ".join(f"{k} {float(v):g}" for k, v in rm.get("metric_weights", {}).items()),
        "rel_tuv_mean": ref.get("tuv", {}).get("mean"), "rel_tuv_sd": ref.get("tuv", {}).get("sd"),
        "rel_adac_mean": ref.get("adac", {}).get("mean"), "rel_adac_sd": ref.get("adac", {}).get("sd"),
        "rel_fallback": rm.get("fallback_score", 5.0),
        "pen_accident": pen.get("accident_confirmed"), "pen_acc_unknown": pen.get("accident_unknown"),
        "pen_owner": pen.get("owner_step"), "pen_no_service": pen.get("no_service_history"),
        "pen_no_tuv": pen.get("no_tuv"), "pen_major": pen.get("major_defect"),
        "pen_minor": pen.get("minor_issue"), "pen_minor_cap": pen.get("minor_issue_cap"),
        "cost_price_best": f"{int(cost_cfg.get('price', {}).get('best_eur', 10000)):,}",
        "cost_price_worst": f"{int(cost_cfg.get('price', {}).get('worst_eur', 15000)):,}",
        "cost_price_pct": fmt_weight(cost_shares.get("price", 0)),
        "cost_rating_pct": fmt_weight(cost_shares.get("price_rating", 0)),
        "cost_running_pct": fmt_weight(cost_shares.get("running", 0)),
        "cost_running_best": f"{int(running_cfg.get('best_eur_yr', 1800)):,}",
        "cost_running_worst": f"{int(running_cfg.get('worst_eur_yr', 3600)):,}",
        "cost_rating_scale": " · ".join(
            f"{esc(PRICE_RATING_LABEL.get(k, k))} = {float(v):g}"
            for k, v in (cost_cfg.get("price_rating_map") or {}).items()),
        "cost_rating_unknown": f"{float(cost_cfg.get('price_rating_unknown', 5)):g}",
        "eq_wishlist_pct": fmt_weight(eq.get("wishlist_weight", 0.7)),
        "eq_year_pct": fmt_weight(eq.get("year_weight", 0.3)),
        "eq_year_worst": eq.get("year_worst"), "eq_year_best": eq.get("year_best"),
        "eq_wishlist_list": ", ".join(
            f"{esc(OPTION_LABELS.get(o, o))} \u00d7{float(v):g}"
            for o, v in sorted(wl.items(), key=lambda kv: (-float(kv[1]), kv[0]))),
        "ab_base": ab.get("base", 5.0), "ab_per_ps": ab.get("per_ps_over_130", 0.06),
        "ab_segment_bonuses": ", ".join(f"{k} +{v}" for k, v in ab.get("segment_bonus", {}).items()),
        "ab_big_bonus": ab.get("big_engine_bonus"), "ab_big_cm3": f"{ab.get('big_engine_cm3', 0):,}",
        "tco_hold_years": tco.get("hold_years", 5), "tco_annual_km": f"{tco.get('annual_km', 6000):,}",
        "tco_fuel_price": tco.get("fuel_price_eur_l"),
        "phev_share_pct": fmt_weight(phev.get("electric_share", 0.6)),
        "phev_kwh": phev.get("kwh_per_100km"), "phev_price_kwh": phev.get("electricity_price_eur_kwh"),
        "phev_petrol": phev.get("petrol_l100_engine"),
        "ml_best_km": f"{int(ml.get('best_km', 30000)):,}", "ml_worst_km": f"{int(ml.get('worst_km', 180000)):,}",
        "loc_origin": esc(lo.get("origin", {}).get("name", "origin")),
        "loc_best": lo.get("best_minutes"), "loc_worst": lo.get("worst_minutes"),
        "loc_fallback": lo.get("fallback_score"),
    }


def _methodology_html(profile: dict, texts: dict, ctx: dict, esc) -> str:
    """The 'how each score is defined' block. All prose comes from report_text.json; each
    criterion links to its per-car breakdown section (mileage and location have none)."""
    w = profile["weights"]
    anchors = {"reliability": "m-reliability", "condition": "m-condition",
               "equipment_fit": "m-equipment", "autobahn": "m-power", "cost": "m-cost"}
    label = text_at(texts, "sections.methodology.details_label", "details")
    parts = []
    for crit in CRITERIA:
        body = fill_text(text_at(texts, f"methodology.{crit}.body", ""), ctx)
        det = fill_text(text_at(texts, f"methodology.{crit}.details", ""), ctx)
        title = esc(CRIT_LABEL[crit])
        anchor = anchors.get(crit)
        head = f'<a href="#{anchor}">{title}</a>' if anchor else title
        det_html = (f'<details class="mm"><summary>{esc(label)}</summary><div>{det}</div></details>'
                    if det else "")
        parts.append(f'<section><h3>{head} <span class="w">weight {fmt_weight(w[crit])}</span></h3>'
                     f'<p>{body}</p>{det_html}</section>')
    heading = esc(text_at(texts, "sections.methodology.heading", "Methodology"))
    sub = fill_text(text_at(texts, "sections.methodology.sub", ""), ctx)
    return (f'<h2>{heading}</h2><p class="sub">{sub}</p>'
            '<div class="meth">' + "".join(parts) + '</div>')

def build_report_html(store: dict, profile: dict, ids: Iterable[str] | None = None,
                      texts: dict | None = None) -> str:
    """A self-contained, minimalistic HTML report: a compact colour-coded ranking table, a
    criteria heatmap, stacked cost bars (value on hover), and a live/all filter. No assets.
    All static prose lives in report_text.json (see load_texts)."""
    import html as _html
    esc = _html.escape
    texts = load_texts() if texts is None else texts
    results = compute_ranking(store, profile, ids=ids)
    weights = profile["weights"]
    tco_cfg = profile.get("tco", {})
    max_total = max((r.tco for r in results), default=1) or 1
    n_live = sum(1 for r in results if r.car.get("active") is True)

    ctx = _report_ctx(profile, esc)
    ctx.update({"date": date.today().isoformat(), "n_ranked": len(results),
                "n_live": n_live, "n_total": len(results)})
    T = lambda key, default="": fill_text(text_at(texts, key, default), ctx)

    p: list[str] = ['<div class="wrap">']
    p.append(f"<h1>{esc(text_at(texts, 'title', 'carcache report'))}</h1>")
    p.append(f'<p class="sub">{T("header.sub", "$date · $n_ranked ranked · $n_live live")}</p>')
    intro = T("intro")
    if intro:
        p.append(f'<p class="intro">{intro}</p>')
    p.append(f'<label class="toggle"><input type="checkbox" id="liveonly" checked> '
             f'{T("header.live_toggle", "show live only ($n_live of $n_total)")}</label>')

    # ---- compact ranking table (interpretable value per cell, coloured by score) ----
    p.append(f'<h2>{esc(text_at(texts, "sections.ranking.heading", "Ranking"))}</h2>')
    p.append(f'<p class="sub">{T("sections.ranking.sub")}</p>')
    p.append('<div class="scroll"><table class="rk-tbl"><thead><tr><th>#</th><th class="l">model</th>'
             '<th title="TÜV Mängelquote + ADAC Pannenstatistik">Reliability</th>'
             '<th title="owners + accident/defect">Condition</th>'
             '<th title="weighted wishlist + registration year">Equipment</th>'
             '<th title="power + segment">Power</th>'
             '<th title="asking price + platform price rating + annual cash running cost">Cost</th>'
             '<th title="odometer (absolute curve)">Mileage</th>'
             '<th title="travel time from the origin to the dealer">Location</th>'
             '<th>Score</th></tr></thead><tbody>')
    for rank, r in enumerate(results, 1):
        car, s = r.car, r.scores
        cond_txt, cond_title = _condition_cell(car, profile)
        aut_txt, aut_title = _autobahn_cell(car, profile)
        rel_inner, rel_title = _reliability_cell(car, profile, s["reliability"])
        eq_inner, eq_title = _equipment_cell(car, profile, s["equipment_fit"])
        loc_inner, loc_title = _location_cell(car, profile)
        cost_inner, cost_title = _cost_cell(car, profile, r)
        url = esc(listing_url(r.cid, car))
        p.append(f'<tr data-active="{_active_attr(car)}"><td class="rk">{rank}</td>'
                 f'<td class="l"><div class="mname"><a href="{url}" target="_blank" rel="noopener">'
                 f'{esc(_display_name(car))}<span class="ext">↗</span></a> {_avail_badge(car)}</div>'
                 f'<div class="msub">€{car.get("price_eur"):,} · {reg_year(car.get("first_reg"))} · '
                 f'{esc(car.get("fuel") or "")}</div></td>'
                 f'<td style="background:{_hsl_good(s["reliability"])}" title="{esc(rel_title)}">{rel_inner}</td>'
                 f'<td class="cond" style="background:{_hsl_good(s["condition"])}" title="{esc(cond_title)}">{cond_txt}</td>'
                 f'<td style="background:{_hsl_good(s["equipment_fit"])}" title="{esc(eq_title)}">{eq_inner}</td>'
                 f'<td style="background:{_hsl_good(s["autobahn"])}" title="{esc(aut_title)}">{aut_txt}</td>'
                 f'<td style="background:{_hsl_good(s["cost"])}" title="{esc(cost_title)}">{cost_inner}</td>'
                 f'<td style="background:{_hsl_good(s["mileage"])}">{(car.get("km") or 0)//1000} k</td>'
                 f'<td style="background:{_hsl_good(s["location"])}" title="{esc(loc_title)}">{loc_inner}</td>'
                 f'<td class="score" style="background:{_hsl_good(r.total)}">{r.total:.2f}</td></tr>')
    p.append("</tbody></table></div>")

    # ---- how stable is this ranking? (influence + weight sensitivity) ----
    if len(results) > 1:
        p.append(f'<h2>{esc(text_at(texts, "sections.stability.heading", "How stable is this ranking?"))}</h2>')
        _ssub = T("sections.stability.sub")
        if _ssub:
            p.append(f'<p class="sub">{_ssub}</p>')
        inf = criteria_influence(results, profile)
        p.append('<div class="scroll"><table class="mx"><thead><tr><th class="l">criterion</th>'
                 '<th>weight</th><th>mean</th><th>sd</th><th>w×sd</th>'
                 '<th title="fraction of the ranking\'s total spread this criterion drives">'
                 'share of ranking</th></tr></thead><tbody>')
        for d in inf:
            p.append(f'<tr><td class="l">{esc(CRIT_LABEL[d["criterion"]])}</td>'
                     f'<td>{d["weight"]:.2f}</td><td>{d["mean"]:.2f}</td><td>{d["sd"]:.2f}</td>'
                     f'<td>{d["influence"]:.3f}</td><td><b>{d["share"]:.0%}</b></td></tr>')
        p.append("</tbody></table></div>")
        sens = weight_sensitivity(results, profile)
        _sname = {r.cid: _display_name(r.car) for r in results}
        p.append(f'<p class="sub">{sens["n"]} re-rankings with every weight independently '
                 f'perturbed ×[{1 - sens["spread"]:.2f}, {1 + sens["spread"]:.2f}] '
                 f'(deterministic seed) — a car at 100% P(#1) wins under every plausible '
                 f'weighting:</p>')
        p.append('<div class="scroll"><table class="mx"><thead><tr><th class="l">model</th>'
                 '<th>P(#1)</th><th>P(top 3)</th></tr></thead><tbody>')
        for cid, cnt in sorted(sens["top3"].items(),
                               key=lambda kv: (-sens["top1"].get(kv[0], 0), -kv[1])):
            p.append(f'<tr><td class="l">{esc(trunc(_sname[cid], 40))}</td>'
                     f'<td>{sens["top1"].get(cid, 0) / sens["n"]:.0%}</td>'
                     f'<td>{cnt / sens["n"]:.0%}</td></tr>')
        p.append("</tbody></table></div>")

    # ---- methodology (links into the per-car breakdown sections below) ----
    p.append(_methodology_html(profile, texts, ctx, esc))

    # ---- criteria heatmap (all criteria + score; no cost column) ----
    p.append(f'<h2>{esc(text_at(texts, "sections.criteria_matrix.heading", "Criteria matrix"))}</h2>')
    p.append(f'<p class="sub">{T("sections.criteria_matrix.sub")}</p>')
    p.append('<div class="scroll"><table class="mx"><thead><tr><th>#</th><th class="l">model</th>')
    for c in CRITERIA:
        p.append(f'<th title="weight {fmt_weight(weights[c])}">{CRIT_LABEL[c]}</th>')
    p.append('<th>SCORE</th></tr></thead><tbody>')
    for rank, r in enumerate(results, 1):
        rb = reliability_basis(r.car, profile)
        rel_t = (esc(f"TÜV {_fmt_num(rb.get('tuv_pct'))}% · ADAC {_fmt_num(rb.get('adac_pannen'))}/1000")
                 if rb else "")
        p.append(f'<tr data-active="{_active_attr(r.car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 34))}</a></td>')
        for c in CRITERIA:
            t = f' title="{rel_t}"' if (c == "reliability" and rel_t) else ""
            p.append(f'<td{t} style="background:{_hsl_good(r.scores[c])}">{r.scores[c]:.1f}</td>')
        p.append(f'<td class="score" style="background:{_hsl_good(r.total)}">{r.total:.2f}</td></tr>')
    p.append("</tbody></table></div>")
    p.append('<p class="legend">' + " · ".join(
        f"<b>{esc(CRIT_LABEL[c])}</b> = {esc(c)} ({fmt_weight(weights[c])})" for c in CRITERIA) + "</p>")

    # ---- reliability derivation (objective per-cell TÜV/ADAC, matched to each listing) ----
    p.append(f'<h2 id="m-reliability">{esc(text_at(texts, "sections.reliability.heading", "Reliability"))}</h2>')
    p.append(f'<p class="sub">{T("sections.reliability.sub")}</p>')
    p.append('<div class="scroll"><table class="mx"><thead><tr>'
             '<th>#</th><th class="l">model</th>'
             '<th>TÜV band</th><th>TÜV %</th><th>s·TÜV</th>'
             '<th>reg yr</th><th>ADAC /1k</th><th>s·ADAC</th>'
             '<th>Reliability</th></tr></thead><tbody>')

    def _cells(comp: dict | None, key_fmt) -> tuple[str, str, str]:
        """(key cell, raw cell, sub-score cell) for one metric component."""
        if not comp:
            return "<td>–</td>", "<td>–</td>", "<td>–</td>"
        keyc = f'<td>{key_fmt(comp["key"])}</td>'
        rawc = f'<td>{_fmt_num(comp["raw"])}</td>'
        subc = f'<td style="background:{_hsl_good(comp["sub"])}">{comp["sub"]:.1f}</td>'
        return keyc, rawc, subc

    for rank, r in enumerate(results, 1):
        d = reliability_detail(r.car, profile)
        comp = d["components"]
        tb, tr_, ts = _cells(comp.get("tuv"), lambda k: f"{k}y")
        yb, ar, as_ = _cells(comp.get("adac"), lambda k: f"’{k[2:]}")
        flag = '<span class="warn">n/a</span>' if d["fallback"] else ""
        p.append(f'<tr data-active="{_active_attr(r.car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 34))}</a></td>'
                 f'{tb}{tr_}{ts}{yb}{ar}{as_}'
                 f'<td class="score" style="background:{_hsl_good(r.scores["reliability"])}">'
                 f'{r.scores["reliability"]:.1f}{flag}</td></tr>')
    p.append("</tbody></table></div>")

    # ---- safety (Euro NCAP) — informational, deliberately unscored ----
    _srows, _sseen = [], set()
    for r in results:
        sb = safety_basis(r.car, profile)
        if not sb:
            continue
        skey = (r.car.get("make"), r.car.get("model"), sb.get("test_year"))
        if skey in _sseen:
            continue
        _sseen.add(skey)
        _srows.append((f"{r.car.get('make')} {r.car.get('model')}", sb))
    if _srows:
        p.append(f'<h2>{esc(text_at(texts, "sections.safety.heading", "Safety (Euro NCAP) — informational"))}</h2>')
        _ssub = T("sections.safety.sub")
        if _ssub:
            p.append(f'<p class="sub">{_ssub}</p>')
        p.append('<div class="scroll"><table class="mx"><thead><tr><th class="l">model</th>'
                 '<th>stars</th><th>adult occupant</th><th>test year</th><th class="l">note</th>'
                 '</tr></thead><tbody>')
        for name, sb in sorted(_srows, key=lambda t: t[0]):
            p.append(f'<tr><td class="l">{esc(name)}</td><td>{"★" * int(sb.get("stars", 0))}</td>'
                     f'<td>{sb.get("adult_pct")}%</td><td>{sb.get("test_year")}</td>'
                     f'<td class="l">{esc(sb.get("note") or "")}</td></tr>')
        p.append("</tbody></table></div>")

    # ---- condition breakdown (what was deducted from 10) ----
    p.append(f'<h2 id="m-condition">{esc(text_at(texts, "sections.condition.heading", "Condition"))}</h2>')
    p.append(f'<p class="sub">{T("sections.condition.sub")}</p>')
    p.append('<div class="scroll"><table class="mx"><thead><tr><th>#</th><th class="l">model</th>'
             '<th class="l">deductions from 10</th><th>Condition</th></tr></thead><tbody>')
    for rank, r in enumerate(results, 1):
        items, score = condition_detail(r.car, profile)
        if items:
            ded = " ".join(
                f'<span class="{"warn" if b in _COND_SEVERE else "tag"}" title="{esc(t)}">{esc(b)}</span>'
                f'\u2009{d:+.1f}' for b, t, d in items)
        else:
            ded = '<span class="tag">clean</span> nothing deducted'
        p.append(f'<tr data-active="{_active_attr(r.car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 34))}</a></td>'
                 f'<td class="l">{ded}</td>'
                 f'<td class="score" style="background:{_hsl_good(score)}">{score:.1f}</td></tr>')
    p.append("</tbody></table></div>")

    # ---- equipment detail (which wishlist items each listing has) ----
    p.append(f'<h2 id="m-equipment">{esc(text_at(texts, "sections.equipment.heading", "Equipment detail"))}</h2>')
    p.append(f'<p class="sub">{T("sections.equipment.sub")}</p>')
    wl = profile.get("wishlist", {})
    if isinstance(wl, list):
        wl = {o: 1 for o in wl}
    opts = sorted(wl.keys(), key=lambda o: (-float(wl[o]), o))
    p.append('<div class="scroll"><table class="eq"><thead><tr><th>#</th><th class="l">model</th>')
    for o in opts:
        lbl = OPTION_LABELS.get(o, o)
        p.append(f'<th class="rot" title="wishlist weight {float(wl[o]):g}">{esc(lbl)}'
                 f'<br><span class="wt">×{float(wl[o]):g}</span></th>')
    p.append('<th>Year</th><th>Equip</th></tr></thead><tbody>')
    for rank, r in enumerate(results, 1):
        car = r.car
        p.append(f'<tr data-active="{_active_attr(car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(car), 30))}</a></td>')
        for o in opts:
            has = _has_option(car, o)
            p.append(f'<td class="{"yes" if has else "no"}">{"✓" if has else "·"}</td>')
        yr = reg_year(car.get("first_reg"))
        ys = year_score(car, profile)
        yrbg = _hsl_good(ys) if ys is not None else "#ededf0"
        p.append(f'<td class="yr" style="background:{yrbg}">{yr or "–"}</td>'
                 f'<td class="eqs" style="background:{_hsl_good(r.scores["equipment_fit"])}">'
                 f'{r.scores["equipment_fit"]:.1f}</td></tr>')
    p.append("</tbody></table></div>")

    # ---- power breakdown (how the autobahn score is built) ----
    p.append(f'<h2 id="m-power">{esc(text_at(texts, "sections.power.heading", "Power"))}</h2>')
    p.append(f'<p class="sub">{T("sections.power.sub")}</p>')
    p.append('<div class="scroll"><table class="mx"><thead><tr><th>#</th><th class="l">model</th>'
             '<th>PS</th><th>segment</th><th>base</th><th>+ power</th><th>+ bonus</th>'
             '<th>Power</th></tr></thead><tbody>')
    for rank, r in enumerate(results, 1):
        d = autobahn_detail(r.car, profile)
        if d["override"] is not None:
            base = pwr = bon = '<span class="tag">map override</span>'
            base, pwr, bon = base, "\u2013", "\u2013"
        else:
            base = f'{d["base"]:.1f}'
            pwr = f'+{d["power_bonus"]:.2f}'
            bon = f'+{d["segment_bonus"]:.1f}' if d["segment_bonus"] else "\u2013"
        seg = d["segment"] + ("" if d["segment"] == "compact" else "")
        p.append(f'<tr data-active="{_active_attr(r.car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 34))}</a></td>'
                 f'<td>{d["ps"] or "\u2013"}</td><td>{esc(seg)}</td>'
                 f'<td>{base}</td><td>{pwr}</td><td>{bon}</td>'
                 f'<td class="score" style="background:{_hsl_good(d["score"])}">{d["score"]:.1f}</td></tr>')
    p.append("</tbody></table></div>")

    p.append(f'<h2 id="m-cost">{esc(text_at(texts, "sections.cost.heading", "Cost of ownership"))}</h2>')
    p.append(f'<p class="sub">{T("sections.cost.sub")}</p>')

    # ---- the three parts of the composite Cost score ----
    p.append(f'<h3>{esc(text_at(texts, "sections.cost.composite_heading", "What Cost is made of"))}</h3>')
    _csub = T("sections.cost.composite_sub")
    if _csub:
        p.append(f'<p class="sub">{_csub}</p>')
    p.append('<div class="scroll"><table class="mx"><thead><tr><th>#</th><th class="l">model</th>'
             f'<th>asking price</th><th>→</th><th>price rating</th><th>→</th>'
             f'<th title="cash TCO components: no depreciation — the asking price already '
             f'has its own third">running cost (cash)</th><th>→</th><th>Cost</th></tr></thead>'
             f'<tbody>')
    for rank, r in enumerate(results, 1):
        items, composite = cost_detail(r.car, profile, r.running)
        sc = {k: v for k, _l, v, _w in items}
        rating = r.car.get("price_rating")
        rating_txt = esc(PRICE_RATING_LABEL.get(rating, "not rated"))
        if rating is None:
            rating_txt = f'{rating_txt} <span class="warn">n/a</span>'
        p.append(f'<tr data-active="{_active_attr(r.car)}"><td>{rank}</td>'
                 f'<td class="l"><a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 34))}</a></td>'
                 f'<td>€{r.car.get("price_eur") or 0:,}</td>'
                 f'<td class="pt" style="background:{_hsl_good(sc["price"])}">{sc["price"]:.1f}</td>'
                 f'<td>{rating_txt}</td>'
                 f'<td class="pt" style="background:{_hsl_good(sc["price_rating"])}">{sc["price_rating"]:.1f}</td>'
                 f'<td title="{esc(_fuel_source_label(r.car, profile))} · TCO incl. depreciation: €{r.tco:,.0f}">'
                 f'€{r.running:,.0f}{"*" if r.tco_estimated else ""}'
                 f'{"" if real_consumption(r.car, profile) is None else " ●"}</td>'
                 f'<td class="pt" style="background:{_hsl_good(sc["running"])}">{sc["running"]:.1f}</td>'
                 f'<td class="score" style="background:{_hsl_good(composite)}">{composite:.1f}</td></tr>')
    p.append("</tbody></table></div>")

    p.append(f'<h3>{esc(text_at(texts, "sections.cost.tco_heading", "Running cost, by component"))}</h3>')
    _tsub = T("sections.cost.tco_sub")
    if _tsub:
        p.append(f'<p class="sub">{_tsub}</p>')
    p.append('<p class="legend">' + "".join(
        f'<span class="sw" style="background:{_COST_COLORS[k]}"></span>{_COST_LABELS[k]}'
        for k in _COST_KEYS) + "</p>")
    for rank, r in enumerate(results, 1):
        segs = "".join(
            f'<div class="seg" data-label="{_COST_LABELS[k]}" data-val="{r.tco_breakdown.get(k, 0):,}" '
            f'style="width:{r.tco_breakdown.get(k, 0) / max_total * 100:.1f}%;background:{_COST_COLORS[k]}" '
            f'title="{_COST_LABELS[k]}: €{r.tco_breakdown.get(k, 0):,}"></div>'
            for k in _COST_KEYS if r.tco_breakdown.get(k, 0) > 0)
        p.append(f'<div class="cr" data-active="{_active_attr(r.car)}">'
                 f'<div class="cr-l">{rank}. {esc(trunc(_display_name(r.car), 30))}</div>'
                 f'<div class="stack">{segs}</div><div class="cr-t">€{r.tco:,.0f}</div></div>')

    # ---- findings: what review.py read out of the seller's description ----
    p.append(f'<h2>{text_at(texts, "sections.notes.heading", "Notes &amp; findings")}</h2>')
    _nsub = T("sections.notes.sub")
    if _nsub:
        p.append(f'<p class="sub">{_nsub}</p>')
    checklist_label = esc(text_at(texts, "sections.notes.checklist_label",
                                  "Before viewing, ask / inspect"))
    for rank, r in enumerate(results, 1):
        findings = [f for f in (r.car.get("findings") or []) if isinstance(f, dict)]
        summary = r.car.get("review_summary")
        checklist = viewing_checklist(r.car, profile)
        if not findings and not summary and not checklist:
            continue
        p.append(f'<div class="nt" data-active="{_active_attr(r.car)}">'
                 f'<b>{rank}. <a href="{esc(listing_url(r.cid, r.car))}" target="_blank" '
                 f'rel="noopener">{esc(trunc(_display_name(r.car), 42))}</a></b>')
        if summary:
            p.append(f'<p class="rsum">{esc(str(summary))}</p>')
        if findings:
            p.append("<ul>")
            # severity is STATED by the reviewer, never guessed from keywords: `note` is
            # context and is styled as such, so it reads as unscored rather than as a defect.
            order = {"major": 0, "minor": 1, "note": 2}
            for f in sorted(findings, key=lambda x: order.get(x.get("severity"), 3)):
                sev = str(f.get("severity", ""))
                scored = "" if sev in ("major", "minor") else " · not scored"
                p.append(f'<li class="{esc(sev)}"><span class="sev">{esc(sev)}</span> '
                         f'<span class="cat">{esc(str(f.get("category", "")))}{scored}</span> '
                         f'{esc(str(f.get("summary", "")))}'
                         f'<span class="ev">{esc(str(f.get("evidence", "")))}</span></li>')
            p.append("</ul>")
        if checklist:
            p.append(f'<details class="mm"><summary>{checklist_label} '
                     f'({len(checklist)})</summary><ul>')
            for item in checklist:
                p.append(f'<li>{esc(item)}</li>')
            p.append("</ul></details>")
        p.append("</div>")

    p.append(f'<p class="foot">{T("footer")}</p></div>')

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>carcache report</title><style>" + _REPORT_CSS + "</style></head>"
            "<body class='liveonly'>"
            + "".join(p) + "<script>" + _REPORT_JS + "</script></body></html>")


def cmd_report(args: argparse.Namespace) -> int:
    store = load_store(args.store)
    profile = load_profile(args.profile)
    if getattr(args, "refresh", False):
        _run_refresh(store, args.ids or None, args.store)
    markdown = build_report(store, profile, ids=args.ids or None)
    out = Path(args.output)
    out.write_text(markdown, encoding="utf-8")
    html_out = out.with_suffix(".html")
    html_out.write_text(build_report_html(store, profile, ids=args.ids or None), encoding="utf-8")
    print(f"wrote {out} ({len(markdown.splitlines())} lines) and {html_out}")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carcache",
        description="Local cache + ranking for used-car listings extracted from mobile.de PDFs.",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="path to cars.json")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), help="path to profile.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="upsert one listing (skips existing id unless --force)")
    p_add.add_argument("id")
    p_add.add_argument("--file", help="path to a JSON file holding the record")
    p_add.add_argument("--json", help="inline JSON record")
    p_add.add_argument("--pdf", help="source PDF to archive into processed/<id>.pdf")
    p_add.add_argument("--force", action="store_true", help="overwrite an already-cached id")
    p_add.set_defaults(func=cmd_add)

    p_show = sub.add_parser("show", help="print one record")
    p_show.add_argument("id")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="compact table of all cached cars")
    p_list.add_argument("--stale-days", type=int, default=None,
                        help="mark records fetched more than N days ago")
    p_list.set_defaults(func=cmd_list)

    p_filter = sub.add_parser(
        "filter", help="query the store, e.g. --towbar fitted_fixed --max-km 110000")
    p_filter.add_argument("conditions", nargs="*",
                          help="repeatable --field value (ranges: --max-km/--min-year/--max-price/…)")
    p_filter.set_defaults(func=cmd_filter)

    p_rank = sub.add_parser("rank", help="hard-filter, score and rank candidates")
    p_rank.add_argument("ids", nargs="*", help="restrict to these ids (default: all)")
    p_rank.add_argument("--stale-days", type=int, default=None,
                        help="mark records fetched more than N days ago")
    p_rank.add_argument("--sensitivity", nargs="?", const=500, type=int, default=None,
                        metavar="N",
                        help="Monte-Carlo weight robustness: N (default 500) re-rankings with "
                             "every weight perturbed ±30%%; prints P(#1)/P(top 3) per car. "
                             "Deterministic (fixed seed).")
    p_rank.add_argument("--refresh", action="store_true",
                        help="re-check availability (network) before ranking")
    p_rank.set_defaults(func=cmd_rank)

    p_report = sub.add_parser("report", help="write a markdown report")
    p_report.add_argument("ids", nargs="*", help="restrict to these ids (default: all)")
    p_report.add_argument("--output", default=str(DEFAULT_REPORT), help="output path")
    p_report.add_argument("--refresh", action="store_true",
                          help="re-check availability (network) before writing")
    p_report.set_defaults(func=cmd_report)

    p_intake = sub.add_parser(
        "intake", help="scan inbox/ PDFs, dedupe by id, archive cached, flag new for extraction")
    p_intake.add_argument("--inbox", default=str(INBOX_DIR), help="folder of incoming PDFs")
    p_intake.add_argument("--processed", default=str(PROCESSED_DIR), help="archive folder")
    p_intake.add_argument("--force", action="store_true",
                          help="treat already-cached ids as new (re-extract)")
    p_intake.set_defaults(func=cmd_intake)

    p_refresh = sub.add_parser("refresh", help="re-check listing availability on mobile.de")
    p_refresh.add_argument("ids", nargs="*", help="restrict to these ids (default: all)")
    p_refresh.set_defaults(func=cmd_refresh)

    p_travel = sub.add_parser(
        "travel", help="fill travel time/distance from the profile origin (geocode + route, once)")
    p_travel.add_argument("ids", nargs="*", help="restrict to these ids (default: all)")
    p_travel.add_argument("--force", action="store_true", help="re-fetch even if already stored")
    p_travel.set_defaults(func=cmd_travel)

    p_setactive = sub.add_parser("set-active", help="manually record availability (live/sold/unknown)")
    p_setactive.add_argument("id")
    p_setactive.add_argument("state", help="live | sold | unknown")
    p_setactive.set_defaults(func=cmd_set_active)

    p_pdfid = sub.add_parser("pdf-id", help="print the mobile.de id extracted from a PDF")
    p_pdfid.add_argument("file")
    p_pdfid.set_defaults(func=cmd_pdf_id)

    p_checkurls = sub.add_parser(
        "check-urls", help="list listing URLs to verify in a browser (past mobile.de's 403)")
    p_checkurls.add_argument("ids", nargs="*", help="restrict to these ids (default: all)")
    p_checkurls.add_argument("--unchecked", action="store_true",
                             help="only never-checked listings (active is null)")
    p_checkurls.add_argument("--stale-days", type=int, default=None,
                             help="only listings not checked within N days")
    p_checkurls.add_argument("--json", action="store_true", help="emit JSON [{id,url}]")
    p_checkurls.set_defaults(func=cmd_check_urls)

    p_apply = sub.add_parser(
        "apply-availability", help='apply a {"<id>": "live|sold|unknown"} JSON file')
    p_apply.add_argument("--file", required=True, help="JSON mapping id -> live|sold|unknown")
    p_apply.set_defaults(func=cmd_apply_availability)

    p_park = sub.add_parser(
        "apply-parkplatz", help="update price + availability from a parkplatz scrape (list of {id,price_eur,available})")
    p_park.add_argument("--file", required=True, help="JSON list [{id, price_eur, available, price_rating}]")
    p_park.set_defaults(func=cmd_apply_parkplatz)

    p_status = sub.add_parser(
        "status", help="show which stored cars drifted from the current extraction/scoring logic")
    p_status.set_defaults(func=cmd_status)

    p_dedupe = sub.add_parser(
        "dedupe", help="mark re-listed duplicates superseded (keeps the newest listing)")
    p_dedupe.add_argument("--dry-run", action="store_true", help="show groups, write nothing")
    p_dedupe.set_defaults(func=cmd_dedupe)

    p_recompute = sub.add_parser(
        "recompute", help="refresh stored score/TCO snapshots at the current scoring version")
    p_recompute.set_defaults(func=cmd_recompute)

    return parser


GLOBAL_VALUE_OPTS = {"--store", "--profile"}


def _split_filter(argv: list[str]) -> tuple[list[str], list[str]] | tuple[None, None]:
    """If the subcommand is `filter`, split argv into (head incl. 'filter', raw conditions).
    argparse cannot capture leading --option tokens as positionals, so we peel the
    conditions off ourselves and hand argparse only the head."""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in GLOBAL_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("--"):  # e.g. --store=path, or -h/--help
            i += 1
            continue
        if tok == "filter":
            return argv[: i + 1], argv[i + 1:]
        return None, None  # some other subcommand
    return None, None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = build_parser()
    head, rest = _split_filter(argv)
    if head is not None:
        args = parser.parse_args(head)
        args.conditions = rest
    else:
        args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
