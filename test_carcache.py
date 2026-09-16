"""Tests for carcache. Run with:  python3 -m pytest -q"""

from __future__ import annotations

import copy
import datetime as _dt
import io
import json
import urllib.error
import zlib
from pathlib import Path

import pytest

import carcache
import extract


def _fake_pdf(cid: str, compressed: bool = True) -> bytes:
    """Minimal bytes that look enough like a mobile.de PDF for id extraction."""
    inner = b"BT (Opel) Tj ET  details.html?id=%s&scopeId=C" % cid.encode()
    if compressed:
        return b"%PDF-1.4\nstream\n" + zlib.compress(inner) + b"\nendstream\n"
    return b"%PDF-1.4\n" + inner + b"\n"

REPO = Path(__file__).resolve().parent


def seed_store() -> dict:
    return carcache.load_store(REPO / "cars.json")


def seed_profile() -> dict:
    return carcache.load_profile(REPO / "profile.json")


def _valid_car(**over) -> dict:
    """A minimal, fully-populated record; override fields per test."""
    base = {
        "make": "Volkswagen", "model": "Golf", "body": "Kombi", "fuel": "Benzin",
        "price_eur": 10000, "first_reg": "01/2019", "km": 100000, "power_ps": 150,
        "emission_class": "Euro6d", "owners": 1, "service_history": True,
        "unfallfrei_declared": True, "tuv": "Neu", "consumption_l100": 6.0,
        "adaptive_cruise": True, "carplay": True, "heated_seats": True,
    }
    base.update(over)
    return carcache.normalize_record(base)


# --------------------------------------------------------------------------------------
# add is idempotent — the core guard against re-processing a PDF
# --------------------------------------------------------------------------------------

def test_add_skips_existing_without_force():
    store = seed_store()
    cid = "458421873"
    before = copy.deepcopy(store[cid])
    changed, msg = carcache.add_listing(store, cid, {"model": "OVERWRITE"}, force=False)
    assert changed is False
    assert "already cached" in msg
    assert store[cid] == before  # untouched


def test_add_force_overwrites_existing():
    store = seed_store()
    cid = "458421873"
    changed, _ = carcache.add_listing(store, cid, {"model": "OVERWRITE"}, force=True)
    assert changed is True
    assert store[cid]["model"] == "OVERWRITE"


def test_add_new_id_fills_schema_defaults():
    store = seed_store()
    changed, _ = carcache.add_listing(store, "123", {"model": "New Kombi"}, force=False)
    assert changed is True
    assert store["123"]["price_eur"] is None      # missing scalar -> null
    assert store["123"]["findings"] == []         # missing list -> []
    assert store["123"]["fetched_date"]           # defaulted to today


def test_normalize_record_drops_stray_id():
    rec = carcache.normalize_record({"id": "999", "model": "X"})
    assert "id" not in rec


def test_add_cli_idempotent(tmp_path):
    store_path = tmp_path / "cars.json"
    store_path.write_text(json.dumps({"111": carcache.normalize_record({"model": "A"})}))
    rc = carcache.main(["--store", str(store_path), "add", "111", "--json", '{"model":"B"}'])
    assert rc == 0
    data = json.loads(store_path.read_text())
    assert data["111"]["model"] == "A"  # unchanged: existing id was skipped

    rc = carcache.main(["--store", str(store_path), "add", "111", "--json", '{"model":"B"}', "--force"])
    assert rc == 0
    data = json.loads(store_path.read_text())
    assert data["111"]["model"] == "B"  # forced overwrite


# --------------------------------------------------------------------------------------
# filter
# --------------------------------------------------------------------------------------

def _filter_store() -> dict:
    """A fixed fixture — filter tests must not depend on the live store, which grows."""
    return {
        "f1": _valid_car(make="Ford", model="Focus", first_reg="03/2020",
                         towbar="capable", adaptive_cruise=True, km=100000),
        "f2": _valid_car(make="Ford", model="Mondeo", first_reg="05/2018",
                         towbar="fitted_fixed", adaptive_cruise=False, km=90000),
        "v1": _valid_car(make="Volkswagen", model="Golf", first_reg="01/2019",
                         towbar="fitted_swivel", adaptive_cruise=True, km=105000),
    }


def test_filter_equality_and_range():
    store = _filter_store()
    conds = carcache.parse_filter_conditions(["--make", "Ford", "--min-year", "2020"])
    assert set(carcache.apply_conditions(store, conds)) == {"f1"}


def test_filter_towbar_exact():
    store = _filter_store()
    conds = carcache.parse_filter_conditions(["--towbar", "fitted_fixed"])
    assert carcache.apply_conditions(store, conds) == ["f2"]


def test_filter_bool_and_max_km():
    store = _filter_store()
    conds = carcache.parse_filter_conditions(["--adaptive_cruise", "true", "--max-km", "100000"])
    ids = carcache.apply_conditions(store, conds)
    assert set(ids) == {"f1"}  # v1 is adaptive but 105k km; f2 is 90k but not adaptive
    for cid in ids:
        assert store[cid]["adaptive_cruise"] is True and store[cid]["km"] <= 100000


def test_filter_case_insensitive_string():
    store = _filter_store()
    conds = carcache.parse_filter_conditions(["--make", "ford"])  # lowercase
    assert len(carcache.apply_conditions(store, conds)) == 2


def test_filter_bare_flag_means_true():
    conds = carcache.parse_filter_conditions(["--service_history"])
    assert conds == [("service_history", "true")]


def test_filter_unknown_field_raises():
    with pytest.raises(ValueError):
        carcache.matches_condition({}, "nonsense_field", "x")


# --------------------------------------------------------------------------------------
# rank
# --------------------------------------------------------------------------------------

def test_rank_is_deterministic():
    store, profile = seed_store(), seed_profile()
    r1 = carcache.compute_ranking(store, profile)
    r2 = carcache.compute_ranking(store, profile)
    assert [r.cid for r in r1] == [r.cid for r in r2]
    assert [r.total for r in r1] == [r.total for r in r2]


def test_rank_sorted_desc_and_all_seed_survive():
    """Every stored car is ranked — there are no filters — except superseded re-listings."""
    store, profile = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, profile)
    superseded = sum(1 for c in store.values() if c.get("superseded_by"))
    assert len(results) == len(store) - superseded
    totals = [r.total for r in results]
    assert totals == sorted(totals, reverse=True)


def test_rank_tie_break_by_id_ascending():
    store = {"200": _valid_car(), "100": _valid_car()}  # identical but for id
    results = carcache.compute_ranking(store, seed_profile())
    assert results[0].total == results[1].total  # genuine tie
    assert [r.cid for r in results] == ["100", "200"]  # broken by id asc


def test_rank_scores_are_bounded():
    store, profile = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, profile)
    for r in results:
        assert 0.0 <= r.total <= 10.0
        for v in r.scores.values():
            assert 0.0 <= v <= 10.0


# --------------------------------------------------------------------------------------
# duplicate detection — a dealer re-listing one car must not be ranked as two
# --------------------------------------------------------------------------------------

def _relisted(**over):
    """Two ids, one physical car — same dealer, model, registration and odometer."""
    base = dict(make="Ford", model="Focus", dealer="SMR-Auto", first_reg="07/2024", km=85800)
    base.update(over)
    return _valid_car(**base)


def test_duplicate_group_found_and_newest_id_wins():
    store = {"459936680": _relisted(price_eur=15500), "460583296": _relisted(price_eur=14950),
             "111": _valid_car(dealer="Other", km=90000)}
    groups = carcache.duplicate_groups(store)
    assert len(groups) == 1
    _key, ids = groups[0]
    assert ids == ["459936680", "460583296"]      # mobile.de ids ascend, so newest is last
    assert carcache.apply_dedupe(store) == [("459936680", "460583296")]
    assert store["459936680"]["superseded_by"] == "460583296"
    assert store["460583296"]["superseded_by"] is None


def test_price_is_not_part_of_the_key():
    """Dealers re-list at a new price; keying on it would miss the duplicate entirely."""
    store = {"1": _relisted(price_eur=15500), "2": _relisted(price_eur=13900)}
    assert len(carcache.duplicate_groups(store)) == 1


def test_different_cars_are_not_duplicates():
    for field, value in [("km", 90000), ("dealer", "Elsewhere"),
                         ("first_reg", "01/2020"), ("model", "Mondeo")]:
        store = {"1": _relisted(), "2": _relisted(**{field: value})}
        assert carcache.duplicate_groups(store) == [], f"{field} should distinguish two cars"


def test_partial_records_never_dedupe():
    """A null must not collide with another null — that would merge unrelated cars."""
    store = {"1": _relisted(dealer=None), "2": _relisted(dealer=None)}
    assert carcache.duplicate_groups(store) == []


def test_superseded_listing_is_excluded_from_ranking():
    store = {"1": _valid_car(), "2": _valid_car(superseded_by="1")}
    assert [r.cid for r in carcache.compute_ranking(store, seed_profile())] == ["1"]


def test_superseded_listing_is_still_reachable_by_id():
    """Excluded from the ranking, not deleted — `rank <id>` must still find it."""
    store = {"1": _valid_car(), "2": _valid_car(superseded_by="1")}
    results = carcache.compute_ranking(store, seed_profile(), ids=["2"])
    assert [r.cid for r in results] == ["2"]


# --------------------------------------------------------------------------------------
# derived criteria sanity
# --------------------------------------------------------------------------------------

def test_condition_clean_beats_dirty():
    prof = seed_profile()
    clean = _valid_car(owners=1, unfallfrei_declared=True, findings=[])
    dirty = carcache.normalize_record(
        {"owners": 3, "unfallfrei_declared": None, "service_history": True, "tuv": None})
    assert carcache.condition(clean, prof) > carcache.condition(dirty, prof)


def _finding(sev, summary="x", cat="mechanical"):
    return {"severity": sev, "category": cat, "summary": summary, "evidence": "e"}


def test_condition_uses_structured_finding_severity():
    """review.py states severity explicitly — no keyword guessing."""
    prof = seed_profile()
    clean = _valid_car(findings=[])
    major = _valid_car(findings=[_finding("major", "coolant leak disclosed")])
    minor = _valid_car(findings=[_finding("minor", "tyres near wear limit")])
    assert carcache.count_findings(major) == (1, 0)
    assert carcache.condition(clean, prof) - carcache.condition(major, prof) >= 3.0
    assert carcache.condition(clean, prof) - carcache.condition(minor, prof) == pytest.approx(0.4)


def test_note_findings_never_penalise():
    """The old prose path made almost every car pick up a 'minor'; notes must cost nothing."""
    prof = seed_profile()
    clean = _valid_car(findings=[])
    noted = _valid_car(findings=[_finding("note", "1.0 EcoBoost — check wet-belt service"),
                                 _finding("note", "cheapest of the batch")])
    assert carcache.count_findings(noted) == (0, 0)
    assert carcache.condition(noted, prof) == carcache.condition(clean, prof)


def test_inconsistency_findings_never_reach_condition():
    """An advert that contradicts itself is a paperwork defect, not a defect of the car."""
    prof = seed_profile()
    clean = _valid_car(findings=[])
    typo = _valid_car(findings=[_finding("minor", "text says 204 PS, data says 150", "inconsistency"),
                                _finding("major", "impossible", "inconsistency")])
    assert carcache.count_findings(typo) == (0, 0)
    assert carcache.condition(typo, prof) == carcache.condition(clean, prof)


def test_declared_accident_is_not_charged_twice():
    """The Fahrzeugzustand tick-box already carries the heavy penalty; an `accident` finding
    restating the same crash must not stack on top of it."""
    prof = seed_profile()
    declared = _valid_car(unfallfrei_declared=False, findings=[])
    both = _valid_car(unfallfrei_declared=False,
                      findings=[_finding("major", "repaired front-end damage", "accident")])
    assert carcache.count_findings(both) == (0, 0)          # finding suppressed
    assert carcache.condition(both, prof) == carcache.condition(declared, prof)


def test_prose_damage_on_an_undeclared_car_is_graded_on_its_merits():
    """The seller ticked 'Unfallfrei' — the only thing the mechanical score sees. Damage found
    in the description is then the ONLY way the car can be charged for it."""
    prof = seed_profile()
    clean = _valid_car(unfallfrei_declared=True, findings=[])
    cosmetic = _valid_car(unfallfrei_declared=True,
                          findings=[_finding("minor", "rear-right panel scuffed", "accident")])
    serious = _valid_car(unfallfrei_declared=True,
                         findings=[_finding("major", "structural rear-end damage", "accident")])
    assert carcache.count_findings(cosmetic) == (0, 1)
    assert carcache.count_findings(serious) == (1, 0)
    assert carcache.condition(clean, prof) - carcache.condition(cosmetic, prof) == pytest.approx(0.4)
    assert carcache.condition(clean, prof) - carcache.condition(serious, prof) >= 3.0


def test_condition_penalises_disclosed_major_defect():
    prof = seed_profile()
    clean = _valid_car(findings=[])
    leak = _valid_car(findings=[_finding("major", "coolant seepage, Kopfdichtung")])
    accident = _valid_car(unfallfrei_declared=False, findings=[])
    assert carcache.condition(clean, prof) - carcache.condition(leak, prof) >= 3.0  # leak tanks it
    assert carcache.condition(accident, prof) <= 6.5                                # accident too
    # neutral spec observations must NOT tank condition
    neutral = _valid_car(findings=[_finding("note", "base 'Drive' trim", "other"),
                                   _finding("note", "'fair' price (not 'good')", "other")])
    assert carcache.condition(neutral, prof) >= 9.0


def test_criteria_dropped_looks_and_comfort():
    assert "looks" not in carcache.CRITERIA and "comfort" not in carcache.CRITERIA
    assert "condition" in carcache.CRITERIA and "condition_history" not in carcache.CRITERIA
    assert "location" in carcache.CRITERIA
    assert len(carcache.CRITERIA) == 7
    assert abs(sum(seed_profile()["weights"].values()) - 1.0) < 1e-9
    # every criterion must be weighted and labelled
    prof = seed_profile()
    for c in carcache.CRITERIA:
        assert c in prof["weights"] and c in carcache.CRIT_LABEL and c in carcache.CRIT_ABBR


def test_build_report_html_features():
    """Assert on STRUCTURE, not on prose — the headings live in report_text.json and are the
    user's to rename, so hardcoding them here makes their edits look like test failures."""
    texts = carcache.load_texts()
    html = carcache.build_report_html(seed_store(), seed_profile(), texts=texts)
    assert html.startswith("<!doctype html>")
    for key in ("sections.ranking.heading", "sections.criteria_matrix.heading",
                "sections.cost.heading"):
        assert carcache.text_at(texts, key) in html
    assert "hsl(" in html               # colour-coded cells
    assert 'id="liveonly"' in html      # live/all toggle
    assert "data-label=" in html        # cost values available on hover
    assert ">cmf<" not in html          # comfort column dropped from the matrix
    # model names link to their mobile.de listings
    a_id = next(iter(seed_store()))
    assert f"details.html?id={a_id}" in html and 'target="_blank"' in html


def test_equipment_fit_counts_only_fitted_towbar():
    prof = seed_profile()
    fitted = _valid_car(towbar="fitted_fixed")
    capable = _valid_car(towbar="capable")
    assert carcache.equipment_fit(fitted, prof) > carcache.equipment_fit(capable, prof)


# --------------------------------------------------------------------------------------
# PDF id extraction (no image cost)
# --------------------------------------------------------------------------------------

def test_extract_id_from_raw_bytes():
    assert carcache.extract_ids_from_bytes(_fake_pdf("455281247", compressed=False)) == ["455281247"]


def test_extract_id_from_flate_stream():
    assert carcache.extract_ids_from_bytes(_fake_pdf("459081695", compressed=True)) == ["459081695"]


def test_pdf_id_reads_file(tmp_path):
    p = tmp_path / "listing.pdf"
    p.write_bytes(_fake_pdf("123456"))
    assert carcache.pdf_id(p) == "123456"


def test_pdf_id_none_when_absent(tmp_path):
    p = tmp_path / "nope.pdf"
    p.write_bytes(b"%PDF-1.4 no ids here")
    assert carcache.pdf_id(p) is None


# --------------------------------------------------------------------------------------
# Inbox intake
# --------------------------------------------------------------------------------------

def test_scan_inbox_archives_cached_and_flags_new(tmp_path):
    inbox = tmp_path / "inbox"
    processed = tmp_path / "processed"
    inbox.mkdir()
    (inbox / "already.pdf").write_bytes(_fake_pdf("111111"))
    (inbox / "brandnew.pdf").write_bytes(_fake_pdf("222222"))
    store = {"111111": carcache.normalize_record({"model": "cached"})}

    items = carcache.scan_inbox(store, inbox, processed)
    by_status = {it.status: it for it in items}

    assert by_status["cached"].cid == "111111"
    assert (processed / "111111.pdf").exists()      # archived out of inbox
    assert not (inbox / "already.pdf").exists()

    assert by_status["new"].cid == "222222"
    assert (inbox / "222222.pdf").exists()           # renamed, awaits extraction
    assert not (processed / "222222.pdf").exists()


def test_scan_inbox_force_treats_cached_as_new(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "x.pdf").write_bytes(_fake_pdf("111111"))
    store = {"111111": carcache.normalize_record({"model": "cached"})}
    items = carcache.scan_inbox(store, inbox, tmp_path / "processed", force=True)
    assert items[0].status == "new"


# --------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------

def test_check_availability_active():
    state, _ = carcache.check_availability("u", fetch=lambda u, timeout=15: (200, u, "<html>Opel Astra</html>"))
    assert state == "active"


def test_check_availability_removal_text():
    state, _ = carcache.check_availability("u", fetch=lambda u, timeout=15: (200, u, "Angebot nicht gefunden"))
    assert state == "inactive"


def test_check_availability_404_is_inactive():
    def raise404(u, timeout=15):
        raise urllib.error.HTTPError(u, 404, "Not Found", {}, io.BytesIO(b""))
    assert carcache.check_availability("u", fetch=raise404)[0] == "inactive"


def test_check_availability_403_is_unknown():
    def raise403(u, timeout=15):
        raise urllib.error.HTTPError(u, 403, "Forbidden", {}, io.BytesIO(b""))
    assert carcache.check_availability("u", fetch=raise403)[0] == "unknown"


def test_check_availability_timeout_is_unknown():
    def boom(u, timeout=15):
        raise TimeoutError("slow")
    assert carcache.check_availability("u", fetch=boom)[0] == "unknown"


def test_refresh_store_updates_and_reports_newly_inactive():
    store = {
        "1": carcache.normalize_record({"model": "live", "active": True}),
        "2": carcache.normalize_record({"model": "gone", "active": True}),
        "3": carcache.normalize_record({"model": "blocked", "active": True}),
    }

    def fake_fetch(url, timeout=15):
        if "id=1" in url:
            return (200, url, "<html>Opel</html>")
        if "id=2" in url:
            return (200, url, "Fahrzeug wurde verkauft")
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b""))

    rows, newly = carcache.refresh_store(store, fetch=fake_fetch)
    assert store["1"]["active"] is True
    assert store["2"]["active"] is False
    assert store["3"]["active"] is True          # unknown never flips the flag
    assert newly == ["2"]
    assert all(store[i]["last_checked"] for i in store)


def test_listing_url_derives_and_prefers_stored():
    assert carcache.listing_url("999") == "https://suchen.mobile.de/fahrzeuge/details.html?id=999"
    assert carcache.listing_url("999", {"listing_url": "https://x/y"}) == "https://x/y"


def test_set_active_cli(tmp_path):
    sp = tmp_path / "cars.json"
    sp.write_text(json.dumps({"1": carcache.normalize_record({"model": "A"})}))
    assert carcache.main(["--store", str(sp), "set-active", "1", "sold"]) == 0
    data = json.loads(sp.read_text())
    assert data["1"]["active"] is False
    assert data["1"]["last_checked"]


def test_apply_availability_batch(tmp_path):
    sp = tmp_path / "cars.json"
    sp.write_text(json.dumps({
        "1": carcache.normalize_record({"model": "A", "active": True}),
        "2": carcache.normalize_record({"model": "B", "active": True}),
    }))
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"1": "sold", "2": "live"}))
    assert carcache.main(["--store", str(sp), "apply-availability", "--file", str(res)]) == 0
    data = json.loads(sp.read_text())
    assert data["1"]["active"] is False
    assert data["2"]["active"] is True
    assert data["1"]["last_checked"] and data["2"]["last_checked"]


def test_apply_parkplatz_updates_price_and_availability(tmp_path):
    sp = tmp_path / "cars.json"
    sp.write_text(json.dumps({
        "1": carcache.normalize_record({"model": "A", "price_eur": 12000, "active": True}),
        "2": carcache.normalize_record({"model": "B", "price_eur": 9000, "active": True}),
    }))
    entries = [
        {"id": "1", "price_eur": 11500, "available": True},    # price drop
        {"id": "2", "price_eur": 9000, "available": False},    # sold
        {"id": "999", "price_eur": 8000, "available": True},   # not in store
    ]
    f = tmp_path / "park.json"
    f.write_text(json.dumps(entries))
    assert carcache.main(["--store", str(sp), "apply-parkplatz", "--file", str(f)]) == 0
    d = json.loads(sp.read_text())
    assert d["1"]["price_eur"] == 11500 and d["1"]["active"] is True
    assert d["2"]["active"] is False
    assert d["1"]["last_checked"] and d["2"]["last_checked"]


def test_apply_parkplatz_summary_reports_changes():
    store = {"1": carcache.normalize_record({"model": "A", "price_eur": 12000, "active": True})}
    r = carcache.apply_parkplatz(
        store,
        [{"id": "1", "price_eur": 11000, "available": True}, {"id": "9", "price_eur": 5000, "available": True}],
        "2026-07-11")
    assert ("1", 12000, 11000) in r["price_changes"]
    assert r["unknown"] == [("9", 5000)]
    assert store["1"]["price_eur"] == 11000


# --------------------------------------------------------------------------------------
# versioning: extraction vs. scoring staleness
# --------------------------------------------------------------------------------------

def test_profile_fingerprint_stable_and_sensitive():
    p = seed_profile()
    assert carcache.profile_fingerprint(p) == carcache.profile_fingerprint(seed_profile())
    p["weights"]["reliability"] += 0.01
    assert carcache.profile_fingerprint(p) != carcache.profile_fingerprint(seed_profile())
    p2 = seed_profile()
    p2["tco"]["fuel_price_eur_l"] = 2.10
    assert carcache.profile_fingerprint(p2) != carcache.profile_fingerprint(seed_profile())


def test_is_extract_stale():
    assert carcache.is_extract_stale({}) is True                                  # pre-versioning
    assert carcache.is_extract_stale({"extract_version": carcache.EXTRACT_VERSION}) is False
    assert carcache.is_extract_stale({"extract_version": carcache.EXTRACT_VERSION - 1}) is True


def test_is_score_stale():
    prof = seed_profile()
    fp = carcache.profile_fingerprint(prof)
    fresh = {"computed": {"scoring_version": carcache.SCORING_VERSION, "profile_fingerprint": fp}}
    assert carcache.is_score_stale(fresh, prof) is False
    assert carcache.is_score_stale({}, prof) is True
    stale_fp = {"computed": {"scoring_version": carcache.SCORING_VERSION, "profile_fingerprint": "deadbeef"}}
    assert carcache.is_score_stale(stale_fp, prof) is True


def test_add_stamps_extract_version():
    store: dict = {}
    carcache.add_listing(store, "1", {"model": "X"})
    assert store["1"]["extract_version"] == carcache.EXTRACT_VERSION


def test_field_ownership_is_disjoint_and_covers_the_schema():
    """The split is the whole safety story: a review may only ever write LLM_FIELDS."""
    assert not set(carcache.PARSED_FIELDS) & set(carcache.LLM_FIELDS)
    for f in carcache.PARSED_FIELDS + carcache.LLM_FIELDS:
        assert f in carcache.SCHEMA_FIELDS, f


def test_merge_reprocessed_rebuilds_parsed_and_keeps_llm_fields():
    record = carcache.normalize_record(
        {"km": 999, "model": "Stale Hand-Written Name", "towbar": "fitted_fixed",
         "review_summary": "written by review.py", "options_from_prose": ["towbar"]})
    parsed = {"km": 110000, "model": "Focus", "towbar": None}
    merged, diffs = carcache.merge_reprocessed(record, parsed)
    assert merged["km"] == 110000            # parsed -> rebuilt from the PDF
    assert merged["model"] == "Focus"        # parsed -> the parser wins, always
    assert merged["review_summary"] == "written by review.py"   # LLM -> preserved
    assert merged["options_from_prose"] == ["towbar"]           # LLM -> preserved
    assert merged["extract_version"] == carcache.EXTRACT_VERSION
    assert ("km", 999, 110000) in diffs


def test_reprocess_lets_the_parser_clear_a_stale_value():
    """A None from the parser means 'the seller did not list this' — a real answer. Keeping a
    stale value here would break the guarantee that a record is a function of its PDF."""
    record = carcache.normalize_record({"towbar": "fitted_swivel", "roof_window": True})
    merged, diffs = carcache.merge_reprocessed(record, {"towbar": None, "roof_window": None})
    assert merged["towbar"] is None and merged["roof_window"] is None
    assert ("towbar", "fitted_swivel", None) in diffs


def test_recompute_cli_writes_and_clears_staleness(tmp_path):
    sp = tmp_path / "cars.json"
    sp.write_text(json.dumps({"1": _valid_car()}))
    pp = tmp_path / "profile.json"
    pp.write_text(json.dumps(seed_profile()))
    assert carcache.main(["--store", str(sp), "--profile", str(pp), "recompute"]) == 0
    data = json.loads(sp.read_text())
    comp = data["1"]["computed"]
    assert comp["scoring_version"] == carcache.SCORING_VERSION
    assert "total" in comp and "tco" in comp
    assert carcache.is_score_stale(data["1"], seed_profile()) is False


# --------------------------------------------------------------------------------------
# extract.py batch + reprocess (text reader injected — no pypdf / no PDFs)
# --------------------------------------------------------------------------------------

def test_batch_inbox_skips_cached_writes_new(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (inbox / "a.pdf").write_bytes(_fake_pdf("111111"))
    (inbox / "b.pdf").write_bytes(_fake_pdf("222222"))
    store = {"111111": carcache.normalize_record({"model": "cached"})}
    reader = lambda p: "Kilometerstand 90.000 km\nLeistung 110 kW (150 PS)\nErstzulassung 03/2019"
    made, skipped, unknown = extract.batch_inbox(store, inbox, out, text_reader=reader)
    assert skipped == ["111111"]
    assert [cid for cid, _, _ in made] == ["222222"]
    skel = json.loads((out / "222222.json").read_text())
    assert skel["km"] == 90000 and skel["power_ps"] == 150


def test_reprocess_store_updates_from_pdf(tmp_path):
    store = {"222222": carcache.normalize_record(
        {"km": 5, "review_summary": "Keep Me", "findings": [{"severity": "major",
         "category": "mechanical", "summary": "leak", "evidence": "e"}]})}
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "222222.pdf").write_bytes(b"x")
    reader = lambda p: "Kilometerstand 90.000 km"
    results = extract.reprocess_store(store, processed, ids=["222222"], text_reader=reader)
    assert results[0][1] == "reprocessed"
    assert store["222222"]["km"] == 90000                    # parsed rebuilt from the PDF
    assert store["222222"]["review_summary"] == "Keep Me"    # LLM field preserved
    assert len(store["222222"]["findings"]) == 1             # no re-review, no API call


def test_reprocess_store_reports_missing_pdf(tmp_path):
    store = {"1": carcache.normalize_record({"model": "x"})}
    results = extract.reprocess_store(store, tmp_path, ids=["1"])
    assert results[0][1] == "missing-pdf"


def test_parse_adaptive_cruise_is_word_bounded():
    # a bare "acc" inside another word must NOT trigger adaptive cruise
    assert extract.parse_listing("Ausstattung: Abstandswarner, Zubehoeraccessoire")["adaptive_cruise"] is None
    assert extract.parse_listing("mit Abstandstempomat")["adaptive_cruise"] is True
    assert extract.parse_listing("Fahrassistenz: ACC serienmäßig")["adaptive_cruise"] is True


def test_parse_heated_wheel_and_variable_floor_variants():
    assert extract.parse_listing("Beheizbares Lenkrad")["heated_wheel"] is True
    assert extract.parse_listing("Laderaumboden verstellbar")["variable_floor"] is True


def test_parse_ignores_similar_vehicles_block():
    # ACC belongs to a recommended *other* car, after the "Ähnliche Fahrzeuge" divider
    text = "Ausstattung\nAbstandswarner\nÄhnliche Fahrzeuge\nFord Focus ST ACC Recaro Kamera"
    assert extract.parse_listing(text)["adaptive_cruise"] is None


def test_parse_roof_window():
    assert extract.parse_listing("Ausstattung Panorama-Dach")["roof_window"] is True
    assert extract.parse_listing("Ausstattung Schiebedach")["roof_window"] is True
    assert extract.parse_listing("Ausstattung Klimaanlage")["roof_window"] is None


# --------------------------------------------------------------------------------------
# TCO components (tax computed exactly; others estimated)
# --------------------------------------------------------------------------------------

def test_kfz_steuer_petrol_pre_2021():
    # 1798cc petrol, CO2 154, reg 2018 -> 18*2 + (154-95)*2 = 36 + 118 = 154
    assert carcache.kfz_steuer(
        {"displacement_cm3": 1798, "co2_gkm": 154, "fuel": "Benzin", "first_reg": "03/2018"}) == 154


def test_kfz_steuer_diesel_rate():
    # 1968cc diesel, CO2 120, reg 2018 -> ceil(19.68)=20 *9.5 + (120-95)*2 = 190 + 50 = 240
    assert carcache.kfz_steuer(
        {"displacement_cm3": 1968, "co2_gkm": 120, "fuel": "Diesel", "first_reg": "01/2018"}) == 240


def test_kfz_steuer_tiered_from_2021():
    # 1498cc petrol, CO2 130, reg 2021 -> 30 + [40 + (130-115)*2.2] = 30 + 73 = 103
    assert carcache.kfz_steuer(
        {"displacement_cm3": 1498, "co2_gkm": 130, "fuel": "Benzin", "first_reg": "06/2021"}) == 103.0


def test_car_segment():
    prof = seed_profile()
    assert carcache.car_segment({"make": "Opel", "model": "Opel Insignia B"}, prof) == "mid"
    assert carcache.car_segment({"make": "VW", "model": "VW Golf Variant"}, prof) == "compact"


def test_depreciation_newer_weaker_brand_costs_more():
    prof = seed_profile()
    strong_old = {"make": "Toyota", "model": "Toyota Corolla", "price_eur": 12000, "first_reg": "01/2017"}
    weak_new = {"make": "Renault", "model": "Renault Mégane", "price_eur": 12000, "first_reg": "01/2023"}
    assert carcache.depreciation_cost(weak_new, prof) > carcache.depreciation_cost(strong_old, prof)


def test_annual_tco_has_all_components():
    prof = seed_profile()
    total, parts, _ = carcache.annual_tco(_valid_car(price_eur=12000), prof)
    assert set(parts) == {"fuel", "tax", "maintenance", "insurance", "depreciation", "hu"}
    assert total == sum(parts.values())
    assert parts["depreciation"] > 0


# --------------------------------------------------------------------------------------
# weighted wishlist (incl. roof window)
# --------------------------------------------------------------------------------------

def test_equipment_fit_weighted_by_option():
    """A heavier wishlist option contributes more. Uses an explicit wishlist so the test
    checks the weighting LOGIC and doesn't break every time the profile is retuned."""
    prof = seed_profile()
    prof["wishlist"] = {"adaptive_cruise": 3, "roof_window": 1}
    off = dict(adaptive_cruise=False, carplay=False, heated_seats=False, towbar=None)
    only_roof = _valid_car(**off, roof_window=True)
    only_acc = _valid_car(**{**off, "adaptive_cruise": True})
    base = _valid_car(**off)
    assert carcache.equipment_fit(only_acc, prof) > carcache.equipment_fit(only_roof, prof)
    assert carcache.equipment_fit(only_roof, prof) > carcache.equipment_fit(base, prof)


def test_equipment_fit_accepts_legacy_list():
    car = _valid_car(carplay=True, roof_window=True)
    assert carcache.equipment_fit(car, {"wishlist": ["carplay", "roof_window"]}) == 10.0


def test_equipment_fit_can_incorporate_year():
    """With year_weight > 0 (not the current profile default — age already reaches the score
    through the reliability cells), a newer car scores higher on the same wishlist."""
    prof = seed_profile()
    prof["equipment"] = {"wishlist_weight": 0.7, "year_weight": 0.3,
                         "year_best": 2025, "year_worst": 2016}
    older = _valid_car(first_reg="01/2017")
    newer = _valid_car(first_reg="01/2024")
    assert carcache.equipment_fit(newer, prof) > carcache.equipment_fit(older, prof)
    # blended score still bounded 0..10
    assert 0.0 <= carcache.equipment_fit(older, prof) <= 10.0


def test_year_weight_zero_is_pure_wishlist():
    car = _valid_car(first_reg="01/2017")
    prof = seed_profile()
    prof["equipment"] = {"wishlist_weight": 1.0, "year_weight": 0.0}
    assert carcache.equipment_fit(car, prof) == carcache.wishlist_share(car, prof)


# --------------------------------------------------------------------------------------
# influence + weight sensitivity — the "can I trust this ranking?" tooling
# --------------------------------------------------------------------------------------

def test_criteria_influence_is_weight_times_spread():
    store, prof = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, prof)
    rows = {d["criterion"]: d for d in carcache.criteria_influence(results, prof)}
    assert set(rows) == set(carcache.CRITERIA)
    for c, d in rows.items():
        assert d["influence"] == pytest.approx(d["weight"] * d["sd"], abs=0.002)
    assert sum(d["share"] for d in rows.values()) == pytest.approx(1.0, abs=0.02)


def test_criteria_influence_zero_spread_means_zero_influence():
    """A criterion every car scores the same on cannot rank anyone, whatever its weight."""
    store, prof = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, prof)
    for r in results:                     # force a constant criterion
        r.scores["mileage"] = 5.0
    rows = {d["criterion"]: d for d in carcache.criteria_influence(results, prof)}
    assert rows["mileage"]["influence"] == 0.0
    assert rows["mileage"]["share"] == 0.0


def test_weight_sensitivity_is_deterministic():
    """rank stays reproducible: the Monte-Carlo uses a fixed seed, so two runs agree."""
    store, prof = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, prof)
    a = carcache.weight_sensitivity(results, prof, n=50)
    b = carcache.weight_sensitivity(results, prof, n=50)
    assert a == b


def test_weight_sensitivity_counts_are_consistent():
    store, prof = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, prof)
    sens = carcache.weight_sensitivity(results, prof, n=40)
    assert sum(sens["top1"].values()) == 40           # exactly one #1 per run
    assert sum(sens["top3"].values()) == 40 * min(3, len(results))
    assert set(sens["top1"]) <= set(sens["top3"])     # a #1 is always in its own top 3


def test_weight_sensitivity_zero_spread_reproduces_the_ranking():
    """With no perturbation every run must reproduce the deterministic ranking exactly."""
    store, prof = seed_store(), seed_profile()
    results = carcache.compute_ranking(store, prof)
    sens = carcache.weight_sensitivity(results, prof, n=10, spread=0.0)
    assert sens["top1"] == {results[0].cid: 10}
    assert sens["top3"] == {r.cid: 10 for r in results[:3]}


# --------------------------------------------------------------------------------------
# real-world consumption — measured data outranks the listing's optimistic WLTP figure
# --------------------------------------------------------------------------------------

def _consumption_profile(**over):
    prof = seed_profile()
    prof["model_consumption"] = {
        "ps_tolerance": 8,
        "entries": [
            {"match": "ford focus", "fuel": "Benzin", "ps": 150, "real_l100": 7.28, "n": 546},
            {"match": "ford focus", "fuel": "Benzin", "ps": 182, "real_l100": 7.62, "n": 199},
            {"match": "seat leon", "fuel": "Hybrid-Benzin", "ps": 204, "real_l100": 3.53,
             "n": 58, "phev": True},
        ],
    }
    prof.update(over)
    return prof


def test_real_consumption_matches_model_fuel_and_power():
    prof = _consumption_profile()
    car = _valid_car(make="Ford", model="Focus", fuel="Benzin", power_ps=150)
    assert carcache.real_consumption(car, prof)["real_l100"] == 7.28


def test_real_consumption_picks_the_closest_power_variant():
    """Two entries for the same model must not be mixed up by engine."""
    prof = _consumption_profile()
    hot = _valid_car(make="Ford", model="Focus", fuel="Benzin", power_ps=180)
    assert carcache.real_consumption(hot, prof)["real_l100"] == 7.62


def test_real_consumption_respects_power_tolerance_and_fuel():
    prof = _consumption_profile()
    far = _valid_car(make="Ford", model="Focus", fuel="Benzin", power_ps=250)
    assert carcache.real_consumption(far, prof) is None          # outside ps_tolerance
    diesel = _valid_car(make="Ford", model="Focus", fuel="Diesel", power_ps=150)
    assert carcache.real_consumption(diesel, prof) is None        # wrong fuel
    other = _valid_car(make="Opel", model="Astra", fuel="Benzin", power_ps=150)
    assert carcache.real_consumption(other, prof) is None         # unlisted model


def test_real_consumption_beats_the_listings_optimistic_wltp():
    """The whole point: a stated 5.5 WLTP must not override a measured 7.28."""
    prof = _consumption_profile()
    car = _valid_car(make="Ford", model="Focus", fuel="Benzin", power_ps=150,
                     consumption_l100=5.5)
    petrol, elec, estimated = carcache.energy_cost(car, prof)
    km = prof["tco"]["annual_km"]
    price = prof["tco"]["fuel_price_eur_l"]
    assert petrol == pytest.approx(7.28 / 100 * km * price)
    assert elec == 0.0
    assert estimated is False        # measured, not modelled


def test_car_without_real_data_still_uses_its_stated_figure():
    prof = _consumption_profile()
    car = _valid_car(make="Opel", model="Astra", fuel="Benzin", power_ps=150,
                     consumption_l100=5.5)
    petrol, _elec, estimated = carcache.energy_cost(car, prof)
    km = prof["tco"]["annual_km"]
    assert petrol == pytest.approx(5.5 / 100 * km * prof["tco"]["fuel_price_eur_l"])
    assert estimated is False


def test_phev_real_figure_is_not_scaled_by_electric_share_again():
    """A PHEV's logged litres already cover the electric-driven km, so the real figure
    replaces the modelled engine consumption whole — while electricity is still charged."""
    prof = _consumption_profile()
    car = _valid_car(make="Seat", model="Leon", fuel="Hybrid-Benzin", power_ps=204,
                     consumption_l100=1.4)
    assert carcache.is_phev(car, prof)
    petrol, elec, estimated = carcache.energy_cost(car, prof)
    km = prof["tco"]["annual_km"]
    assert petrol == pytest.approx(3.53 / 100 * km * prof["tco"]["fuel_price_eur_l"])
    assert elec > 0            # still buys electricity
    assert estimated is False


def test_real_consumption_entries_are_well_formed():
    """Guard the shipped data: every entry needs a plausible figure and a real sample."""
    entries = seed_profile().get("model_consumption", {}).get("entries", [])
    assert entries, "profile should ship real-world consumption data"
    for e in entries:
        assert 2.0 <= e["real_l100"] <= 15.0, e
        assert e["n"] >= 10, e                 # sample-size gate
        assert e["fuel"] and e["match"] and e["ps"]


# --------------------------------------------------------------------------------------
# safety basis — informational Euro NCAP lookup, generation picked by registration year
# --------------------------------------------------------------------------------------

def test_safety_basis_picks_generation_by_registration_year():
    prof = seed_profile()
    leon_iii = _valid_car(make="Seat", model="Leon", first_reg="03/2017")
    leon_iv = _valid_car(make="Seat", model="Leon", first_reg="06/2021")
    assert carcache.safety_basis(leon_iii, prof)["test_year"] == 2012
    assert carcache.safety_basis(leon_iv, prof)["test_year"] == 2020


def test_safety_basis_unknown_model_is_empty():
    prof = seed_profile()
    assert carcache.safety_basis(_valid_car(make="Dacia", model="Duster"), prof) == {}


def test_safety_is_not_a_scored_criterion():
    """Every catalogue model is 5-star and NCAP percentages are not comparable across
    protocol years — scoring them would rank test vintage, not safety."""
    assert "safety" not in carcache.CRITERIA
    assert "safety" not in seed_profile()["weights"]


# --------------------------------------------------------------------------------------
# viewing checklist — a None is an answer for scoring, but a question for a buyer
# --------------------------------------------------------------------------------------

def test_checklist_flags_unstated_facts():
    car = _valid_car(service_history=None, tuv=None, owners=None,
                     unfallfrei_declared=None, price_rating=None)
    items = " ".join(carcache.viewing_checklist(car, seed_profile()))
    for expect in ("Accident-free", "Service history", "HU/TÜV", "previous owners",
                   "no rating"):
        assert expect in items


def test_checklist_is_short_for_a_fully_stated_clean_car():
    car = _valid_car(price_rating="gut")
    assert carcache.viewing_checklist(car, seed_profile()) == []


def test_checklist_turns_findings_into_inspection_items():
    car = _valid_car(price_rating="gut", findings=[
        {"severity": "major", "category": "mechanical",
         "summary": "Seller discloses coolant loss", "evidence": "Kühlmittelverlust"},
        {"severity": "minor", "category": "inconsistency",
         "summary": "Text says 204 PS, data table says 150 PS", "evidence": "204 PS"},
        {"severity": "note", "category": "other",
         "summary": "One owner, German vehicle", "evidence": "1 Halter"},
    ])
    items = carcache.viewing_checklist(car, seed_profile())
    assert any(i.startswith("Inspect (major): Seller discloses coolant loss") for i in items)
    assert any(i.startswith("Clarify the contradiction:") for i in items)
    # notes are context, not tasks
    assert not any("German vehicle" in i for i in items)


def test_checklist_flags_unfitted_towbar_only_when_wishlisted():
    prof = seed_profile()
    car = _valid_car(price_rating="gut", towbar="prep")
    assert any("Towbar" in i for i in carcache.viewing_checklist(car, prof))
    no_wish = json.loads(json.dumps(prof))
    no_wish["wishlist"].pop("towbar", None)
    assert not any("Towbar" in i for i in carcache.viewing_checklist(car, no_wish))


def test_mileage_score_absolute_and_smooth():
    prof = seed_profile()
    prof["mileage"] = {"best_km": 30000, "worst_km": 180000}
    assert carcache.mileage_score(30000, prof) == 10.0
    assert carcache.mileage_score(180000, prof) == 0.0
    assert carcache.mileage_score(10000, prof) == 10.0    # clamped above the top
    assert carcache.mileage_score(200000, prof) == 0.0    # clamped below the floor
    # monotonic and mid-range is not near-zero (the old min-max over-penalised)
    assert carcache.mileage_score(69575, prof) > carcache.mileage_score(110000, prof)
    assert carcache.mileage_score(110000, prof) > 3.0


def test_location_score_curve_and_fallback():
    prof = seed_profile()
    prof["location"] = {"best_minutes": 90, "worst_minutes": 480, "avg_kmh": 95,
                        "fallback_score": 5.0}
    berlin = _valid_car(location="Berlin", travel_minutes=1, travel_mode="car")
    near = _valid_car(location="Heimburg", travel_minutes=159, travel_mode="car")
    far = _valid_car(location="Mössingen", travel_minutes=432, travel_mode="car")
    assert carcache.location_score(berlin, prof) == 10.0          # Berlin/Brandenburg -> 10
    assert carcache.location_score(far, prof) < carcache.location_score(near, prof) < 10.0
    assert carcache.location_score(_valid_car(travel_minutes=600), prof) == 0.0   # clamped
    # no travel data -> flagged fallback
    assert carcache.location_score(_valid_car(), prof) == 5.0


def test_location_falls_back_to_distance_when_no_time():
    """With only a distance stored, travel time is derived at avg_kmh."""
    prof = seed_profile()
    car = _valid_car(travel_km=190)                      # ~120 min at 95 km/h
    mins, mode = carcache.travel_minutes_for(car, prof)
    assert 110 < mins < 130 and mode == "car_km"
    assert 0.0 < carcache.location_score(car, prof) <= 10.0


def test_phev_energy_includes_electricity():
    """A plug-in's WLTP litres must not be used raw — electricity is a real cost."""
    prof = seed_profile()
    phev = _valid_car(fuel="Hybrid-Benzin", consumption_l100=1.6, model="Seat Leon e-Hybrid")
    assert carcache.is_phev(phev, prof) is True
    petrol, elec, estimated = carcache.energy_cost(phev, prof)
    assert elec > 0 and estimated is True
    # the naive WLTP reading would be ~1.6/100*6000*1.85 = 178 EUR; the modelled energy is far higher
    assert petrol + elec > 400
    _tco, parts, _e = carcache.annual_tco(phev, prof)
    assert parts["electricity"] > 0


def test_phev_badge_in_variant_field_is_detected():
    """The plug-in badge (GTE, e-Hybrid) usually lives in `variant`, not `model`. A GTE with
    no stated consumption must still be recognised as a plug-in, not scored as a plain hybrid."""
    prof = seed_profile()
    gte = _valid_car(fuel="Hybrid-Benzin", consumption_l100=None,
                     make="Volkswagen", model="Passat Variant", variant="GTE")
    assert carcache.is_phev(gte, prof) is True
    petrol, elec, _ = carcache.energy_cost(gte, prof)
    assert elec > 0            # modelled as a plug-in, electricity charged
    ehyb = _valid_car(fuel="Hybrid-Benzin", consumption_l100=None,
                      make="Seat", model="Leon", variant="FR e-Hybrid")
    assert carcache.is_phev(ehyb, prof) is True


def test_full_hybrid_is_not_treated_as_plug_in():
    """A self-charging/mild hybrid burns only petrol — its stated figure is used as-is."""
    prof = seed_profile()
    hev = _valid_car(fuel="Hybrid-Benzin", consumption_l100=4.4, model="Ford Mondeo 2.0 Hybrid")
    assert carcache.is_phev(hev, prof) is False
    petrol, elec, _ = carcache.energy_cost(hev, prof)
    assert elec == 0.0 and petrol > 0
    _tco, parts, _e = carcache.annual_tco(hev, prof)
    assert "electricity" not in parts


def test_wishlist_has_no_filtered_out_options():
    """carplay/heated_seats are search filters, not differentiators — they must not be scored."""
    wl = seed_profile()["wishlist"]
    assert "carplay" not in wl and "heated_seats" not in wl
    assert all(isinstance(v, (int, float)) for v in wl.values())   # no stray _comment key


# --------------------------------------------------------------------------------------
# Cost — a composite of asking price, the platform's price verdict, and running cost
# --------------------------------------------------------------------------------------

def _cost_profile(**over):
    """A profile with an explicit cost block. Pinned here, not read from profile.json: the
    curve and the component weights are the user's to retune, and a test must not fail because
    they did."""
    cfg = {"weights": {"price": 1, "price_rating": 1, "running": 1},
           "price": {"best_eur": 10000, "worst_eur": 15000},
           "running": {"best_eur_yr": 2000, "worst_eur_yr": 4000},
           "price_rating_map": {"sehr_gut": 10, "gut": 7.5, "fair": 5, "erhoeht": 2.5, "hoch": 0},
           "price_rating_unknown": 5}
    cfg.update(over)
    prof = seed_profile()
    prof["cost"] = cfg
    return prof


def test_price_score_is_an_absolute_clamped_curve():
    prof = _cost_profile()          # best_eur 10000 -> 10, worst_eur 15000 -> 0
    assert carcache.price_score(10000, prof) == 10.0
    assert carcache.price_score(15000, prof) == 0.0
    assert carcache.price_score(12500, prof) == pytest.approx(5.0)
    assert carcache.price_score(5000, prof) == 10.0     # clamped, not 20
    assert carcache.price_score(99000, prof) == 0.0     # clamped, not negative


def test_price_score_absent_price_is_worst_not_best():
    """A missing price must never look like a bargain."""
    assert carcache.price_score(None, _cost_profile()) == 0.0


def test_price_rating_maps_all_five_buckets():
    prof = _cost_profile()
    got = {r: carcache.price_rating_score({"price_rating": r}, prof)
           for r in ("sehr_gut", "gut", "fair", "erhoeht", "hoch")}
    assert got == {"sehr_gut": 10, "gut": 7.5, "fair": 5, "erhoeht": 2.5, "hoch": 0}


def test_unrated_listing_scores_neutral_not_zero():
    """Silence is not a bad price — an unrated car must not be punished as if it were."""
    prof = _cost_profile()
    assert carcache.price_rating_score({"price_rating": None}, prof) == 5
    assert carcache.price_rating_score({"price_rating": "nonsense"}, prof) == 5


def test_cost_is_the_weighted_mean_of_its_three_parts():
    prof = _cost_profile()                                   # equal thirds
    car = _valid_car(price_eur=12500, price_rating="gut")     # price 5.0, rating 7.5
    items, composite = carcache.cost_detail(car, prof, 2000)  # running 2000 -> 10.0
    assert [v for _k, _l, v, _w in items] == [5.0, 7.5, 10.0]
    assert [round(w, 4) for _k, _l, _v, w in items] == [0.3333, 0.3333, 0.3333]
    assert composite == pytest.approx((5.0 + 7.5 + 10.0) / 3, abs=0.01)


def test_cost_component_weights_are_relative_not_fractions():
    """30/40/30 and 3/4/3 must mean the same thing — they are normalised, not read as shares."""
    car = _valid_car(price_eur=12500, price_rating="gut")
    common = dict(price={"best_eur": 10000, "worst_eur": 15000},
                  running={"best_eur_yr": 2000, "worst_eur_yr": 4000},
                  price_rating_map={"gut": 7.5}, price_rating_unknown=5)
    a = carcache.cost_score(car, _cost_profile(
        weights={"price": 30, "price_rating": 40, "running": 30}, **common), 2000)
    b = carcache.cost_score(car, _cost_profile(
        weights={"price": 3, "price_rating": 4, "running": 3}, **common), 2000)
    assert a == pytest.approx(b)
    assert a == pytest.approx(0.3 * 5.0 + 0.4 * 7.5 + 0.3 * 10.0, abs=0.01)


def test_a_single_component_can_be_weighted_to_zero():
    prof = _cost_profile(weights={"price": 2, "price_rating": 0, "running": 0},
                         price={"best_eur": 10000, "worst_eur": 15000},
                         price_rating_map={"sehr_gut": 10}, price_rating_unknown=5)
    car = _valid_car(price_eur=12500, price_rating="sehr_gut")
    assert carcache.cost_score(car, prof, 4000) == pytest.approx(5.0)


def test_cheap_car_can_still_be_poor_value():
    """The point of the composite: price and the market's verdict on that price disagree, and
    both count. With equal weights a cheap-but-overpriced car ties a dear-but-well-priced one."""
    prof = _cost_profile()
    cheap_overpriced = _valid_car(price_eur=10000, price_rating="hoch")     # 10 + 0
    dear_bargain = _valid_car(price_eur=15000, price_rating="sehr_gut")     # 0 + 10
    a = carcache.cost_score(cheap_overpriced, prof, 3000)
    b = carcache.cost_score(dear_bargain, prof, 3000)
    assert a == pytest.approx(b, abs=0.01)
    # ...and neither ties a car that is both cheap AND well-priced
    best = _valid_car(price_eur=10000, price_rating="sehr_gut")
    assert carcache.cost_score(best, prof, 3000) > a


def test_cost_weight_and_weights_still_sum_to_one():
    w = seed_profile()["weights"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert "running_cost" not in w and "cost" in carcache.CRITERIA


def test_running_cost_score_is_an_absolute_clamped_curve():
    prof = _cost_profile()          # best 2000 -> 10, worst 4000 -> 0
    assert carcache.running_cost_score(2000, prof) == 10.0
    assert carcache.running_cost_score(4000, prof) == 0.0
    assert carcache.running_cost_score(3000, prof) == pytest.approx(5.0)
    assert carcache.running_cost_score(500, prof) == 10.0     # clamped
    assert carcache.running_cost_score(9000, prof) == 0.0     # clamped
    assert carcache.running_cost_score(None, prof) == 0.0     # unknown is worst, not best


def test_cash_running_excludes_exactly_depreciation():
    parts = {"fuel": 700, "tax": 100, "maintenance": 800, "insurance": 600,
             "depreciation": 1300, "hu": 65}
    assert carcache.cash_running(parts) == 700 + 100 + 800 + 600 + 65


def test_depreciation_does_not_reach_the_cost_score():
    """Depreciation is derived from the asking price, which already has its own third —
    scoring it too would charge the same euros twice. Two cars with identical cash costs
    must get the same Cost score however their depreciation differs."""
    prof = _cost_profile()
    car = _valid_car(price_eur=12500, price_rating="gut")
    cheap_dep = {"fuel": 700, "tax": 100, "maintenance": 800, "insurance": 600,
                 "depreciation": 900, "hu": 65}
    dear_dep = dict(cheap_dep, depreciation=1600)
    a = carcache.cost_score(car, prof, carcache.cash_running(cheap_dep))
    b = carcache.cost_score(car, prof, carcache.cash_running(dear_dep))
    assert a == b


def test_scores_are_candidate_set_independent():
    """Every criterion is an absolute curve or per-listing lookup, so a car's scores and
    total must be identical whether it is ranked alone or with the whole catalogue."""
    store, prof = seed_store(), seed_profile()
    full = {r.cid: r for r in carcache.compute_ranking(store, prof)}
    some_cid = sorted(full)[0]
    alone = carcache.compute_ranking(store, prof, ids=[some_cid])[0]
    assert alone.scores == full[some_cid].scores
    assert alone.total == full[some_cid].total


def test_parse_all_five_price_ratings():
    for text, want in (("14.950 € Sehr guter Preis", "sehr_gut"), ("Guter Preis", "gut"),
                       ("Fairer Preis", "fair"), ("Erhöhter Preis", "erhoeht"),
                       ("Hoher Preis", "hoch"), ("no verdict here", None)):
        assert extract.parse_listing(text)["price_rating"] == want


def test_mileage_score_independent_of_candidate_set():
    """Absolute curve => a car's mileage score does not change with the other candidates."""
    prof = seed_profile()
    lone = {"1": _valid_car(km=100000)}
    mixed = {"1": _valid_car(km=100000), "2": _valid_car(km=40000), "3": _valid_car(km=119000)}
    s_lone = carcache.compute_ranking(lone, prof)[0].scores["mileage"]
    r_mixed = {r.cid: r for r in carcache.compute_ranking(mixed, prof)}
    assert r_mixed["1"].scores["mileage"] == s_lone


def test_map_lookup_ignores_diacritics():
    """mobile.de's header writes 'Megane'; the basis is keyed 'renault mégane'. A miss here is
    SILENT — the car just takes the fallback — so the fold must hold in both directions."""
    prof = seed_profile()
    megane = _valid_car(make="Renault", model="Megane", first_reg="01/2022")
    accented = _valid_car(make="Renault", model="Mégane", first_reg="01/2022")
    assert carcache.reliability_detail(megane, prof)["fallback"] is False
    assert carcache.reliability_score(megane, prof) == carcache.reliability_score(accented, prof)


def test_fold_strips_case_and_accents():
    assert carcache._fold("Mégane") == "megane"
    assert carcache._fold("Citroën C4") == "citroen c4"
    assert carcache._fold("SKODA Octavia") == "skoda octavia"


def test_reliability_basis_lookup_longest_match():
    prof = seed_profile()
    insignia = _valid_car(make="Opel", model="Insignia")
    astra = _valid_car(make="Opel", model="Opel Astra ST")
    b_ins = carcache.reliability_basis(insignia, prof)
    b_ast = carcache.reliability_basis(astra, prof)
    # the model-specific rows resolve distinctly (neither collapses to a shared "opel" row)
    assert b_ins.get("adac_by_year", {}).get("2017") == 37.1
    assert b_ast.get("adac_by_year", {}).get("2017") == 11.6
    assert "tuv_by_age" in b_ast and b_ast is not b_ins


def test_reliability_per_cell_lookup_by_age_and_year():
    """TÜV is picked by age band, ADAC by registration year; both feed the score."""
    prof = seed_profile()
    car = _valid_car(make="Skoda", model="Skoda Octavia Combi", first_reg="01/2019")
    d = carcache.reliability_detail(car, prof, today=_dt.date(2026, 1, 1))  # age 7 -> band 6-7
    assert d["components"]["tuv"]["key"] == "6-7"
    assert d["components"]["adac"]["key"] == "2019"
    assert d["components"]["adac"]["raw"] == 8.9      # official Octavia 2019 cell


def test_report_html_full_labels_and_new_sections():
    html = carcache.build_report_html(seed_store(), seed_profile())
    for label in ("Reliability", "Condition", "Equipment", "Power", "Cost", "Mileage"):
        assert f">{label}</th>" in html
    # anchors are structural (methodology links target them); headings are user-editable prose
    for anchor in ("m-reliability", "m-condition", "m-equipment", "m-power", "m-cost"):
        assert f'id="{anchor}"' in html and f'href="#{anchor}"' in html
    assert "Mängelquote" in html                      # ADAC/TÜV basis surfaced in the table
    assert ">rel</th>" not in html                    # old abbreviations gone from headers


def test_report_text_config_drives_the_page():
    """Static prose comes from report_text.json, with $placeholders filled from the profile."""
    texts = carcache.load_texts()
    assert texts, "report_text.json should exist"
    assert carcache.text_at(texts, "sections.ranking.heading") == "Ranking"
    html = carcache.build_report_html(seed_store(), seed_profile(), texts=texts)
    assert carcache.text_at(texts, "title") in html
    assert 'class="intro"' in html                       # the summary block under the title
    assert "$rel_gain" not in html and "$loc_origin" not in html   # placeholders got filled

    # editing the config changes the page, without touching Python
    custom = json.loads(json.dumps(texts))
    custom["sections"]["ranking"]["heading"] = "Shortlist"
    out = carcache.build_report_html(seed_store(), seed_profile(), texts=custom)
    assert ">Shortlist</h2>" in out and ">Ranking</h2>" not in out


def test_report_text_missing_file_falls_back():
    """A missing/incomplete text config must not break the report."""
    assert carcache.load_texts("does-not-exist.json") == {}
    html = carcache.build_report_html(seed_store(), seed_profile(), texts={})
    assert html.startswith("<!doctype html>") and ">Ranking</h2>" in html


def test_weights_are_shown_as_percentages():
    assert carcache.fmt_weight(0.2) == "20%" and carcache.fmt_weight(0.115) == "11.5%"
    prof = seed_profile()
    html = carcache.build_report_html(seed_store(), prof)
    assert f"weight {carcache.fmt_weight(prof['weights']['cost'])}" in html
    assert "weight 0.2" not in html and "(0.2" not in html


def test_reliability_derived_older_scores_lower():
    """Same model, older registration -> lower reliability (age is baked into the listing)."""
    prof = seed_profile()
    newer = _valid_car(make="Skoda", model="Skoda Octavia Combi", first_reg="01/2024")
    older = _valid_car(make="Skoda", model="Skoda Octavia Combi", first_reg="01/2017")
    r_new = carcache.reliability_score(newer, prof, today=_dt.date(2026, 1, 1))
    r_old = carcache.reliability_score(older, prof, today=_dt.date(2026, 1, 1))
    assert r_new > r_old + 1.0                         # a clear, not marginal, gap
    assert 0.0 <= r_old <= 10.0 and 0.0 <= r_new <= 10.0


def test_reliability_better_model_scores_higher_same_age():
    """At equal age, the model with better TÜV+ADAC figures scores higher."""
    prof = seed_profile()
    vw = _valid_car(make="Volkswagen", model="VW Golf", first_reg="01/2019")
    insignia = _valid_car(make="Opel", model="Opel Insignia B", first_reg="01/2019")
    assert carcache.reliability_score(vw, prof) > carcache.reliability_score(insignia, prof)


def test_reliability_fallback_when_no_basis_figure():
    """A make with no basis figure gets the neutral fallback and is flagged."""
    prof = seed_profile()
    unknown = _valid_car(make="Tesla", model="Tesla Model Y")
    d = carcache.reliability_detail(unknown, prof)
    assert d["fallback"] is True
    assert d["score"] == prof["reliability_model"]["fallback_score"]
    # and the ranking table flags it
    store = {"x": unknown}
    assert '<span class="warn">n/a</span>' in carcache.build_report_html(store, prof)


def test_reliability_uses_available_metric_when_one_missing():
    """A model with only TÜV cells (no ADAC, e.g. a discontinued model) is scored on TÜV alone."""
    prof = seed_profile()
    prof["model_reliability_basis"] = {"skoda": {"tuv_by_age": {"6-7": 12.0}}}   # adac absent
    car = _valid_car(make="Skoda", model="Skoda Octavia", first_reg="01/2020")
    d = carcache.reliability_detail(car, prof, today=_dt.date(2026, 1, 1))
    assert d["fallback"] is False and "tuv" in d["components"] and "adac" not in d["components"]


# --------------------------------------------------------------------------------------
# independent verification tool
# --------------------------------------------------------------------------------------

def test_verify_store_flags_wrong_km(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "1.pdf").write_bytes(b"x")
    store = {"1": carcache.normalize_record(
        {"model": "X", "km": 50000, "power_ps": 150, "first_reg": "03/2019"})}
    reader = lambda p: "Kilometerstand 90.000 km\nErstzulassung 03/2019\nLeistung 110 kW (150 PS)"
    cid, status, issues = extract.verify_store(store, processed, ids=["1"], text_reader=reader)[0]
    assert status == "MISMATCH"
    assert any(f.startswith("km") for f, _, _ in issues)


def test_verify_store_ok_when_matching(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "1.pdf").write_bytes(b"x")
    store = {"1": carcache.normalize_record(
        {"model": "X", "body": "Kombi", "km": 90000, "power_ps": 150, "first_reg": "03/2019"})}
    reader = lambda p: "Kategorie Kombi\nKilometerstand 90.000 km\nErstzulassung 03/2019\nLeistung 110 kW (150 PS)"
    assert extract.verify_store(store, processed, ids=["1"], text_reader=reader)[0][1] == "ok"
