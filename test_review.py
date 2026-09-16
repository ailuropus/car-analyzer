"""Tests for review.py — the Claude API listing review.

All offline: no API key, no network. The API call is exercised through a stub client that
returns a canned response in the real SDK's shape.
"""

from __future__ import annotations

import base64
import json
import os
from unittest import mock
from pathlib import Path

import pytest

import carcache
import review

REPO = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# stub SDK objects (shaped like anthropic's response)
# --------------------------------------------------------------------------------------

class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Usage:
    input_tokens, output_tokens = 1234, 567


class _Response:
    def __init__(self, payload, stop_reason="end_turn"):
        self.content = [_Block(json.dumps(payload))]
        self.stop_reason = stop_reason
        self.usage = _Usage()
        self.model = "claude-opus-4-8"


class _Messages:
    def __init__(self, payload, stop_reason="end_turn"):
        self._payload, self._stop = payload, stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self._payload, self._stop)

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return type("C", (), {"input_tokens": 50_000})()


class _Client:
    def __init__(self, payload=None, stop_reason="end_turn"):
        self.messages = _Messages(payload or _OK, stop_reason)


_OK = {
    "options_from_prose": ["towbar"],
    "findings": [
        {"severity": "major", "category": "mechanical",
         "summary": "Seller discloses coolant loss.", "evidence": "Kühlmittelverlust"},
        {"severity": "note", "category": "usage",
         "summary": "One previous owner.", "evidence": "1 Hand"},
    ],
    "notable_options": ["Towbar (detachable)", "Panoramic roof"],
    "summary": "A tidy estate with a disclosed coolant leak.",
}


@pytest.fixture()
def pdf(tmp_path, monkeypatch):
    """A stand-in PDF. Text mode is the default, so stub the text layer rather than build a
    real PDF; --with-images tests read the bytes."""
    p = tmp_path / "460583296.pdf"
    p.write_bytes(b"%PDF-1.4\nfake\n")
    monkeypatch.setattr(review, "text_layer_block", lambda _p: {
        "type": "text", "text": "<listing>Kilometerstand 85.800 km\nKühlmittelverlust</listing>"})
    return p


@pytest.fixture()
def real_pdf():
    """An actual archived listing, for the text-layer extraction path."""
    p = REPO / "processed" / "460583296.pdf"
    if not p.exists():
        pytest.skip("no archived PDF available")
    return p


# --------------------------------------------------------------------------------------
# .env loading (the API key never belongs in a tracked file)
# --------------------------------------------------------------------------------------

def test_dotenv_parses_comments_quotes_and_export(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nANTHROPIC_API_KEY="sk-ant-quoted"\n'
                   "export OTHER=plain\nBAD_LINE_NO_EQUALS\n")
    for k in ("ANTHROPIC_API_KEY", "OTHER"):
        monkeypatch.delenv(k, raising=False)
    loaded = review.load_dotenv(env)
    assert set(loaded) == {"ANTHROPIC_API_KEY", "OTHER"}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted"   # quotes stripped
    assert os.environ["OTHER"] == "plain"                        # `export ` stripped


def test_real_env_var_beats_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    assert review.load_dotenv(env) == []                          # nothing overridden
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"


def test_dotenv_missing_file_is_fine(tmp_path):
    assert review.load_dotenv(tmp_path / "nope.env") == []


def test_dotenv_is_gitignored():
    """A committed key is a leaked key — this must never regress."""
    patterns = (REPO / ".gitignore").read_text().split()
    assert ".env" in patterns
    assert ".env.example" not in patterns or "!.env.example" in patterns


# --------------------------------------------------------------------------------------
# prompt + schema
# --------------------------------------------------------------------------------------

def test_prompt_loads_and_strips_editor_notes():
    prompt = review.load_prompt()
    assert prompt and "You are reviewing a single used-car listing" in prompt
    assert "Edit freely" not in prompt          # the editor-facing header is stripped
    for word in ("major", "minor", "note"):     # severity contract is stated
        assert word in prompt


def test_schema_is_strict_and_matches_the_scorer():
    s = review.REVIEW_SCHEMA
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"findings", "options_from_prose", "notable_options", "summary"}
    f = review.FINDING_SCHEMA
    assert f["additionalProperties"] is False
    assert set(f["required"]) == {"severity", "category", "summary", "evidence"}
    # the severities the prompt promises are exactly the ones condition() understands
    assert f["properties"]["severity"]["enum"] == list(carcache.CONDITION_SEVERITIES)


def test_default_is_text_only_not_page_images(pdf):
    """The cost fix: page images are ~5x the tokens, so text is the default."""
    content = review.build_messages(pdf)[0]["content"]
    assert content[0]["type"] == "text"                 # not "document"
    assert not any(b["type"] == "document" for b in content)
    assert "No photos are included" in content[1]["text"]


def test_with_images_sends_base64_pdf_before_the_text(pdf):
    content = review.build_messages(pdf, with_images=True)[0]["content"]
    assert content[0]["type"] == "document" and content[1]["type"] == "text"
    src = content[0]["source"]
    assert src["type"] == "base64" and src["media_type"] == "application/pdf"
    assert "\n" not in src["data"]
    assert base64.standard_b64decode(src["data"]) == pdf.read_bytes()
    assert "including its photos" in content[1]["text"]


def test_text_layer_extracts_a_real_listing_and_trims_similar_vehicles(real_pdf):
    block = review.text_layer_block(real_pdf)
    text = block["text"]
    assert block["type"] == "text"
    assert "Kilometerstand" in text and "85.800" in text     # the real car's data is there
    assert "Ähnliche Fahrzeuge" not in text                  # the other-cars block is trimmed


def test_text_mode_is_far_cheaper_than_images(real_pdf):
    """Guards the whole point of text mode — chars are a fair proxy for tokens here."""
    text_chars = len(review.text_layer_block(real_pdf)["text"])
    pdf_b64_chars = len(review.pdf_document_block(real_pdf)["source"]["data"])
    assert pdf_b64_chars > text_chars * 20


def test_oversized_pdf_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "MAX_REQUEST_MB", 0.000001)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"x" * 5000)
    with pytest.raises(SystemExit, match="request limit"):
        review.pdf_document_block(big)


# --------------------------------------------------------------------------------------
# request shape
# --------------------------------------------------------------------------------------

def test_review_pdf_request_shape(pdf):
    client = _Client()
    review.review_pdf(pdf, client=client)
    kw = client.messages.calls[0]
    assert kw["model"] == review.DEFAULT_MODEL
    assert kw["thinking"] == {"type": "adaptive"}          # off by default on Opus 4.8 otherwise
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"] is review.REVIEW_SCHEMA
    assert kw["output_config"]["effort"] == "high"
    assert kw["system"] == review.load_prompt()


def test_review_pdf_returns_validated_findings(pdf):
    out = review.review_pdf(pdf, client=_Client())
    assert [f["severity"] for f in out["findings"]] == ["major", "note"]
    assert out["notable_options"] == ["Towbar (detachable)", "Panoramic roof"]
    assert out["usage"]["input"] == 1234 and out["usage"]["usd"] > 0
    assert out["model"] == "claude-opus-4-8" and out["prompt_sha"]
    assert out["raw_text"]                      # verbatim response kept for audit


def test_review_output_feeds_condition_directly(pdf):
    """The whole point: review output drops into a record and scores with no keyword matching."""
    out = review.review_pdf(pdf, client=_Client())
    car = carcache.normalize_record({"model": "X", "owners": 1, "unfallfrei_declared": True,
                                     "service_history": True, "tuv": "Neu",
                                     "findings": out["findings"]})
    prof = carcache.load_profile(REPO / "profile.json")
    assert carcache.count_findings(car) == (1, 0)         # the 'note' is not counted
    assert carcache.condition(car, prof) < 7.0            # the major defect lands


def test_prose_options_reach_equipment_fit(pdf):
    """An option the seller only DESCRIBES must still score — the parser never sees it."""
    out = review.review_pdf(pdf, client=_Client())
    assert out["options_from_prose"] == ["towbar"]
    car = carcache.normalize_record({"model": "X", "towbar": None,
                                     "options_from_prose": out["options_from_prose"]})
    assert carcache._has_option(car, "towbar") is True
    blind = carcache.normalize_record({"model": "X", "towbar": None})
    assert carcache._has_option(blind, "towbar") is False


def test_prose_options_are_restricted_to_the_wishlist_vocabulary():
    """A key equipment_fit cannot look up would score nothing while looking like it had."""
    assert set(review.wishlist_options()) == set(
        carcache.load_profile(REPO / "profile.json")["wishlist"])
    with pytest.raises(ValueError, match="unknown option"):
        review.validate({"findings": [], "notable_options": [], "summary": "",
                         "options_from_prose": ["heated_cupholder"]})


def test_review_only_writes_llm_fields(pdf):
    """The hard invariant: a review may never overwrite a field the PDF parser owns."""
    out = review.review_pdf(pdf, client=_Client())
    car = carcache.normalize_record({"make": "Ford", "model": "Focus", "km": 85800})
    review.apply_to_record(car, out)
    assert car["model"] == "Focus" and car["make"] == "Ford" and car["km"] == 85800
    assert car["reviewed_by"] == "claude-opus-4-8"   # the LLM id lands here, not in `model`
    assert car["review_summary"] and car["findings"]


def test_apply_to_record_refuses_to_write_a_parsed_field():
    """A guard, not a comment: `out` has a `model` key meaning the LLM id, which collides by
    name with the car's model — the assert is what stops that becoming a silent corruption."""
    car = carcache.normalize_record({"model": "Focus"})
    with pytest.raises(AssertionError, match="LLM_FIELDS"):
        with mock.patch.object(carcache, "LLM_FIELDS", ["findings"]):
            review.apply_to_record(car, {**_OK, "model": "claude-sonnet-5", "source": "text",
                                         "prompt_sha": "abc", "options_from_prose": []})


# --------------------------------------------------------------------------------------
# validation + failure modes
# --------------------------------------------------------------------------------------

def test_validate_rejects_unknown_severity():
    with pytest.raises(ValueError, match="unknown severity"):
        review.validate({"findings": [{"severity": "critical", "category": "mechanical",
                                       "summary": "s", "evidence": "e"}]})


def test_validate_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown category"):
        review.validate({"findings": [{"severity": "major", "category": "vibes",
                                       "summary": "s", "evidence": "e"}]})


def test_validate_tolerates_empty_review():
    out = review.validate({"findings": [], "options_from_prose": [],
                           "notable_options": [], "summary": ""})
    assert out == {"findings": [], "options_from_prose": [], "notable_options": [], "summary": ""}


def test_refusal_is_surfaced_not_swallowed(pdf):
    with pytest.raises(SystemExit, match="refusal"):
        review.review_pdf(pdf, client=_Client(stop_reason="refusal"))


def test_truncated_response_is_surfaced(pdf):
    with pytest.raises(SystemExit, match="max_tokens"):
        review.review_pdf(pdf, client=_Client(stop_reason="max_tokens"))


def test_estimate_reports_tokens_and_cost(pdf):
    n, usd = review.estimate(pdf, client=_Client())
    assert n == 50_000
    inp, outp = review.prices(review.DEFAULT_MODEL)
    assert usd == pytest.approx(50_000 / 1e6 * inp + 1200 / 1e6 * outp)


# --------------------------------------------------------------------------------------
# target selection — every re-review is a paid API call
# --------------------------------------------------------------------------------------

def _reviewed(**over):
    car = {"reviewed_at": "2026-07-15", "findings": [], "review_summary": "clean"}
    car.update(over)
    return car


def test_a_clean_car_is_not_re_reviewed(tmp_path, monkeypatch):
    """An empty `findings` list is a correct answer, not a missing one. Testing the list's
    truthiness re-billed every clean car on every run."""
    monkeypatch.setattr(review, "PROCESSED_DIR", tmp_path)
    (tmp_path / "1.pdf").write_bytes(b"%PDF")
    assert review._targets({"1": _reviewed()}, None, force=False) == []


def test_an_unreviewed_car_is_a_target(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "PROCESSED_DIR", tmp_path)
    (tmp_path / "1.pdf").write_bytes(b"%PDF")
    store = {"1": {"reviewed_at": None, "findings": []}}
    assert [cid for cid, _ in review._targets(store, None, force=False)] == ["1"]


def test_force_re_reviews_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "PROCESSED_DIR", tmp_path)
    (tmp_path / "1.pdf").write_bytes(b"%PDF")
    assert len(review._targets({"1": _reviewed()}, None, force=True)) == 1


def test_a_car_with_no_pdf_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "PROCESSED_DIR", tmp_path)
    assert review._targets({"1": {"reviewed_at": None}}, None, force=False) == []
