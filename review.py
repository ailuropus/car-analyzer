#!/usr/bin/env python3
"""review.py — have Claude read a listing PDF and return structured findings.

This replaces the old hand-written `flags` prose (a leftover from when a human/LLM read the
PDFs by eye) with a reproducible API call that returns a fixed JSON shape the scorer consumes
directly. The key difference from the old prose: the model states an explicit `severity`, so
`condition()` no longer has to keyword-match English sentences to guess how bad something is.

    setup:   .venv/bin/pip install -r requirements.txt
             export ANTHROPIC_API_KEY=...        (or: ant auth login)

    usage:   review.py --dry-run --store              # what would it cost? (no findings written)
             review.py --store                        # review every car missing findings
             review.py --store 460583296 --force      # re-review specific ids
             review.py --pdf processed/460583296.pdf  # one PDF, print JSON, write nothing

Needs `anthropic` + `pypdf`; carcache.py stays stdlib-only and never imports this.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import carcache

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "review_prompt.md"
PROCESSED_DIR = BASE_DIR / "processed"
REVIEWS_DIR = BASE_DIR / "reviews"          # audit trail: one JSON per review run
ENV_PATH = BASE_DIR / ".env"

# Claude Opus 4.8. Structured outputs + PDF input + adaptive thinking are all supported here.
# Chosen by experiment, not by vibes: 5 listings x 3 models, graded on whether each `evidence`
# quote actually appears in the source. Sonnet 5 matched Opus 4.8's scoring-relevant output on
# 4 of 5 listings (both caught the undisclosed crash damage and graded it the same), had zero
# ungrounded quotes, and costs ~a third less. Haiku 4.5 never used `note` at all — it collapsed
# context into `minor`, which over-penalises. Re-run scratchpad/experiment.py after prompt edits.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "high"
MAX_TOKENS = 16000
MAX_REQUEST_MB = 32  # hard API limit on the whole request; base64 inflates a PDF by ~33%

# $ per 1M tokens (input, output). Used for the --dry-run estimate and the recorded usage.usd.
MODEL_PRICES = {
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-sonnet-5":   (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00, 5.00),
}
DEFAULT_PRICE = (5.00, 25.00)


# Adaptive thinking and `effort` are 4.6+ features. Older models (Haiku 4.5, Sonnet 4.5) reject
# both with a 400 and need the legacy fixed thinking budget instead. Structured outputs work on
# all of these. Keep this table rather than assuming — a wrong guess is a hard API error.
ADAPTIVE_THINKING_MODELS = ("claude-fable-5", "claude-mythos-5", "claude-opus-4-8",
                            "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-5",
                            "claude-sonnet-4-6")
LEGACY_THINKING_BUDGET = 4000   # must be < MAX_TOKENS, min 1024


def request_tuning(model: str, effort: str) -> tuple[dict, dict]:
    """(thinking, output_config) shaped for this model's generation."""
    fmt = {"type": "json_schema", "schema": REVIEW_SCHEMA}
    if model in ADAPTIVE_THINKING_MODELS:
        return {"type": "adaptive"}, {"format": fmt, "effort": effort}
    # pre-4.6: fixed thinking budget, and `effort` is rejected outright
    return {"type": "enabled", "budget_tokens": LEGACY_THINKING_BUDGET}, {"format": fmt}


def prices(model: str) -> tuple[float, float]:
    """(input, output) $/Mtok.

    The API echoes a DATED id back (`claude-haiku-4-5-20251001`) even when you request the alias,
    so match on prefix — an exact-key lookup silently fell through to the Opus fallback and
    over-reported Haiku's cost by ~5x. Unknown models still fall back to the priciest tier, so a
    miss over-estimates rather than under-estimates."""
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    for alias, price in MODEL_PRICES.items():
        if model.startswith(alias):
            return price
    return DEFAULT_PRICE

SEVERITIES = ["major", "minor", "note"]
CATEGORIES = ["accident", "mechanical", "corrosion", "wear", "service_history",
              "usage", "warranty", "inconsistency", "other"]

# The contract with the scorer. `severity` is what condition() reads; keep it in sync with
# carcache.CONDITION_SEVERITIES. Strict mode requires additionalProperties:false + required.
FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": SEVERITIES,
                     "description": "major = disclosed defect costing real money or affecting "
                                    "safety; minor = real but bounded disclosed issue; "
                                    "note = context, not a defect (never penalised)."},
        "category": {"type": "string", "enum": CATEGORIES},
        "summary": {"type": "string",
                    "description": "One factual, specific sentence in English."},
        "evidence": {"type": "string",
                     "description": "Short supporting quote from the listing, in the original "
                                    "German, or a pointer to the structured field/photo."},
    },
    "required": ["severity", "category", "summary", "evidence"],
    "additionalProperties": False,
}

# The wishlist vocabulary, straight from the profile so the two can never drift apart. Options
# are reported as ENUM KEYS rather than free text: `equipment_fit` looks options up by key, and
# a model writing "Anhängerkupplung (abnehmbar)" would score nothing while looking like it had.
def wishlist_options() -> list[str]:
    try:
        wishlist = json.loads((BASE_DIR / "profile.json").read_text(encoding="utf-8"))["wishlist"]
    except (OSError, ValueError, KeyError):
        return []
    return sorted(wishlist if isinstance(wishlist, list) else wishlist.keys())


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        # Options the seller DESCRIBES but did not tick in the structured equipment list. The
        # parser can only see the list, so without this an option mentioned solely in prose is
        # invisible to scoring. Unioned into equipment_fit by carcache._has_option().
        "options_from_prose": {"type": "array",
                               "items": {"type": "string", "enum": wishlist_options()}},
        "notable_options": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["findings", "options_from_prose", "notable_options", "summary"],
    "additionalProperties": False,
}

USER_TEXT = (
    "Review this mobile.de listing and return the structured findings, notable options, and "
    "summary described in your instructions. Report only what this listing actually discloses "
    "about this car. Returning an empty findings array is correct if nothing is disclosed."
)


def load_prompt(path: Path = PROMPT_PATH) -> str:
    """The system prompt lives in review_prompt.md so it can be tuned without touching code.
    Everything before the first '---' is editor-facing notes and is stripped."""
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition("\n---\n")
    return (body if sep else text).strip()


def prompt_sha(path: Path = PROMPT_PATH) -> str:
    """Short hash of the prompt, stored on every review so you can tell which prompt version
    produced a finding (and re-review only what is stale after you edit review_prompt.md)."""
    import hashlib
    return hashlib.sha1(load_prompt(path).encode("utf-8")).hexdigest()[:8]


def save_review(cid: str, payload: dict, reviews_dir: Path = REVIEWS_DIR) -> Path:
    """Persist one review verbatim so a human can audit what the model actually said.

    Two files per run:
      reviews/<id>.json                       -> the ACTIVE review (what cars.json reflects)
      reviews/history/<id>__<ts>__<model>.json -> append-only; never overwritten
    The store keeps only what scoring needs; the full text, evidence and usage live here."""
    reviews_dir.mkdir(exist_ok=True)
    (reviews_dir / "history").mkdir(exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    active = reviews_dir / f"{cid}.json"
    active.write_text(body, encoding="utf-8")
    stamp = payload.get("reviewed_at_utc", "").replace(":", "").replace("-", "")[:15] or "unknown"
    model = str(payload.get("model", "?")).replace("/", "-")
    (reviews_dir / "history" / f"{cid}__{stamp}__{model}.json").write_text(body, encoding="utf-8")
    return active


def load_review(cid: str, reviews_dir: Path = REVIEWS_DIR) -> dict | None:
    p = reviews_dir / f"{cid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pdf_document_block(pdf: Path) -> dict:
    """A base64 PDF content block. Must be placed BEFORE the text block in the message.

    Costly: the API renders EVERY PAGE to an image (~7 pages per mobile.de listing here), and a
    high-res page image can run several thousand tokens. Prefer text_layer_block()."""
    raw = pdf.read_bytes()
    data = base64.standard_b64encode(raw).decode("utf-8")  # no newlines
    mb = len(data) / 1_000_000
    if mb > MAX_REQUEST_MB:
        raise SystemExit(f"{pdf.name}: {mb:.1f} MB base64 exceeds the {MAX_REQUEST_MB} MB "
                         f"request limit — shrink the PDF or use text mode (the default).")
    return {"type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data}}


def text_layer_block(pdf: Path) -> dict:
    """The PDF's text layer only — no page images. ~5x cheaper than sending the PDF.

    mobile.de PDFs carry a real text layer holding everything a review needs (structured data,
    the seller's description, the equipment list); the page images are dealer marketing photos.
    Reuses extract._relevant() so the trailing 'similar vehicles' block is trimmed and the model
    never reviews the wrong car."""
    import extract
    text = extract._relevant(extract.read_text(pdf))
    if not text.strip():
        raise SystemExit(f"{pdf.name}: no text layer — re-run with --with-images")
    return {"type": "text", "text": f"<listing source=\"{pdf.name}\">\n{text}\n</listing>"}


def build_messages(pdf: Path, with_images: bool = False) -> list[dict]:
    """Text-only by default; --with-images sends the whole PDF (page images included)."""
    source = pdf_document_block(pdf) if with_images else text_layer_block(pdf)
    note = ("The listing PDF is attached, including its photos."
            if with_images else
            "The listing's text layer is below. No photos are included, so make no findings "
            "about anything only a photo could show.")
    return [{"role": "user", "content": [source, {"type": "text", "text": f"{note}\n\n{USER_TEXT}"}]}]


def load_dotenv(path: Path = ENV_PATH) -> list[str]:
    """Load `KEY=value` lines from .env into os.environ. Returns the keys it set.

    Deliberately minimal (no python-dotenv dependency, no interpolation or multi-line values) —
    it exists so an API key can live in a gitignored file instead of a shell profile. A real
    environment variable always wins, so `ANTHROPIC_API_KEY=... review.py` still overrides .env."""
    if not path.exists():
        return []
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"warning: {path.name} is readable by other users — chmod 600 {path.name}",
              file=sys.stderr)
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:  # never clobber a real env var
            os.environ[key] = val
            loaded.append(key)
    return loaded


def _client():
    try:
        import anthropic
    except ModuleNotFoundError:
        raise SystemExit("review.py needs the anthropic SDK:\n"
                         "  .venv/bin/pip install -r requirements.txt")
    load_dotenv()
    # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
    # Construction never fails: the SDK only resolves credentials at request time, and then
    # raises a bare TypeError. Check up front so the first failure is a useful sentence.
    client = anthropic.Anthropic()
    if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)):
        raise SystemExit(
            f"no Anthropic credentials found. Easiest: put your key in {ENV_PATH.name} "
            f"(gitignored)\n"
            f"    printf 'ANTHROPIC_API_KEY=sk-ant-...\\n' > {ENV_PATH.name} && chmod 600 {ENV_PATH.name}\n"
            "  or export ANTHROPIC_API_KEY=... in the shell, or `ant auth login`.\n"
            "  Get a key at platform.claude.com -> Settings -> API keys.")
    return client


def validate(data: dict) -> dict:
    """Defence in depth: the schema is enforced server-side, but a bad severity silently
    mis-scores a car, so re-check the fields the scorer depends on."""
    findings = []
    for f in data.get("findings") or []:
        sev, cat = f.get("severity"), f.get("category")
        if sev not in SEVERITIES:
            raise ValueError(f"unknown severity {sev!r} (expected one of {SEVERITIES})")
        if cat not in CATEGORIES:
            raise ValueError(f"unknown category {cat!r}")
        findings.append({"severity": sev, "category": cat,
                         "summary": str(f.get("summary", "")).strip(),
                         "evidence": str(f.get("evidence", "")).strip()})
    # An option key the wishlist does not define would score nothing and read as a silent miss,
    # so reject it loudly rather than storing a value equipment_fit will never look up.
    known = set(wishlist_options())
    prose_opts = []
    for o in data.get("options_from_prose") or []:
        if str(o) not in known:
            raise ValueError(f"unknown option {o!r} (expected one of {sorted(known)})")
        prose_opts.append(str(o))

    return {
        "findings": findings,
        "options_from_prose": sorted(set(prose_opts)),
        "notable_options": [str(o).strip() for o in (data.get("notable_options") or []) if str(o).strip()],
        "summary": str(data.get("summary", "")).strip(),
    }


def apply_to_record(car: dict, out: dict) -> dict:
    """Write a review onto a car record — the ONLY path from an LLM to the store.

    Enforces the invariant the whole extract/review split rests on: a review may write
    carcache.LLM_FIELDS and nothing else. Fields parsed from the PDF are a pure function of that
    PDF, and a model must never be able to overwrite one — that is what keeps a record
    reproducible instead of resampled. The assert is not decoration: `out` carries a `model` key
    meaning the LLM's model id, which collides by name with the car's `model` (its make/model),
    so a careless `car.update(out)` here would quietly rewrite every car as 'claude-sonnet-5'."""
    writes = {
        "findings": out["findings"],
        "options_from_prose": out["options_from_prose"],
        "notable_options": out["notable_options"] or car.get("notable_options") or [],
        "review_summary": out["summary"],
        "reviewed_by": out["model"],            # the LLM id -> reviewed_by, never `model`
        "reviewed_source": out["source"],
        "review_prompt_sha": out["prompt_sha"],
        "reviewed_at": carcache.date.today().isoformat(),
    }
    stray = set(writes) - set(carcache.LLM_FIELDS)
    if stray:
        raise AssertionError(f"review.py may only write LLM_FIELDS; refused: {sorted(stray)}")
    car.update(writes)
    return car


def _friendly(exc: Exception, model: str, what: str) -> SystemExit:
    """Turn an SDK exception into one actionable line instead of a traceback."""
    import anthropic
    if isinstance(exc, anthropic.AuthenticationError):
        return SystemExit(f"{what}: auth rejected (401) — the key in .env or ANTHROPIC_API_KEY "
                          f"is wrong or revoked. Check platform.claude.com -> API keys.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return SystemExit(f"{what}: this key may not use {model!r} (403).")
    if isinstance(exc, anthropic.NotFoundError):
        return SystemExit(f"{what}: model {model!r} not found — try --model claude-sonnet-5")
    if isinstance(exc, anthropic.RateLimitError):
        retry = exc.response.headers.get("retry-after", "60")
        return SystemExit(f"{what}: rate limited; retry after {retry}s")
    if isinstance(exc, anthropic.APIStatusError):
        return SystemExit(f"{what}: API error {exc.status_code}: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):
        return SystemExit(f"{what}: network error reaching the API")
    return SystemExit(f"{what}: {exc.__class__.__name__}: {exc}")


def review_pdf(pdf: Path, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
               client=None, with_images: bool = False) -> dict:
    """Send one listing to Claude; return the validated review dict. Text-only unless
    with_images (see build_messages)."""
    import anthropic

    client = client or _client()
    thinking, output_config = request_tuning(model, effort)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=load_prompt(),
            thinking=thinking,                      # off unless set explicitly on Opus 4.8
            output_config=output_config,
            messages=build_messages(pdf, with_images),
        )
    except anthropic.APIError as exc:
        raise _friendly(exc, model, pdf.name)

    if response.stop_reason == "refusal":
        raise SystemExit(f"model declined this listing (stop_reason=refusal) — {pdf.name}")
    if response.stop_reason == "max_tokens":
        raise SystemExit(f"hit max_tokens on {pdf.name}; raise MAX_TOKENS")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise SystemExit(f"no text block in the response for {pdf.name}")
    import datetime
    out = validate(json.loads(text))
    usage_in, usage_out = response.usage.input_tokens, response.usage.output_tokens
    # Everything needed to audit or reproduce this review later.
    out.update({
        "listing_id": pdf.stem,
        "model": response.model,
        "effort": effort,
        "source": "pdf+images" if with_images else "text",
        "prompt_sha": prompt_sha(),
        "reviewed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "usage": {"input": usage_in, "output": usage_out,
                  "usd": round(usage_in / 1e6 * prices(response.model)[0]
                               + usage_out / 1e6 * prices(response.model)[1], 4)},
        "raw_text": text,
    })
    return out


def estimate(pdf: Path, model: str = DEFAULT_MODEL, client=None,
             with_images: bool = False) -> tuple[int, float]:
    """(input tokens, $ estimate) for one listing. With images the page renders dominate, so
    always check before a bulk run."""
    import anthropic
    client = client or _client()
    try:
        counted = client.messages.count_tokens(model=model, system=load_prompt(),
                                               messages=build_messages(pdf, with_images))
    except anthropic.APIError as exc:
        raise _friendly(exc, model, f"{pdf.name} (estimate)")
    n = counted.input_tokens
    p_in, p_out = prices(model)
    # Output is small (a few findings); assume ~1200 tokens for the estimate.
    return n, n / 1e6 * p_in + 1200 / 1e6 * p_out


def _targets(store: dict, ids: list[str] | None, force: bool) -> list[tuple[str, Path]]:
    """Cars still needing a review call, as (id, pdf).

    "Reviewed" means `reviewed_at` is set — NOT that `findings` is non-empty. An empty findings
    list is a correct, complete answer (the prompt says so outright: most listings disclose
    nothing), and testing the list's truthiness re-billed every clean car on every run, forever.
    A re-review is a paid API call, so the skip has to key on whether the call happened."""
    out = []
    for cid in (ids or list(store)):
        car = store.get(cid)
        if car is None:
            print(f"  {cid}: not in the store", file=sys.stderr)
            continue
        if car.get("reviewed_at") and not force:
            continue
        pdf = PROCESSED_DIR / f"{cid}.pdf"
        if not pdf.exists():
            print(f"  {cid}: no {pdf.name} to review", file=sys.stderr)
            continue
        out.append((cid, pdf))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Have Claude review listing PDFs into structured findings.")
    ap.add_argument("--pdf", help="review a single PDF and print the JSON (writes nothing)")
    ap.add_argument("--store", nargs="*", metavar="ID",
                    help="review cars in cars.json from processed/<id>.pdf (default: all missing findings)")
    ap.add_argument("--force", action="store_true", help="re-review cars that already have findings")
    ap.add_argument("--dry-run", action="store_true", help="count tokens + estimate cost, make no review call")
    ap.add_argument("--with-images", action="store_true",
                    help="send the whole PDF incl. every page image (~5x the cost); default is "
                         "the text layer only, which carries everything a review needs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args(argv)

    if args.pdf:
        pdf = Path(args.pdf)
        if args.dry_run:
            n, usd = estimate(pdf, args.model, with_images=args.with_images)
            print(f"{pdf.name}: ~{n:,} input tokens, ~${usd:.2f}")
            return 0
        print(json.dumps(review_pdf(pdf, args.model, args.effort, with_images=args.with_images),
                         ensure_ascii=False, indent=2))
        return 0

    if args.store is None:
        ap.error("pass --pdf <file> or --store [ids...]")

    store = carcache.load_store()
    targets = _targets(store, args.store or None, args.force)
    if not targets:
        print("nothing to review (use --force to re-review)")
        return 0

    if args.dry_run:
        total = 0.0
        client = _client()
        for cid, pdf in targets:
            n, usd = estimate(pdf, args.model, client, args.with_images)
            total += usd
            print(f"  {cid}  ~{n:>8,} tok  ~${usd:>5.2f}")
        mode = "PDF + page images" if args.with_images else "text layer only"
        print(f"\n{len(targets)} listing(s) · ~${total:.2f} total at {args.model} ({mode})\n"
              f"(no findings written — drop --dry-run to run for real)")
        return 0

    client = _client()
    done = 0
    for cid, pdf in targets:
        out = review_pdf(pdf, args.model, args.effort, client, args.with_images)
        save_review(cid, out)                       # full text + evidence + usage -> reviews/
        apply_to_record(store[cid], out)
        sev = {s: sum(1 for f in out["findings"] if f["severity"] == s) for s in SEVERITIES}
        print(f"  {cid}  {sev['major']} major · {sev['minor']} minor · {sev['note']} note"
              f"   ({out['usage']['input']:,} in / {out['usage']['output']:,} out"
              f" · ${out['usage']['usd']:.3f}) -> reviews/{cid}.json")
        done += 1
        carcache.save_store(store)  # save as we go; a mid-run failure keeps prior work

    print(f"\nreviewed {done} listing(s) with {args.model} -> cars.json "
          f"(run `carcache.py recompute` to refresh scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
