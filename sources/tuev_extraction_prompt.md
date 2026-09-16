# ChatGPT extraction prompt for the AUTO BILD TÜV-Report 2026

Paste everything below the line into ChatGPT, then attach report pages in batches
(3–6 pages per message works best). For follow-up batches just write: "Same task,
next batch." Save each JSON reply to a file (e.g. `tuev_batch_01.json`) and hand the
files to carcache/Claude for ingestion into `model_reliability_basis`.

---

You are a meticulous data-extraction assistant. I will attach scanned pages from the
German used-car inspection report **"AUTO BILD TÜV-Report 2026"** in batches. Your job
is to transcribe them into structured JSON, completely and faithfully. This data feeds
an automated car-scoring pipeline, so **accuracy beats completeness beats brevity** —
a wrong number is worse than a null.

## Ground rules

1. **Transcribe, never infer.** Copy every number exactly as printed. Convert German
   decimal commas to points (5,4 → 5.4). If a value is unreadable or cut off, use
   `null` and explain in `notes`. Never estimate, average, or fill gaps from your own
   knowledge of cars.
2. **German text stays verbatim.** Do not translate. Keep umlauts. Transcribe the
   page's full running text, including headlines, sub-headlines, photo captions, and
   margin notes.
3. **One JSON object per model.** If a page covers two models, output two objects.
4. **Output raw JSON only** — no markdown fences, no commentary before or after.
5. **End every reply with a `coverage` object** listing which attached pages you
   processed and anything you skipped or could not read.

## Page types (detect automatically)

**A. Model page** — one car model with its photo, editorial text, and a data block of
TÜV inspection results by vehicle age class.

**B. Statistics table** — a large grid: rows = models, columns = age classes
(2–3, 4–5, 6–7, 8–9, 10–11, 12–13 Jahre), cells = defect rate in % ("erhebliche
Mängel") and usually a rank ("Platz"). Extract **every row**, exactly as printed.

**C. Table of contents / editorial pages** — transcribe the structure (sections, page
numbers, vehicle classes) or the full text respectively.

## Output schema

```json
{
  "source": {
    "publication": "AUTO BILD TÜV-Report 2026",
    "pages": [87],
    "content_type": "model_page | statistics_table | toc | editorial"
  },
  "models": [
    {
      "make": "Opel",
      "model": "Astra",
      "generation_note": "build years / generation exactly as the page states them, else null",
      "vehicle_class": "Kompaktklasse",
      "defect_rates": {
        "2-3":   {"pct": null, "rank": null, "of": null, "avg_km_tkm": null},
        "4-5":   {"pct": null, "rank": null, "of": null, "avg_km_tkm": null},
        "6-7":   {"pct": null, "rank": null, "of": null, "avg_km_tkm": null},
        "8-9":   {"pct": null, "rank": null, "of": null, "avg_km_tkm": null},
        "10-11": {"pct": null, "rank": null, "of": null, "avg_km_tkm": null},
        "12-13": {"pct": null, "rank": null, "of": null, "avg_km_tkm": null}
      },
      "strengths": ["verbatim German bullet", "..."],
      "weaknesses": ["verbatim German bullet", "..."],
      "typical_defects": [
        {"area": "e.g. Achsaufhängung, Beleuchtung, Bremse, Ölverlust — as printed",
         "detail": "verbatim German detail if given",
         "age_classes": "which ages it concerns, if stated",
         "vs_average": "above | below | average | null"}
      ],
      "prose_verbatim": "The COMPLETE editorial text of the page in German: headline, body text, photo captions, boxes. Preserve paragraph breaks with \\n\\n.",
      "notes": "anything ambiguous, unreadable, or unusual about this page; null if none"
    }
  ],
  "table_rows": [
    {"make": "...", "model": "...", "vehicle_class": "...",
     "rates_pct": {"2-3": null, "4-5": null, "6-7": null, "8-9": null, "10-11": null, "12-13": null},
     "ranks":     {"2-3": null, "4-5": null, "6-7": null, "8-9": null, "10-11": null, "12-13": null}}
  ],
  "toc": [
    {"section": "...", "vehicle_class": "...", "page": null}
  ],
  "coverage": {
    "pages_processed": [],
    "pages_skipped_or_unreadable": [],
    "warnings": []
  }
}
```

Omit `table_rows` on model pages, omit `models` on pure table pages, omit `toc`
except on contents pages. In `defect_rates`, include **only the age classes the page
actually shows** — do not pad with nulls for classes the page omits, and add classes
if the report prints ones not listed here.

The defect rate ("Mängelquote") means the share of vehicles with **erhebliche
Mängel** at the Hauptuntersuchung. If the page distinguishes further categories
(e.g. "verkehrsunsicher", "geringe Mängel"), capture them in `notes` or as extra
keys — never mix them into `pct`.

## Priority models — double-check these especially carefully

Volkswagen Golf, Volkswagen Passat, Skoda Octavia, Skoda Superb, Seat Leon,
Hyundai i30, Hyundai i40, Kia Optima, Kia Ceed, Opel Astra, Opel Insignia,
Ford Focus, Ford Mondeo, Renault Mégane, Renault Talisman.

Extract every model you see (the full dataset is wanted), but if one of the priority
models appears on a page, re-read its numbers twice before writing them.
