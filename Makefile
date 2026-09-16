# Convenience wrapper. Uses the repo-local .venv if present (needed for extract.py and
# tests); carcache.py itself is stdlib-only and runs under plain python3 too.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: help setup test intake extract-inbox reprocess verify status recompute rank report list clean

help:
	@echo "carcache targets:"
	@echo "  make setup                         create .venv, install pypdf + pytest"
	@echo "  make test                          run the test suite"
	@echo "  make intake                        scan inbox/ (dedupe PDFs by id)"
	@echo "  make extract-inbox                 text-extract ALL new inbox PDFs -> <id>.json"
	@echo "  make extract PDF=inbox/<id>.pdf OUT=<id>.json   one PDF -> record skeleton"
	@echo "  make status                        show extract/scoring drift of stored cars"
	@echo "  make recompute                     refresh stored score snapshots"
	@echo "  make reprocess [ALL=1]             rebuild stale (or all) records from processed/ PDFs"
	@echo "  make verify                        audit stored records against their archived PDFs"
	@echo "  make rank                          rank all cached cars"
	@echo "  make report                        write report.md"
	@echo "  make list                          list cached cars"
	@echo "  make clean                         remove __pycache__ / .pytest_cache"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q

intake:
	$(PY) carcache.py intake

extract:
	$(PY) extract.py $(PDF) $(if $(OUT),--json $(OUT))

extract-inbox:
	$(PY) extract.py --inbox

status:
	$(PY) carcache.py status

recompute:
	$(PY) carcache.py recompute

reprocess:
	$(PY) extract.py --reprocess $(if $(ALL),--all)

verify:
	$(PY) extract.py --verify

rank:
	$(PY) carcache.py rank

report:
	$(PY) carcache.py report

list:
	$(PY) carcache.py list

clean:
	rm -rf __pycache__ .pytest_cache
