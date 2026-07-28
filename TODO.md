# Plan: Improve handling of poorly-written / grammatically-incorrect text

## Steps

- [x] Step 0: Create TODO.md with plan tracking
- [x] Step 1: Add `normalize_bad_grammar()` helper function
- [x] Step 2: Relax `is_low_quality_sentence()` — removed alphabetic ratio check, increased navigation noise thresholds (≥4/0.30), added length-based override (≥12 words, ≥8 meaningful → auto-pass)
- [x] Step 3: Fix `has_summary_quality_issue()` — raised min source word threshold to 500, lowered min summary ratio to 0.25, restricted repeated word noise check to short summaries only, added ≥2 sentence guard for duplicate check
- [x] Step 4: Relax `polish_summary_sentences()` — kept existing logic as-is (it's already loose enough for BART-generated summaries)
- [x] Step 5: Improve `generate_summary()` — added `normalize_bad_grammar()` pre-processing, added relaxed text retry path before extractive fallback
- [x] Step 6: Ready to test — run `streamlit run app.py`
- [x] ✅ All implementations complete — code is ready for testing

