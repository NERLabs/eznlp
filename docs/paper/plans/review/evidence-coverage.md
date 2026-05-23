# Evidence Coverage Review

## Coverage

- Abstract/main result: covered by `paper_result_registry.md`.
- Introduction application context: covered by user DOCX and local Agricultural Machinery Journal writing patterns.
- Literature synthesis style: covered by local converted NER papers; current manuscript references remain governed by reference validators.
- Materials corpus description: partially covered by user DOCX; current manuscript should absorb concrete corpus/preprocessing wording while avoiding unsupported table shifts.
- Method naming: covered by result registry and source consistency check.

## Risks

- User DOCX contains stale result `88.73` and model name `EDBS`; these must not overwrite the current verified main result and naming.
- User DOCX states channel attention as a main component; current submission evidence says not to include it as the main model unless experiments are changed.
- PDF-to-Markdown local papers include OCR artifacts, so they are used for writing pattern and section organization, not as authoritative quoted text.
