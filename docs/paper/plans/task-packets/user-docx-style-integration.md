# Task Packet

## Scope

Integrate usable content from the user-provided DOCX and local PDF-to-Markdown papers into the current Agricultural Machinery Journal manuscript, focusing on Abstract, Introduction, and Materials/Methods.

## Files to read

- `融合专家词典与边界选择的红枣栽培命名实体识别方法.docx`
- `docs/paper/user_draft_融合专家词典与边界选择的红枣栽培NER_抽取稿.md`
- `docs/paper/农业机械学报_红枣NER_投稿稿.md`
- `docs/paper/paper_result_registry.md`
- `_2DATA/papers/李春春 等 - 2025 - 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法/auto/李春春 等 - 2025 - 基于BERT-BiLSTM-CRF的茶树病虫害命名实体识别方法.md`
- `_2DATA/papers/路阳 等 - 2026 - 基于曼哈顿注意力机制的水稻病虫害命名实体识别/auto/路阳 等 - 2026 - 基于曼哈顿注意力机制的水稻病虫害命名实体识别.md`
- `_2DATA/papers/蒲攀 等 - 2024 - 融合动态词典特征和CBAM的苹果病虫害命名实体识别方法/auto/蒲攀 等 - 2024 - 融合动态词典特征和CBAM的苹果病虫害命名实体识别方法.md`
- `_2DATA/papers/吴钊 等 - 2025 - 基于多特征融合的苹果栽培命名实体识别/auto/吴钊 等 - 2025 - 基于多特征融合的苹果栽培命名实体识别.md`

## Files allowed to edit

- `docs/paper/农业机械学报_红枣NER_投稿稿.md`
- `docs/paper/农业机械学报同类NER论文写法对照.md`
- `docs/paper/research_writing_skill_audit_2026-05-23.md`
- `docs/paper/submission_readiness_audit.md`
- Derived DOCX/PDF/submission package files after regeneration.

## Required skills

paper-orchestration, writing-core, evidence-driven-writing, verification.

## Evidence/data inputs

Use `paper_result_registry.md` for all performance numbers. Use user DOCX and local papers for structure and wording patterns only where they do not conflict with the registry or source consistency evidence.

## Required artifacts

- Updated manuscript with stronger corpus/preprocessing and problem-chain wording.
- Updated audit noting use of user DOCX and local papers.
- Regenerated DOCX/PDF/submission package.
- Passing validation reports.

## Rejection checks

- No `EDBS` in final manuscript.
- No `88.73` as main result.
- No claim that channel attention is part of the submitted main model.
- No new table numbering unless all validators and captions are updated.
- No copied OCR artifacts from local converted papers.

## Validation commands

```bash
python3 docs/paper/tools/run_all_submission_checks.py --report docs/paper/full_submission_check_report_2026-05-23.md
python3 docs/paper/tools/validate_submission_package.py docs/paper/submission_package --report docs/paper/submission_package/evidence/package_validation_draft_2026-05-23.md
unzip -t docs/paper/农业机械学报_红枣NER_投稿交接包_2026-05-23.zip
```
