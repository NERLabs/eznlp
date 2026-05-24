# Progress

## Stage

S1/S2/S5 mixed: evidence-driven revision of Introduction and Materials/Methods, followed by submission verification.

## 2026-05-23 User DOCX Revision Pass

- Located user draft: `融合专家词典与边界选择的红枣栽培命名实体识别方法.docx`.
- Extracted draft text to `docs/paper/user_draft_融合专家词典与边界选择的红枣栽培NER_抽取稿.md`.
- Confirmed user draft contains usable abstract/introduction/materials prose, but also contains stale/conflicting items: `EDBS`, channel attention as main component, and old or single-run result values.
- Local paper pool inspected: target Agricultural Machinery Journal NER papers and the apple cultivation NER paper converted from PDF to Markdown.

### Capability-use audit

- Required skills: paper-orchestration, writing-core, evidence-driven-writing, verification.
- Skills actually used: paper-orchestration, writing-core, evidence-driven-writing, verification.
- Inputs consumed: current manuscript, user DOCX extracted text, local Markdown papers under `datasets/raw/papers/`, result registry.
- Inputs not used and why: no sub-agent output; current available delegation tool requires explicit user request for sub-agents.
- Artifacts produced: project overview, outline, chapter architecture, progress log, evidence map, section blueprint, task packet.
- Verification run: pending after manuscript revision.
- Remaining risk: true author/front-matter information and Word/WPS visual final check still require user-side input.

## 2026-05-23 Peer Review Closure

- Round 2 review created: `docs/paper/peer_review_round2.md`.
- Round 2 revision log created: `docs/paper/revision_log_round2.md`.
- Round 3 re-review created: `docs/paper/peer_review_round3.md`.
- Round 3 revision log created: `docs/paper/revision_log_round3.md`.
- Main text changes from Round 2: removed process-style wording, replaced unsupported significance wording, acknowledged ResumeNER negative transfer result, and added annotation-agreement limitation.
- Round 3 experiment sufficiency pass created `experiment_sufficiency_analysis_2026-05-23.md` plus group/seed CSV files.
- Main text was updated with paired t-test evidence after raw three-seed results were located for `G_bilstm_baseline`, `CRF_nodict_bertwwm`, `CRF_nodict`, and `Q_bs_focal`.
- Re-review judgment: manuscript text is at "minor revision before submission" level; remaining blockers are real front-matter data and Word/WPS final display inspection.
