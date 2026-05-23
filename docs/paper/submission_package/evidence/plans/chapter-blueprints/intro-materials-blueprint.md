# Introduction And Materials Blueprint

## Paragraph 1

- Role: Red jujube application context.
- Main claim: Red jujube cultivation knowledge is dispersed in technical texts, and structured entity extraction supports knowledge graph, retrieval and QA.
- Evidence IDs: U-DOCX, AMJ-TEA, CSAE-APPLE.
- Contrast or transition: Move from crop knowledge demand to NER.
- Forbidden content: Broad claims about national strategy without citation or data.

## Paragraph 2

- Role: Existing NER route synthesis.
- Main claim: Agricultural NER studies commonly improve contextual representation, data construction, and external feature fusion, but red jujube entities remain domain-specific and long-tailed.
- Evidence IDs: AMJ-TEA, AMJ-RICE, AMJ-APPLE-CBAM, CSAE-APPLE.
- Contrast or transition: Move from general routes to shortcomings.
- Forbidden content: Paper-by-paper list without synthesis.

## Paragraph 3

- Role: Bottleneck cascade.
- Main claim: Red jujube NER faces three coupled bottlenecks: domain term representation, boundary-level global decision, and class imbalance.
- Evidence IDs: U-DOCX, REG, CODE.
- Contrast or transition: Leads directly to EDBP design.
- Forbidden content: Claiming external manual expert knowledge base or channel attention as the submitted model.

## Materials 1.1

- Role: Corpus and annotation protocol.
- Main claim: RJND was built from red jujube cultivation books and related technical texts through text extraction, correction, sentence segmentation, BMES annotation, manual checking, and 8:1:1 split.
- Evidence IDs: U-DOCX, CSAE-APPLE, REG.
- Contrast or transition: Leads to entity category table.
- Forbidden content: Adding unsupported corpus counts or shifting table numbering.

## Method 1.2-1.4

- Role: Model architecture and naming alignment.
- Main claim: EDBP combines MacBERT contextual representation, BMES expert dictionary features, and a boundary prediction decoder implemented as boundary-selection/span classification.
- Evidence IDs: U-DOCX, CODE, REG.
- Contrast or transition: Follow with formulas and symbol explanations.
- Forbidden content: EDBS/EDBP mixed naming, BiLSTM as main model if not in final configuration.
