# AI collaboration

> **This file must be YOUR record.** The brief grades how *you* steered the AI. Below is a draft structure
> using the real design choices in this repo. For each decision, replace the evidence with **real excerpts
> from your own chat** (copy-paste them verbatim) and rewrite "Why" in your own words so you can defend it live.
> Export your full chat history into `chat_history/` (export files or share links), starting at your first prompt.
> Do not paste invented dialogue. Delete decisions you did not actually make or change.

## Decision 1: how the app is used
AI suggested: _(e.g. command-line script)_
I decided: a single local web page: pick folder, drop photo, see names
Why: office staff are not technical; the brief says UI is scored
Evidence: chat excerpt #1

## Decision 2: no training, use pre-trained embeddings + nearest neighbour
AI suggested: _(e.g. fine-tune a classifier / use a cloud API)_
I decided: pre-trained 128-d face embeddings, compare by distance
Why: 1 photo per person, unseen people at demo, offline rule
Evidence: chat excerpt #2

## Decision 3: library choice (dlib via face_recognition)
AI suggested: _(e.g. InsightFace / DeepFace)_
I decided: dlib-bin + face_recognition_models
Why: models bundled in pip, no compiler, no runtime download. See RESEARCH.md
Evidence: _paste test/install attempt_

## Decision 4: handling "unknown"
AI suggested: _(e.g. library default 0.6)_
I decided: 0.50 plus a live strictness slider
Why: demo contains people not in the folder; wrong names are worse than "Unknown"
Evidence: _your threshold experiments_

## Decision 5: one person can only be matched once per photo
Why: prevents two faces both labelled with the same name
Evidence: `tests/test_pipeline.py`

## Decision 6: fix photo rotation (EXIF) and report bad folder files
Why: phone photos are often stored sideways; silent failures would hurt the live demo

## Decision 7: UI choices (dashed orange = unknown, solid teal = known, list synced with photo, save/CSV export)
Why: colour is not the only cue; staff can verify each face visually

## Chat excerpt #1
> me: _paste_
> AI: _paste_

## Chat excerpt #2
> me: _paste_
> AI: _paste_

## Chat excerpt #3
> me: _paste_
> AI: _paste_
