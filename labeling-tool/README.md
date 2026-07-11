# Labeling tool (versioned snapshot)

The **working copy** lives at the sibling path `PadelLabs/labeling-tool/index.html`
(next to `videos/<Player>/*.MOV`, which are never committed). This folder is the
version-controlled snapshot so the tool's history travels with the ML repo.

After editing the working copy, sync + commit:

    cp ../labeling-tool/index.html labeling-tool/index.html

Current feature set: session/video pairing, timeline + markers, stroke
validation with top-3 confidence, and the active-learning **auto-accept**
triage (conf >= 0.85 AND top1-top2 margin >= 0.30, tunable; `review_mode`
provenance column in the exported `_validation.csv`).
