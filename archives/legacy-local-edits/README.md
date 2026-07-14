# Preserved Legacy Local Edits

This folder preserves the only unique uncommitted change found in the old
`github-ready` checkout before that checkout was retired on 2026-07-14.

The old `.gitignore` was byte-for-byte identical to current `main`. The old
modified `Tools/github_colab_sync.py` was already preserved by commit `d914917`
(`Store Week 5 optimizer resume state as release assets`) and was later
hardened on `main`.

The remaining Week 5 notebook edit was not committed elsewhere. It is preserved
as [week5-resumable-release-assets.patch](week5-resumable-release-assets.patch),
which applies to blob `47f4ce9` of
`Week 5/notebooks/week5_retain_regularized_unlearning_resumable.ipynb`.

The patch is retained for provenance only. Its direct upload/download behavior
was superseded by the centralized release-asset handling in
`Tools/github_colab_sync.py`; it is not applied to the current notebook.
