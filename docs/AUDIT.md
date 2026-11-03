# Audit of the previous version and what was changed

This file records the problems found in the earlier implementation (commit `b23dc72`) and
how each was resolved. It matters for any publication: **numbers, figures and timing tables
produced by the earlier code should not be reused.**

## Problems found

| # | Area | Problem | Effect |
|---|------|---------|--------|
| 1 | Shot detection | Shot times computed as `frame / 25`, regardless of the video's real frame rate. | On a 29.97 fps video every timestamp drifts later, by ~17% of elapsed time (≈60 s off after 6 minutes). This is the main reason extracted clips did not match the scenes. |
| 2 | Shot detection | TransNetV2 weights in the repository were Git-LFS pointer files (133 bytes), not weights. | A fresh clone could not run shot detection at all. |
| 3 | Scene segmentation | Described as "LGSS", but implemented as cosine similarity of the *dialogue text* of adjacent shots only. Shots without speech received a zero vector (similarity 0 → always a boundary); no visual information was used. | Scene boundaries depended almost entirely on Whisper output. |
| 4 | Scene merging | Any scene with fewer than 4 shots was merged into its neighbour, repeatedly. | Most videos collapsed into a handful of very long scenes (the committed sample: 8 shots → 1 scene). |
| 5 | Boundary refinement | Start snapped to the previous sentence start and end to the next sentence end independently. | Neighbouring scenes overlapped (e.g. scene 1 ended at 112.4 s, scene 2 started at 103.2 s). |
| 6 | Importance model | Trained on CLIP embeddings, but at inference fed the **first 512 raw pixel values** of the mean frame (`frames.mean(0).view(-1)[:512]`). | Visual importance scores were unrelated to the model's training; effectively noise. |
| 7 | Importance model | 80/20 `random_split` over 20 s segments, so segments of the same video appeared in both train and validation. No test set; no standard metric. | Reported validation numbers were optimistic and not comparable to the literature. |
| 8 | Training logs | `importance_scoring/training_history.json` contains hand-typed round values (0.0260, 0.0220, 0.0190, …). | Not a real training log. |
| 9 | Timing tables | `Processing_Time_Breakdown.txt` and the GPU/CPU table were hard-coded constants (and internally inconsistent: "linear" scaling but 20 min → 90 min CPU). | Not measured. |
| 10 | Text / audio scores | Text importance was dominated by `0.3 × word count`; the code also contained unused keyword lists. | Scenes with more words ranked higher regardless of content. |
| 11 | Repository | Custom scene code lived as untracked files inside the `SceneSeg` git submodule. | It was never pushed; clones were missing the pipeline's stages 3–7. |

## What the current version does instead

* **Shots**: TransNetV2 (official PyTorch weights, checksum-verified), frames decoded at the
  video's real constant frame rate; gradual transitions are cut at their centre so shots tile
  the video exactly. Verified frame-accurate by inspecting frames at ±1 frame of each cut.
* **Scenes**: training-free multimodal boundary scoring (CLIP visual contrast + Sentence-BERT
  dialogue contrast + speech-continuity prior), robust per-video normalisation, greedy
  selection with min/max scene length, then sentence-aware refinement that keeps scenes
  contiguous. See `svs/scenes.py`.
* **Importance**: CLIP features at 2 fps → temporal Transformer trained on TVSum with 5-fold
  cross-validation (every video tested exactly once, inner validation split for early
  stopping), evaluated with the standard rank-correlation protocol (Otani et al., CVPR 2019)
  and F1@15%, against human, random and ablation baselines (`training/`). All curves and
  tables are generated from real logs by `training/export_results.py`.
* **Summary**: shot-level knapsack maximising importance × time under a length budget.
* **Timing**: every run records measured per-stage wall-clock times in `result.json`.

## Removed

`SLIDES_DUMP/` (slide screenshots), `Visualization_of_steps/` and `analysis_graphs/` (figures
from the old pipeline / hard-coded data), `Processing_Time_Breakdown.txt`, the old
`preprocessing/`, `scene_segmentation/` (submodule), `refine_scenes/`, `synthesis/`,
`importance_scoring/` trees, root `main.py`/`server.py`/`index.html`/`style.css`, and
`copy_tvsum_segments.py`. Everything remains in git history; untracked local files were moved
to a backup folder outside the repository.

## Still to do before publication

* **Scene segmentation has no quantitative evaluation yet.** TVSum has no scene labels. To
  report scene-boundary accuracy, evaluate on a dataset with scene annotations (e.g. BBC
  Planet Earth, Baraldi et al. 2015; or MovieNet/MovieScenes) using AP / mIoU / F1 of
  boundaries.
* Audio and dialogue scores are interpretable heuristics, not learned; the paper should
  describe them as such, or evaluate them (e.g. on a dataset with narration such as
  QFVS or a user study).
