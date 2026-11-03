# Ranking model: what it is and how accurate it is

## What the model does

The visual part of the clip ranking comes from a small neural network that looks at a video
two frames per second and gives each moment an importance score between 0 and 1. A clip's
visual score is the average over its frames.

* Input: CLIP ViT-B/32 image embeddings (512 numbers per frame), 2 frames per second.
* Network: a 2-layer Transformer encoder (width 256, 4 heads, dropout 0.3) followed by a
  linear layer and a sigmoid. About 1.2 million parameters.
* Loss: mean squared error plus a Pearson correlation term, so it learns both the level and
  the ordering of importance.
* Optimiser: AdamW, learning rate 1e-4, weight decay 0.01, gradient clipping at 1.0.

The final clip score also mixes in an audio score (loudness and onset strength) and a dialogue
score (how close the clip's speech is to the overall topic of the video). Those two are simple
rules, not trained, and they are not part of the accuracy numbers below. You can change their
weight in the app.

## Training data

TVSum (Song et al., CVPR 2015): 50 YouTube videos, 2 to 10 minutes each, 10 categories, 5
videos per category. Every frame was rated from 1 to 5 by 20 different people. The training
target is the average of the 20 ratings, scaled to 0 to 1 within each video.

## How it was tested

* 5-fold cross-validation. The 50 videos are split into 5 groups of 10. Each group is tested
  once by a model that never saw it, so every video gets a fair held-out prediction.
* Inside each fold, 8 of the 40 training videos are held back to decide when to stop
  training. The test videos are never used for any decision.
* Everything is repeated with 3 random seeds and reported as mean plus or minus standard
  deviation.
* Metrics follow Otani et al. (CVPR 2019). Kendall τ and Spearman ρ measure how well the
  predicted ranking of moments agrees with each person's ranking, averaged over the 20
  people. F1 compares a 15% length summary built from the prediction with the summary built
  from each person's ratings.

## Results

| Model | Kendall τ | Spearman ρ | F1 at 15% |
|---|---:|---:|---:|
| Human (one person vs. another) | 0.177 | 0.204 | 0.493 |
| Random scores | 0.000 | 0.000 | 0.516 |
| Segment MLP (the project's original design) | 0.165 ± 0.004 | 0.207 ± 0.005 | 0.532 ± 0.002 |
| Frame MLP | 0.194 ± 0.003 | 0.254 ± 0.004 | 0.556 ± 0.002 |
| Transformer without position encoding | 0.195 ± 0.004 | 0.255 ± 0.005 | 0.526 ± 0.003 |
| Transformer, MSE loss only | 0.188 ± 0.003 | 0.247 ± 0.005 | 0.529 ± 0.001 |
| **Transformer (used in the app)** | **0.191 ± 0.003** | **0.250 ± 0.004** | **0.516 ± 0.002** |

What this means in plain words:

* **It agrees with people about as well as people agree with each other, slightly better.**
  Two random annotators reach τ = 0.177. The model reaches τ = 0.191 against them.
* The human row is computed by our own evaluation code and matches the published value
  (0.177 and 0.204), which is a check that the metric code is correct.
* The correlation is positive on 48 of the 50 held-out videos (median τ = 0.21).
* It is clearly better than the original design: a paired Wilcoxon test over the 50 videos
  gives p = 0.009.
* It is **not** clearly better than a per-frame MLP (p = 0.36). The temporal Transformer
  does not add measurable accuracy on TVSum. The gain over the original design comes from
  frame-level CLIP features and frame-level training.
* F1 does not separate the models from random scores. This is a known flaw of F1 on TVSum
  (Otani et al. 2019), so τ and ρ are the numbers to rely on.

### By category (Kendall τ, held-out)

| Category | τ |
|---|---:|
| Bike tricks | 0.291 |
| Grooming an animal | 0.280 |
| Changing a vehicle tire | 0.246 |
| Making a sandwich | 0.229 |
| Beekeeping | 0.210 |
| Getting a vehicle unstuck | 0.140 |
| Parade | 0.134 |
| Dog show | 0.125 |
| Parkour | 0.108 |
| Flash mob | 0.106 |

The model is strongest on instructional and activity videos where the important part looks
different from the rest, and weakest on crowd events (parades, flash mobs) where most frames
look alike.

## Released weights

`weights/importance_transformer.pt` is trained on all 50 videos for 6 epochs, the median of
the early-stopping epochs chosen during cross-validation. Its accuracy is described by the
cross-validation numbers above; there is no separate test set left after using all 50 videos.

## Limits

* TVSum is small (50 videos) and made of user-generated YouTube clips. Accuracy on films,
  lectures, sports broadcasts and so on has not been measured.
* The scene segmentation step has no ground truth in TVSum and is not covered by these
  numbers.
* The audio and dialogue scores are heuristics.

## Reproduce

```bash
python -m training.prepare_tvsum --tvsum_dir path/to/ydata-tvsum50-v1_1 --out data/tvsum_clip.h5
python -m training.train_importance --data data/tvsum_clip.h5
python -m training.export_results --data data/tvsum_clip.h5 --tvsum_dir path/to/ydata-tvsum50-v1_1
```

Raw numbers per video, fold and seed: `training/results/tvsum_cv_results.json`.
Training curves: `training/results/logs/`. Figures: `docs/figures/`.

## Sanity checks

`python -m training.sanity_checks` verifies that the five test folds are disjoint, cover every
video once, and never overlap with the training or early-stopping videos. It also repeats the
whole training with each video's labels shuffled in time. That control scores τ = 0.020 and
ρ = 0.027, essentially zero, so the real result (τ = 0.191) comes from the video content and not
from the evaluation setup. Results: `training/results/sanity_checks.json`.

## Do the audio and dialogue scores help?

`python -m training.eval_fusion` combines the held-out visual predictions with the audio and
dialogue scores on TVSum, with weights fixed in advance.

| Configuration | Kendall τ | Spearman ρ |
|---|---:|---:|
| Visual only | 0.187 | 0.244 |
| Audio only | 0.015 | 0.019 |
| Dialogue only (46 videos with speech) | -0.001 | 0.001 |
| Visual + audio (0.8 / 0.2) | 0.186 | 0.243 |
| Visual + dialogue (0.8 / 0.2) | 0.178 | 0.233 |
| All three, app default (0.6 / 0.15 / 0.25) | 0.168 | 0.220 |
| All three, equal | 0.127 | 0.167 |

On TVSum they do not help: audio and dialogue alone are close to random, and adding them lowers
agreement with the annotators slightly. TVSum is mostly visual, user-recorded footage, so this
does not settle whether they help on narration-heavy video, but it is the honest result on this
benchmark. Results: `training/results/fusion_eval.json`.
