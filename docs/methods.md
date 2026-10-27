# Methods and interpretation

## Inputs, checkpoints and deterministic extraction

The fixed `images/manifest.json` records exact URLs, dimensions and SHA-256 hashes for 20 COCO val2017 photographs. Missing inputs are fetched and byte-verified. The original six visual indices and hero input remain unchanged. All scientific settings are in `config.yaml`; every model ID has a pinned revision, including `facebook/dinov2-giant` (`611a9d42f2335e0f921f1e313ad3c1b7178d206d`) and `facebook/dinov2-with-registers-giant` (`8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb`). Giant weights were verified against their official LFS SHA-256 digests during preparation.

Bicubic squash-resize produces 518×518 RGB pixels. ImageNet mean/std normalization, seed 42, float32, eval/inference mode, deterministic algorithms and TF32-off settings match the original runs. We request eager attention and all hidden states. Large/Base/Giant have 24/12/40 blocks. A single image is processed at a time on the A100 40 GB; peak torch-allocated GPU memory is recorded per model. Libraries remain pinned in `requirements.txt`. Reported peaks are torch-allocated bytes, not minimum device capacity; allocator reservation, driver contexts and other workspaces add overhead.

## Token layout and the two feature spaces

The sequence is `[CLS] + patches` or `[CLS] + 4 registers + patches`. Every spatial feature slice removes all prefixes and asserts exactly 1,369 patches. Prefix tokens are retained for the attention computation and recorded separately for norm analysis.

| Space label | Last-layer tensor | Intermediate-layer convention |
|---|---|---|
| `raw` | `hidden_states[-1]`, before the final output LayerNorm | Direct block/embedding hidden state |
| `normalized` | `last_hidden_state`, after the learned final LayerNorm | Apply the **same learned final LayerNorm** to each raw hidden state |

The normalized intermediate curves are an explicit diagnostic transformation, not the model's internal per-block pre-attention LayerNorm outputs. Every forward checks `model.layernorm(hidden_states[-1]) == last_hidden_state`. LayerNorm removes a token's channel mean/scale, then applies learned affine weights; equal final token norms are not guaranteed. Giant's raw high-norm patches become a normalized low-norm group after this operation, whereas Large retains a small direction-dependent normalized high-norm group. This does not establish the training mechanism.

## A — Last-layer norms and thresholds

All patch L2 norms are pooled across images. Counts, means, population std, maxima and quantiles are exported **in both spaces**. Histograms share bins/ranges between the model pair, with log y-axes. Raw and normalized color scales are not conflated.

The historical normalized reference is **68.0**. The additional raw reference is the pre-existing fixed diagnostic cutoff **150.0**, explicitly **not** a universal artifact definition. Fixed 60.0/80.0/150.0 and original 100.0/150.0/200.0 checks are saved for both spaces. Full floating-point reference and observed-gap values stay in config; CSV/JSON retain full precision. Observed-gap values document this calibration run and are re-estimated from the data; the reference fields control fixed cutoffs.

The gap search sorts each model/space's norms, considers percentile indices 90–99.99, leaves at least three samples above a candidate, and finds the largest adjacent multiplicative jump. A jump ≥1.2 is a clear gap under this exploratory rule. Its geometric midpoint is reported; otherwise an unused 256-bin Otsu-on-log-norm candidate is retained transparently. A single extreme Base raw patch does not form a population of at least three under this rule.

For Large normalized outputs, the observed gap is 55.7–83.0 and the cutoff is 68.0 (21 patches). For Giant raw outputs, the gap is 86.8–126.0 with midpoint 104.6 (442 patches); fixed 150.0 selects 439. No corresponding separated Giant register-model raw patch mode is detected. An absolute raw cutoff can count many ordinary register-model patches because model scales differ: this is why `above_reference_count` is not named an artifact count.

Median-relative sensitivity uses every configured multiple (1.25, 1.5 and 2), not only a selected successful comparison. A separate retained 1.5×-median table maintains continuity. Tail/gap definitions were exploratory, not preregistered.

The lower-tail search uses the same adjacent-ratio rule over configured percentiles 0.01–10, again requiring at least three samples. Giant normalized output has a lower gap from **27.9 to 50.9**, with split **37.7** and **443** low-norm tokens. All **442** raw high-gap patches are in this set. Their normalized median is **19.5**, compared with **58.6** for other raw patches. This is a **signature reversal**, not disappearance of the feature-map artifacts. `feature_space_transform.csv` records the paired overlap and conditional medians for all models.

## B — Layer tails and prefix-token distributions

At each stage, pool patch norms over the same 20 images **separately by model and feature space**. The cutoff is mean + k×population std, k=3 and `ddof=0`. Layer 0 is the embedding; subsequent indices denote complete block outputs. Register/CLS norms never influence patch thresholds.

`layer_statistics.csv` contains patch-tail statistics in both spaces. `prefix_token_norms.csv` records each image/layer/CLS/register norm. `layer_token_statistics.csv` places patch and individual prefix summaries next to each other. Each register-model prefix figure combines final token-norm histograms with per-layer patch median/10–90%/99th-percentile curves, CLS means, and **four separate register means**. Histograms show counts with different population sizes (27,380 patches versus 20 samples per prefix token), rather than implying equal sample sizes.

Final-layer summaries, where R1…R4 are sequence positions rather than fixed semantic roles:

| Register model | Space | Patch median | CLS mean | R1 mean | R2 mean | R3 mean | R4 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | raw | 140.1 | 125.0 | 96.3 | 222.6 | 99.4 | 313.1 |
| Base | normalized | 24.8 | 24.4 | 27.7 | 23.0 | 25.5 | 30.1 |
| Large | raw | 284.2 | 193.7 | 846.0 | 167.5 | 170.9 | 191.6 |
| Large | normalized | 23.5 | 23.0 | 20.0 | 23.6 | 23.0 | 21.7 |
| Giant | raw | 115.4 | 148.1 | 329.2 | 167.4 | 1346.3 | 97.0 |
| Giant | normalized | 52.6 | 50.4 | 75.2 | 52.3 | 49.7 | 53.8 |

The large raw register norms are expected to be kept separate from image patches. Their magnitude is consistent with registers offering non-spatial computation slots, but norm plots alone do not prove how the model uses them.

## Why Large has the higher raw register-model 3σ rate

The raw no-register model has mean/std **96.4/9.1** and a pooled cutoff **123.7**, selecting **36** patches across **10** images. The register model has mean/std **286.2/40.1** and cutoff **406.3**, selecting **216** patches entirely from image **19**, COCO 000000001761 (airplanes/bridge/sky).

1. **Not global scale alone.** For a positive global rescaling, the mean and std rescale together, so membership in a mean + 3σ tail is unchanged. Median rescaling was explicitly checked and leaves the counts identical.
2. **The pooled image mixture matters.** The sky image has raw mean **384.5**, std **24.9**, and its own cutoff **459.2**, above its maximum **455.7**. Its image-specific rule selects **zero**. A broad group of ordinary image-conditioned values is being selected by the global rule.
3. **Specific patches, not leaked prefix tokens.** The 216 selected grid locations occur in one image, not a fixed coordinate repeated across images. They are real patch tokens after explicit prefix removal. There are no normalized-reference outliers in that register model. The 36 raw no-register tail patches also have **zero overlap** with its 21 normalized-reference outliers.
4. **Not mainly channel-mean/DC energy.** On the register-model tail, only **0.039%** of squared raw norm on average comes from the per-token channel mean. Centered feature magnitude remains large. The code verifies the squared-norm decomposition.

Sensitivity to k, median multiples and a MAD-based rule is fully exported. Per-image counts, coordinates, and every selected token's raw/normalized norm are saved. This explains the statistical reversal; it is not a causal explanation of pretraining or proof that every statistical tail token is computationally repurposed.

## C — Spatial features, attention and PCA

Each of the same six inputs has raw and normalized norm/PCA figures for all three sizes. Within a figure, paired norm maps share a linear scale across both models and all six images. PCA uses three full-SVD components fitted independently per image/model, with per-channel min–max RGB scaling. Raw and normalized PCA colors cannot be interpreted as aligned semantic classes.

Final-layer CLS attention is averaged across heads, sliced to patch columns, and not renormalized after removing prefix columns. Attention weights are independent of applying a final output LayerNorm. Shared log probability color scales begin at 1e-6; values below that display bound are under-range colors, not altered saved probabilities.

The installed, commit-pinned FeatLens package computes cosine to the same center patch and full-token attention rollout. `src/attention.py` imports `featlens.methods.cosine_similarity_map` and `featlens.attention._rollout`. The latter is a private helper, so the dependency is pinned to an exact commit: the public `featlens.attention()` entry point captures timm models, whereas this experiment already has Hugging Face attention tensors. No FeatLens code is copied into this repository. For rollout, mean heads, add identity residual, row-normalize, and compose all layers; no attention edges are discarded. CLS/register transitions remain until the final CLS-to-patch slice. Only head averaging is moved to GPU first; the FeatLens mathematics is retained. The final feature-space label affects cosine features but not attention/rollout probabilities. Old normalized figure URLs are retained as explicitly labeled aliases.

## Outside-object statistic and image bootstrap

Official pycocotools decodes the union of COCO instance and crowd masks. A patch is outside objects when its center falls outside this union in original image coordinates. This complement is an imperfect background proxy and covers **68.4%** of all patch centers in the sample.

The statistic is the **pooled number of selected outside patches / pooled number of selected patches**. With seed 42, draw 1,000 index vectors, each sampling 20 entire images with replacement. For each draw, repeat the image's whole selected-token/outside-token counts and recompute the ratio. Report 2.5th/97.5th percentiles (95% interval). The same draw indices are used across models/spaces/rules; the unit is **image**, not independent patch.

| Selection | Observed outside fraction | 95% image-bootstrap interval |
|---|---:|---:|
| Large no registers, normalized reference >68.0 | 19/21 (90.5%) | 75.0–100.0% |
| Giant no registers, raw gap >104.6 | 352/442 (79.6%) | 65.1–92.1% |
| Giant no registers, normalized low gap <37.7 | 353/443 (79.7%) | 65.1–92.2% |
| Giant no registers, raw fixed >150.0 | 349/439 (79.5%) | 64.9–92.1% |

Location tables also include the detected **lower-gap** selection, with `selection_direction=below`; all ordinary high-tail/reference rules use `above`. Zero selected tokens for a given rule makes the fraction and interval **undefined**, not zero. Zero-denominator bootstrap draws are excluded from conditional percentile estimation and their number is recorded. In very sparse cases (e.g. a single observed Base raw patch), a degenerate conditional interval must not be interpreted as precise population evidence. Main Large/Giant rows have all 1,000 valid draws. The intervals describe sampling uncertainty within these 20 images and do not correct incomplete semantic labels or sample-selection bias.

## Reproduction and limitations

The default command recomputes all six models. `--reuse` only reuses arrays that contain the expanded prefix measurements; incompatible old arrays trigger extraction again. Exact settings, source hashes, versions/hardware, pinned revisions, and validation are saved per run. A full check asserts both feature-space feature/norm consistency, patch/prefix shapes, last-layer norm consistency, attention mass, raw centered/mean norm decomposition, bootstrap bounds, and PNG/PDF integrity.

Only 20 fixed images, three model sizes, one resolution, no retraining, and no seed/checkpoint sweep. Comparing different public checkpoints does not isolate every pretraining difference. No norm threshold, relative tail, attention plot, PCA map or background proxy by itself establishes a causal mechanism. All four original Base/Large final outputs are preserved numerically; new conclusions concern the newly measured Giant/raw/prefix behavior and the quantified uncertainty.

## FeatLens dependency migration

The published experiment originally used attributed copies of the same three FeatLens functions. The pip dependency replaces those copies at the identical upstream commit. The migration check compares function ASTs, both feature spaces for all 72 saved cosine maps (six models × six images × two spaces), and rollout outputs with and without registers. Existing result provenance remains the record of the original experiment; future runs also record the installed FeatLens version in `environment.json`. See [`results/data/featlens_dependency_validation.json`](../results/data/featlens_dependency_validation.json) for the checks. Run `python -m unittest discover -s tests -v` for the dependency integration checks.
