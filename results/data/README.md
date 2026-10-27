# Result data and feature spaces

`python scripts/run_all.py` regenerates the study for all six models and both spaces:

- **raw:** residual-stream block output before final LayerNorm.
- **normalized:** after the model's learned final LayerNorm. For intermediate stages this same final normalization is explicitly applied to each raw hidden state.

Every norm table has a `feature_space` column or explicitly named raw/normalized columns. Attention probability tables use `attention_probabilities`; image metadata and masks are feature-space independent.

| Table | Meaning |
|---|---|
| `summary.csv` | Final patch mean/std/max/quantiles, own 3σ tails, reference exceedances, and observed upper/lower gap populations in both spaces |
| `thresholds.json` | Per-size/per-space baseline gap diagnostics and separately labeled reference thresholds |
| `threshold_sensitivity.csv`, `original_threshold_diagnostic.csv` | All configured fixed cutoffs in both spaces |
| `scale_adjusted_tail.csv`, `scale_adjusted_sensitivity.csv` | Own-median-relative rules, including all configured multiples |
| `layer_statistics.csv` | Patch-only, per-model/per-layer/per-space tails |
| `layer_token_statistics.csv` | Patch, CLS and individual register summaries next to one another, per layer and space |
| `prefix_token_norms.csv` | Every image/layer/CLS/register norm, explicitly labeled by space |
| `per_image.csv`, `*_patch_norms.csv` | Per-image statistics and both final patch norm columns/reference flags |
| `*_attention_mass.csv` | Original CLS/register/patch attention probability mass, independent of final feature normalization |
| `redundancy.csv` | Neighboring input-embedding cosine grouped by the final reference event in each space |
| `feature_space_transform.csv` | Overlap of raw high-gap and normalized low-gap patches, with conditional normalized medians |
| `large_tail_investigation.json`, `large_tail_*.csv` | Rule/scale sensitivity, driving images, coordinates and every pooled raw-tail token |
| `background_proxy_summary.csv`, `background_proxy_per_image.csv` | Outside-object fractions for references, 3σ rules, observed gaps and fixed cutoffs in both spaces |
| `bootstrap_outside_fraction.csv`, `bootstrap_settings.json` | Seeded image-bootstrap samples/settings and explicit undefined/valid-draw accounting |
| `models.json`, `environment.json`, `run_provenance.json`, `validation.json` | Pinned revisions, versions/hardware and peak torch GPU memory, settings/source hashes, and domain checks |
| `featlens_dependency_validation.json` | One-time dependency migration audit: exact upstream commit, unchanged function ASTs, 72 saved cosine maps, and full-grid rollout parity with/without registers |
| `image_sources.json`, `image_attribution.csv` | Exact input hashes, dimensions, source URLs and licenses |

References are **68.0 normalized** and **150.0 raw**. The raw reference is a scale diagnostic, not a universal artifact definition. Machine-readable values retain full precision. The observed Giant raw gap is **104.6**, with the full value also recorded in config. See `../../docs/methods.md`.

Large uncommitted NPZ arrays are regenerated locally:

| Key | Shape / meaning |
|---|---|
| `layer_norms` | `[20, blocks+1, 1369]`, **raw** patch norms (stage 0 embedding) |
| `layer_normalized_norms` | Same shape, after applying the learned final LayerNorm at each stage |
| `final_norms` | `[20,1369]`, final **raw** patch norms |
| `normalized_norms` | `[20,1369]`, final **normalized** patch norms |
| `prefix_norms_raw`, `prefix_norms_normalized` | `[20,blocks+1,1+registers]`; CLS first, then four separate registers where present |
| `features`, `raw_features` | `[6,1369,hidden_size]`, final normalized/raw patch features |
| `pca_rgb`, `raw_pca_rgb` and variance-ratio counterparts | Corresponding independently fitted PCA previews |
| `cosine`, `raw_cosine` | Corresponding FeatLens cosine-to-center maps |
| `raw_feature_mean`, `raw_centered_norms` | Final raw channel means and centered L2 norms for the squared-norm decomposition |
| `embedding_neighbor_cosine` | `[20,37,37]`, mean input-embedding cosine to available 2–4 neighbors |
| `selected_indices` | Six image indices in manifest order |
| `cls_attention`, `rollout`, `rollout_mass` | Feature-space-independent probability maps and prefix/spatial mass |

`background_proxy_masks.npz` stores patch-center complements of COCO object masks. `bootstrap_image_indices.npz` stores the shared seeded 1000×20 image draws; they can also be regenerated from config.

Photos and image-containing figures retain their original terms; see `../../images/SOURCES.md`. The original four-model environment audit is preserved as `../../docs/validation_v0.1.json`; current checks concern six models and both spaces.
