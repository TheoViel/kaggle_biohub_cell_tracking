# BioHub Cell Tracking During Development

**Kaggle slug:** `biohub-cell-tracking-during-development`  
**Deadline:** 2026-09-29 | **Prize:** $60,000 | **Teams:** ~1582

## Task

Detect cells, link them across frames, and identify division events in 3D+time fluorescence microscopy of zebrafish embryos. The output is a tracking graph of nodes (cell detections) and edges (temporal links + divisions).

## Evaluation

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

- **Edge Jaccard:** Predicted nodes are matched to GT nodes per timepoint via optimal bipartite assignment on scaled centroid distance (max 7.0 µm; physical scale z=1.625, y=x=0.40625 µm/voxel). A predicted edge is TP when both endpoints match GT nodes connected by a GT edge. TP / (TP + FP + FN), adjusted by a penalty for over-predicting total node count.
- **Division Jaccard:** A division is a node with 2+ outgoing edges. For each GT division, the predicted graph must cover the pre-split stage and touch both daughter lineages. Micro-averaged across all samples.
- Scores can exceed 1.0 due to sparse GT labeling.

## Data Format

### Images
- Format: `.zarr` v3 directories, single array at `0/` with shape `(T, Z, Y, X)` — typically `(100, 64, 256, 256)` in `uint16`
- Chunked one timepoint at a time: `(1, 64, 256, 256)`, compressed with blosc/zstd
- Physical voxel scale: **z=1.625 µm, y=0.40625 µm, x=0.40625 µm**

### Ground Truth (train only)
- Format: `.geff` directories (graph exchange format, also Zarr v3)
- `nodes/ids` — node ID array
- `nodes/props/{t,z,y,x}/values` — integer centroid coordinates in voxels
- `edges/ids` — shape `(N, 2)`, columns `(source_id, target_id)`
- Annotations are **sparse** — not every cell in every frame is labeled
- `estimated_number_of_nodes` in `.geff` metadata gives an estimate of true cell count

### File Structure
```
train/   — paired .zarr (image) + .geff (tracking graph) per sample
test/    — .zarr only; hidden test set swapped in at rerun time (~same size as train)
sample_submission.csv
```

### Embryo Identity
Folder names: `{embryo_id}_{field_of_view}` (e.g., `44b6_0049_0438_1330_1273`).  
**Train and test sets are embryo-disjoint.**

## Submission Format

CSV with two row types:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,node,2,1,33,130,125,-1,-1
2,6bba,edge,-1,-1,-1,-1,-1,1,2
```

- **node rows:** `node_id` + integer centroid voxel coords `(t, z, y, x)`; set `source_id`/`target_id` to `-1`
- **edge rows:** `source_id` + `target_id` referencing node IDs; set `node_id`, `t`, `z`, `y`, `x` to `-1`
- Every dataset in the test set must appear in the submission
- `id` is a consecutive throwaway index

## Metric Details

*Source: [royerlab/kaggle-cell-tracking-competition/metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)*

### Edge Jaccard

Three-step process:

1. **Node matching** — Predicted nodes are paired to GT nodes by centroid distance (max 7 µm threshold) using optimal bipartite assignment (one-to-one).
2. **Edge matching** — A predicted edge is TP when both endpoints match GT nodes connected by a GT edge. FP when an endpoint matches a GT node connected to a *different* counterpart. FN for unmatched GT edges.
3. **Calculation** — `Edge Jaccard = TP / (TP + FP + FN)`

#### Adjusted Edge Jaccard

Penalizes over-predicting nodes:

```
adjusted_jaccard = max(0, jaccard · (1 − a · (T_pred − T_true) / T_true))
```

- `T_pred` = predicted node count
- `T_true` = estimated true node count (from `.geff` metadata)
- `a = 0.1` (penalty weight)

### Division Jaccard

A **cell division** is a node with exactly two outgoing edges. Evaluation uses a local window of: `grandparent → dividing parent → children → grandchildren`, allowing predicted forks within ±1 timepoint of the actual split.

**True Positive** — A predicted fork succeeds when:
1. A predicted node matches the dividing parent or its predecessor (local parent anchor)
2. Each GT daughter lineage matches a *different* predicted branch (bipartite)
3. The parent is the fork or its immediate predecessor (directed local topology)
4. Child branches reference consistent GT connected components
5. Direct children maintain single-parent relationships (unmerged branches)

**False Positive** — A predicted fork is FP when it matches a GT node area but fails topology requirements, child branches reference different GT components, branches merge, or it goes unmatched after bipartite pairing. Predicted forks with no local GT evidence at all are ignored.

### Aggregation

Results are **micro-averaged**: per-sample TP, FP, and FN counts are summed across the full split *before* Jaccard is computed (so larger samples contribute proportionally).

- **Adjusted edge Jaccard** — weight-averaged by sample size `w_i = TP_i + FP_i + FN_i`
- **Division Jaccard** — computed from summed division metrics across all videos

**Final score:**

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```
