# Multi-Agent Motion Forecasting on Argoverse 2

The first milestone is to understand the scenario data and verify the boundary
between observed motion (input) and future motion (target). No machine learning
model is trained in this step.

## Task

- **Input:** past motion of the focal agent and surrounding actors
- **Output:** future 2D trajectory of the focal agent
- **Dataset:** Argoverse 2 Motion Forecasting
- **Future metrics:** ADE (mean displacement error over the future path) and
  FDE (displacement error at the final predicted point)

Each scenario contains actor tracks in map coordinates. The `observed` flag in
the data marks each state as past input or future ground truth; the split is not
hard-coded to a presumed timestep.

## Setup

Python 3.10 or newer is required.

```powershell
cd C:\Dev\motion-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Data

The `data/` directory is intentionally excluded from Git. The working copy has
500 training scenarios under `data/train/` and the 500-scenario validation
subset under `data/val/`; the first validation scenario also has its local map
archive. The validation IDs are recorded in `splits/val_scenario_ids.txt` so
experiments can use the same scenarios. The downloader sorts all scenario IDs,
takes the first N, and downloads only `scenario_<id>.parquet` files:

```bash
python scripts/download_av2_subset.py \
  --split train \
  --num-scenarios 20000 \
  --output data/train

python scripts/download_av2_subset.py \
  --split val \
  --num-scenarios 500 \
  --output data/val
```

For map-aware experiments, install the optional AV2 map API and include the
vector-map JSON archives while downloading:

```powershell
python -m pip install -e ".[maps,train]"
python scripts/download_av2_subset.py --split train --num-scenarios 20000 --output data/train_20k --include-maps
python scripts/download_av2_subset.py --split val --num-scenarios 500 --output data/val --include-maps
```

## Inspect and visualize

From the project root, the scripts use the first Parquet scenario found under
`data/val/` by default:

```powershell
python scripts/inspect_scenario.py
python scripts/visualize_scenario.py
```

You can pass a specific Parquet file or a directory containing scenarios:

```powershell
python scripts/inspect_scenario.py data/val/<scenario-id>/scenario_<scenario-id>.parquet
python scripts/visualize_scenario.py data/val/<scenario-id> --output outputs/example.png
```

The plot shows the focal agent's past as a solid line, its future ground truth
as a dashed line, the constant-velocity prediction as a dotted line, and
surrounding actors as thin grey trajectories.

## First milestone

Take an AV2 scenario, print its track structure, and save a bird's-eye view of
the focal agent's past, future ground truth, and baseline prediction.

## Constant Velocity baseline

The first baseline estimates velocity from the mean of the last five
displacements and extrapolates linearly at AV2's 10 Hz sampling interval.
Coordinates and ADE/FDE values are in meters. Run it on the available
validation subset:

```powershell
python scripts/evaluate_constant_velocity.py --data data/val
```

Use `--limit 500` to cap evaluation at 500 scenarios. ADE averages Euclidean
error over future timesteps; FDE measures the error at the final future point.

## First learned model: focal-agent MLP

The dataset uses 50 observed and 60 future positions. In the agent-centric
representation, the last observed position becomes `(0, 0)` and the estimated
direction of motion is rotated onto the local `+x` axis. Each history step
contains `[x, y, dx, dy]`, where displacement is measured in local coordinates
and the first history step uses zero displacement because it has no predecessor.
The inverse transform restores predictions to world coordinates for plots.
The MLP architecture stays the same: flatten history, pass it through two ReLU
hidden layers, and predict the complete future path in one pass. Training uses
coordinate-wise MSE; ADE and FDE remain the evaluation metrics.

Install PyTorch if it is not already available (Colab has it preinstalled):

```powershell
python -m pip install -e ".[train]"
```

Train on the local 500-scene training subset and evaluate on the same 500
validation scenarios used for the earlier models:

```powershell
python scripts/train_mlp.py --train-data data/train --val-data data/val --device cpu --epochs 20 --batch-size 64 --representation agent-centric --output checkpoints/mlp_agent_centric.pt
python scripts/evaluate_mlp.py --data data/val --checkpoint checkpoints/mlp_agent_centric.pt
```

Training uses the 500-scenario training subset under `data/train/`. A longer
run and GPU use the same scripts and checkpoint format; for example, pass
`--device cuda --epochs 20 --batch-size 256`.

### Validation comparison

The MLP checkpoints below are from 20-epoch CPU runs on 500 training scenarios.
Each checkpoint is selected by the lowest validation ADE. `basic` reproduces
MLP v1; `agent-centric` is the new coordinate and feature representation.

| Model | Train scenes | ADE ↓ (m) | FDE ↓ (m) |
| --- | ---: | ---: | ---: |
| Constant Velocity | — | 4.762 | 12.194 |
| MLP basic | 500 | 5.605 | 13.255 |
| MLP agent-centric | 500 | 4.446 | 11.154 |

## GRU sequence model

The GRU reads the same 50-step agent-centric `[x, y, dx, dy]` sequence as the
MLP. Its final hidden state is mapped directly to all 60 future positions, so
there is no autoregressive feedback loop. It uses the same MSE training loss
and ADE/FDE evaluation as the MLP.

CPU smoke run on the current 500 training scenarios:

```powershell
python scripts/train_gru.py --train-data data/train --val-data data/val --device cpu --epochs 2 --batch-size 256 --hidden-dim 128 --seed 42 --num-workers 0 --run-name gru-500-smoke --output checkpoints/gru-500-smoke.pt
python scripts/evaluate_gru.py --data data/val --checkpoint checkpoints/gru-500-smoke.pt
```

For a small-data comparison closer to the 20-epoch MLP run, train with
`--epochs 20 --batch-size 64`; this gives both models roughly the same number
of optimizer updates on 500 examples.

Training seeds Python, NumPy, PyTorch, CUDA, and the shuffled sampler. Worker
seeds derive from PyTorch's seeded generator. Each run writes its history to
`runs/<run_name>/metrics.csv` (`epoch,train_loss,val_ade,val_fde`) and saves a
three-panel learning curve (train MSE, validation ADE, validation FDE) beside
it. Checkpoints remain under `checkpoints/` and use the lowest validation ADE.
For Colab, the same scripts accept
`--device cuda --num-workers 4`; increase the training set and pass
`--train-scenarios 5000` or `--train-scenarios 20000` when the selected training
directory contains that many Parquet files. Set a distinct `--run-name` for
each experiment, for example `gru-500`, `gru-5k`, and `gru-20k`.

| Model | Train scenes | ADE ↓ (m) | FDE ↓ (m) |
| --- | ---: | ---: | ---: |
| Constant Velocity | — | 4.762 | 12.194 |
| MLP agent-centric | 500 | 4.446 | 11.154 |
| GRU agent-centric | 500 | 5.798 | 14.048 |
| GRU agent-centric | 5,000 | 3.861 | 10.035 |
| GRU agent-centric | 20,000 | 3.714 | 9.742 |

5k/20k trained in Google Colab; all models evaluated on the same 500 validation scenarios.

## Social GRU

`SocialTrajectoryDataset` adds the eight actors that are nearest to the focal
agent at the final observed timestep. It returns focal history `[50, 4]`,
neighbor histories `[8, 50, 5]`, an actor-level `neighbor_mask` `[8]`, and the
focal future `[60, 2]`. Neighbor features are local-frame `[x, y, dx, dy, valid]`;
the per-timestep `valid` bit distinguishes missing states from a real actor at
the zero-padded coordinate. A shared neighbor GRU encodes each actor,
then masked mean pooling produces the social context joined with the focal GRU
state.

Smoke train and evaluation on the local 500-scene subsets:

```bash
python scripts/train_social_gru.py \
  --train-data data/train --val-data data/val \
  --device cpu --epochs 1 --batch-size 256 \
  --train-scenarios 500 --num-neighbors 8 --run-name social-gru-smoke \
  --output checkpoints/social_gru.pt

python scripts/evaluate_social_gru.py \
  --data data/val --checkpoint checkpoints/social_gru.pt --device cpu
```

The training objective remains coordinate-wise MSE, and validation uses the
same ADE/FDE metrics as the single-agent GRU.

## Interaction GRU

`InteractionTrajectoryGRU` replaces mean pooling with a masked attention
module over the shared neighbor-GRU embeddings. Attention scores are computed
from focal context and each neighbor embedding. Optional physics features add
relative position and velocity, distance, signed closing speed (positive when
approaching), and time to closest approach. Velocities use AV2's 10 Hz
timesteps. Padding neighbors are masked before softmax; scenes with no
neighbors receive a zero social context.

Train the attention-only and attention-plus-physics variants with identical
data and hyperparameters, changing only the interaction-feature flag:

```bash
python scripts/train_interaction_gru.py \
  --train-data data/train_20k --val-data data/val \
  --train-scenarios 20000 --epochs 20 --batch-size 256 \
  --hidden-dim 128 --num-neighbors 8 --interaction-features false \
  --seed 42 --device cuda --run-name interaction-attention \
  --output checkpoints/interaction_gru.pt

python scripts/train_interaction_gru.py \
  --train-data data/train_20k --val-data data/val \
  --train-scenarios 20000 --epochs 20 --batch-size 256 \
  --hidden-dim 128 --num-neighbors 8 --interaction-features true \
  --seed 42 --device cuda --run-name interaction-physics \
  --output checkpoints/interaction_gru_physics.pt

python scripts/evaluate_interaction_gru.py \
  --data data/val --checkpoint checkpoints/interaction_gru.pt --device cuda

python scripts/evaluate_interaction_gru.py \
  --data data/val --checkpoint checkpoints/interaction_gru_physics.pt --device cuda
```

Evaluation reports ADE/FDE and prints the top attended neighbors for three
scenarios by default. Checkpoints, per-epoch CSVs, and learning curves retain
the interaction-feature setting for reproducibility.

To compare ground truth, Constant Velocity, MLP, and GRU on representative
straight, turning, braking, and sharp-maneuver examples:

```powershell
python scripts/visualize_model_cases.py --data data/val --mlp-checkpoint checkpoints/mlp_agent_centric.pt --gru-checkpoint checkpoints/gru.pt
```

The case selector uses future ground truth only to choose illustrative plots;
those labels are not fed to either model.

The Colab workflow is in `notebooks/train_av2_colab.ipynb`. Set its repository
URL and Drive dataset path, then run its cells to train all three data sizes
with the same seed and validation subset. The notebook saves checkpoints,
per-epoch metrics, and the combined scaling plot to Drive.

## Map-Aware Interaction GRU

The map-aware model adds 16 nearby lane centerlines to the focal and neighbor
GRU encoders. Each centerline is resampled to 20 points and transformed into
the focal agent's local frame. A shared point MLP and max pooling encode each
lane, and focal-conditioned masked attention pools lane and neighbor context.
The map cache is precomputed once (and ignored by Git), so vector-map JSON is
not parsed on every training epoch. This uses the official AV2 static-map API
to load each `log_map_archive_<id>.json` and read lane-segment centerlines.

Precompute lane tensors, then train and evaluate with the same 20k training
scenarios, 500 validation scenarios, seed 42, and CPU setup:

```powershell
python scripts/precompute_map_context.py --data data/train_20k --output cache/train_20k_maps.npz --max-lanes 16 --points-per-lane 20
python scripts/precompute_map_context.py --data data/val --output cache/val_500_maps.npz --max-lanes 16 --points-per-lane 20
python scripts/train_map_aware_gru.py --train-data data/train_20k --val-data data/val --train-map-cache cache/train_20k_maps.npz --val-map-cache cache/val_500_maps.npz --train-scenarios 20000 --epochs 20 --batch-size 256 --hidden-dim 128 --num-neighbors 8 --seed 42 --device cpu --run-name map-aware-20k-cpu --output checkpoints/map_aware_interaction_gru_20k.pt
python scripts/evaluate_map_aware_gru.py --data data/val --map-cache cache/val_500_maps.npz --checkpoint checkpoints/map_aware_interaction_gru_20k.pt --device cpu
```

Validation comparison (same 500 validation scenarios):

| Model | Train scenes | ADE ↓ (m) | FDE ↓ (m) |
| --- | ---: | ---: | ---: |
| GRU agent-centric | 20,000 | 3.812 | 9.925 |
| Social GRU mean | 20,000 | 3.801 | 9.726 |
| Social attention | 20,000 | 3.776 | 9.720 |
| Map-aware interaction GRU | 20,000 | 3.531 | 8.821 |

The map-aware run used 20 epochs on CPU; best validation ADE was at epoch 20.
Its per-epoch metrics and learning curve are saved in
`runs/map-aware-20k-cpu/metrics.csv` and
`runs/map-aware-20k-cpu/learning_curves.png`; the checkpoint also records its
best-epoch validation metrics. Evaluation prints lane-attention weights for
three validation scenes.

## Multimodal decoder

`MultimodalForecaster` keeps the map-aware focal, social, and map encoders and
replaces only the single-path decoder. Six learned mode embeddings produce six
full futures; a probability head predicts logits for selecting a mode. Training
uses best-of-K coordinate MSE, cross-entropy for the best mode, and a small
pairwise endpoint diversity penalty. The checkpoint is selected by validation
minADE@K. Evaluation reports probability-selected top-1 ADE/FDE, oracle
minADE@K/minFDE@K, and MissRate@K (a miss is a best final-point error above the
configurable threshold).

Train and evaluate with the precomputed map caches:

```powershell
python scripts/train_multimodal.py --train-data data/train_20k --val-data data/val --train-scenarios 20000 --epochs 20 --batch-size 256 --num-modes 6 --classification-weight 0.5 --diversity-weight 0.05 --miss-threshold 2.0 --seed 42 --device cpu --run-name multimodal-20k-cpu --output checkpoints/multimodal_20k.pt
python scripts/evaluate_multimodal.py --data data/val --map-cache cache/val_500_maps.npz --checkpoint checkpoints/multimodal_20k.pt --device cpu --miss-threshold 2.0
```

Results on 20,000 training scenarios and the same 500 validation scenarios,
with seed 42 and CPU training (best checkpoint: epoch 20):

| Model / metric | ADE ↓ (m) | FDE ↓ (m) | MissRate@6 ↓ |
| --- | ---: | ---: | ---: |
| Map-aware single trajectory | 3.531 | 8.821 | — |
| Multimodal probability top-1 | 3.911 | 9.960 | — |
| Multimodal oracle min@6 | 1.733 | 3.813 | 0.708 |

The oracle metrics show that the six outputs cover futures closer to ground
truth than the single-path model, while probability-selected top-1 is weaker
in this run. Per-epoch loss and validation history is in
`runs/multimodal-20k-cpu/metrics.csv`; its learning curve is saved beside it.

### Quality-aware scoring ablation

The soft-scoring variant keeps the encoder, decoder, best-of-K regression,
diversity penalty, data, seed, and training schedule fixed. Only the
probability-head objective changes: detached per-mode quality
`ADE + 0.5 * FDE` is converted to a target distribution with temperature 1.0,
then trained with KL divergence. The soft target is recalculated from each
batch's current trajectory predictions.

```powershell
python scripts/train_multimodal.py --train-data data/train_20k --val-data data/val --train-scenarios 20000 --epochs 20 --batch-size 256 --num-modes 6 --classification-weight 0.5 --diversity-weight 0.05 --scoring-loss soft --scoring-temperature 1.0 --miss-threshold 2.0 --seed 42 --device cpu --run-name multimodal-soft-20k-cpu --output checkpoints/multimodal_soft_20k.pt
python scripts/evaluate_multimodal.py --data data/val --map-cache cache/val_500_maps.npz --checkpoint checkpoints/multimodal_soft_20k.pt --device cpu --miss-threshold 2.0
```

Both checkpoints are selected by validation minADE@6, then evaluated on the
same 500 validation scenarios:

| Multimodal scoring | minADE@6 ↓ (m) | minFDE@6 ↓ (m) | Top-1 ADE ↓ (m) | Top-1 FDE ↓ (m) | MissRate@6 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard winner CE (epoch 20) | 1.733 | 3.813 | 3.911 | 9.960 | 0.708 |
| Soft quality targets (epoch 13) | 1.794 | 4.168 | 4.187 | 10.727 | 0.794 |

On this run the soft targets did not improve confidence selection or oracle
coverage. The per-epoch results are in `runs/multimodal-soft-20k-cpu/metrics.csv`.

### Frozen trajectory-aware reranker

The reranker is trained as a separate second stage from the hard-winner
checkpoint. The multimodal generator is frozen and run once to cache its six
candidates and concatenated focal/social/map scene embedding. A trajectory MLP
encodes each full candidate path; a scorer combines that embedding with scene
context and predicts a score per candidate. Cross-entropy targets the candidate
with the lowest `ADE + 0.5 * FDE`. Reranker checkpoints are selected by
validation top-1 ADE. Since generated trajectories are cached and never
updated, minADE/minFDE and MissRate must remain identical to the generator.

```powershell
python scripts/train_reranker.py --train-data data/train_20k --val-data data/val --train-scenarios 20000 --generator-checkpoint checkpoints/multimodal_20k.pt --epochs 20 --batch-size 256 --reranker-hidden-dim 128 --seed 42 --device cpu --run-name trajectory-reranker-20k-cpu --output checkpoints/trajectory_reranker_20k.pt
python scripts/evaluate_reranker.py --data data/val --map-cache cache/val_500_maps.npz --generator-checkpoint checkpoints/multimodal_20k.pt --reranker-checkpoint checkpoints/trajectory_reranker_20k.pt --device cpu --miss-threshold 2.0
```

Both models use the same 20,000 training scenarios and 500 validation
scenarios. The reranker best checkpoint was selected at epoch 10:

| Scoring | minADE@6 ↓ (m) | minFDE@6 ↓ (m) | Top-1 ADE ↓ (m) | Top-1 FDE ↓ (m) | MissRate@6 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original hard head | 1.733 | 3.813 | 3.911 | 9.960 | 0.708 |
| Trajectory-aware reranker | 1.733 | 3.813 | 3.774 | 9.564 | 0.708 |

The reranker modestly improves candidate selection while preserving oracle
coverage exactly. Its epoch history is in
`runs/trajectory-reranker-20k-cpu/metrics.csv`.

To draw the trained MLP beside the Constant Velocity prediction and ground
truth, pass its checkpoint to the visualizer:

```powershell
python scripts/visualize_scenario.py --checkpoint checkpoints/mlp_agent_centric.pt
```

To reproduce the earlier XY-only MLP, pass `--representation basic` while
training; evaluation reads the representation from the saved checkpoint.
