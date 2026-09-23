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
  --train-scenarios 500 --run-name social-gru-smoke \
  --output checkpoints/social_gru.pt

python scripts/evaluate_social_gru.py \
  --data data/val --checkpoint checkpoints/social_gru.pt --device cpu
```

The training objective remains coordinate-wise MSE, and validation uses the
same ADE/FDE metrics as the single-agent GRU.

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

To draw the trained MLP beside the Constant Velocity prediction and ground
truth, pass its checkpoint to the visualizer:

```powershell
python scripts/visualize_scenario.py --checkpoint checkpoints/mlp_agent_centric.pt
```

To reproduce the earlier XY-only MLP, pass `--representation basic` while
training; evaluation reads the representation from the saved checkpoint.
