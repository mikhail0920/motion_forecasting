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
archive. To add more scenarios, download the matching
`scenario_<id>.parquet` and `log_map_archive_<id>.json` files into a folder
under `data/val/` from the public Argoverse S3 bucket:

`s3://argoverse/datasets/av2/motion-forecasting/val/<scenario-id>/`

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
| MLP agent-centric | 10k+ | TBD | TBD |

To draw the trained MLP beside the Constant Velocity prediction and ground
truth, pass its checkpoint to the visualizer:

```powershell
python scripts/visualize_scenario.py --checkpoint checkpoints/mlp_agent_centric.pt
```

To reproduce the earlier XY-only MLP, pass `--representation basic` while
training; evaluation reads the representation from the saved checkpoint.
