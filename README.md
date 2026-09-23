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

The `data/` directory is intentionally excluded from Git. A small validation
subset of 500 scenario Parquet files is in the working copy under `data/val/`;
the first scenario also has its local map archive. To add more scenarios,
download the matching
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

The dataset uses 50 observed and 60 future positions. Both are shifted by the
last observed position, which becomes `(0, 0)`. The MLP flattens the history,
passes it through two ReLU hidden layers, and predicts the complete future path
in one forward pass. Training minimizes coordinate-wise MSE; ADE and FDE remain
the reported evaluation metrics.

Install PyTorch if it is not already available (Colab has it preinstalled):

```powershell
python -m pip install -e ".[train]"
```

Train and evaluate on the same local 500-scene validation subset:

```powershell
python scripts/train_mlp.py --train-data data/train --val-data data/val --device cpu --epochs 20 --batch-size 64 --output checkpoints/mlp.pt
python scripts/evaluate_mlp.py --data data/val --checkpoint checkpoints/mlp.pt
```

Training uses the 500-scenario training subset under `data/train/`. A longer
run and GPU use the same scripts and checkpoint format; for example, pass
`--device cuda --epochs 20 --batch-size 256`.

### Validation comparison

These MLP values are from a 20-epoch CPU run on 500 training scenarios. The
checkpoint is selected by the lowest validation ADE.

| Model | ADE ↓ (m) | FDE ↓ (m) |
| --- | ---: | ---: |
| Constant Velocity | 4.762 | 12.194 |
| MLP (20 epochs) | 5.605 | 13.255 |

To draw the trained MLP beside the Constant Velocity prediction and ground
truth, pass its checkpoint to the visualizer:

```powershell
python scripts/visualize_scenario.py --checkpoint checkpoints/mlp.pt
```
