"""Interactive inference-only player for Argoverse 2 motion forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from motion_forecasting.data import load_scenario
from motion_forecasting.demo.scenario_player import (
    animated_figure,
    focal_state,
    load_map_cache,
    load_predictor,
    predict_scenario,
    prediction_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def cached_scenario(path: str):
    return load_scenario(path)


@st.cache_data(show_spinner="Loading lane map cache…")
def cached_map_cache(path: str):
    return load_map_cache(path)


@st.cache_resource(show_spinner="Loading model checkpoint…")
def cached_predictor(model_name: str, checkpoint_dir: str):
    return load_predictor(model_name, checkpoint_dir)


def _absolute_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


st.set_page_config(
    page_title="Argoverse Motion Player",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
      [data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e2e8f0;
        padding: .8rem 1rem; border-radius: .8rem; }
      div[data-testid="stCaptionContainer"] { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interactive Motion Forecasting Player")
st.caption(
    "Argoverse 2 · metric bird’s-eye view · inference only · future remains hidden until revealed"
)

with st.sidebar:
    st.header("Scenario & model")
    data_text = st.text_input("Scenario data folder", value="data/val")
    checkpoint_text = st.text_input("Checkpoint folder", value="checkpoints")
    map_text = st.text_input("Map cache", value="cache/val_500_maps.npz")

data_dir = _absolute_path(data_text)
checkpoint_dir = _absolute_path(checkpoint_text)
map_cache_path = _absolute_path(map_text)
if not data_dir.is_dir():
    st.error(f"Scenario folder not found: {data_dir}")
    st.stop()

scenario_files = sorted(data_dir.rglob("scenario_*.parquet"))
if not scenario_files:
    st.error(f"No scenario_*.parquet files found under {data_dir}")
    st.stop()

scenario_by_name = {
    f"{path.stem.removeprefix('scenario_')} · {path.parent.name}": path
    for path in scenario_files
}
with st.sidebar:
    scenario_label = st.selectbox("Scenario", list(scenario_by_name))
    checkpoint_options = {
        "Constant Velocity": None,
        "GRU": checkpoint_dir / "gru_20k_cpu.pt",
        "Map-aware": checkpoint_dir / "map_aware_interaction_gru_20k.pt",
        "Final multimodal": checkpoint_dir / "lane_conditioned_20k.pt",
    }
    available_models = [
        name
        for name, path in checkpoint_options.items()
        if path is None or path.is_file()
    ]
    if "Final multimodal" in available_models and not (
        checkpoint_dir / "trajectory_reranker_lane_conditioned_20k.pt"
    ).is_file():
        available_models.remove("Final multimodal")
    model_name = st.radio("Model", available_models, index=max(0, len(available_models) - 1))
    st.markdown("---")
    st.caption("The app loads checkpoints and runs inference only. It never starts training.")

scenario_path = scenario_by_name[scenario_label]
scenario = cached_scenario(str(scenario_path))
scenario_key = f"{scenario['scenario_id']}::{model_name}"
if st.session_state.get("active_scenario_key") != scenario_key:
    st.session_state.active_scenario_key = scenario_key
    st.session_state.reveal_ground_truth = False

needs_map = model_name in {"Map-aware", "Final multimodal"}
map_cache = None
if needs_map:
    if not map_cache_path.is_file():
        st.error(f"Map cache not found: {map_cache_path}")
        st.stop()
    try:
        map_cache = cached_map_cache(str(map_cache_path))
        if scenario["scenario_id"] not in map_cache["indices"]:
            st.error(
                f"Scenario {scenario['scenario_id']} is not in {map_cache_path}. "
                "Select a matching scenario split/cache pair."
            )
            st.stop()
    except Exception as exc:
        st.error(f"Could not load map cache: {exc}")
        st.stop()

try:
    predictor = (
        None
        if model_name == "Constant Velocity"
        else cached_predictor(model_name, str(checkpoint_dir))
    )
    result = predict_scenario(scenario, model_name, predictor, map_cache)
except Exception as exc:
    st.exception(exc)
    st.stop()

top_score = float(result["scores"][result["top_index"]])
with st.sidebar:
    st.markdown("### Focal agent")
    state = focal_state(scenario)
    col_speed, col_heading = st.columns(2)
    col_speed.metric("Speed", f"{state['speed']:.1f} m/s")
    col_heading.metric("Heading", f"{state['heading'] * 180 / 3.141592653589793:.0f}°")
    st.metric("Actors at cutoff", int(state["neighbors"]))
    st.markdown("### Selected prediction")
    st.metric("Top-1 score", f"{top_score:.1%}")
    st.caption("Reranker softmax score" if len(result["scores"]) > 1 else "Single-trajectory model")
    if len(result["scores"]) > 1:
        st.markdown("#### Hypothesis scores")
        for index, score in enumerate(result["scores"]):
            selected = "●" if index == result["top_index"] else "○"
            st.write(f"{selected} **#{index + 1}**　{float(score):.1%}")
    if result["neighbor_weights"]:
        st.markdown("### Social attention")
        attention_rows = sorted(
            result["neighbor_weights"].items(), key=lambda row: row[1], reverse=True
        )
        table = pd.DataFrame(
            [{"Neighbor": track_id, "Attention": f"{weight:.3f}"} for track_id, weight in attention_rows]
        )
        st.dataframe(table, hide_index=True, width="stretch", height=240)

left, right = st.columns([4.2, 1.0], gap="large")
with right:
    st.markdown("### Ground truth")
    if st.session_state.reveal_ground_truth:
        if st.button("Hide Ground Truth", use_container_width=True):
            st.session_state.reveal_ground_truth = False
            st.rerun()
    elif st.button("Reveal Ground Truth", type="primary", use_container_width=True):
        st.session_state.reveal_ground_truth = True
        st.rerun()
    st.caption("GT and error metrics are hidden until you reveal them.")

    if st.session_state.reveal_ground_truth:
        metrics = prediction_metrics(result)
        st.markdown("### Forecast metrics")
        st.metric("Top-1 ADE", f"{metrics['top1_ade']:.2f} m")
        st.metric("Top-1 FDE", f"{metrics['top1_fde']:.2f} m")
        st.metric("Oracle minADE", f"{metrics['oracle_min_ade']:.2f} m")
        st.metric("Oracle minFDE", f"{metrics['oracle_min_fde']:.2f} m")
        miss_label = "MISS" if metrics["miss"] else "HIT"
        st.metric("Miss / Hit (2 m)", miss_label)

with left:
    st.markdown(
        "**Timeline:** 0–5 s observed motion · at cutoff predictions appear · "
        "continue playback to 11 s. Use the Plotly timeline and ▶ Play control."
    )
    figure = animated_figure(
        scenario,
        result,
        reveal_ground_truth=st.session_state.reveal_ground_truth,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "Top-1 is highlighted with a solid line. Faint dotted lines show the other "
        "hypotheses; lane geometry and actor positions use scene coordinates in meters."
    )
