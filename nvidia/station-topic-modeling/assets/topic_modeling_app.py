"""Interactive BERTopic dashboard with GPU-accelerated UMAP + HDBSCAN.

Re-runs only the post-embedding steps when hyperparameters change.
Embeddings + preprocessed texts are cached to disk under ./.cache/.

Run with:
    ./run_app.sh
or:
    streamlit run topic_modeling_app.py
"""
from __future__ import annotations

import torch
torch.cuda.init()

# ── GPU accel must be installed BEFORE umap/hdbscan/pandas imports ───────────
import cuml.accel
cuml.accel.install()

import cudf.pandas
cudf.pandas.install()

import copy
import io
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

ROOT       = Path(__file__).parent
DATA_PATH  = ROOT / "Electronics.jsonl.gz"
CACHE_DIR  = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="GPU BERTopic — interactive",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GPU-Accelerated BERTopic — interactive hyperparameter explorer")
st.caption(
    "Tweak UMAP/HDBSCAN parameters in the sidebar and click **Refit topics** "
    "to refresh every visualization. Embeddings are cached, so only the "
    "post-embedding pipeline reruns."
)


# ── Cached preprocessing + embedding ─────────────────────────────────────────

def _cache_paths(nrows: int) -> tuple[Path, Path]:
    return (
        CACHE_DIR / f"preprocessed_texts_{nrows}.pkl",
        CACHE_DIR / f"embeddings_{nrows}.npy",
    )


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_or_build_embeddings(nrows: int) -> tuple[list[str], np.ndarray]:
    """Load preprocessed texts + embeddings from disk, building once if absent."""
    text_path, embed_path = _cache_paths(nrows)
    if text_path.exists() and embed_path.exists():
        with text_path.open("rb") as fh:
            texts = pickle.load(fh)
        embeddings = np.load(embed_path)
        return texts, embeddings

    # Build from scratch — slow path
    with st.status(f"Loading {nrows:,} rows from {DATA_PATH.name}…", expanded=True) as status:
        if not DATA_PATH.exists():
            st.error(f"Missing {DATA_PATH}. Download it first (see notebook cell).")
            st.stop()
        t0 = time.time()
        df = pd.read_json(DATA_PATH, lines=True, nrows=nrows, compression="gzip")
        st.write(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

        status.update(label="Preprocessing text…")
        t0 = time.time()
        texts = [_preprocess(str(t)) for t in df["text"].tolist()]
        st.write(f"Preprocessed {len(texts):,} reviews in {time.time() - t0:.1f}s")

        status.update(label=f"Encoding with {EMBED_MODEL_NAME} (this is the slow step)…")
        t0 = time.time()
        model = get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)
        st.write(f"Embedded in {time.time() - t0:.1f}s — shape {embeddings.shape}")

        status.update(label="Caching to disk…")
        with text_path.open("wb") as fh:
            pickle.dump(texts, fh)
        np.save(embed_path, embeddings)
        status.update(label="Embeddings ready.", state="complete")

    return texts, embeddings


# ── Sidebar: hyperparameters ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    nrows = st.number_input(
        "Number of documents",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Documents to embed and cluster. Changing this rebuilds the embedding cache.",
    )

    st.header("UMAP")
    umap_n_components = st.slider("n_components", 2, 20, 5)
    umap_n_neighbors  = st.slider("n_neighbors", 2, 200, 15)
    umap_min_dist     = st.slider("min_dist", 0.0, 1.0, 0.0, step=0.05)
    umap_metric       = st.selectbox("metric", ["cosine", "euclidean", "correlation"], index=0)

    st.header("HDBSCAN")
    hdbscan_min_cluster_size = st.slider("min_cluster_size", 5, 500, 50, step=5)
    hdbscan_min_samples      = st.slider("min_samples", 1, 100, 10)
    hdbscan_metric           = st.selectbox(
        "metric (HDBSCAN)", ["euclidean"], index=0,
        help="cuML HDBSCAN only supports euclidean.",
    )

    st.header("Visualizations")
    barchart_top_n = st.slider("barchart: top N topics", 4, 32, 8)
    heatmap_top_n  = st.slider("heatmap: top N topics", 10, 500, 100, step=10)
    datamap_pct    = st.slider(
        "datamap: doc sample %", 0.1, 10.0, 1.0, step=0.1,
        help="Document datamap is slow — sample a fraction of docs for plotting.",
    )

    refit = st.button("Refit topics", type="primary", width="stretch")


# ── Embeddings (one-time, cached) ────────────────────────────────────────────
texts, embeddings = load_or_build_embeddings(int(nrows))
st.success(f"Embeddings ready: {embeddings.shape[0]:,} docs × {embeddings.shape[1]} dims")


# ── Refit BERTopic on demand ─────────────────────────────────────────────────
def fit_topic_model() -> BERTopic:
    umap_model = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        gen_min_span_tree=True,
        prediction_data=True,
    )
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    with st.status("Fitting BERTopic (UMAP → HDBSCAN → c-TF-IDF)…", expanded=True) as status:
        t0 = time.time()
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
        st.write(f"Fit complete in {time.time() - t0:.1f}s · {len(set(topics))} topics found")
        status.update(label="BERTopic ready.", state="complete")
    return topic_model


if refit or "topic_model" not in st.session_state:
    st.session_state.topic_model = fit_topic_model()
    st.session_state.params = dict(
        nrows=int(nrows),
        umap=(umap_n_components, umap_n_neighbors, umap_min_dist, umap_metric),
        hdbscan=(hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_metric),
    )

topic_model: BERTopic = st.session_state.topic_model
params = st.session_state.params


# ── Show current params + topic table ────────────────────────────────────────
with st.expander("Current fit parameters", expanded=False):
    st.json(params)

topic_info = topic_model.get_topic_info()
c1, c2, c3 = st.columns(3)
c1.metric("Topics (incl. outliers)", f"{len(topic_info):,}")
c2.metric("Outlier docs (topic -1)",
          f"{int(topic_info.loc[topic_info['Topic'] == -1, 'Count'].sum()):,}")
c3.metric("Clustered docs",
          f"{int(topic_info.loc[topic_info['Topic'] != -1, 'Count'].sum()):,}")


# ── Datamap helpers ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=4)
def reduce_2d(n: int, fit_key: str) -> np.ndarray:
    """UMAP the first `n` embeddings to 2D, shared by both datamap renderers."""
    return UMAP(n_neighbors=15, n_components=2, min_dist=0.15,
                metric="cosine").fit_transform(embeddings[:n])


@st.cache_data(show_spinner=False, max_entries=4)
def build_datamap(_model: BERTopic, n: int, interactive: bool, offline: bool,
                  label_top_n: int, fit_key: str):
    """Render the datamap for the first `n` docs. `fit_key` keys the cache."""
    temp_model = copy.copy(_model)
    temp_model.topics_ = _model.topics_[:n]
    kwds = dict(
        reduced_embeddings=reduce_2d(n, fit_key),
        interactive=interactive,
    )
    if interactive:
        # offline_mode inlines deck.gl/d3/arrow/jquery + fonts into the page
        # instead of pulling them from unpkg.com at view time. Costs ~3 MB of
        # HTML but means the map still draws if the browser can't reach a CDN.
        kwds["int_datamap_kwds"] = {"offline_mode": offline}
    else:
        # Static label placement costs ~0.4s per distinct label: a 250-topic
        # sample takes ~110s, the top 25 take ~3s. Unselected topics still get
        # plotted, just unlabelled.
        ranked = _model.get_topic_info().sort_values("Count", ascending=False)
        kwds["topics"] = [t for t in ranked["Topic"].tolist() if t != -1][:label_top_n]

    fig = temp_model.visualize_document_datamap(texts[:n], **kwds)

    if interactive:
        # str(fig) is the standalone HTML page. Do NOT use _repr_html_() here:
        # that wraps the page in a fixed 1200x750 iframe, which Streamlit then
        # nests inside its own component iframe — the map ends up clipped.
        return str(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    return buf.getvalue()


def render_datamap() -> None:
    st.subheader("Document datamap")
    n = max(100, int(len(texts) * datamap_pct / 100.0))
    st.caption(f"Plotting {n:,} of {len(texts):,} documents.")

    mode_col, opt_col = st.columns([2, 3])
    with mode_col:
        interactive = st.radio(
            "Renderer", ["Interactive (WebGL)", "Static image"],
            horizontal=True, label_visibility="collapsed",
        ) == "Interactive (WebGL)"
    with opt_col:
        if interactive:
            label_top_n = 0
            offline = st.checkbox(
                "Bundle JS + fonts into the page (offline mode)", value=True,
                help="Interactive datamaps normally load deck.gl, d3 and Google Fonts "
                     "from a CDN at view time. If your browser can't reach unpkg.com "
                     "the map renders blank — keep this on to inline them instead. "
                     "The first render downloads and caches the bundle (needs network "
                     "on the machine running the app).",
            )
        else:
            offline = False
            label_top_n = st.slider(
                "Label the top N topics", 5, 100, 25, step=5,
                help="Static rendering places every label with matplotlib, which "
                     "costs roughly 0.4s per distinct topic. Labelling all ~250 "
                     "topics in a sample takes minutes; the rest are still plotted, "
                     "just without a label.",
            )

    spinner = ("Rendering datamap…" if interactive else
               f"Rendering static datamap ({label_top_n} labels, ~{label_top_n // 8 + 2}s)…")
    try:
        with st.spinner(spinner):
            result = build_datamap(topic_model, n, interactive, offline,
                                   label_top_n, str(params))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Datamap render failed: {type(exc).__name__}: {exc}")
        if offline:
            st.info("If this is a network error, untick offline mode — the CDN "
                    "bundle can't be downloaded without internet access.")
        return

    if interactive:
        st.components.v1.html(result, height=820, scrolling=False)
        st.download_button(
            "Download standalone HTML",
            result, file_name="document_datamap.html", mime="text/html",
            help="If the map above is blank, open this file directly — the "
                 "browser console will show why (usually WebGL or a blocked CDN).",
        )
    else:
        st.image(result, width="stretch")


# ── Visualizations ───────────────────────────────────────────────────────────
# One view at a time, chosen server-side. st.tabs renders every tab body on
# every run, and the datamap's deck.gl canvas draws once while its tab is still
# hidden — it never redraws when you switch to it, so the map comes up blank
# until some other widget forces a rerun. Rendering only the selected view
# means the canvas is always mounted visible.
VIEWS = [
    "Intertopic map",
    "Top words barchart",
    "Similarity heatmap",
    "Document datamap",
    "Topic table",
]
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")

if view == "Intertopic map":
    st.subheader("Intertopic distance map")
    try:
        fig = topic_model.visualize_topics()
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"visualize_topics failed: {exc}")

elif view == "Top words barchart":
    st.subheader(f"Top words for top {barchart_top_n} topics")
    fig = topic_model.visualize_barchart(top_n_topics=barchart_top_n)
    st.plotly_chart(fig, width="stretch")

elif view == "Similarity heatmap":
    st.subheader(f"Topic similarity heatmap (top {heatmap_top_n} topics)")
    fig = topic_model.visualize_heatmap(top_n_topics=heatmap_top_n)
    st.plotly_chart(fig, width="stretch")

elif view == "Document datamap":
    render_datamap()

elif view == "Topic table":
    st.subheader("Topic info")
    st.dataframe(topic_info.to_pandas() if hasattr(topic_info, "to_pandas") else topic_info,
                 width="stretch", height=600)
