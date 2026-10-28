"""BrandPulse AI - Streamlit dashboard.

Run from the project root:   streamlit run app.py
Requires Streamlit >= 1.37 (uses st.fragment for the live-updating section).
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import LABELS
from src.preprocess import clean_tweet
from src.tokenizer import SimpleTokenizer, pad_sequences

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RAW_CSV = ROOT / "data" / "raw" / "Tweets.csv"

COLORS = {"negative": "#e15759", "neutral": "#9d9da1", "positive": "#59a14f"}
EMOJI = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
HISTORY_TWEETS = 600  # simulated tweets spread over the last 24 hours

FALLBACK_TWEETS = [
    "The service was amazing! Crew went above and beyond.",
    "I waited 4 hours just to get a cold burger.",
    "Flight delayed again and nobody tells us anything.",
    "Thanks for the quick rebooking, really appreciate it!",
    "Does anyone know the baggage allowance for international flights?",
    "Lost my luggage and customer support is useless.",
    "Smooth check-in and boarding, great experience.",
    "Oh great, another cancelled flight. Just what I needed.",
    "Landed on time, friendly staff, comfortable seats.",
    "Can I change my seat after online check-in?",
    "Worst airline experience of my life. Never flying again.",
    "Love the new in-flight entertainment options!",
]

st.set_page_config(page_title="BrandPulse AI", page_icon="📡", layout="wide")

FONT_SCALE = 1.25  # 1.0 = default, 1.25 = 25% bigger, 1.5 = 50% bigger

st.markdown(
    f"""
    <style>
        html {{ font-size: {FONT_SCALE * 100}%; }}
        [data-testid="stMetricValue"] {{ font-size: 2.6rem; }}
        [data-testid="stMetricLabel"] p {{ font-size: 1.1rem; }}
        [data-testid="stCaptionContainer"] p {{ font-size: 1rem; }}
        button[data-baseweb="tab"] p {{ font-size: 1.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------------
# Model loading (cached so models are loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classical():
    path = MODELS_DIR / "tfidf_logreg.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_lstm():
    paths = [
        MODELS_DIR / n
        for n in ("lstm_model.keras", "tokenizer.json", "lstm_config.json")
    ]
    if not all(p.exists() for p in paths):
        return None
    from tensorflow import keras  # imported lazily: only needed if the LSTM exists

    model = keras.models.load_model(paths[0])
    tokenizer = SimpleTokenizer.from_json(paths[1])
    config = json.loads(paths[2].read_text())
    return model, tokenizer, config


def predict_classical(texts):
    pipe = load_classical()
    probs = pipe.predict_proba([clean_tweet(t) for t in texts])
    order = [list(pipe.classes_).index(label) for label in LABELS]
    return probs[:, order]


def predict_lstm(texts):
    model, tokenizer, config = load_lstm()
    cleaned = [clean_tweet(t, remove_stopwords=False) for t in texts]
    X = pad_sequences(tokenizer.texts_to_sequences(cleaned), config["max_len"])
    return model.predict(X, verbose=0)


def available_models():
    models = {}
    if load_classical() is not None:
        models["TF-IDF + Logistic Regression"] = predict_classical
    if load_lstm() is not None:
        models["BiLSTM (Keras)"] = predict_lstm
    return models


@st.cache_data
def load_tweet_pool():
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)["text"].dropna().drop_duplicates().tolist()
    return FALLBACK_TWEETS


def classify(texts, predictor, times):
    probs = predictor(texts)
    return pd.DataFrame(
        {
            "time": times,
            "text": texts,
            "sentiment": [LABELS[i] for i in probs.argmax(axis=1)],
            "confidence": probs.max(axis=1),
        }
    )


# ---------------------------------------------------------------------------
# Simulated tweet stream
# ---------------------------------------------------------------------------
def init_feed(predictor, pool):
    """Back-fill ~24 hours of history so the trend line has something to show."""
    texts = [random.choice(pool) for _ in range(HISTORY_TWEETS)]
    now = pd.Timestamp.now()
    times = [now - pd.Timedelta(seconds=random.uniform(0, 24 * 3600)) for _ in texts]
    return classify(texts, predictor, times).sort_values("time").reset_index(drop=True)


def add_tweets(k, predictor, pool):
    texts = random.sample(pool, min(k, len(pool)))
    new = classify(texts, predictor, [pd.Timestamp.now()] * len(texts))
    feed = pd.concat([st.session_state.feed, new], ignore_index=True)
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    st.session_state.feed = feed[feed["time"] >= cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("📡 BrandPulse AI")
st.caption("Real-time Voice-of-the-Customer sentiment monitoring")

models = available_models()
if not models:
    st.error(
        "No trained models found in `models/`. Run `notebooks/01_Preprocessing_and_Classical.ipynb` "
        "(and optionally `02_Deep_Learning_LSTM.ipynb`) first."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    model_name = st.radio("Model", list(models.keys()))
    running = st.toggle("▶ Live stream", value=False)
    per_tick = st.slider("Tweets per update", 1, 10, 3)
    every = st.slider("Seconds between updates", 1, 10, 2)
    if st.button("Reset feed"):
        st.session_state.pop("feed", None)
    st.divider()
    st.caption(
        "Tweet source: "
        + ("data/raw/Tweets.csv" if RAW_CSV.exists() else "built-in sample tweets")
        + ". The stream is simulated by replaying tweets with fresh timestamps."
    )

predictor = models[model_name]
pool = load_tweet_pool()

if st.session_state.get("feed_model") != model_name or "feed" not in st.session_state:
    with st.spinner("Classifying the last 24 hours of tweets..."):
        st.session_state.feed = init_feed(predictor, pool)
        st.session_state.feed_model = model_name


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_try, tab_report = st.tabs(
    ["📈 Live dashboard", "✍️ Try your own text", "📊 Model report"]
)

with tab_live:

    @st.fragment(run_every=every if running else None)
    def live_dashboard():
        if running:
            add_tweets(per_tick, predictor, pool)
        feed = st.session_state.feed

        counts = feed["sentiment"].value_counts().reindex(LABELS, fill_value=0)
        total = int(counts.sum())
        pct = counts / max(total, 1) * 100
        net = pct["positive"] - pct["negative"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tweets (24h)", f"{total:,}")
        m2.metric("Positive", f"{pct['positive']:.1f}%")
        m3.metric("Negative", f"{pct['negative']:.1f}%")
        m4.metric("Net sentiment", f"{net:+.1f}")

        left, right = st.columns([1, 2])
        with left:
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                hole=0.45,
                color=counts.index,
                color_discrete_map=COLORS,
                title="Sentiment distribution",
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        with right:
            hourly = feed.assign(hour=feed["time"].dt.floor("60min"))
            table = (
                hourly.groupby(["hour", "sentiment"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=LABELS, fill_value=0)
            )
            share = table.div(table.sum(axis=1), axis=0) * 100
            long = share.reset_index().melt(
                id_vars="hour", var_name="sentiment", value_name="share"
            )
            fig = px.line(
                long,
                x="hour",
                y="share",
                color="sentiment",
                markers=True,
                color_discrete_map=COLORS,
                title="Sentiment trend - last 24 hours (% of tweets per hour)",
            )
            fig.update_layout(yaxis_title="% of tweets", xaxis_title=None)
            st.plotly_chart(fig)

        st.subheader("Incoming tweets")
        recent = feed.tail(10).iloc[::-1].copy()
        recent["time"] = recent["time"].dt.strftime("%H:%M:%S")
        recent["sentiment"] = recent["sentiment"].map(lambda s: f"{EMOJI[s]} {s}")
        recent["confidence"] = recent["confidence"].map(lambda c: f"{c:.0%}")
        st.dataframe(
            recent[["time", "sentiment", "confidence", "text"]], hide_index=True
        )

    live_dashboard()

with tab_try:
    st.subheader("Classify any text")
    text = st.text_area("Enter a tweet", "The service was amazing!", height=100)
    if st.button("Analyze", type="primary") and text.strip():
        st.caption(
            f"Cleaned text (classical pipeline): `{clean_tweet(text) or '(empty)'}`"
        )
        cols = st.columns(len(models))
        for col, (name, fn) in zip(cols, models.items()):
            probs = fn([text])[0]
            label = LABELS[int(np.argmax(probs))]
            with col:
                st.markdown(f"**{name}**")
                st.markdown(f"### {EMOJI[label]} {label.title()} ({probs.max():.0%})")
                fig = px.bar(
                    x=LABELS,
                    y=probs,
                    color=LABELS,
                    color_discrete_map=COLORS,
                    labels={"x": "", "y": "probability"},
                    range_y=[0, 1],
                )
                fig.update_layout(showlegend=False, height=260, margin=dict(t=10))
                st.plotly_chart(fig)
    st.caption(
        "Tip: try sarcasm, e.g. 'Oh great, another delayed flight. Just what I needed.'"
    )

with tab_report:
    st.subheader("Classical vs Deep Learning")
    csv = REPORTS_DIR / "model_comparison.csv"
    if csv.exists():
        st.dataframe(pd.read_csv(csv), hide_index=True)
    else:
        st.info("Run both notebooks to generate `reports/model_comparison.csv`.")
    c1, c2 = st.columns(2)
    for col, fname, title in [
        (c1, "cm_classical.png", "Classical"),
        (c2, "cm_lstm.png", "BiLSTM"),
    ]:
        if (REPORTS_DIR / fname).exists():
            col.markdown(f"**{title} - confusion matrix**")
            col.image(str(REPORTS_DIR / fname))
