"""
AI vs Human Text Classifier — Gradio App
=========================================
Accepts a raw text/article, extracts 24 linguistic features, and runs the
saved Random Forest model (models/best_model.pkl) to predict whether the
text was written by a Human or an AI.
"""

from __future__ import annotations

import pathlib
import joblib
import pandas as pd
import gradio as gr

from features import TextAnalyzer

# ─── Model & analyzer ────────────────────────────────────────────────────────

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "best_model.pkl"
MODEL = joblib.load(MODEL_PATH)
ANALYZER = TextAnalyzer()

# ─── Feature order (must match training order exactly) ───────────────────────

FEATURES = [
    "perplexity", "burstiness", "sentence_len_variance", "transition_density",
    "subjectivity", "flesch_ease", "gunning_fog", "ttr", "hapax_ratio",
    "bigram_diversity", "trigram_diversity", "passive_voice_density", "nv_ratio",
    "adj_density", "adv_density", "pronoun_density", "preposition_density",
    "stop_word_density", "word_count", "avg_word_length", "avg_sentence_length",
    "punctuation_density", "contraction_density", "typo_density",
]

# ─── Prediction function ──────────────────────────────────────────────────────

def classify_text(text: str):
    text = text.strip()
    if not text:
        return "⚠️ Please enter some text.", "", ""

    features = ANALYZER.extract_all_features(text)
    X = pd.DataFrame([features])[FEATURES]

    prediction = int(MODEL.predict(X)[0])

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        confidence = float(proba[prediction]) * 100
    else:
        confidence = 100.0

    label = "🤖 AI-Generated" if prediction == 1 else "👤 Human-Written"

    rows = "| Feature | Value |\n|---|---|\n"
    for feat, val in features.items():
        rows += f"| `{feat}` | {val:.4f} |\n" if isinstance(val, float) else f"| `{feat}` | {val} |\n"

    return label, f"{confidence:.1f}%", rows


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#label_out textarea {
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
}
"""

with gr.Blocks(title="AI vs Human Text Classifier") as demo:
    gr.Markdown(
        """
        # 🔍 AI vs Human Text Classifier
        Paste any article or essay below. The model will extract **24 linguistic features**
        and predict whether the text was written by a **Human** or an **AI**.
        """,
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📄 Input Text",
                placeholder="Paste your article or essay here…",
                lines=18,
                max_lines=40,
            )
            classify_btn = gr.Button("🚀 Classify", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_out = gr.Textbox(
                label="🏷️ Prediction",
                interactive=False,
                elem_id="label_out",
                lines=2,
            )
            confidence_out = gr.Textbox(
                label="📊 Model Confidence",
                interactive=False,
                lines=1,
            )

    with gr.Accordion("🔬 Feature Breakdown", open=False):
        feature_out = gr.Markdown()

    classify_btn.click(
        fn=classify_text,
        inputs=[text_input],
        outputs=[label_out, confidence_out, feature_out],
    )

    gr.Examples(
        examples=[
            [
                "The impact of climate change on global agriculture is multifaceted and far-reaching. "
                "Rising temperatures alter precipitation patterns, shift growing seasons, and increase the "
                "frequency of extreme weather events. These changes threaten food security in vulnerable "
                "regions while simultaneously opening new agricultural possibilities in previously inhospitable "
                "areas. Policymakers must balance adaptation strategies with mitigation efforts to ensure "
                "sustainable food production for future generations."
            ],
            [
                "so i was thinking bout this thing that happened to me last week. my friend jake "
                "totally forgot we were supposed to meet up and i was waiting like 45 mins in the cold!! "
                "ugh. anyways we talked it out and he said sorry so its fine i guess. stuff like this "
                "happens alot but its still annoying u know? anyway we ended up getting pizza which was rly good"
            ],
        ],
        inputs=[text_input],
        label="Try an example",
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css=CSS,
    )
