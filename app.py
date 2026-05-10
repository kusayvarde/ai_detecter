"""
AI vs Human Text Classifier — Gradio App
=========================================
Accepts a raw text/article, extracts 24 linguistic features, and runs the
saved Random Forest model (../models/best_model.pkl) to predict whether the
text was written by a Human or an AI.
"""

from __future__ import annotations

import math
import pathlib
import joblib
import re
import string
from collections import Counter

import gradio as gr
import numpy as np
import pandas as pd

# ─── Model path ──────────────────────────────────────────────────────────────

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "best_model.pkl"

MODEL = joblib.load(MODEL_PATH)

# ─── Feature list (must match training order exactly) ────────────────────────

FEATURES = [
    "perplexity",
    "burstiness",
    "sentence_len_variance",
    "transition_density",
    "subjectivity",
    "flesch_ease",
    "gunning_fog",
    "ttr",
    "hapax_ratio",
    "bigram_diversity",
    "trigram_diversity",
    "passive_voice_density",
    "nv_ratio",
    "adj_density",
    "adv_density",
    "pronoun_density",
    "preposition_density",
    "stop_word_density",
    "word_count",
    "avg_word_length",
    "avg_sentence_length",
    "punctuation_density",
    "contraction_density",
    "typo_density",
]

# ─── Small helper sets ────────────────────────────────────────────────────────

_PUNCT_SET = set(string.punctuation)

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "not", "as", "if", "so",
}

_PRONOUNS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
}

_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "above", "after", "against", "along", "among", "around", "before",
    "behind", "below", "beneath", "beside", "between", "beyond", "during",
    "except", "inside", "into", "near", "off", "onto", "outside", "over",
    "past", "since", "through", "throughout", "under", "until", "up",
    "upon", "within", "without",
}

_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll",
    "you'd", "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd", "that's", "there's", "here's",
    "what's", "who's", "how's", "let's", "that'll", "there'll",
}

_TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "consequently", "additionally", "alternatively", "subsequently",
    "similarly", "likewise", "conversely", "meanwhile", "accordingly",
    "hence", "thus", "nonetheless", "notwithstanding", "otherwise",
    "in contrast", "in addition", "as a result", "for example",
    "for instance", "on the other hand", "in conclusion",
}

# Common English words for a very lightweight perplexity proxy
_COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
}

# Passive-voice heuristic: "was/were/is/are/been + <word>ed"
_PASSIVE_RE = re.compile(
    r"\b(?:was|were|is|are|been|being|be)\s+\w+ed\b", re.IGNORECASE
)

# Simple adjective/adverb heuristics (suffix-based — rough approximation)
_ADJ_SUFFIXES = ("ful", "less", "ous", "ive", "al", "ary", "ic", "able", "ible")
_ADV_SUFFIXES = ("ly",)


# ─── Per-feature extractor functions ─────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?'."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stripping punctuation."""
    return re.findall(r"\b[a-z']+\b", text.lower())


def get_perplexity(text: str) -> float:
    """
    Lightweight perplexity proxy: fraction of tokens NOT in a common-word
    vocabulary, converted to a log-scale 'surprise' score.
    A higher value means the text uses rarer vocabulary (AI tends higher).
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    unknown_rate = sum(1 for t in tokens if t not in _COMMON_WORDS) / len(tokens)
    # Map [0, 1] → a perplexity-like value in [1, 100]
    return float(np.exp(unknown_rate * np.log(100)))


def get_burstiness(text: str) -> float:
    """
    Burstiness of sentence lengths: (std − mean) / (std + mean).
    Human text is 'bursty' (large variance bursts); AI text is steadier.
    Returns 0 when std+mean ≈ 0.
    """
    sentences = _split_sentences(text)
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    m, s = float(np.mean(lengths)), float(np.std(lengths))
    denom = s + m
    return (s - m) / denom if denom else 0.0


def get_sentence_len_variance(text: str) -> float:
    """Variance of word-counts across sentences."""
    sentences = _split_sentences(text)
    lengths = [len(s.split()) for s in sentences]
    return float(np.var(lengths)) if len(lengths) >= 2 else 0.0


def get_transition_density(text: str) -> float:
    """Fraction of sentences that start with a transition word/phrase."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    count = 0
    for s in sentences:
        s_lower = s.lower()
        for tw in _TRANSITION_WORDS:
            if s_lower.startswith(tw):
                count += 1
                break
    return count / len(sentences)


def get_subjectivity(text: str) -> float:
    """
    Rough subjectivity proxy: ratio of opinion-loaded words to total words.
    Uses a small hand-crafted lexicon.
    """
    opinion_words = {
        "good", "bad", "great", "terrible", "wonderful", "awful", "amazing",
        "horrible", "excellent", "poor", "best", "worst", "love", "hate",
        "beautiful", "ugly", "important", "interesting", "boring", "exciting",
        "think", "believe", "feel", "seem", "appear", "consider", "suggest",
        "recommend", "prefer", "certainly", "probably", "perhaps", "clearly",
        "obviously", "unfortunately", "fortunately", "surprisingly",
    }
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in opinion_words) / len(tokens)


def get_flesch_ease(text: str) -> float:
    """
    Flesch Reading Ease score.
    206.835 − 1.015*(words/sentences) − 84.6*(syllables/words)
    Syllable count: rough heuristic (vowel groups).
    """
    tokens = _tokenize(text)
    sentences = _split_sentences(text)
    if not tokens or not sentences:
        return 0.0

    def syllable_count(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    words_per_sent = len(tokens) / len(sentences)
    syllables_per_word = sum(syllable_count(t) for t in tokens) / len(tokens)
    return 206.835 - 1.015 * words_per_sent - 84.6 * syllables_per_word


def get_gunning_fog(text: str) -> float:
    """
    Gunning Fog Index.
    0.4 * [(words/sentences) + 100*(complex_words/words)]
    Complex word = 3+ syllables (excluding common suffixes).
    """
    tokens = _tokenize(text)
    sentences = _split_sentences(text)
    if not tokens or not sentences:
        return 0.0

    def syllable_count(word: str) -> int:
        word = word.lower()
        return max(len(re.findall(r"[aeiouy]+", word)), 1)

    complex_count = sum(1 for t in tokens if syllable_count(t) >= 3)
    words_per_sent = len(tokens) / len(sentences)
    complex_ratio = complex_count / len(tokens)
    return 0.4 * (words_per_sent + 100 * complex_ratio)


def get_ttr(text: str) -> float:
    """Type-Token Ratio: unique tokens / total tokens."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_hapax_ratio(text: str) -> float:
    """Hapax legomena ratio: words appearing exactly once / total tokens."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(tokens)


def get_bigram_diversity(text: str) -> float:
    """Unique bigrams / total bigrams."""
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / len(bigrams)


def get_trigram_diversity(text: str) -> float:
    """Unique trigrams / total trigrams."""
    tokens = _tokenize(text)
    if len(tokens) < 3:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return len(set(trigrams)) / len(trigrams)


def get_passive_voice_density(text: str) -> float:
    """Passive constructions per sentence (regex heuristic)."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    matches = len(_PASSIVE_RE.findall(text))
    return matches / len(sentences)


def get_nv_ratio(text: str) -> float:
    """
    Noun-to-verb ratio proxy.
    Nouns: words ending in common noun suffixes or proper nouns.
    Verbs: words ending in common verb suffixes.
    Returns nouns / (verbs + 1) to avoid division by zero.
    """
    tokens = _tokenize(text)
    noun_sfx = ("tion", "ness", "ment", "ity", "er", "or", "ist", "ism", "ence", "ance")
    verb_sfx = ("ing", "ed", "ize", "ise", "fy", "en")
    nouns = sum(1 for t in tokens if any(t.endswith(s) for s in noun_sfx))
    verbs = sum(1 for t in tokens if any(t.endswith(s) for s in verb_sfx))
    return nouns / (verbs + 1)


def get_adj_density(text: str) -> float:
    """Fraction of tokens that look like adjectives (suffix heuristic)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if any(t.endswith(s) for s in _ADJ_SUFFIXES)) / len(tokens)


def get_adv_density(text: str) -> float:
    """Fraction of tokens that look like adverbs (ending in '-ly')."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t.endswith("ly")) / len(tokens)


def get_pronoun_density(text: str) -> float:
    """Fraction of tokens that are personal pronouns."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _PRONOUNS) / len(tokens)


def get_preposition_density(text: str) -> float:
    """Fraction of tokens that are prepositions."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _PREPOSITIONS) / len(tokens)


def get_stop_word_density(text: str) -> float:
    """Fraction of tokens that are stop words."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _STOP_WORDS) / len(tokens)


def get_word_count(text: str) -> int:
    """Total word count."""
    return len(text.split())


def get_avg_word_length(text: str) -> float:
    """Mean character length of words."""
    words = text.split()
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def get_avg_sentence_length(text: str) -> float:
    """Mean word count per sentence."""
    sentences = _split_sentences(text)
    words = text.split()
    if not sentences:
        return 0.0
    return len(words) / len(sentences)


def get_punctuation_density(text: str) -> float:
    """Punctuation characters / total characters."""
    if not text:
        return 0.0
    return sum(1 for c in text if c in _PUNCT_SET) / len(text)


def get_contraction_density(text: str) -> float:
    """Fraction of tokens that are contractions."""
    tokens = re.findall(r"\b[a-z']+\b", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _CONTRACTIONS) / len(tokens)


def get_typo_density(text: str) -> float:
    """
    Rough typo proxy: fraction of tokens that are very short (≤2 chars)
    AND not in the common-word vocabulary (likely noise/typos).
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    typos = sum(
        1 for t in tokens
        if len(t) <= 2 and t not in _COMMON_WORDS and not t.isdigit()
    )
    return typos / len(tokens)


# ─── Master feature extractor ─────────────────────────────────────────────────

def get_features(text: str) -> dict[str, float]:
    """
    Extract all 24 features from raw text and return them as an ordered dict
    whose keys match FEATURES exactly.
    """
    return {
        "perplexity":           get_perplexity(text),
        "burstiness":           get_burstiness(text),
        "sentence_len_variance":get_sentence_len_variance(text),
        "transition_density":   get_transition_density(text),
        "subjectivity":         get_subjectivity(text),
        "flesch_ease":          get_flesch_ease(text),
        "gunning_fog":          get_gunning_fog(text),
        "ttr":                  get_ttr(text),
        "hapax_ratio":          get_hapax_ratio(text),
        "bigram_diversity":     get_bigram_diversity(text),
        "trigram_diversity":    get_trigram_diversity(text),
        "passive_voice_density":get_passive_voice_density(text),
        "nv_ratio":             get_nv_ratio(text),
        "adj_density":          get_adj_density(text),
        "adv_density":          get_adv_density(text),
        "pronoun_density":      get_pronoun_density(text),
        "preposition_density":  get_preposition_density(text),
        "stop_word_density":    get_stop_word_density(text),
        "word_count":           get_word_count(text),
        "avg_word_length":      get_avg_word_length(text),
        "avg_sentence_length":  get_avg_sentence_length(text),
        "punctuation_density":  get_punctuation_density(text),
        "contraction_density":  get_contraction_density(text),
        "typo_density":         get_typo_density(text),
    }


# ─── Prediction function ──────────────────────────────────────────────────────

def classify_text(text: str):
    """
    Main Gradio handler.
    Returns:
        label     – "🤖 AI-Generated" or "👤 Human-Written"
        confidence – formatted confidence string
        feature_table – markdown table of extracted features
    """
    text = text.strip()
    if not text:
        return "⚠️ Please enter some text.", "", ""

    features = get_features(text)
    X = pd.DataFrame([features])[FEATURES]

    prediction = int(MODEL.predict(X)[0])

    # Try to get probability; fall back to plain predict if not available
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        confidence = float(proba[prediction]) * 100
    else:
        confidence = 100.0  # unknown

    label = "🤖 AI-Generated" if prediction == 1 else "👤 Human-Written"

    # Build a nice feature breakdown table
    rows = "| Feature | Value |\n|---|---|\n"
    for feat, val in features.items():
        if isinstance(val, float):
            rows += f"| `{feat}` | {val:.4f} |\n"
        else:
            rows += f"| `{feat}` | {val} |\n"

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
