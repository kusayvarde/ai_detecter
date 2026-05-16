from __future__ import annotations

import re
import string

import nltk
import numpy as np
import torch
import textstat
from textblob import TextBlob
from spellchecker import SpellChecker
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("stopwords", quiet=True)


class TextAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt_model.eval()

        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words("english"))

    def get_perplexity(self, text: str) -> float:
        text = str(text)[:1024]
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            loss = self.gpt_model(encodings.input_ids, labels=encodings.input_ids).loss
        return torch.exp(loss).item()

    def get_burstiness(self, text: str) -> float:
        sentences = nltk.sent_tokenize(str(text))
        if len(sentences) <= 1:
            return 0.0
        lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        return float(np.std(lengths))

    def get_sentence_len_variance(self, text: str) -> float:
        sentences = nltk.sent_tokenize(str(text))
        if len(sentences) <= 1:
            return 0.0
        return float(np.var([len(s.split()) for s in sentences]))

    def get_transition_density(self, text: str) -> float:
        transition_words = {
            "however", "nevertheless", "nonetheless", "conversely", "instead",
            "otherwise", "similarly", "whereas", "moreover", "furthermore",
            "additionally", "subsequently", "meanwhile", "finally", "therefore",
            "consequently", "accordingly", "thus", "hence", "for instance",
            "specifically", "notably", "resultantly",
        }
        words = str(text).lower().split()
        if not words:
            return 0.0
        return sum(1 for w in words if w in transition_words) / len(words)

    def get_subjectivity(self, text: str) -> float:
        return TextBlob(str(text)).sentiment.subjectivity

    def get_flesch_ease(self, text: str) -> float:
        return textstat.flesch_reading_ease(str(text))

    def get_gunning_fog(self, text: str) -> float:
        return textstat.gunning_fog(str(text))

    def get_ttr(self, text: str) -> float:
        words = nltk.word_tokenize(str(text).lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def get_hapax_ratio(self, text: str) -> float:
        words = nltk.word_tokenize(str(text).lower())
        if not words:
            return 0.0
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        return sum(1 for v in counts.values() if v == 1) / len(words)

    def get_bigram_diversity(self, text: str) -> float:
        words = str(text).lower().split()
        if len(words) < 2:
            return 1.0
        bi = list(ngrams(words, 2))
        return len(set(bi)) / len(bi)

    def get_trigram_diversity(self, text: str) -> float:
        words = str(text).lower().split()
        if len(words) < 3:
            return 1.0
        tri = list(ngrams(words, 3))
        return len(set(tri)) / len(tri)

    def get_passive_voice_density(self, tags: list, word_count: int) -> float:
        be_verbs = {"am", "is", "are", "was", "were", "be", "been", "being"}
        count = sum(
            1 for i in range(len(tags) - 1)
            if tags[i][0].lower() in be_verbs and tags[i + 1][1] == "VBN"
        )
        return count / word_count

    def get_nv_ratio(self, tags: list) -> float:
        nouns = sum(1 for _, t in tags if t.startswith("NN"))
        verbs = sum(1 for _, t in tags if t.startswith("VB"))
        return nouns / verbs if verbs else float(nouns)

    def get_adj_density(self, tags: list, word_count: int) -> float:
        return sum(1 for _, t in tags if t.startswith("JJ")) / word_count

    def get_adv_density(self, tags: list, word_count: int) -> float:
        return sum(1 for _, t in tags if t.startswith("RB")) / word_count

    def get_pronoun_density(self, tokens: list[str], word_count: int) -> float:
        pronouns = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
        return sum(1 for w in tokens if w.lower() in pronouns) / word_count

    def get_preposition_density(self, tags: list, word_count: int) -> float:
        return sum(1 for _, t in tags if t == "IN") / word_count

    def get_stop_word_density(self, tokens: list[str], word_count: int) -> float:
        return sum(1 for w in tokens if w.lower() in self.stop_words) / word_count

    def get_avg_word_length(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return sum(len(t) for t in tokens) / len(tokens)

    def get_punctuation_density(self, text: str) -> float:
        text_str = str(text)
        if not text_str:
            return 0.0
        return sum(1 for c in text_str if c in string.punctuation) / len(text_str)

    def get_contraction_density(self, text: str, word_count: int) -> float:
        return len(re.findall(r"\b\w+['']\w+\b", str(text).lower())) / word_count

    def get_typo_density(self, tokens: list[str], word_count: int) -> float:
        return len(self.spell.unknown(tokens)) / word_count

    def extract_all_features(self, text: str) -> dict[str, float]:
        tokens = nltk.word_tokenize(str(text))
        sentences = nltk.sent_tokenize(str(text))
        tags = nltk.pos_tag(tokens)

        word_count = max(len(tokens), 1)
        sent_count = max(len(sentences), 1)

        return {
            "perplexity":            self.get_perplexity(text),
            "burstiness":            self.get_burstiness(text),
            "sentence_len_variance": self.get_sentence_len_variance(text),
            "transition_density":    self.get_transition_density(text),
            "subjectivity":          self.get_subjectivity(text),
            "flesch_ease":           self.get_flesch_ease(text),
            "gunning_fog":           self.get_gunning_fog(text),
            "ttr":                   self.get_ttr(text),
            "hapax_ratio":           self.get_hapax_ratio(text),
            "bigram_diversity":      self.get_bigram_diversity(text),
            "trigram_diversity":     self.get_trigram_diversity(text),
            "passive_voice_density": self.get_passive_voice_density(tags, word_count),
            "nv_ratio":              self.get_nv_ratio(tags),
            "adj_density":           self.get_adj_density(tags, word_count),
            "adv_density":           self.get_adv_density(tags, word_count),
            "pronoun_density":       self.get_pronoun_density(tokens, word_count),
            "preposition_density":   self.get_preposition_density(tags, word_count),
            "stop_word_density":     self.get_stop_word_density(tokens, word_count),
            "word_count":            word_count,
            "avg_word_length":       self.get_avg_word_length(tokens),
            "avg_sentence_length":   word_count / sent_count,
            "punctuation_density":   self.get_punctuation_density(text),
            "contraction_density":   self.get_contraction_density(text, word_count),
            "typo_density":          self.get_typo_density(tokens, word_count),
        }
