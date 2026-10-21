"""Lab 02 - Smart Keyboard V1: n-gram language model with add-k smoothing (no UI code here).

P(w_i | w_{i-n+1} ... w_{i-1}) = (count(context + w) + k) / (count(context) + k * V)
"""
import csv
import math
import re
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


# ----------------------------------------------------------------- data
def tokenize_text(text):
    """Tokenize English text while keeping punctuation as separate tokens."""
    text = re.sub(r"=+", " ", text)  # remove WikiText heading markers
    text = re.sub(r"\s+", " ", text).strip().lower()
    return re.findall(r"[a-z]+(?:['’][a-z]+)?|[0-9]+|[^\w\s]", text)


def load_documents(path=None, source="huggingface", max_lines=12000):
    """Load a real text corpus from Hugging Face and group lines into documents.

    Dataset: Salesforce/wikitext, config wikitext-2-raw-v1, train split.
    The first call downloads the dataset; later calls use the Hugging Face cache.
    A local CSV path can still be passed to load the original/custom corpus.
    """
    docs = defaultdict(list)
    if path:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tokens = tokenize_text(row["sentence"])
                if tokens:
                    docs[int(row["doc_id"])].append(tokens)
        return dict(docs)

    if source != "huggingface":
        raise ValueError("source must be 'huggingface' or pass a local CSV path")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Le package 'datasets' manque. Installez-le avec : pip install datasets"
        ) from exc

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    doc_id = 0
    line_count = 0
    current_doc = []
    for row in dataset:
        text = row.get("text", "")
        tokens = tokenize_text(text)
        if not tokens:
            continue
        current_doc.append(tokens)
        line_count += 1
        if len(current_doc) >= 20:
            docs[doc_id] = current_doc
            current_doc = []
            doc_id += 1
        if line_count >= max_lines:
            break
    if current_doc:
        docs[doc_id] = current_doc
    if not docs:
        raise RuntimeError("Le corpus Hugging Face n'a fourni aucun texte exploitable.")
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    """Fixed, reproducible split BY DOCUMENT (the last 20 % of documents are the test set)."""
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    train = [s for i in ids[:cut] for s in docs[i]]
    test = [s for i in ids[cut:] for s in docs[i]]
    return train, test


# ----------------------------------------------------------------- model
class NGramModel:
    def __init__(self, sentences, n=3, k=1.0):
        self.n, self.k = n, k
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}   # BOS is never predicted
        self.V = len(self.vocab)
        self.ctx_counts = defaultdict(Counter)       # context tuple -> Counter(next word)
        for s in sentences:
            padded = [BOS] * (n - 1) + list(s) + [EOS]
            for i in range(n - 1, len(padded)):
                self.ctx_counts[tuple(padded[i - n + 1:i])][padded[i]] += 1
        self.ctx_totals = {c: sum(cnt.values()) for c, cnt in self.ctx_counts.items()}

    # -- helpers
    def clean(self, words):
        """Map out-of-vocabulary words to <unk> (so the model never crashes on new words)."""
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words):
        words = [BOS] * (self.n - 1) + self.clean(words)
        return tuple(words[-(self.n - 1):]) if self.n > 1 else ()

    def prob(self, ctx, word):
        counts = self.ctx_counts.get(ctx)
        total = self.ctx_totals.get(ctx, 0)
        num = (counts[word] if counts else 0) + self.k
        den = total + self.k * self.V
        return num / den if den > 0 else 0.0         # k = 0 and unseen context -> probability 0

    def next_word_distribution(self, words):
        ctx = self.context(words)
        return {w: self.prob(ctx, w) for w in self.vocab if w != UNK}

    def top_k(self, words, k=3):
        dist = self.next_word_distribution(words)
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def seen_context(self, words):
        return self.context(words) in self.ctx_counts

    # -- evaluation
    def perplexity(self, sentences):
        """exp( - mean log P(token) ) over every token of every sentence (EOS included).
        Returns (perplexity, share_of_zero_probability_tokens)."""
        log_sum, n_tok, zeros = 0.0, 0, 0
        for s in sentences:
            words = self.clean(s)
            padded = [BOS] * (self.n - 1) + words + [EOS]
            for i in range(self.n - 1, len(padded)):
                p = self.prob(tuple(padded[i - self.n + 1:i]), padded[i])
                n_tok += 1
                if p <= 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        if zeros:
            return float("inf"), zeros / n_tok
        return math.exp(-log_sum / n_tok), 0.0


class InterpolatedModel:
    """p = l3 * p_trigram + l2 * p_bigram + l1 * p_unigram  (a smoother alternative to hard back-off)."""

    def __init__(self, sentences, k=0.1, lambdas=(0.6, 0.3, 0.1)):
        self.models = [NGramModel(sentences, n, k) for n in (3, 2, 1)]
        self.lambdas = lambdas
        self.vocab, self.V, self.n = self.models[0].vocab, self.models[0].V, 3

    def clean(self, words):
        return self.models[0].clean(words)

    def prob_words(self, history, word):
        history = [BOS, BOS] + history
        p = 0.0
        for lam, m in zip(self.lambdas, self.models):
            ctx = tuple(history[len(history) - (m.n - 1):]) if m.n > 1 else ()
            p += lam * m.prob(ctx, word)
        return p

    def top_k(self, words, k=3):
        hist = self.clean(words)
        dist = {w: self.prob_words(hist, w) for w in self.vocab if w != UNK}
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def perplexity(self, sentences):
        log_sum, n_tok = 0.0, 0
        for s in sentences:
            words = self.clean(s)
            for i, w in enumerate(words + [EOS]):
                log_sum += math.log(self.prob_words(words[:i], w))
                n_tok += 1
        return math.exp(-log_sum / n_tok), 0.0


def backoff_top_k(tri, bi, uni, words, k=3):
    """Hard back-off for suggestions: use the highest order whose context was actually seen."""
    for model in (tri, bi, uni):
        if model.seen_context(words) or model.n == 1:
            return model.top_k(words, k), f"{model.n}-gram"
    return [], "none"


def top_k_accuracy(model, sentences, k=3):
    """Share of positions where the true next word is among the k suggestions."""
    hits = total = 0
    failures = []
    for s in sentences:
        for i, true_w in enumerate(s + [EOS]):
            guess = [w for w, _ in model.top_k(s[:i], k)]
            truth = true_w if true_w in model.vocab else UNK
            total += 1
            if truth in guess:
                hits += 1
            elif len(failures) < 200:
                failures.append((" ".join(s[:i]) or "<start>", true_w, guess))
    return hits / total, failures


if __name__ == "__main__":
    docs = load_documents()
    train, test = split_by_document(docs)
    m = NGramModel(train, n=3, k=0.1)
    print("after 'the girls':", m.top_k(["the", "girls"]))
    print("test perplexity  :", round(m.perplexity(test)[0], 2))
