import pandas as pd
import re
import string
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# 1. LOAD YOUR DATASET


file_path = "Reviews.csv"  
df = pd.read_csv(file_path)


possible_text_cols = [col for col in df.columns if df[col].dtype == 'object']
if len(possible_text_cols) == 0:
    raise ValueError("No text columns found in the dataset.")
text_column = possible_text_cols[0]

df = df[[text_column]].dropna()
df[text_column] = df[text_column].astype(str)
corpus = df[text_column].tolist()

print("Dataset loaded successfully.")
print("Number of rows:", len(corpus))
print("Using text column:", text_column)
print("Sample text:", corpus[0])
print()


# 2. PREPROCESS DATA


def preprocess(sentences):
    clean_corpus = []
    for s in sentences:
        s = s.lower()
        s = re.sub(r"[^a-z\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            clean_corpus.append(s)
    return clean_corpus

clean_corpus = preprocess(corpus)
print("Data preprocessing complete.")
print("Sample after cleaning:", clean_corpus[0])
print()


# 3. BUILD AUTOCOMPLETE MODEL


def build_prefix_index(sentences, max_pref_len=20, top_k=5):
    prefix_dict = defaultdict(Counter)
    for s in sentences:
        for token in s.split():
            for i in range(1, min(len(token), max_pref_len)+1):
                prefix_dict[token[:i]][token] += 1
    prefix_index = {p: [w for w, _ in c.most_common(top_k)] for p, c in prefix_dict.items()}
    return prefix_index

prefix_index = build_prefix_index(clean_corpus)

def autocomplete(prefix):
    prefix = prefix.lower()
    return prefix_index.get(prefix, ["<no suggestion>"])


# 4. BUILD AUTOCORRECT MODEL


def words(text):
    return re.findall(r'\w+', ' '.join(clean_corpus))

WORDS = Counter(words(clean_corpus))
alphabet = string.ascii_lowercase

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known(words_):
    return set(w for w in words_ if w in WORDS)

def candidates(word):
    return known([word]) or known(edits1(word)) or [word]

def correction(word):
    cands = candidates(word)
    return max(cands, key=lambda w: WORDS[w]) if cands else word

# 5. CREATE SYNTHETIC ERROR DATASET


def introduce_typos(sentence, typo_rate=0.1):
    words_ = sentence.split()
    new_words = []
    for w in words_:
        if random.random() < typo_rate and len(w) > 2:
            pos = random.randint(0, len(w)-2)
            w = w[:pos] + random.choice(alphabet) + w[pos+1:]
        new_words.append(w)
    return ' '.join(new_words)

sample_corpus = clean_corpus[:1000] if len(clean_corpus) > 1000 else clean_corpus
typo_sentences = [introduce_typos(s) for s in sample_corpus]

df_eval = pd.DataFrame({
    "original": sample_corpus,
    "noisy": typo_sentences
})


# 6. AUTOCORRECT EVALUATION


def word_accuracy(originals, noisy):
    total, correct = 0, 0
    for o, n in zip(originals, noisy):
        for w_o, w_n in zip(o.split(), n.split()):
            if correction(w_n) == w_o:
                correct += 1
            total += 1
    return correct / total if total else 0

acc = word_accuracy(sample_corpus, typo_sentences)
print("Autocorrect Accuracy: {:.2f}%".format(acc*100))


# 7. AUTOCOMPLETE EVALUATION


def next_word_accuracy(sentences, top_k=3):
    total, correct = 0, 0
    for s in sentences:
        tokens = s.split()
        if len(tokens) < 1:
            continue
        prefix = tokens[0][:2]
        target = tokens[0]
        suggestions = autocomplete(prefix)
        if target in suggestions[:top_k]:
            correct += 1
        total += 1
    return correct / total if total else 0

topk_acc = next_word_accuracy(sample_corpus)
print("Autocomplete Top-3 Accuracy: {:.2f}%".format(topk_acc*100))

# 8. VISUALIZATION


metrics = {
    "Autocorrect Accuracy": acc,
    "Autocomplete Top-3 Accuracy": topk_acc
}

plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values(), color=["#9ad0f5", "#f5b7b1"])
plt.ylabel("Accuracy")
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.show()


# 9. INTERACTIVE MODE (Optional)


def interactive_mode():
    print("Interactive Autocomplete & Autocorrect Mode")
    while True:
        text = input("\nType something (or 'exit'): ")
        if text.lower() == 'exit':
            print("Goodbye!")
            break
        last_word = text.split()[-1] if text else ""
        if last_word:
            print("Autocorrect:", correction(last_word))
            prefix = last_word[:2] if len(last_word) >= 2 else last_word
            print("Autocomplete:", autocomplete(prefix))


