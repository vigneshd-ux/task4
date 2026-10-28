
import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "support_tickets.csv"
SEED = 23

def clean_text(text):
    """Lowercase, remove punctuation, and tokenize into simple word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def make_dataset():
    """Create 450 short labeled tickets: 150 for each of three categories."""
    rng = random.Random(SEED)
    examples = {
        "billing": [
            "I was charged twice for my monthly subscription",
            "Please explain this unexpected payment on my account",
            "I need a refund for the duplicate invoice",
            "My card payment failed but money was taken",
            "The billing amount is incorrect on the latest statement",
            "Please cancel the renewal and return my charge",
            "I cannot update my payment method in billing settings",
            "The subscription price changed without notice",
            "I need a receipt for the transaction on my card",
            "There is a problem with my invoice and account balance",
        ],
        "technical": [
            "The mobile app crashes when I open my profile",
            "I cannot log in because the password reset fails",
            "The website shows an error when I upload a file",
            "My screen freezes during checkout in the application",
            "The search feature is not loading any results",
            "I forgot my password and the sign in link is broken",
            "The latest update made the app very slow",
            "Notifications stopped working on my device",
            "I get a server error when saving my settings",
            "The dashboard will not open in my browser",
        ],
        "delivery": [
            "My package has not arrived and tracking is delayed",
            "The order was delivered to the wrong address",
            "Please tell me when my shipment will arrive",
            "The courier left my parcel outside in the rain",
            "Tracking says delivered but I did not receive the order",
            "I need to change the shipping address for my package",
            "The delivery is late and the tracking page is not updated",
            "My box arrived damaged and an item is missing",
            "Can you arrange a return pickup for this order",
            "The shipment is still waiting at the warehouse",
        ],
    }   

    endings = ["please help", "thank you", "this is urgent", "please advise", "today"]
    rows = []
    for label, templates in examples.items():
        for i in range(150):
            text = templates[i % len(templates)]
            if i % 2 == 0:
                text += " " + rng.choice(endings)
            rows.append((text, label))
    rng.shuffle(rows)
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return rows


def train_skipgram_word2vec(tokenized_sentences, vector_size=50, window=2, epochs=60):
    """Train skip-gram with negative sampling using PyTorch embedding layers."""
    counts = Counter(token for sentence in tokenized_sentences for token in sentence)
    vocabulary = {word: i for i, (word, _) in enumerate(counts.most_common())}
    indexed = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in tokenized_sentences]
    pairs = []
    for sentence in indexed:
         for center_position, center in enumerate(sentence):
            left = max(0, center_position - window)
            right = min(len(sentence), center_position + window + 1)
            for context_position in range(left, right):
                if context_position != center_position:
                    pairs.append((center, sentence[context_position]))

    torch.manual_seed(SEED)
    input_vectors = torch.nn.Embedding(len(vocabulary), vector_size)
    output_vectors = torch.nn.Embedding(len(vocabulary), vector_size)
    torch.nn.init.uniform_(input_vectors.weight, -0.5 / vector_size, 0.5 / vector_size)
    torch.nn.init.zeros_(output_vectors.weight)
    optimizer = torch.optim.Adam(list(input_vectors.parameters()) + list(output_vectors.parameters()), lr=0.02)
    pair_array = torch.tensor(pairs, dtype=torch.long)
    negative_probabilities = torch.tensor(
        [counts[word] ** 0.75 for word in vocabulary], dtype=torch.float32
    )
    negative_probabilities /= negative_probabilities.sum()
    batch_size = 512

    for _ in range(epochs):
        order = torch.randperm(len(pair_array))
        for start in range(0, len(order), batch_size):
            batch = pair_array[order[start : start + batch_size]]
            centers, contexts = batch[:, 0], batch[:, 1]
            negatives = torch.multinomial(negative_probabilities, len(centers) * 5, replacement=True)
            negatives = negatives.reshape(len(centers), 5)
            center_vectors = input_vectors(centers)
            positive_scores = (center_vectors * output_vectors(contexts)).sum(dim=1)
            negative_scores = torch.bmm(
                output_vectors(negatives), center_vectors.unsqueeze(2)
            ).squeeze(2)
            loss = -(torch.nn.functional.logsigmoid(positive_scores).mean()
                     + torch.nn.functional.logsigmoid(-negative_scores).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    vectors = input_vectors.weight.detach().numpy()
    return vocabulary, vectors

def nearest_words(word, vocabulary, vectors, count=5):
    if word not in vocabulary:
        return []
    matrix = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    scores = matrix @ matrix[vocabulary[word]]
    scores[vocabulary[word]] = -np.inf
    return [list(vocabulary)[i] for i in np.argsort(scores)[-count:][::-1]]


def analogy(a, b, c, vocabulary, vectors):
    """Return the nearest vector to a - b + c, excluding input words."""
    missing = [word for word in (a, b, c) if word not in vocabulary]
    if missing:
        return None, missing
    query = vectors[vocabulary[a]] - vectors[vocabulary[b]] + vectors[vocabulary[c]]
    query /= np.linalg.norm(query) + 1e-12
    matrix = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    scores = matrix @ query
    for word in (a, b, c):
        scores[vocabulary[word]] = -np.inf
    result = list(vocabulary)[int(np.argmax(scores))]
    return result, []

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)

    rows = make_dataset()
    texts = [text for text, _ in rows]
    labels = np.array([label for _, label in rows])
    indices = np.arange(len(texts))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_texts = [texts[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=clean_text, token_pattern=None)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    baseline.fit(train_texts, train_labels)
    tfidf_predictions = baseline.predict(test_texts)
    print(f"Dataset: {len(rows)} labeled tickets ({len(set(labels))} classes), split {len(train_indices)}/{len(test_indices)}")
    print("TF-IDF baseline: accuracy {:.3f}, macro F1 {:.3f}".format(
        accuracy_score(test_labels, tfidf_predictions),
        f1_score(test_labels, tfidf_predictions, average="macro"),
    ))
    class_names = baseline.named_steps["classifier"].classes_
    feature_names = baseline.named_steps["tfidf"].get_feature_names_out()
    coefficients = baseline.named_steps["classifier"].coef_
    for label, weights in zip(class_names, coefficients):
        top_words = feature_names[np.argsort(weights)[-10:][::-1]]
        print(f"Top words for {label}: {', '.join(top_words)}")
    print("Confusion matrix (label order: {}):".format(", ".join(class_names)))
    print(confusion_matrix(test_labels, tfidf_predictions, labels=class_names))

    # These extra sentences introduce the vocabulary needed to demonstrate
    # gender/royalty analogies; they are not used as labeled classifier data.
    analogy_sentences = [
        "king is a man and queen is a woman",
        "a king is a male ruler and a queen is a female ruler",
        "the king and the queen are royal rulers",
        "man is to king as woman is to queen",
        "king man woman queen prince princess ruler royal male female",
    ] * 80
    embedding_training_texts = train_texts + analogy_sentences
    embedding_tokens = [clean_text(text) for text in embedding_training_texts]
    vocabulary, vectors = train_skipgram_word2vec(embedding_tokens)

    for word in ("billing", "delivery"):
        print(f"5 nearest words to {word}: {nearest_words(word, vocabulary, vectors)}")
    answer, missing = analogy("king", "man", "woman", vocabulary, vectors)
    if answer:
        print(f"king - man + woman -> {answer}")
    else:
        print("Analogy could not be calculated; missing words:", missing)

    # Average token vectors to get one fixed-size sentence representation.
    def sentence_vector(text):
        tokens = [word for word in clean_text(text) if word in vocabulary]
        if not tokens:
            return np.zeros(vectors.shape[1])
        return np.mean([vectors[vocabulary[word]] for word in tokens], axis=0)

    x_train = np.vstack([sentence_vector(texts[i]) for i in train_indices])
    x_test = np.vstack([sentence_vector(texts[i]) for i in test_indices])
    embedding_classifier = LogisticRegression(max_iter=1000, random_state=SEED)
    embedding_classifier.fit(x_train, train_labels)
    embedding_predictions = embedding_classifier.predict(x_test)
    print("Averaged Word2Vec: accuracy {:.3f}, macro F1 {:.3f}".format(
        accuracy_score(test_labels, embedding_predictions),
        f1_score(test_labels, embedding_predictions, average="macro"),
    ))
    print("Generated dataset CSV:", DATA_FILE)


if __name__ == "__main__":
    main()
