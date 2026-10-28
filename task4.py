
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_DIR = OUT / "sentiment-model"
BASE_MODEL = "distilbert-base-uncased"
SEED = 42


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "sentiment.csv")
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df.label
    )
    y_train, y_test = train_df.label.to_numpy(), test_df.label.to_numpy()
    x_train, x_test = train_df.text.tolist(), test_df.text.tolist()

    # Fit and measure the classical baseline using the same holdout split.
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    xtr = vectorizer.fit_transform(x_train)
    baseline = LogisticRegression(max_iter=1000, random_state=SEED).fit(xtr, y_train)
    baseline_train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    base_pred = baseline.predict(vectorizer.transform(x_test))
    base_infer_s = (time.perf_counter() - t0) / len(x_test)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Tokenizer demonstration for two sentences, including the special tokens.
    demo = ["I love this film!", "The service was not good."]
    encoded = tokenizer(demo, padding=True, truncation=True)
    tokenizer_demo = []
    for i, sentence in enumerate(demo):
        tokenizer_demo.append({
            "sentence": sentence,
            "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][i]),
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
        })
    (OUT / "tokenizer-output.json").write_text(json.dumps(tokenizer_demo, indent=2))

    label_names = ["negative", "positive"]
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).rename_column("label", "labels")
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).rename_column("label", "labels")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(OUT / "trainer-checkpoints"),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        fp16=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    t0 = time.perf_counter()
    trainer.train()
    transformer_train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_result = trainer.evaluate()
    transformer_infer_s = (time.perf_counter() - t0) / len(test_ds)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    rows = [
        {"model": "TF-IDF + Logistic Regression", "accuracy": accuracy_score(y_test, base_pred),
         "macro_f1": f1_score(y_test, base_pred, average="macro"),
         "training_time_s": baseline_train_s, "inference_time_ms_per_example": base_infer_s * 1000},
        {"model": "DistilBERT (Trainer, 3 epochs)", "accuracy": eval_result["eval_accuracy"],
         "macro_f1": eval_result["eval_f1"], "training_time_s": transformer_train_s,
         "inference_time_ms_per_example": transformer_infer_s * 1000},
    ]
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))

    # Required pipeline demonstrations (three examples for each task).
    sentiment_pipe = pipeline("sentiment-analysis", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR))
    ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    sentiments = ["I enjoyed the film.", "The meal was awful.", "It was fine, nothing special."]
    entities = ["Mary Smith works at Acme in London.",
                "Google opened an office in Paris.",
                "Ada Lovelace was born in London."]
    (OUT / "pipeline-examples.json").write_text(json.dumps({
        "sentiment_analysis": [{"text": s, "result": r} for s, r in zip(sentiments, sentiment_pipe(sentiments))],
        "ner": [{"text": s, "entities": r} for s, r in zip(entities, ner_pipe(entities))],
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved model: {MODEL_DIR}")


if __name__ == "__main__":
    main()
