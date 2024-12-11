import random
from cleantext import clean
from simplemma import simple_tokenizer, lemmatize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
from multiprocessing import Pool
from model_data import TenderData, CountryModelData, LanguageModelData
from sklearn.dummy import DummyClassifier
import os
from database_login import TABLE_NAME


RANDOM_SEED = 69
MAX_NUM_CHARACTERS = 50000


def clean_text(t):
    t = re.sub(r"<.*?>", " ", t)
    # t = re.sub(r"[^a-zA-Z0-9.!,?;: ]+", " ", t)
    t = re.sub(" +", " ", t)
    t = t.lower()
    t = t[:MAX_NUM_CHARACTERS]
    return t


class Trainer:
    def return_input(example, language):
        text = (
            example[3] + example[4]
            if TABLE_NAME == "dataset"
            else example[2] + example[3]
        )
        text = clean_text(text)

        tokens = simple_tokenizer(text)
        lemmatized_tokens = [lemmatize(token, lang=language) for token in tokens]
        return tokens, lemmatized_tokens

    def check_example(example):
        if TABLE_NAME == "dataset":
            if (example[2] is None and example[3] is None and example[4] is None) or (
                example[2] == "" and example[3] == "" and example[4] == ""
            ):
                return False
        elif TABLE_NAME == "inference":
            if (example[2] is None and example[3] is None) or (
                example[2] == "" and example[3] == ""
            ):
                return False
        return True

    def train(dataset, language, stop_words=[], deleted_words=[]):
        print(deleted_words)
        examples = []
        inference_examples = []
        print("Cleaning data...")
        for example in tqdm(dataset):
            if not Trainer.check_example(example):
                continue
            tokens, lemmatized_tokens = Trainer.return_input(example, language)
            if example[5] is not None:
                examples.append(
                    {
                        "original": " ".join(tokens),
                        "input_text": " ".join(lemmatized_tokens),
                        "label": int(example[5]),
                        "tender_id": str(example[7]),
                    }
                )
            else:
                inference_examples.append(
                    {
                        "original": " ".join(tokens),
                        "input_text": " ".join(lemmatized_tokens),
                        "label": 2,
                        "tender_id": str(example[7]),
                    }
                )

        train_ratio = 0.8
        random.seed(RANDOM_SEED)
        random.shuffle(examples)

        print(len(examples), len(inference_examples))

        num_train = int(len(examples) * train_ratio)
        train_examples = examples  # [:num_train] taking everything for train
        test_examples = inference_examples

        train_texts = [example["input_text"] for example in train_examples]
        train_labels = np.array([example["label"] for example in train_examples])
        test_texts = [example["input_text"] for example in test_examples]

        if len(stop_words + deleted_words) == 0:
            print("Obtaining stop words...")
            vectorizer = TfidfVectorizer(max_df=0.05, min_df=2)
            _ = vectorizer.fit_transform(train_texts)
            stop_words = list(vectorizer.stop_words_)

        print("Training model...")
        vectorizer = TfidfVectorizer(stop_words=stop_words + deleted_words)
        train_features = vectorizer.fit_transform(train_texts)

        clf = LogisticRegression(
            random_state=RANDOM_SEED, class_weight="balanced", C=0.3
        ).fit(train_features, train_labels)

        all_texts = [example["input_text"] for example in examples + inference_examples]
        all_tender_ids = [
            example["tender_id"] for example in examples + inference_examples
        ]
        all_labels = np.array(
            [example["label"] for example in examples + inference_examples]
        )
        all_features = vectorizer.transform(all_texts)
        all_preds = clf.predict(all_features)
        all_predict_probas = clf.predict_proba(all_features)[:, 1]

        tender_data = TenderData(
            all_features, all_preds, all_predict_probas, all_labels, all_tender_ids
        )
        language_model_data = LanguageModelData(
            clf, vectorizer, stop_words, deleted_words, tender_data
        )
        print("Success")

        return language_model_data
