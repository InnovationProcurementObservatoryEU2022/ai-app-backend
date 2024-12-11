import matplotlib.pyplot as plt
import pickle
import sklearn
import os
import numpy as np
import uuid
import io
import psycopg2
import trainer
import pickle
from database_login import DBNAME, USER, PASSWORD, HOST, PORT
import codecs
from model_data import CountryModelData
from config import NUM_WORDS
from tqdm import tqdm
from database_login import TABLE_NAME


alpha2name = {
    "UK": "United Kingdom",
    "DE": "Germany",
    "HR": "Croatia",
    "AT": "Austria",
    "NL": "The Netherlands",
    "IT": "Italy",
    "FR": "France",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HU": "Hungary",
    "PT": "Portugal",
    "BG": "Bulgaria",
    "LV": "Latvia",
    "NO": "Norway",
    "EL": "Greece",
    "PL": "Poland",
    "SK": "Slovakia",
    "DK": "Denmark",
    "LT": "Lithuania",
    "ES": "Spain",
    "SI": "Slovenia",
    "CZ": "Czechia",
    "IT": "Italy",
    "CY": "Cyprus",
    "NL": "The Netherlands",
    "SE": "Sweden",
    "EE": "Estonia",
    "FI": "Finland",
    "DE": "Germany",
    "LU": "Luxembourg",
    "IE": "Ireland",
    "RO": "Romania",
    "MT": "Malta",
    "FR": "France",
    "AT": "Austria",
    "LU": "Luxembourg",
    "UK": "United Kingdom",
    "CH": "Switzerland",
}

# map country to language tokenizer
country2language = {
    "HR": "hbs",
    "BE": "nl",
    "BG": "bg",
    "HU": "hu",
    "PT": "pt",
    "LV": "lv",
    "NO": "nn",
    "EL": "el",
    "PL": "pl",
    "SK": "sk",
    "DK": "da",
    "LT": "lt",
    "ES": "es",
    "SI": "sl",
    "CZ": "cs",
    "IT": "it",
    "CY": "el",
    "NL": "nl",
    "SE": "sv",
    "EE": "et",
    "FI": "fi",
    "DE": "de",
    # "LU": "lb",  only one class
    "IE": "en",
    "RO": "ro",
    "MT": "en",
    "FR": "fr",
    "AT": "de",
    "LU": "de",
    "UK": "en",
    "CH": "fr",
}


class PostgresCountryModel:
    def __init__(self) -> None:
        print(f"Connecting to {TABLE_NAME}")
        self.connect_database()
        self.cur.execute(f"select distinct country_iso from {TABLE_NAME}")
        countries = self.cur.fetchall()
        self.close_database_connection()

        countries = [country[0] for country in countries]
        countries = list(filter(lambda country: country in country2language, countries))
        # countries = ["HR", "BE", "BG", "HU", "PT", "LV", "NO"]

        print(f"Supported countries: {countries}")

        if not os.path.exists("data"):
            os.makedirs("data")

        for country in countries:
            saved_path = os.path.join("data", f"{country}.pickle")
            if not os.path.exists(saved_path):
                country_dataset = self.fetch_dataset(country)
                language = country2language[country]
                try:
                    language_model_data = trainer.Trainer.train(
                        country_dataset, language
                    )
                    current_country_model_data = CountryModelData(
                        country,
                        {language: language_model_data},
                    )
                    current_country_model_data.save()
                    self.update_predictions(language_model_data.tender_data, country)
                except Exception as e:
                    print(
                        f"The following error occured during preprocessing for country: {country}, error: {e}"
                    )
                    del country2language[country]

        country_model_data = {}
        countries = list(filter(lambda country: country in country2language, countries))
        for country in countries:
            model_data = CountryModelData.load(country)
            country_model_data[country] = model_data
        self.country_model_data = country_model_data

        print(f"Country model data: {self.country_model_data.keys()}")

        self.detailed_country_data = {}

        self.global_data = {}
        for country in self.country_model_data:
            self.calculate_global_data(country)

    def connect_database(self):
        self.conn = psycopg2.connect(
            dbname=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
        )
        self.cur = self.conn.cursor()

    def close_database_connection(self):
        self.cur.close()
        self.conn.close()

    def update_predictions(self, tender_data, country):
        self.connect_database()
        print("Updating predictions...")
        for tender_id, prediction, predict_probas in tqdm(
            zip(
                tender_data.tender_ids,
                tender_data.predictions,
                tender_data.predict_probas,
            )
        ):
            self.cur.execute(
                f"UPDATE {TABLE_NAME} SET innovation_prediction_wo_docs={predict_probas:.5f} WHERE country_iso='{country}' AND dgcnect_tender_id={tender_id}"
            )
            prediction = int(prediction)
            self.cur.execute(
                f"UPDATE {TABLE_NAME} SET innovation_prediction={prediction} WHERE country_iso='{country}' AND dgcnect_tender_id={tender_id}"
            )
            self.conn.commit()
        self.close_database_connection()

    def retrain_country(self, country, deleted_words=[], reenabled_words=[]):
        print(f"Processing country: {country}")
        country_dataset = self.fetch_dataset(country)
        language = country2language[country]
        country_model_data = self.country_model_data[country]
        for reenabled_word in reenabled_words:
            if (
                reenabled_word
                in country_model_data.language_to_model_data[language].deleted_words
            ):
                country_model_data.language_to_model_data[
                    language
                ].deleted_words.remove(reenabled_word)
        language_model_data = trainer.Trainer.train(
            country_dataset,
            language,
            stop_words=country_model_data.language_to_model_data[language].stop_words,
            deleted_words=country_model_data.language_to_model_data[
                language
            ].deleted_words
            + deleted_words,
        )

        new_country_model_data = CountryModelData(
            country,
            {language: language_model_data},
        )
        new_country_model_data.save()

        self.country_model_data[country] = new_country_model_data
        self.calculate_global_data(country)

        tender_data = language_model_data.tender_data
        self.update_predictions(tender_data, country)
        print()

    def fetch_dataset(self, country):
        print("Fetching data...")
        self.connect_database()
        self.cur.execute(f"SELECT * FROM {TABLE_NAME} where country_iso='{country}'")
        dataset = self.cur.fetchall()
        self.close_database_connection()

        return dataset

    def fetch_tender(self, country, tender_id):
        self.connect_database()
        self.cur.execute(
            f"SELECT * FROM {TABLE_NAME} where country_iso='{country}' AND dgcnect_tender_id={tender_id}"
        )
        example = self.cur.fetchall()
        self.close_database_connection()

        return example[0]

    def infer_model(self, country, example):
        language = country2language[country]
        country_model_data = self.country_model_data[country]
        language_model_data = country_model_data.language_to_model_data[language]

        tokens, lemmatized_tokens = trainer.Trainer.return_input(example, language)
        print(len(tokens), len(lemmatized_tokens))
        features = language_model_data.vectorizer.transform(
            [" ".join(lemmatized_tokens)]
        )
        # pred = language_model_data.classifier.predict(features)[0]
        pred_proba = language_model_data.classifier.predict_proba(features)[0]
        pred_index = pred_proba.argmax(-1)
        pred = pred_proba[pred_index]

        return tokens, lemmatized_tokens, features, pred_index, pred

    def annotate_tender(self, country, tender_id, annotation):
        self.connect_database()
        self.cur.execute(
            f"UPDATE {TABLE_NAME} SET innovation_label={annotation} WHERE country_iso='{country}' AND dgcnect_tender_id={tender_id}"
        )
        self.conn.commit()
        self.close_database_connection()

        language = country2language[country]
        country_model_data = self.country_model_data[country]
        tender_data = country_model_data.language_to_model_data[language].tender_data
        tender_index = tender_data.tender_ids.index(tender_id)
        tender_data.labels[tender_index] = annotation
        country_model_data.save()
        print("annotated")

    def get_countries_data(self):
        retval = []
        print(self.country_model_data.keys())
        for key in self.country_model_data.keys():
            language = country2language[key]
            country_model_data = self.country_model_data[key]
            tender_data = country_model_data.language_to_model_data[
                language
            ].tender_data
            metadata_dict = {
                "NumExamples": tender_data.predictions.shape[0],
                "NumInnovative": tender_data.labels[tender_data.labels < 2]
                .sum()
                .item(),
                "NumNonInnovative": (
                    tender_data.predictions.shape[0]
                    - tender_data.labels[tender_data.labels < 2].sum()
                ).item(),
            }
            retval.append(
                {
                    "CountryName": alpha2name[key],
                    "Country2Alpha": key,
                    "Metadata": metadata_dict,
                }
            )
        return retval

    def calculate_details_for_country(self, country):
        language = country2language[country]
        country_model_data = self.country_model_data[country]
        tender_data = country_model_data.language_to_model_data[language].tender_data
        metadata_dict = {
            "NumExamples": tender_data.predictions.shape[0],
            "NumInnovative": tender_data.labels[tender_data.labels < 2].sum().item(),
            "NumNonInnovative": (
                tender_data.predictions.shape[0]
                - tender_data.labels[tender_data.labels < 2].sum()
            ).item(),
        }
        selected_prediction_type_dict = {
            "TruePositive": [],
            "TrueNegative": [],
            "FalsePositive": [],
            "FalseNegative": [],
            "UnlabeledPositive": [],
            "UnlabeledNegative": [],
        }
        for i, (all_label, all_pred, all_tender_id) in enumerate(
            zip(tender_data.labels, tender_data.predictions, tender_data.tender_ids)
        ):
            if all_label == 1:
                if all_label == all_pred:
                    selected_prediction_type_dict["TruePositive"].append(
                        str(all_tender_id)
                    )
                else:
                    selected_prediction_type_dict["FalseNegative"].append(
                        str(all_tender_id)
                    )
            elif all_label == 0:
                if all_label == all_pred:
                    selected_prediction_type_dict["TrueNegative"].append(
                        str(all_tender_id)
                    )
                else:
                    selected_prediction_type_dict["FalsePositive"].append(
                        str(all_tender_id)
                    )
            else:
                if all_pred == 0:
                    selected_prediction_type_dict["UnlabeledNegative"].append(
                        str(all_tender_id)
                    )
                else:
                    selected_prediction_type_dict["UnlabeledPositive"].append(
                        str(all_tender_id)
                    )
        self.detailed_country_data[country] = {
            "Metadata": metadata_dict,
            "Details": selected_prediction_type_dict,
        }

    def get_country_data(self, country):
        self.calculate_details_for_country(country=country)
        return self.detailed_country_data[country]

    def calculate_global_data(self, country, n_words=200):
        score_key = []
        language = country2language[country]
        country_model_data = self.country_model_data[country]
        language_model_data = country_model_data.language_to_model_data[language]
        tender_data = language_model_data.tender_data
        clf, vectorizer = language_model_data.classifier, language_model_data.vectorizer
        all_features, all_tender_ids = tender_data.features, tender_data.tender_ids

        score_key = []
        for key, index in vectorizer.vocabulary_.items():
            score_key.append((key, clf.coef_[0][index], index))
        score_key = sorted(score_key, reverse=True, key=lambda k: k[1])

        top_score_key_tenders = []
        for token, score, word_index in score_key[:n_words]:
            if score < 0:
                break
            tender_appears = all_features[:, word_index].nonzero()[0]
            tender_id_appears = [
                all_tender_ids[tender_appear] for tender_appear in tender_appears
            ]
            top_score_key_tenders.append((token, score, tender_id_appears))

        bottom_score_key_tenders = []
        for token, score, word_index in score_key[-n_words:]:
            if score > 0:
                break
            tender_appears = all_features[:, word_index].nonzero()[0]
            tender_id_appears = [
                all_tender_ids[tender_appear] for tender_appear in tender_appears
            ]
            bottom_score_key_tenders.append((token, score, tender_id_appears))
        bottom_score_key_tenders = bottom_score_key_tenders[::-1]

        self.global_data[country] = {
            "TopWords": top_score_key_tenders,
            "BottomWords": bottom_score_key_tenders,
            "DeletedWords": language_model_data.deleted_words,
        }

    def get_global_data(self, country):
        return self.global_data[country]

    def get_tender_data(self, country, tender_id):
        language = country2language[country]
        country_model_data = self.country_model_data[country]
        language_model_data = country_model_data.language_to_model_data[language]
        clf, vectorizer = language_model_data.classifier, language_model_data.vectorizer

        example = self.fetch_tender(country, tender_id)
        (
            original_words,
            lemma_words,
            features,
            tender_prediction,
            tender_prediction_probability,
        ) = self.infer_model(country, example)

        # preprocessed_str = vectorizer.build_preprocessor()(" ".join(original_words))
        # original_words = vectorizer.build_tokenizer()(preprocessed_str)

        # preprocessed_str = vectorizer.build_preprocessor()(" ".join(lemma_words))
        # lemma_words = vectorizer.build_tokenizer()(preprocessed_str)

        original_words = vectorizer.build_preprocessor()(
            " ".join(original_words)
        ).split(" ")
        lemma_words = vectorizer.build_preprocessor()(" ".join(lemma_words)).split(" ")

        tender_prediction = tender_prediction.tolist()
        tender_label = 2
        if example[5] is not None:
            tender_label = int(example[5])
        word_scores = features.multiply(clf.coef_[0]).tocsr()
        scored_words = []
        # total_score = features.dot(clf.coef_[0])
        word_score = {}
        lemma_original = {}
        for i, (original_word, lemma_word) in enumerate(
            zip(original_words, lemma_words)
        ):
            score = 0.0
            if lemma_word in vectorizer.vocabulary_:
                word_index = vectorizer.vocabulary_[lemma_word]
                score = word_scores[0, word_index].item()

            if lemma_word not in word_score:
                word_score[lemma_word] = 0
            if lemma_word not in lemma_original:
                lemma_original[lemma_word] = []
            word_score[lemma_word] = score
            lemma_original[lemma_word].append(original_word)

            if original_word == "of" and score != 0:
                print(original_word, score, lemma_word)
            scored_words.append([original_word, score])

        fig, ax = plt.subplots()
        word_score = dict(sorted(word_score.items(), key=lambda k: k[1]))
        vis_words = []
        current_sum = 0
        bias = np.array(clf.intercept_)
        ax.axvline(x=-bias[0], color="red", label="decision boundary")
        ax.text(
            -bias[0] + 0.05, 5, "decision boundary", rotation=90, color="r", va="center"
        )
        ax.axvline(x=0, color="black", label="zero", linestyle="dashed")

        current_word_index = 0
        top_keys = list(word_score.keys())  # [-NUM_WORDS:]
        top_keys.reverse()
        positive_other_sum = 0.0
        for i, lemma_word in enumerate(top_keys):
            # if word_score[lemma_word] == 0:
            #    continue

            if i < NUM_WORDS and word_score[lemma_word] > 0:
                ax.barh(
                    current_word_index,
                    current_sum + word_score[lemma_word],
                    align="center",
                    color="r",
                )
                ax.barh(current_word_index, current_sum, align="center", color="white")
                current_sum += word_score[lemma_word]
                vis_words.append(lemma_original[lemma_word][0].lower())
                ax.text(
                    current_sum + 0.1,
                    current_word_index,
                    str(round(word_score[lemma_word], 2)),
                    color="r",
                    va="center",
                )
                current_word_index += 1
            elif word_score[lemma_word] > 0:
                positive_other_sum += word_score[lemma_word]
            else:
                break
        ax.barh(NUM_WORDS, current_sum + positive_other_sum, align="center", color="r")
        ax.barh(NUM_WORDS, current_sum, align="center", color="white")
        current_sum += positive_other_sum
        vis_words.append("remaining POSITIVE")
        ax.text(
            current_sum + 0.1,
            current_word_index,
            str(round(positive_other_sum, 2)),
            color="r",
            va="center",
        )
        current_word_index += 1

        bot_keys = list(word_score.keys())  # [:NUM_WORDS]
        negative_other_sum = 0.0
        # bot_keys.reverse()
        for i, lemma_word in enumerate(bot_keys):
            # if word_score[lemma_word] == 0:
            #    continue
            if i < NUM_WORDS and word_score[lemma_word] < 0:
                if current_sum > 0:
                    zero_to_pos = current_sum + word_score[lemma_word]
                    ax.barh(current_word_index, current_sum, align="center", color="b")
                    if zero_to_pos > 0:
                        ax.barh(
                            current_word_index,
                            zero_to_pos,
                            align="center",
                            color="white",
                        )
                    else:
                        ax.barh(
                            current_word_index, current_sum, align="center", color="b"
                        )
                        ax.barh(
                            current_word_index, zero_to_pos, align="center", color="b"
                        )
                else:
                    ax.barh(
                        current_word_index,
                        current_sum + word_score[lemma_word],
                        align="center",
                        color="b",
                    )
                    ax.barh(
                        current_word_index, current_sum, align="center", color="white"
                    )
                ax.text(
                    current_sum + 0.1,
                    current_word_index,
                    str(round(word_score[lemma_word], 2)),
                    color="blue",
                    va="center",
                )
                current_word_index += 1
                current_sum += word_score[lemma_word]
                vis_words.append(lemma_original[lemma_word][0].lower())
            elif word_score[lemma_word] < 0:
                negative_other_sum += word_score[lemma_word]
            else:
                break
        # draw remainder of negative
        if current_sum > 0:
            zero_to_pos = current_sum + negative_other_sum
            ax.barh(current_word_index, current_sum, align="center", color="b")
            if zero_to_pos > 0:
                ax.barh(current_word_index, zero_to_pos, align="center", color="white")
            else:
                ax.barh(current_word_index, current_sum, align="center", color="b")
                ax.barh(current_word_index, zero_to_pos, align="center", color="b")
        else:
            ax.barh(
                current_word_index,
                current_sum + negative_other_sum,
                align="center",
                color="b",
            )
            ax.barh(current_word_index, current_sum, align="center", color="white")
        ax.text(
            current_sum + 0.1,
            current_word_index,
            str(round(negative_other_sum, 2)),
            color="blue",
            va="center",
        )
        current_word_index += 1
        current_sum += negative_other_sum
        vis_words.append("remaining NEGATIVE")

        ax.set_yticks(
            np.linspace(0, current_word_index + 1, current_word_index + 1),
            labels=vis_words + [""],
        )
        ax.invert_yaxis()
        fig.tight_layout()

        filename = uuid.uuid4()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_image = buf.read().hex()
        b64_image = codecs.encode(codecs.decode(byte_image, "hex"), "base64").decode()
        plt.close()

        return {
            "WordScores": scored_words,
            "Plot": b64_image,
            "Prediction": tender_prediction,
            "PredictionProbability": tender_prediction_probability,
            "Label": tender_label,
        }
