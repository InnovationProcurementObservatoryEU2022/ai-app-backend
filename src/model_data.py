import os
import pickle


class LanguageModelData:
    """Class that stores required objects for model training/inference on a specific language."""

    def __init__(self, classifier, vectorizer, stop_words, deleted_words, tender_data):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.stop_words = stop_words
        self.deleted_words = deleted_words
        self.tender_data = tender_data


class TenderData:
    """Class that stores data connected to individual tenders. Used for frontend visualization."""

    def __init__(self, features, predictions, predict_probas, labels, tender_ids):
        self.features = features
        self.predictions = predictions
        self.predict_probas = predict_probas
        self.labels = labels
        self.tender_ids = tender_ids


class CountryModelData:
    """Helper class for mapping countries to their respective model trained on individual language"""

    @classmethod
    def load(cls, country, save_start_path="./data"):
        with open(os.path.join(save_start_path, country + ".pickle"), "rb") as f:
            load_country_model_data = pickle.load(f)
        return load_country_model_data

    def __init__(self, country, language_to_model_data, save_start_path="./data"):
        self.country = country
        self.language_to_model_data = language_to_model_data
        self.save_start_path = save_start_path

    def save(self):
        """Save this object to a file."""
        with open(
            os.path.join(self.save_start_path, self.country + ".pickle"), "wb"
        ) as f:
            pickle.dump(self, f)
