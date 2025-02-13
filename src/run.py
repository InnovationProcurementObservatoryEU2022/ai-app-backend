from flask import Flask, request, abort
from flask_restx import Resource, Api, fields
from model import PostgresCountryModel
from waitress import serve
from flask_cors import CORS
import time


app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version="1.0",
    title="DGCNECT Tender Visualization",
    description="API for visualizing model outputs",
)

model = PostgresCountryModel()

dgcnect_ns = api.namespace(
    "dgcnect", description="Used to visualize specific outputs of the TF-IDF model"
)

countries_model = api.model(
    "Country", {"CountryName": fields.String, "Country2Alpha": fields.String}
)

country_details_model = api.model(
    "CountryDetails", {"Country": countries_model, "Details": fields.Arbitrary}
)

stop_words = api.model("StopWords", {"StopWords": fields.List(fields.String)})
annotation = api.model("Annotation", {"Annotation": fields.Integer})

question_model = api.model("Question", {"QuestionText": fields.String})
predicted_intent = api.model(
    "PredictedIntent", {"IntentID": fields.String, "Confidence": fields.Float}
)
resource_fields = api.model(
    "Intent",
    {
        "IntentID": fields.String,
        "Questions": fields.List(fields.Nested(question_model)),
    },
)


@dgcnect_ns.route("/countries_data")
class CountryData(Resource):
    @api.response(200, "Success", [countries_model])
    @api.response(400, "Error")
    def get(self):
        """Get descriptives for all countries (number of examples, number of (non)innovative tenders, etc.)

        Returns:
            Dict: Descriptives for a country used for frontend
        """
        try:
            return model.get_countries_data()
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/country_details/<string:country2alpha>")
class CountryDetails(Resource):
    def get(self, country2alpha: str):
        """Fetch stats for a single country

        Args:
            country2alpha (str): Country to fetch stats for

        Returns:
            Dict: Fetched stats
        """
        try:
            return model.get_country_data(country=country2alpha)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/global_explanation/<string:country2alpha>")
class GlobalExplanation(Resource):
    def get(self, country2alpha: str):
        """Get global importance scores for a country

        Args:
            country2alpha (str): Country to fetch the global data for

        Returns:
            Dict: Global data"""
        try:
            return model.get_global_data(country=country2alpha)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/tender_details/<string:country2alpha>/<string:tender_id>")
class CountryDetails(Resource):
    def get(self, country2alpha: str, tender_id: str):
        """Get data used for single tender visualization. Includes per-token importances,
        prediction information and an image of the importance plot for that tender.

        Args:
            country (str): Country of the tender.
            tender_id (str): Tender ID

        Returns:
            Dict: Data used for single tender visualization."""
        try:
            return model.get_tender_data(country=country2alpha, tender_id=tender_id)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/retrain_country/<string:country2alpha>")
class RetrainCountry(Resource):
    @api.expect(stop_words)
    def post(self, country2alpha: str):
        """Retrain the model for a particular country. Optionally disable tokens given by deleted_words,
        and reenable disabled tokens via reenabled_words (these two parameters are connected to the global token importances)

        Args:
            country2alpha (str): Country to retrain
            stop_words (StopWords): Words to remove and/or to reenable in the vocab"""
        data = request.get_json()
        try:
            if "ReEnabledWords" not in data:
                reenabled_words = []
            else:
                reenabled_words = data["ReEnabledWords"]
            model.retrain_country(
                country=country2alpha,
                deleted_words=data["StopWords"],
                reenabled_words=reenabled_words,
            )
            return 200, "Success"
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/annotate_tender/<string:country2alpha>/<string:tender_id>")
class AnnotateTender(Resource):
    @api.expect(annotation)
    def post(self, country2alpha: str, tender_id: str):
        """Manually annotate a tender and save its label to the database

        Args:
            country (str): Country of the tender
            tender_id (str): Tender ID
            annotation (Annotation): Label (0 or 1) (non-innovative or innovative)
        """
        data = request.get_json()
        annotation = data["Annotation"]
        try:
            model.annotate_tender(country2alpha, tender_id, annotation)
            return 200, "Success"
        except Exception as e:
            abort(400, str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
    # serve(app=app, host="0.0.0.0", port=7000)
