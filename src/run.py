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
        try:
            return model.get_countries_data()
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/country_details/<string:country2alpha>")
class CountryDetails(Resource):
    # @api.response(200, "Success", [country_details_model])
    # @api.response(400, "Error")
    def get(self, country2alpha):
        try:
            return model.get_country_data(country=country2alpha)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/global_explanation/<string:country2alpha>")
class GlobalExplanation(Resource):
    # @api.response(200, "Success", [country_details_model])
    # @api.response(400, "Error")
    def get(self, country2alpha):
        try:
            return model.get_global_data(country=country2alpha)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/tender_details/<string:country2alpha>/<string:tender_id>")
class CountryDetails(Resource):
    # @api.response(200, "Success", [country_details_model])
    # @api.response(400, "Error")
    def get(self, country2alpha, tender_id):
        print(country2alpha, tender_id)
        try:
            return model.get_tender_data(country=country2alpha, tender_id=tender_id)
        except Exception as e:
            abort(400, str(e))


@dgcnect_ns.route("/retrain_country/<string:country2alpha>")
class RetrainCountry(Resource):
    # @api.response(200, "Success", [country_details_model])
    # @api.response(400, "Error")
    @api.expect(stop_words)
    def post(self, country2alpha):
        data = request.get_json()
        print(country2alpha, data)
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
    # @api.response(200, "Success", [country_details_model])
    # @api.response(400, "Error")
    @api.expect(annotation)
    def post(self, country2alpha, tender_id):
        data = request.get_json()
        annotation = data["Annotation"]
        print(country2alpha, annotation)
        try:
            model.annotate_tender(country2alpha, tender_id, annotation)
            print("run success")
            # try:
            return 200, "Success"
        # return model.retrain_country(country=country2alpha, stop_words=stop_words)
        except Exception as e:
            abort(400, str(e))


"""@chatbot_ns.route("/query")
class Query(Resource):
    @api.response(200, "Success", predicted_intent)
    @api.response(400, "Error")
    @api.expect(question_model)
    def post(self):
        try:
            data = request.get_json()
            text = data["QuestionText"]
            intent_id, similarity = model.query(text, 1)[0]
            return {
                "PredictedIntent": {"IntentID": intent_id, "Confidence": similarity}
            }
        except Exception as e:
            abort(400, str(e))


@chatbot_ns.route("/query/<int:top_k>")
class QueryTopK(Resource):
    @api.response(200, "Success", [predicted_intent])
    @api.response(400, "Error")
    @api.expect(question_model)
    def post(self, top_k):
        try:
            data = request.get_json()
            text = data["QuestionText"]
            intent_sim_scores = model.query(text, top_k)
            return [
                {"PredictedIntent": {"IntentID": intent_id, "Confidence": similarity}}
                for intent_id, similarity in intent_sim_scores
            ]
        except Exception as e:
            abort(400, str(e))"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
    # serve(app=app, host="0.0.0.0", port=7000)
