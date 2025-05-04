import ydf
import numpy as np
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Cargar los modelos
model_relaks = ydf.load_model("relaks_v03")

api = Api(app, title="Relaks API", version="1.0", description="API to predict categories of relaxation techniques.")

# Namespace para Relaks
relaks_ns = api.namespace("Relaks", description="Endpoints for the Relaks model.")

# Modelo de datos esperado para Relaks
relaks_model = api.model("RelaksInput", {
    "age": fields.Integer(required=True, description="Age ( '0'('>=30'), '1'('<30') )"),
    "gender": fields.Integer(required=True, description="Gender ( '0'('FEMALE') , 1'('MALE') )"),
    "spo2": fields.Float(required=True, description="Blood oxygen level"),
    "bpm": fields.Float(required=True, description="Heart rate"),
    "sleep_duration": fields.Float(required=True, description="Sleep duration in milliseconds"),
    "ALERT": fields.Integer(required=True, description="Alert status ('0' , '1')"),
    "HAPPY": fields.Integer(required=True, description="Happiness status ('0' , '1')"),
    "SAD": fields.Integer(required=True, description="Sadness status ('0' , '1')"),
    "TENSE/ANXIOUS": fields.Integer(required=True, description="Anxiety status ('0' , '1')"),
    "TIRED": fields.Integer(required=True, description="Tiredness status ('0' , '1')"),
    "stai_stress_category": fields.Integer(required=True, description="STAI stress category ( '1'('Below Average') , '2'('Average'), '3'('Over Average') )"),
})

# Accuracy estático
RELASK_ACCURACY = 0.8184281842818428 # 81.8%

@relaks_ns.route("/predict")
class RelaksPredict(Resource):
    @relaks_ns.expect(relaks_model)
    def post(self):

        try:
            data = request.get_json()
            input_data = {
                "AGE": [data['age']],
                "GENDER": [data['gender']],
                "spo2": [data['spo2']],
                "bpm": [data['bpm']],
                "sleep_duration": [data['sleep_duration']],
                "ALERT": [data['ALERT']],
                "HAPPY": [data['HAPPY']],
                "SAD": [data['SAD']],
                "TENSE/ANXIOUS": [data['TENSE/ANXIOUS']],
                "TIRED": [data['TIRED']],
                "stai_stress_category": [data['stai_stress_category']],
            }
            prediction_probabilities = model_relaks.predict(input_data)
            class_labels = [3, 2, 1]
            predicted_index = np.argmax(prediction_probabilities, axis=1)[0]
            predicted_class = class_labels[predicted_index]
            return {"predicted_category": predicted_class}
        except Exception as e:
            return {"error": str(e)}, 500
        

@relaks_ns.route("/accuracy")
class RelaksAccuracy(Resource):
    def get(self):
        return {"model": "Relaks", "accuracy": f"{RELASK_ACCURACY * 100:.1f}%"}

if __name__ == '__main__':
    port = int(os.environ.get(8080, 5000))  # Toma el puerto de Railway o usa 5000
    app.run(host="0.0.0.0", port=8080)