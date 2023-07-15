from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Union
import joblib
import json

# description will apear in the doc
description = """
![GetAround](https://lever-client-logos.s3.amazonaws.com/2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png)
\n\n
[GetAround](https://www.getaround.com/?wpsrc=Google+Organic+Search) est l'Airbnb pour les voitures. Vous pouvez louer des voitures à n'importe qui pour quelques heures à quelques jours ! Fondée en 2009,
cette entreprise a connu une croissance rapide. En 2019, ils comptent plus de 5 millions d'utilisateurs et environ
\n\n
L'objectif de l'API Getaround  est d'aider les utilisateurs à estimer la valeur de location quotidienne de leur voiture en fonction des critères de celle-ci.

## Preview

* `/preview` quelques lignes aléatoires dans l'historique

## ML-Model-Prediction
 
* `/predict` insérez les détails de votre voiture pour recevoir une estimatio du prix quotidien de la voiture de locationn basée sur notre modèle de prédiction.

"""

# tags to identify different endpoints
tags_metadata = [
    {
        "name": "Preview",
        "description": "Prévisualiser les cas aléatoires dans l'ensemble de données",
    },

    {
        "name": "ML-Model-Prediction",
        "description": "Estimer le prix de location basé sur un modèle d'apprentissage automatique formé avec des données historiques et le modèle XGBoost"
    }
]

app = FastAPI(
    title="🚗 Getaround API",
    description=description,
    version="1.0",
    openapi_tags=tags_metadata
)


class PredictionFeatures(BaseModel):
    model_key: str
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


@app.get("/", tags=["Preview"])
async def random_data(rows: int = 10):
    fname = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv"
    df = pd.read_csv(fname, index_col=0, low_memory=False)
    sample = df.sample(rows)
    response = sample.to_json(orient='records')
    return response

# preparing labels that will be replaced as other
list_model_other = ['Maserati', 'Suzuki', 'Porsche', 'Ford',
                    'KIA Motors', 'Alfa Romeo', 'Fiat',
                    'Lexus', 'Lamborghini', 'Mini', 'Mazda',
                    'Honda', 'Yamaha', 'Other']

list_fuel_other = ['hybrid_petrol', 'electro', 'other']

list_color_other = ['green', 'orange', 'other']


def other_re(x, list_):
    y = x
    if x in list_:
        y = 'others'
    return y


msg = """ 
    Error! PLease check your input. It should be in json format. Example input:\n\n
    "model_key": "Volkswagen",\n
    "mileage": 17500,\n
    "engine_power": 190,\n
    "fuel": "diesel",\n
    "paint_color": "black",\n
    "car_type": "convertible",\n
    "private_parking_available": true,\n
    "has_gps": true,\n
    "has_air_conditioning": true,\n
    "automatic_car": true,\n
    "has_getaround_connect": true,\n
    "has_speed_regulator": true,\n
    "winter_tires": true\n
    """


@app.post("/predict", tags=["ML-Model-Prediction"])
async def predict(predictionFeatures: PredictionFeatures):
 
    if predictionFeatures.json:
        # Conversion des donnees en dataframe
        df = pd.DataFrame(dict(predictionFeatures), index=[0])
        
        preprocessor = joblib.load('models/preprocessor.joblib') # preprocessing model
        model = joblib.load('models/xgb_model.joblib') # xgboost model

        try:
            # pretraitement
            processed_predictionFeatures = preprocessor.transform(df)

            # application du model 
            prediction = model.predict(processed_predictionFeatures)

            # Resultat
            rental_price_per_day = prediction.tolist()[0]
            
            response = {f"Prix ​​de location prédit par jour pour votre voiture: {round(rental_price_per_day, 2)} USD"}
        except:
            response = json.dumps({"message": msg})
        return response
    else:
        msg = json.dumps({"message": msg})
        return msg

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)