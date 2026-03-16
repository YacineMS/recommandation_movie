import mlflow.pyfunc
import pandas as pd


class SurpriseSVDWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, algo):
        self.algo = algo

    def predict(self, context, model_input: pd.DataFrame):
        if not {"userid", "movieid"}.issubset(model_input.columns):
            raise ValueError("Le DataFrame doit contenir 'userid' et 'movieid'")

        preds = []
        for _, row in model_input.iterrows():
            # garder le type identique à l'entraînement
            est = self.algo.predict(row["userid"], row["movieid"]).est
            preds.append(est)

        return pd.Series(preds)
