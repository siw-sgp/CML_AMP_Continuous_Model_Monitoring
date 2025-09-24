# ###########################################################################
#  CDSW version of AMP predict.py
# ###########################################################################

import pickle
import pandas as pd
import cdsw   # use cdsw instead of cml
from src.utils import col_order

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict(data_input):
    """
    CDSW predict function
    """

    # Convert dict representation back to dataframe for inference
    df = pd.DataFrame.from_records([data_input["record"]])
    df = df[col_order].drop("price", axis=1)

    # Track raw input values of features
    active_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "waterfront",
        "zipcode",
        "condition",
        "view",
    ]
    cdsw.track_metric(
        "input_features", df[active_features].to_dict(orient="records")[0]
    )

    # Run prediction
    result = model.predict(df).item()

    # Track prediction
    cdsw.track_metric("predicted_result", result)

    # Return result in CDSW-style (dict is fine)
    return {"prediction": result}
