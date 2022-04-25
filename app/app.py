import pandas as pd
import joblib
from fastapi import FastAPI
import uvicorn
from pydantic.main import BaseModel
from fastapi.encoders import jsonable_encoder

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the API'}

class Data(BaseModel):
    account_days_in_rem_12_24m: int
    account_days_in_term_12_24m: int
    age: int
    avg_payment_span_0_12m : int
    merchant_category: str
    merchant_group: str
    has_paid: int
    max_paid_inv_0_12m : int
    max_paid_inv_0_24m: int
    name_in_email : str
    num_active_div_by_paid_inv_0_12m : int
    num_arch_dc_0_12m : int
    num_arch_dc_12_24m : int
    num_arch_ok_0_12m : int
    num_arch_ok_12_24m : int
    num_unpaid_bills : int
    status_last_archived_0_24m : int
    status_2nd_last_archived_0_24m : int
    status_3rd_last_archived_0_24m :int
    status_max_archived_0_12_months : int
    status_max_archived_0_24_months : int
    sum_paid_inv_0_12m : int



@app.post('/predict')
def predict_status(data : Data):
    try:
        loaded_model = joblib.load("credit_default.joblib")
        
        
        df = pd.DataFrame(jsonable_encoder(data), index=[0])

        print(df)
        

        # Make prediction
        status = loaded_model.predict(df)
        prob = loaded_model.predict_proba(df)[:,1]

        if status[0] == 0:
            result = "Negative"
        elif status[0] == 1:
            result = "Pozitive"

       
        model_prediction = {
            'info': 'success',
            'status': status[0],
            'prob' : prob[0],
            'result' : result
        }

    except ValueError as ve:
        model_prediction = {
            'error_code' : '-1',
            "info": str(ve)
        }

    return str(model_prediction)


if __name__ == "__main__":
    # Changed it from app to 'main:app' to reload changes automatically.
    uvicorn.run('app:app', host="0.0.0.0", port=80, reload=True)