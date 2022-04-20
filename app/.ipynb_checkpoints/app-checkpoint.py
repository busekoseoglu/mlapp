import numpy as np
import json
import sys
import os
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
    account_amount_added_12_24m : int
    account_days_in_dc_12_24m : float
    account_days_in_rem_12_24m: float
    account_days_in_term_12_24m: float
    age: int
    avg_payment_span_0_12m : float
    merchant_category: str
    merchant_group: str
    has_paid: int
    max_paid_inv_0_12m : float
    max_paid_inv_0_24m: float
    name_in_email : str
    num_active_div_by_paid_inv_0_12m : float
    num_active_inv : int
    num_arch_dc_0_12m : int
    num_arch_dc_12_24m : int
    num_arch_ok_0_12m : int
    num_arch_ok_12_24m : int
    num_arch_rem_0_12m : int
    num_arch_written_off_0_12m : float
    num_arch_written_off_12_24m : float
    num_unpaid_bills : int
    status_last_archived_0_24m : int
    status_2nd_last_archived_0_24m : int
    status_3rd_last_archived_0_24m :int
    status_max_archived_0_6_months : int
    status_max_archived_0_12_months : int
    status_max_archived_0_24_months : int
    recovery_debt : int
    sum_capital_paid_account_0_12m : int
    sum_capital_paid_account_12_24m : int
    sum_paid_inv_0_12m : int
    time_hours : float


@app.post('/predict')
def predict_status(data : Data):
    try:
        loaded_model = joblib.load("credit_default.joblib")
        
        
        dataframe = pd.DataFrame(jsonable_encoder(data), index=[0])
        
        train_columns = ['account_amount_added_12_24m',
 'account_days_in_dc_12_24m',
 'account_days_in_rem_12_24m',
 'account_days_in_term_12_24m',
 'age',
 'avg_payment_span_0_12m',
 'has_paid',
 'max_paid_inv_0_12m',
 'max_paid_inv_0_24m',
 'num_active_div_by_paid_inv_0_12m',
 'num_active_inv',
 'num_arch_dc_0_12m',
 'num_arch_dc_12_24m',
 'num_arch_ok_0_12m',
 'num_arch_ok_12_24m',
 'num_arch_rem_0_12m',
 'num_arch_written_off_0_12m',
 'num_arch_written_off_12_24m',
 'num_unpaid_bills',
 'status_last_archived_0_24m',
 'status_2nd_last_archived_0_24m',
 'status_3rd_last_archived_0_24m',
 'status_max_archived_0_6_months',
 'status_max_archived_0_12_months',
 'status_max_archived_0_24_months',
 'recovery_debt',
 'sum_capital_paid_account_0_12m',
 'sum_capital_paid_account_12_24m',
 'sum_paid_inv_0_12m',
 'time_hours',
 'merchant_category_Automotive Parts & Accessories',
 'merchant_category_Bags & Wallets',
 'merchant_category_Body & Hair Care',
 'merchant_category_Books & Magazines',
 'merchant_category_Car electronics',
 'merchant_category_Children Clothes & Nurturing products',
 'merchant_category_Children toys',
 'merchant_category_Cleaning & Sanitary',
 'merchant_category_Collectibles',
 'merchant_category_Concept stores & Miscellaneous',
 'merchant_category_Cosmetics',
 'merchant_category_Costumes & Party supplies',
 'merchant_category_Dating services',
 'merchant_category_Decoration & Art',
 'merchant_category_Dietary supplements',
 'merchant_category_Digital services',
 'merchant_category_Diversified Health & Beauty products',
 'merchant_category_Diversified Home & Garden products',
 'merchant_category_Diversified Jewelry & Accessories',
 'merchant_category_Diversified children products',
 'merchant_category_Diversified electronics',
 'merchant_category_Diversified entertainment',
 'merchant_category_Diversified erotic material',
 'merchant_category_Electronic equipment & Related accessories',
 'merchant_category_Erotic Clothing & Accessories',
 'merchant_category_Event tickets',
 'merchant_category_Food & Beverage',
 'merchant_category_Fragrances',
 'merchant_category_Furniture',
 'merchant_category_General Shoes & Clothing',
 'merchant_category_Hobby articles',
 'merchant_category_Household electronics (whitegoods/appliances)',
 'merchant_category_Jewelry & Watches',
 'merchant_category_Kitchenware',
 'merchant_category_Music & Movies',
 'merchant_category_Musical Instruments & Equipment',
 'merchant_category_Non',
 'merchant_category_Office machines & Related accessories (excl. computers)',
 'merchant_category_Personal care & Body improvement',
 'merchant_category_Pet supplies',
 'merchant_category_Pharmaceutical products',
 'merchant_category_Plants & Flowers',
 'merchant_category_Prescription optics',
 'merchant_category_Prints & Photos',
 'merchant_category_Safety products',
 'merchant_category_Sex toys',
 'merchant_category_Sports gear & Outdoor',
 'merchant_category_Tobacco',
 'merchant_category_Tools & Home improvement',
 'merchant_category_Travel services',
 'merchant_category_Underwear',
 'merchant_category_Video Games & Related accessories',
 'merchant_category_Wheels & Tires',
 'merchant_category_Wine, Beer & Liquor',
 'merchant_category_Youthful Shoes & Clothing',
 'merchant_group_Children Products',
 'merchant_group_Clothing & Shoes',
 'merchant_group_Electronics',
 'merchant_group_Entertainment',
 'merchant_group_Erotic Materials',
 'merchant_group_Food & Beverage',
 'merchant_group_Health & Beauty',
 'merchant_group_Home & Garden',
 'merchant_group_Intangible products',
 'merchant_group_Jewelry & Accessories',
 'merchant_group_Leisure, Sport & Hobby',
 'name_in_email_F+L',
 'name_in_email_F1+L',
 'name_in_email_Initials',
 'name_in_email_L',
 'name_in_email_L1+F',
 'name_in_email_Nick',
 'name_in_email_no_match']
        
        dataframe = oneHotEncoding(dataframe, ['merchant_category', 'merchant_group','name_in_email'])
        
        for i in range(len(train_columns)):
            if train_columns[i] not in dataframe.columns.to_list():
                dataframe[train_columns[i]] = 0
        

        # Make prediction
        status = loaded_model.predict([np.array(list(dataframe.iloc[0]))])

       
        model_prediction = {
            'info': 'success',
            'status': status,
        }

    except ValueError as ve:
        model_prediction = {
            'error_code' : '-1',
            "info": str(ve)
        }

    return str(model_prediction)

        
def oneHotEncoding(df, columnsList):
    one_hot_encoded_data = pd.get_dummies(df, columns = columnsList, 
                                          drop_first=True)
    return one_hot_encoded_data




