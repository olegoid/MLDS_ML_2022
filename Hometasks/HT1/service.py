import joblib
import pandas as pd
import re

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()
model = joblib.load('my_model.pkl')
ohe = OneHotEncoder()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame()
    
    df['name'] = [item.name]
    df['year'] = [item.year]
    df['selling_price'] = [item.selling_price]
    df['km_driven'] = [item.km_driven]
    df['fuel'] = [item.fuel]
    df['seller_type'] = [item.seller_type]
    df['transmission'] = [item.transmission]
    df['owner'] = [item.owner]
    df['mileage'] = [item.mileage]
    df['engine'] = [item.engine]
    df['max_power'] = [item.max_power]
    df['torque'] = [item.torque]
    df['seats'] = [item.seats]
    
    df = pd.concat([df])
    df = groom_data(df)
    df = encode_categorical_predictors(df)
    df = df.iloc[-1].to_frame().T
    
    y_pred = model.predict(df)
    
    return float(y_pred[0])


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return 'ok'

def groom_data(df):
    # clean df
    df = clean_mileage(df)
    df = clean_engine(df)
    df = clean_max_power(df)
    
    # cast engine and seats to int
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    
    # drop columns
    df = df.drop('selling_price', axis=1)
    df = df.drop('torque', axis=1)
    df = df.drop('name', axis=1)
    df = df.drop('fuel', axis=1)
    
    # use medians of max_torque_rpm and torque here to reduce complexity
    df['max_torque_rpm'] = 2400.0
    df['torque'] = 150.0
    
    return df

def clean_mileage(df):
    df['mileage'] = df['mileage'].astype(str)
    df['mileage'] = df['mileage'].map(lambda x: x.rstrip(' kmpl'))
    df['mileage'] = df['mileage'].map(lambda x: x.rstrip(' km/kg'))
    
    df['mileage'] = df['mileage'].apply(test_apply)
    
    return df

def clean_engine(df):
    df['engine'] = df['engine'].astype(str)
    df['engine'] = df['engine'].map(lambda x: x.rstrip(' C'))
    
    df['engine'] = df['engine'].apply(test_apply)
    
    return df

def clean_max_power(df):
    df['max_power'] = df['max_power'].astype(str)
    df['max_power'] = df['max_power'].map(lambda x: x.rstrip(' bhp'))

    # cast string to float
    df['max_power'] = df['max_power'].apply(test_apply)

    return df

def test_apply(x):
    try:
        return float(x)
    except ValueError:
        return None

def one_hot_encode(df, column_name, column_to_drop):
    # owner
    transformed = ohe.fit_transform(df[[column_name]])
    df[ohe.categories_[0]] = transformed.toarray()

    df = df.drop(column_name, axis=1)
    
    if column_to_drop in df.columns:
        df = df.drop(column_to_drop, axis=1)
    
    return df

def encode_categorical_predictors(df):
    # owner
    df = encode_owner(df)

    # transmission
    df = encode_transmission(df)

    # seller_type
    df = encode_seller_type(df)
    
    # seats
    df = encode_seats(df)
    
    return df

def encode_seller_type(df):
    df = one_hot_encode(df, 'seller_type', 'Trustmark Dealer')
    
    if 'Dealer' not in df.columns:
        df['Dealer'] = 0
    
    if 'Individual' not in df.columns:
        df['Individual'] = 0
    
    return df

def encode_owner(df):
    df = one_hot_encode(df, 'owner', 'First Owner')
    
    if 'Second Owner' not in df.columns:
        df['Second Owner'] = 0
    
    if 'Third Owner' not in df.columns:
        df['Third Owner'] = 0
    
    if 'Fourth & Above Owner' not in df.columns:
        df['Fourth & Above Owner'] = 0
    
    if 'Test Drive Car' not in df.columns:
        df['Test Drive Car'] = 0
    
    return df

def encode_transmission(df):
    df = one_hot_encode(df, 'transmission', 'Automatic')
    
    if 'Manual' not in df.columns:
        df['Manual'] = 0
    
    return df

def encode_seats(df):
    df['seats'] = df['seats'].astype(object)

    transformed = ohe.fit_transform(df[['seats']])
    df[ohe.categories_[0]] = transformed.toarray()

    df = df.drop('seats', axis=1)
    if 4 in df.columns:
        df = df.drop('seats', axis=1)
        
    df.rename(columns={2: '2', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 14: '14'}, inplace=True)
    
    if '2' not in df.columns:
        df['2'] = 0
    
    if '5' not in df.columns:
        df['5'] = 0
    
    if '6' not in df.columns:
        df['6'] = 0
        
    if '7' not in df.columns:
        df['7'] = 0
        
    if '8' not in df.columns:
        df['8'] = 0
        
    if '9' not in df.columns:
        df['9'] = 0
        
    if '10' not in df.columns:
        df['10'] = 0
        
    if '14' not in df.columns:
        df['14'] = 0
    
    return df