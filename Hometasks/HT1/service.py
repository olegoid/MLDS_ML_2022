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
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

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
    
    df['year'] = [item.year]
    df['selling_price'] = [item.selling_price]
    df['km_driven'] = [item.km_driven]
    df['fuel'] = [item.fuel]
    df['seller_type'] = [item.seller_type]
    df['transmission'] = [item.transmission]
    df['owner'] = [item.owner]
    df['mileage'] = [item.mileage]
    df['engine'] = [item.engine]
    df['max_power'] = [item.engine]
    df['torque'] = [item.torque]
    df['seats'] = [item.seats]
    
    df = pd.concat([df_test, df])
    df = groom_data(df)
    df = encode_categorical_predictors(df)
    df = df.iloc[-1]
    
    print(df)
    
    return 'ok'


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return 'ok'

def groom_data(df):
    # clean df
    df = clean_mileage(df)
    df = clean_engine(df)
    df = clean_max_power(df)
    df = fill_the_gaps(df)
    
    # cast engine and seats to int
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    
    # drop_selling_price
    df = df.drop('selling_price', axis=1)
    df = df.drop('torque', axis=1)
    
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
    df = df.drop(column_to_drop, axis=1)
    
    return df

def encode_categorical_predictors(df):
    # owner
    df = one_hot_encode(df, 'owner', 'First Owner')

    # transmission
    df = one_hot_encode(df, 'transmission', 'Automatic')

    # seller_type
    df = one_hot_encode(df, 'seller_type', 'Trustmark Dealer')
    
    df['seats'] = df['seats'].astype(object)

    transformed = ohe.fit_transform(df[['seats']])
    df[ohe.categories_[0]] = transformed.toarray()

    df = df.drop('seats', axis=1)
    df = df.drop(4, axis=1)
    df.rename(columns={2: '2', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 14: '14'}, inplace=True)
    
    return df
    
def fill_the_gaps(df):
    # mileage
    median = df['mileage'].median()
    df['mileage'].fillna(median, inplace=True)

    # engine
    median = df['engine'].median()
    df['engine'].fillna(median, inplace=True)

    # max_power
    median = df['max_power'].median()
    df['max_power'].fillna(median, inplace=True)

    # seats
    median = df['seats'].median()
    df['seats'].fillna(median, inplace=True)
    
    return df