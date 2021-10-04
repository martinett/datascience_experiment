from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

def load_data():
    df = pd.read_csv("./data/raw/mars-2014-complete.csv", sep=";", decimal=",", encoding="latin-1")
    return df

def clean_data(df):
    df = df.drop(columns=['date_maj'])
    df = df[df['co2'].isna()==False]
    df = df.dropna(axis=1)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    return df

def split_xy(df):
    y = df['co2']
    x = df.drop(columns=['co2'])
    return x,y

def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=56)
    return x_train, x_test, y_train, y_test

def setup_model(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)

st.title('Datascience Experiment - random forest')
df = load_data()
st.write(df)
df = clean_data(df)
x, y = split_xy(df)
x_train, x_test, y_train, y_test = split_train_test(x, y)
y_pred = setup_model(x_train, x_test, y_train, y_test)
res = pd.DataFrame({"pred": y_pred, "y": y_test})
res["erreur"] = res.apply(lambda x : abs(x["pred"]-x["y"]), axis=1)
st.write(res)
st.write("La moyenne des erreurs est de " + str(res["erreur"].mean()))