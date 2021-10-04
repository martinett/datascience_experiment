import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised import AutoML

df_co2 = pd.read_csv("./data/raw/mars-2014-complete.csv", sep=";", encoding="latin-1")
# print(df_co2.shape)

df_co2 = df_co2.drop(columns=['date_maj'])
df_co2 = df_co2[df_co2['co2'].isna()==False]

y = df_co2['co2'].values
x = df_co2.drop(columns=['co2'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = AutoML(total_time_limit=5*60, mode='Explain', random_state=42, ml_task='regression')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
model.report()
