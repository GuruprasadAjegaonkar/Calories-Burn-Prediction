import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
data = pd.read_csv('calories.csv')


label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])


X = data.drop(columns=['User_ID', 'Calories'])
y = data['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'calories_burnt_model.pkl')
joblib.dump(label_encoder, 'gender_encoder.pkl')

print("Model and label encoder saved successfully!")
