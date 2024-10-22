import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('food_delivery_dataset.csv')


print(data)
data['Delivery Time'] = data['Delivery Time'].fillna(data['Delivery Time'].median())
data['Average_Cost'] = data['Average_Cost'].replace({'₹': '', ',': ''}, regex=True).astype(float)
data['Minimum_Order'] = data['Minimum_Order'].replace({'₹': '', ',': ''}, regex=True).astype(float)

label_encoder = LabelEncoder()
data['Cuisines'] = label_encoder.fit_transform(data['Cuisines'])
data['Location'] = label_encoder.fit_transform(data['Location'])
data['Restaurant'] = label_encoder.fit_transform(data['Restaurant'])


X = data[['Cuisines', 'Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews', 'Location', 'Restaurant']]
y = data['Delivery Time']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
