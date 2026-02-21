
import pandas as pd

df = pd.read_csv("./datasets/ridge_housing_data.csv")

# 2. Charts
#   i) SqFt_Living vs Price_USD
#   ii) Total_Rooms vs Price_USD
#   iii) Luxury_Score vs Price_USD

import matplotlib.pylab as plt

plt.figure(figsize=(8,8))
plt.scatter(x="SqM_Living",y="Price_USD", data=df,alpha=0.5, color="blue")
plt.xlabel("SqM_Living")
plt.ylabel("Price_USD")
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(x="Total_Rooms",y="Price_USD", data=df,alpha=0.5, color="blue")
plt.xlabel("Total_Rooms")
plt.ylabel("Price_USD")
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(x="Luxury_Score",y="Price_USD", data=df,alpha=0.5, color="blue")
plt.xlabel("Luxury_Score")
plt.ylabel("Price_USD")
plt.show()

#3 Feateures, Label and Split the data

features = ["SqM_Living", "Total_Rooms", "Luxury_Score"]
label = "Price_USD"
X = df[features]
y = df[label] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

#4 Train the data set using ridge regrssion

from sklearn.linear_model import Ridge, LinearRegression
model = Ridge(alpha=100)
# model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)

y_pred=model.predict(X_test)

i = 0
print('SqM_Living\tTotal_Rooms\tLuxury_Score\tPrice_USD\tPredicted_Price_USD')

while i < 10:
    print(f"{X_test.iloc[i]['SqM_Living']}\t{X_test.iloc[i]['Total_Rooms']}\t{X_test.iloc[i]['Luxury_Score']}\t{y_test.iloc[i]}\t{y_pred[i]:.2f}")
    i += 1

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted (Regression Line)')
plt.show()

from sklearn import metrics
print("--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, y_pred):.2f}")
print(f"Median Absolute Error: {metrics.median_absolute_error(y_test, y_pred):.2f}")
print(f"Explained Variance Score: {metrics.explained_variance_score(y_test, y_pred):.2f}")
print(f"R2 Score: {metrics.r2_score(y_test, y_pred):.2f}")