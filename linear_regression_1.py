import pandas as pd

df=pd.read_csv("./datasets/advanced_housing_data.csv")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="Square_Feet",
    y="Price_USD",
    data=df,
    alpha=0.5
)

plt.title('square feet vs price usd')
plt.xlabel('square feet')
plt.ylabel('price usd')
plt.show()

features = ['Square_Feet','House_Age']
label='Price_USD'

x = df[features]
y = df[label]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=35)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)
#print(y_pred)
i = 0
print('Square_Feet\tHouse_Age\tPrice_USD\tPredicted_Price_USD')

while i < 10:
    print(f"{X_test.iloc[i]['Square_Feet']}\t{X_test.iloc[i]['House_Age']}\t{Y_test.iloc[i]}\t{y_pred[i]:.2f}")
    i += 1

plt.figure(figsize=(10,5))
plt.scatter(Y_test, y_pred, alpha=0.5, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted (Regression Line)')
plt.show()

from sklearn import metrics
print("--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(Y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(Y_test, y_pred):.2f}")
print(f"Median Absolute Error: {metrics.median_absolute_error(Y_test, y_pred):.2f}")
print(f"Explained Variance Score: {metrics.explained_variance_score(Y_test, y_pred):.2f}")
print(f"R2 Score: {metrics.r2_score(Y_test, y_pred):.2f}")