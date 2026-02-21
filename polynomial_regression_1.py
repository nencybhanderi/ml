
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv("./datasets/polynomial_fuel_data.csv")

plt.figure(figsize=(12, 12))
plt.title("Speed_kmh VS Fuel_Consumed")
plt.xlabel("Speed_kmh")
plt.ylabel("Fuel_Consumed")
plt.scatter(
    x="Speed_kmh",
    y="Fuel_Consumed",
    data=df,
    color="blue",
    alpha=0.4
)

plt.show()

features = ["Speed_kmh"]
label = "Fuel_Consumed"

pre = PolynomialFeatures(degree=2)
X_poly = pre.fit_transform(df[features])
print(X_poly)

X = X_poly
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = LinearRegression()
model.fit(X_train,y_train)

print(model.coef_)

y_pred=model.predict(X_test)

i = 0
print('Speed_kmh\tFuel_Consumed\tPredicted_Fuel_Consumed')

while i < 10:
    print(f"{y_test.iloc[i]}\t{y_pred[i]:.2f}")
    i += 1

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Fuel')
plt.ylabel('Predicted Fuel')
plt.title('Actual vs. Predicted (Regression Line)')
plt.show()

print("--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, y_pred):.2f}")
print(f"Median Absolute Error: {metrics.median_absolute_error(y_test, y_pred):.2f}")
print(f"Explained Variance Score: {metrics.explained_variance_score(y_test, y_pred):.2f}")
print(f"R2 Score: {metrics.r2_score(y_test, y_pred):.2f}")