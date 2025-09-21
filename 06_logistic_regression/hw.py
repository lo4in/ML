import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/titanic.csv')



df = df[['Survived', 'Sex', 'Age', 'Fare', 'Pclass']]
df['Age'] = df['Age'].fillna(df['Age'].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df.head())
Y = np.array(df['Survived'])
X = df[['Sex', 'Age', 'Fare', 'Pclass']]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_sc = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    df_sc, Y, test_size=0.2, random_state=42, stratify=Y
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(Y_pred)

# plt.plot(Y_test)
# plt.plot(Y_pred)
# plt.show()


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Коэффициенты модели:", model.coef_)
print("Свободный член:", model.intercept_)