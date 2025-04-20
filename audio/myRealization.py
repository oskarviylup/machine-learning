import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from itertools import product

class CustomLinearRegression:
    def __init__(self, fit_intercept=True, alpha=0.0, normalize=False):
        self.fit_intercept = fit_intercept
        self.alpha = alpha  # L2-регуляризация
        self.normalize = normalize
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = X.copy()
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец единиц

        # Решение через нормальное уравнение + L2-регуляризация
        I = np.eye(X.shape[1])
        I[0, 0] = 0  # Не регуляризируем свободный член
        self.coefficients = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]

    def predict(self, X):
        X = X.copy()
        if self.normalize:
            X = self.scaler.transform(X)

        if self.fit_intercept:
            return X @ self.coefficients + self.intercept
        else:
            return X @ self.coefficients



df = pd.read_csv('final_data_set_ready_for_model.csv')

x = df.drop(columns=['price'])
df['price'] = np.log1p(df['price'])
y = df['price']


x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=df['engine_type'])
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42,
                                                stratify=x_temp['engine_type'])


one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
one_hot_encoder.fit(df[['brand', 'engine_type']])

# Трансформируем каждую выборку
encoded_train = one_hot_encoder.transform(x_train[['brand', 'engine_type']]).toarray()
encoded_val = one_hot_encoder.transform(x_val[['brand', 'engine_type']]).toarray()
encoded_test = one_hot_encoder.transform(x_test[['brand', 'engine_type']]).toarray()

# Преобразуем в DataFrame с одинаковыми именами столбцов
columns = one_hot_encoder.get_feature_names_out(['brand', 'engine_type'])
encoded_train = pd.DataFrame(encoded_train, columns=columns, index=x_train.index)
encoded_val = pd.DataFrame(encoded_val, columns=columns, index=x_val.index)
encoded_test = pd.DataFrame(encoded_test, columns=columns, index=x_test.index)

# Объединяем с остальными признаками
x_train = pd.concat([x_train.drop(columns=['brand', 'engine_type']), encoded_train], axis=1)
x_val = pd.concat([x_val.drop(columns=['brand', 'engine_type']), encoded_val], axis=1)
x_test = pd.concat([x_test.drop(columns=['brand', 'engine_type']), encoded_test], axis=1)

# Сетка гиперпараметров
fit_intercept_options = [True, False]
alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]  # L2-регуляризация
normalize_options = [True, False]

best_mse = float('inf')
best_params = None
best_model = None

for fit_intercept, alpha, normalize in product(fit_intercept_options, alpha_values, normalize_options):
    model = CustomLinearRegression(fit_intercept=fit_intercept, alpha=alpha, normalize=normalize)
    model.fit(x_train.values, y_train.values)
    val_predictions = model.predict(x_val.values)

    mse = mean_squared_error(y_val, val_predictions)

    if mse < best_mse:
        best_mse = mse
        best_params = {'fit_intercept': fit_intercept, 'alpha': alpha, 'normalize': normalize}
        best_model = model

print(f"Best parameters: {best_params}")
print(f"Best validation MSE: {best_mse}")

test_predictions = best_model.predict(x_test.values)
mse_test = mean_squared_error(y_test, test_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)

print(f"Test MSE: {mse_test}, Test MAE: {mae_test}")