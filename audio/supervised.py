import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

df = pd.read_csv('final_data_set_ready_for_model.csv')

x = df.drop(columns=['price'])
df['price'] = np.log1p(df['price'])
y = df['price']

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42, stratify=df['engine_type']
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=x_temp['engine_type']
)
print('ytrain', y_train)
# Обучаем OneHotEncoder на всем датасете
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



def baseline():
    # mean
    mean_price = y_train.mean()

    baseline_predictions_mean = [mean_price] * len(y_val)

    mse_mean = mean_squared_error(y_val, baseline_predictions_mean)
    mae_mean = mean_absolute_error(y_val, baseline_predictions_mean)
    print(f"Baseline (Mean) -> MSE: {mse_mean}, MAE: {mae_mean}")

    # median
    median_price = y_train.median()

    baseline_predictions_median = [median_price] * len(y_val)

    mse_median = mean_squared_error(y_val, baseline_predictions_median)
    mae_median = mean_absolute_error(y_val, baseline_predictions_median)
    print(f"Baseline (Median) -> MSE: {mse_median}, MAE: {mae_median}")


def linear_model():
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(x_train, y_train)

    val_predictions = linear_regression_model.predict(x_val)

    mse = mean_squared_error(y_val, val_predictions)
    mae = mean_absolute_error(y_val, val_predictions)

    print(f"Validation MSE: {mse}, Validation MAE: {mae}")


def grid_search():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())  # Используем Ridge-регрессию вместо обычной
    ])

    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(x_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MSE: {-grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    val_predictions_gs = best_model.predict(x_val)
    mse_gs = mean_squared_error(y_val, val_predictions_gs)
    print(f"Validation (GridSearch) MSE: {mse_gs}")


def evaluate_on_test():
    best_model = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
        {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'model__fit_intercept': [True, False]},
        scoring='neg_mean_squared_error',
        cv=5
    ).fit(x_train, y_train).best_estimator_

    test_predictions = best_model.predict(x_test)
    mse_test = mean_squared_error(y_test, test_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)

    print(f"Test MSE: {mse_test}, Test MAE: {mae_test}")

print('xtrain', x_train.dtypes)
print(x_val.dtypes)
print(x_test.dtypes)

baseline()
linear_model()
grid_search()
evaluate_on_test()