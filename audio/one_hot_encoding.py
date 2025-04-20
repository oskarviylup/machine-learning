import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('final_data_set_ready_for_model.csv')


def one_hot_encoding(data, columns):

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    transformed_data = []

    for col in columns:
        transformed_column = one_hot_encoder.fit_transform(data[[col]])
        column_names = [f"{col}_{category}" for category in one_hot_encoder.categories_[0]]

        transformed_columns = pd.DataFrame(transformed_column, columns=column_names, index=data.index)

        transformed_data.append(transformed_columns)

    data = pd.concat([data] + transformed_data, axis=1)

    data = data.drop(columns=columns)

    return data

df = one_hot_encoding(df, ['brand', 'engine_type'])

df.to_csv('final_data_set.csv', index=False)



'''
Best parameters: {'fit_intercept': True, 'alpha': 10, 'normalize': True}
Best validation MSE: 0.12659523427313565
Test MSE: 0.11844389087994946, Test MAE: 0.25126200846354607

'''
