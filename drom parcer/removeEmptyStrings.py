import pandas as pd

# Загрузка данных из файла CSV
df = pd.read_csv("data_set_2.csv", header=None)

df = df.fillna('')

# Фильтрация строк: убираем строки, где все значения в определенных колонках пустые или содержат "---", "No price", "- мест"
filtered_df = df[~df.apply(lambda row: row.str.contains(r'No price', regex=False).all(), axis=1)]

# Сохранение отфильтрованных данных обратно в CSV
filtered_df.to_csv("filtered_file.csv", index=False, header=False)

print("Фильтрация завершена. Данные сохранены в 'filtered_file.csv'.")
