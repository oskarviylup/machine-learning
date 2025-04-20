import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv('final_data_set_ready_for_model.csv')

df['engine_type'].value_counts().plot(kind='bar')
plt.title('Частота типов двигателя')
plt.xlabel('Тип двигателя')
plt.ylabel('Количество')
plt.show()

df.plot(kind='scatter', x='year_of_creation', y='price', alpha=0.5)
plt.title('dependency beetween years of creation and price')
plt.xlabel('Years of creation')
plt.ylabel('Prices')
plt.show()

y = df['price']
y_log = np.log1p(y)
plt.figure(figsize=(10, 6))
sns.histplot(y_log, kde=True, bins=30, color='blue')
plt.title('Распределение цен', fontsize=16)
plt.xlabel('Цена', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.show()


Q1 = y_log.quantile(0.25)
Q3 = y_log.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Нижняя граница: {lower_bound}, Верхняя граница: {upper_bound}")

outliers = y_log[(y_log < lower_bound) | (y_log > upper_bound)]
print(f"Число выбросов: {len(outliers)}")




