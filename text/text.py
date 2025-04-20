import youtokentome as yttm
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('scoutieDataset_127.csv')

# Убедитесь, что текстовые данные корректны
texts = dataset['text'].dropna().tolist()

# Сохранение текстов во временный файл для обучения модели BPE
text_file_path = "texts_for_bpe.txt"
with open(text_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(texts))

# Шаг 1: Обучение модели BPE
bpe_model_path = "bpe_model.yttm"
yttm.BPE.train(data=text_file_path, vocab_size=10000, model=bpe_model_path)

# Загрузка обученной модели
bpe = yttm.BPE(model=bpe_model_path)

# Шаг 2: Токенизация текстов
tokenized_texts = bpe.encode(texts, output_type=yttm.OutputType.ID)

# Шаг 3: Padding и Truncation
max_length = 128
padded_texts = [
    seq[:max_length] + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]
    for seq in tokenized_texts
]

# Шаг 4: Разделение данных на train и test
train_texts, test_texts = train_test_split(padded_texts, test_size=0.2, random_state=42)

# Вывод информации о разделении
print(f"Размер тренировочного набора: {len(train_texts)}")
print(f"Размер тестового набора: {len(test_texts)}")
print(f"Пример токенизированного текста (длина {max_length} токенов): {padded_texts[0]}")