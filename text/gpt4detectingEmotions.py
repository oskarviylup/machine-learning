from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Создаем OpenAI клиент
#openai.api_key = "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"
client = OpenAI(api_key="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");
# Функция для разметки токсичности
def get_toxicity_label(text):
    prompt = f"""Ты — NLP-модель для определения токсичности текста.
    Тебе дан анекдот. Определи, является ли он токсичным.
    **Токсичный текст** — это грубые, оскорбительные, расистские, сексистские или унижающие достоинство высказывания.
    Ответь только **'токсичный'** или **'не токсичный'**.

    Анекдот: {text}"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    label = response["choices"][0]["message"]["content"].strip().lower()
    return 1 if label == "токсичный" else 0




# Загружаем модель
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Функция разметки
def get_toxicity_label_sec(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    score = torch.sigmoid(logits).numpy()[0][0]  # Оцениваем вероятность токсичности
    return 1 if score > 0.5 else 0

import openai

openai.api_key = "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"

def get_toxicity_label_tre(text):
    response = client.moderations.create(input=text)
    is_toxic = response["results"][0]["category_scores"]["hate"] > 0.5  # Можно менять порог
    return 1 if is_toxic else 0

# Загружаем датасет
dataset = pd.read_csv('scoutieDataset_127.csv')
texts = dataset['text'].dropna().tolist()

# Размечаем датасет
labels = []
for text in tqdm(texts):
    labels.append(get_toxicity_label_tre(text))

# Добавляем разметку в датасет
dataset['toxic'] = labels

# Сохраняем размеченный датасет
dataset.to_csv("scoutieDataset_with_toxic.csv", index=False)

print("Разметка завершена!")
