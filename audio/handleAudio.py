import os
import pandas as pd
import librosa.feature
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE


DATA_PATH = "cv-corpus-17.0-delta-2024-03-15/en"
CLIPS_PATH = os.path.join("clips17")
CSV_PATH = os.path.join("all_validated.tsv")

# Загружаем таблицу
df = pd.read_csv(CSV_PATH, sep="\t")

# Фильтруем только те строки, где есть метка пола
df = df[df["gender"].notna()]
df = df[df["gender"].isin(["male_masculine", "female_feminine"])]

# Создаем список файлов и меток (0 = male, 1 = female)
df["file_path"] = df["path"].apply(lambda x: os.path.join(CLIPS_PATH, x))
df["label"] = df["gender"].map({"male_masculine": 0, "female_feminine": 1})

# Убираем строки с несуществующими файлами
df = df[df["file_path"].apply(os.path.exists)]

# Сохраняем только нужные колонки
df = df[["file_path", "label"]]

print(df.head())


def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]  # Обрезаем слишком длинные
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")  # Паддинг
    return mfcc


features, labels = [], []

for _, row in df.iterrows():
    mfcc = extract_mfcc(row["file_path"])
    if mfcc is not None:
        features.append(mfcc)
        labels.append(row["label"])

# Преобразуем в numpy массивы и сохраняем
features = np.array(features)
labels = np.array(labels)

np.save("features.npy", features)
np.save("labels.npy", labels)

print(df["label"].value_counts())
print(f"✅ Обработано {len(features)} аудиофайлов!")

# Загружаем подготовленные данные
features = np.load("features.npy")
labels = np.load("labels.npy")


# Датасет для PyTorch
class VoiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mfcc = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc.T, label  # Транспонируем, чтобы размерность была (time_steps, n_mfcc)


# Гиперпараметры
input_size = 13  # Количество MFCC
hidden_size = 128
batch_size = 32
lr = 0.001
epochs = 10

# Создаём общий датасет
dataset = VoiceDataset(features, labels)

# Разбиваем датасет на обучающую и валидационную выборки (например, 80%/20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader-ы
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Определяем модель
class GenderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GenderClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 2)  # Два класса

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch_size, hidden_size)
        x = h_n[-1]  # Берём последний слой: (batch_size, hidden_size)
        x = self.dropout(x)
        return self.fc(x)


# Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderClassifier(input_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # Обучающий цикл
    model.train()
    train_losses = []
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    # Валидационный цикл
    model.eval()
    val_losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            val_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    avg_val_loss = np.mean(val_losses)

    # Вычисляем метрики
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


# Функция для извлечения эмбеддингов
def extract_embeddings(model, dataset, batch_size=32):
    model.eval()  # переводим модель в режим оценки
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)  # размер: (batch, time_steps, n_mfcc)
            # Извлекаем скрытое состояние из LSTM.
            # Здесь повторяем часть логики модели:
            # Получаем выход LSTM: (_, (h_n, _))
            _, (h_n, _) = model.lstm(batch_features)
            # Берем последний слой (последнее время)
            embeddings = h_n[-1]  # размер: (batch, hidden_size)
            embeddings = model.dropout(embeddings)  # применяем dropout, как в модели
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_labels

embeddings, labels = extract_embeddings(model, dataset, batch_size=32)

# Применяем t-SNE для сведения размерности до 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Построение scatter plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("t-SNE: 2D-визуализация аудио эмбеддингов")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
# Добавляем легенду с названиями классов (например, 0 = male, 1 = female)
legend1 = plt.legend(*scatter.legend_elements(), title="Класс")
plt.gca().add_artist(legend1)
plt.show()

print("✅ Обучение завершено!")


'''import os
import pandas as pd
import torch
import torchaudio
import librosa
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Пути к папкам с разными датасетами
DATASET_DIRS = [
    "cv-corpus-17.0-delta-2024-03-15/en",
    "cv-corpus-18.0-delta-2024-06-14/en",
    "cv-corpus-20.0-delta-2024-12-06/en"
]

# Гиперпараметры
BATCH_SIZE = 32
N_MFCC = 13
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Простая LSTM модель для классификации пола по голосу
class GenderClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GenderClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return self.sigmoid(x)


# Кастомный Dataset для аудиофайлов
class AudioDataset(Dataset):
    def __init__(self, df, clips_path, n_mfcc=N_MFCC):
        self.df = df
        self.clips_path = clips_path
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.clips_path, row["path"])

        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return mfcc, label


# Функция обучения
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for mfcc, labels in train_loader:
        mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def collate_fn(batch):
    mfccs, labels = zip(*batch)

    # Определяем максимальную длину MFCC в текущем батче
    max_length = max(mfcc.shape[0] for mfcc in mfccs)

    # Дополняем все MFCC нулями до max_length
    padded_mfccs = [torch.cat([mfcc, torch.zeros(max_length - mfcc.shape[0], mfcc.shape[1])]) for mfcc in mfccs]

    # Преобразуем список в единый тензор
    padded_mfccs = torch.stack(padded_mfccs)
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_mfccs, labels


# Загружаем предобученную модель или создаем новую
input_size = N_MFCC
model = GenderClassifier(input_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# 🔥 Последовательное обучение на нескольких датасетах
for dataset_dir in DATASET_DIRS:
    print(f"\n🔄 Обучаем модель на датасете: {dataset_dir}")

    # Загружаем validated.tsv
    df = pd.read_csv(os.path.join(dataset_dir, "validated.tsv"), sep="\t")

    # Оставляем только нужные метки пола
    df = df[df["gender"].isin(["male_masculine", "female_feminine"])]
    df["label"] = df["gender"].map({"male_masculine": 0, "female_feminine": 1})

    # Разделяем на train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Создаём датасеты и DataLoader'ы
    clips_path = os.path.join(dataset_dir, "clips")
    train_dataset = AudioDataset(train_df, clips_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 🔥 Обучаем модель на текущем датасете
    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion)
        print(f"📉 Epoch {epoch + 1}, Loss: {loss:.4f}")

    # После обучения на одном датасете — сохраняем веса
    torch.save(model.state_dict(), f"model_after_{dataset_dir.split('/')[-1]}.pth")

print("\n✅ Модель обучена на всех датасетах!")

'''



'''class VoiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Нейросеть
class GenderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GenderClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Два класса: мужской/женский

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(1))
        return self.fc(h_n[-1])


# Гиперпараметры
input_size = 13  # MFCC
hidden_size = 64
batch_size = 32
lr = 0.001
epochs = 10

# Загружаем данные
features = np.load("features.npy")
labels = np.load("labels.npy")
train_dataset = VoiceDataset(features, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderClassifier(input_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

test_file = "test_voice.wav"
test_features = extract_features(test_file)
test_tensor = torch.tensor(test_features, dtype=torch.float32).to(device).unsqueeze(0)
prediction = model(test_tensor).argmax(dim=1).item()

print("Мужской" if prediction == 0 else "Женский")'''