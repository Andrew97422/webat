Audio na habre: https://habr.com/ru/articles/672094/
https://github.com/olivecha/guitarsounds
tsfresh

Examples:
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import os

def load_audio_files(directory):
    data = []
    labels = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(subdir, file)
                label = os.path.basename(subdir)  # Имя папки как метка класса
                y, sr = librosa.load(path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                data.append(np.mean(mfccs.T, axis=0))
                labels.append(label)
    return np.array(data), labels

# Путь к вашей директории с аудио файлами
directory_path = 'path/to/your/audio_files'
features, labels = load_audio_files(directory_path)

# Преобразуем данные в DataFrame для визуализации
df_features = pd.DataFrame(features)
df_features['label'] = labels

# Построим графики распределения признаков для каждого класса
plt.figure(figsize=(12, 6))
for i in range(13):  # Предполагая 13 MFCC
    plt.subplot(3, 5, i+1)
    sns.boxplot(x='label', y=df_features[i], data=df_features)
    plt.title(f'MFCC {i+1}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from scipy import stats

def remove_outliers(df, features):
    z_scores = np.abs(stats.zscore(df[features]))
    mask = (z_scores < 3).all(axis=1)
    return df[mask]

feature_columns = [i for i in range(13)]
df_cleaned = remove_outliers(df_features, feature_columns)

# Разделение данных
X = df_cleaned.drop('label', axis=1)
y = df_cleaned['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Оценка модели
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


guitarsounds
Многоклассовая классификация:
Статистические методы:

Z-оценка: Выбросы определяются как точки данных, которые имеют z-оценку (стандартное отклонение от среднего) больше определенного порога, обычно 3 или -3.
Межквартильный диапазон (IQR): Выбросы определяются как точки за пределами диапазона, вычисляемого как [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR], где Q1 и Q3 — первый и третий квартили.
Визуализационные методы:

Boxplot: Визуальный метод, где выбросы отображаются как точки за пределами «усов» диаграммы.
Scatterplot: Помогает визуально идентифицировать выбросы на двухмерных данных, особенно если выбросы сильно отклоняются от скопления данных.
Машинное обучение и кластеризация:

Методы кластеризации (например, K-средние): Выбросы могут быть определены как точки, которые не принадлежат ни к одному из кластеров или находятся далеко от центроидов кластеров.
Алгоритмы обучения, устойчивые к выбросам: Такие как Isolation Forest или Local Outlier Factor (LOF), специально разработаны для выявления и обработки выбросов.
Переменные пороговые значения:

Определение выбросов с помощью доменно-специфичных порогов или правил, основанных на понимании данных и контекста задачи.
Методы на основе временных рядов:

Скользящее среднее и медианные фильтры: Сглаживают временные ряды и помогают определить и скорректировать или удалить выбросы.
Детрендирование: Удаление трендов из данных, чтобы выявить аномальные колебания.
Дополнительные подходы:

Линейная регрессия: Моделирование данных и анализ остатков регрессионной модели для выявления аномально больших остатков.
Энтропийные методы: Анализ изменений энтропии в данных для определения аномалий.

Когда мы работаем со сложными данными, где \(X\) является матрицей больших размеров, а \(y\) — массивом, важно адаптировать методы обработки выбросов для работы с многомерными данными. Давайте рассмотрим примеры применения некоторых методов для таких случаев.

### 1. Z-оценка для многомерных данных

Для многомерных данных Z-оценка применяется по каждому признаку отдельно.

```python
import numpy as np

# Пример данных: Матрица X (1000 samples, 10 features), y - массив
np.random.seed(0)
X = np.random.normal(size=(1000, 10))
X[::100] = np.random.normal(loc=10, size=(10, 10))  # Добавим выбросы
y = np.random.choice([0, 1], size=1000)

mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)
z_scores = (X - mean) / std_dev

# Определяем выбросы
threshold = 3
outliers = np.where(np.abs(z_scores) > threshold)
print("Выбросы по Z-оценке:", outliers)
```

### 2. Isolation Forest для многомерных данных

Isolation Forest можно применять ко всей матрице признаков.

```python
from sklearn.ensemble import IsolationForest

# Используем Isolation Forest
isolation_forest = IsolationForest(contamination=0.01)
outliers = isolation_forest.fit_predict(X)

# Индексы выбросов
outlier_indices = np.where(outliers == -1)[0]
print("Выбросы по Isolation Forest:", outlier_indices)
```

### 3. PCA и анализ выбросов

Метод главных компонент (PCA) может быть использован для выявления выбросов в сжатом представлении данных.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Строим график для визуальной оценки выбросов
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', label='Обычные точки')
plt.scatter(X_pca[outlier_indices, 0], X_pca[outlier_indices, 1], c='r', label='Выбросы')
plt.title('PCA с выбросами')
plt.legend()
plt.show()
```

### 4. K-средние и выбросы в многомерных данных

Мы можем использовать расстояние до центроидов для определения выбросов.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

distances = kmeans.transform(X)
dist_to_centroid = np.min(distances, axis=1)

threshold_distance = np.percentile(dist_to_centroid, 95)
outlier_indices = np.where(dist_to_centroid > threshold_distance)[0]
print("Выбросы по K-средним:", outlier_indices)
```

### 5. Линейная регрессия и выбросы в многомерных данных

Линейная регрессия также может быть использована для анализа остатков:

```python
from sklearn.linear_model import LinearRegression

# Предположим, что y зависит от X по линейной модели
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

threshold = np.percentile(np.abs(residuals), 95)
outlier_indices = np.where(np.abs(residuals) > threshold)[0]
print("Выбросы по остатку регрессии:", outlier_indices)
```

Эти примеры демонстрируют, как можно адаптировать различные методы для работы с многомерными данными для выявления и обработки выбросов. Каждый метод может быть полезен в зависимости от структуры ваших данных и специфики задачи анализа.

Выбор метода для обнаружения и обработки выбросов зависит от специфики данных и задач анализа. Вот рекомендации по использованию каждого из методов в зависимости от конкретных сценариев:

### 1. Z-оценка

**Когда использовать**:
- Данные приблизительно нормально распределены.
- Набор данных не очень большой (в противном случае, вычисление может стать дорогостоящим).
- Неожиданные значения находятся в пределах одного или двух признаков.

**Когда избегать**:
- Данные распределены асимметрично или содержат множество выбросов, которые могут сместить среднее и стандартное отклонение.

### 2. Isolation Forest

**Когда использовать**:
- Данные многомерные и возможно сильное взаимодействие между признаками.
- Число признаков значительно больше, чем выборка (имеется высокая размерность).
- Нужно выявить выбросы в больших наборах данных.

**Когда избегать**:
- Когда выбросы нужно обнаружить в малом наборе данных, поскольку модель может неправильно обучиться.

### 3. PCA и анализ выбросов

**Когда использовать**:
- Вы хотите снизить размерность данных и выявить выбросы в низкоразмерном пространстве.
- Данные сложно визуализировать из-за их высокой размерности, но нужно понять общую картину.

**Когда избегать**:
- Признаки данных не линейно зависимы.
- Выбросы не проявляют себя в первых нескольких главных компонентах.

### 4. K-средние и выбросы

**Когда использовать**:
- Данные имеют хорошо определенные кластеры, и выбросы проявляются как точки, удаленные от этих кластеров.
- Набор данных среднего размера, где изначально известное число кластеров.

**Когда избегать**:
- Данные обладают сложной структурой, не поддающейся делению на четкие кластеры.
- Когда выбросы представляют собой кластеры, а не отдельные точки.

### 5. Линейная регрессия и анализ остатков

**Когда использовать**:
- Необходимо анализировать зависимость переменной от множества предикторов.
- Данные имеют приблизительно линейные зависимости между предикторами и целевой переменной.

**Когда избегать**:
- Данные сильно нелинейные, и линейная модель не подходит для их анализа.
- Высокая мультиколлинеарность среди признаков, что может усложнить интерпретацию модели.

Эти рекомендации обеспечивают основу для выбора подходящего метода обработки выбросов в зависимости от характеристик вашего набора данных и поставленных задач. Однако всегда полезно тестировать несколько подходов и оценивать их результаты на конкретных данных.
