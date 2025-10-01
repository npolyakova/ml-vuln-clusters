import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Случайная кластеризация
def random_baseline(labels_true, n_clusters=None):
    n_samples = len(labels_true)
    if n_clusters is None:
        # Оцениваем число кластеров как sqrt(n_samples) или берем из истинных меток
        # n_clusters = len(np.unique(labels_true))
        n_clusters = 14

    labels_pred = np.random.randint(0, n_clusters, n_samples)
    return labels_pred

# Полное совпадение по названию
def exact_match_baseline(descriptions):
    clusters = defaultdict(list)
    for i, desc in enumerate(descriptions):
        clusters[desc].append(i)

    # Создаем массив меток кластеров
    labels = np.zeros(len(descriptions))
    for cluster_id, indices in enumerate(clusters.values()):
        for idx in indices:
            labels[idx] = cluster_id

    return labels

# Совпадение по ключевым словам
def tfidf_kmeans_baseline(descriptions, true_labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    # Векторизация
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(descriptions)

    # Кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_pred = kmeans.fit_predict(X_tfidf)

    return labels_pred

# Совпадение по первым словам
def first_words_baseline(descriptions, n_words=5):
    clusters = defaultdict(list)
    for i, desc in enumerate(descriptions):
        key = ' '.join(desc.split()[:n_words]).lower()
        clusters[key].append(i)

    labels = np.zeros(len(descriptions))
    for cluster_id, indices in enumerate(clusters.values()):
        for idx in indices:
            labels[idx] = cluster_id

    return labels

def compare_results(true_values, values):
    correct_count = 0
    for true_item in true_values:
        for item in values:
            if item == true_item:
                correct_count += 1
    return correct_count / len(true_values)


text_file = open("test-data.txt", "r")
lines = text_file.read().split(';')

true_clusters = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                 7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,  10, 10, 10,
                 10, 10, 10, 10, 10, 8, 11, 11, 11, 12, 12, 13, 13, 13, 14]

# Проверка рандомной кластеризации
print("Проверка рандомной кластеризации")
print(compare_results(true_clusters, random_baseline(lines)), "%")

# Проверка полного совпадения текста
print("Проверка полного совпадения текста")
print(compare_results(true_clusters, exact_match_baseline(lines)), "%")

# Проверка совпадения по ключевым словам
print("Проверка совпадения по ключевым словам")
print(compare_results(true_clusters, tfidf_kmeans_baseline(lines, true_clusters, 12)), "%")

# Проверка совпадения по первым словам
print("Проверка совпадения по первым словам")
print(compare_results(true_clusters, first_words_baseline(lines, 1)), "%")