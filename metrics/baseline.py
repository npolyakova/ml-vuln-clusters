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

true_clusters_by_type = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                         7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10,
                         10, 10, 10, 10, 10, 8, 11, 11, 11, 12, 12, 13, 13, 13, 14]
true_clusters_by_assignee = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 3,
                             1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 4, 2, 2, 2, 2,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 2, 2, 3, 2, 1,
                             1, 1, 3, 3, 3, 1]

# Проверка рандомной кластеризации
print("Успех рандомной кластеризации по типам ", compare_results(true_clusters_by_type, random_baseline(lines)), "%")
print("Успех рандомной кластеризации по ответственным ", compare_results(true_clusters_by_assignee, random_baseline(lines, 4)), "%")

# Проверка полного совпадения текста
print("Успех полного совпадения текста", compare_results(true_clusters_by_type, exact_match_baseline(lines)), "%")

# Проверка совпадения по ключевым словам
print("Проверка совпадения по ключевым словам при делении по типам", compare_results(true_clusters_by_type, tfidf_kmeans_baseline(lines, true_clusters_by_type, 12)), "%")
print("Проверка совпадения по ключевым при делении по ответственным", compare_results(true_clusters_by_assignee, tfidf_kmeans_baseline(lines, true_clusters_by_assignee, 12)), "%")

# Проверка совпадения по первым словам
print("Проверка совпадения по первым словам при делении по типам ", compare_results(true_clusters_by_type, first_words_baseline(lines, 1)), "%")
print("Проверка совпадения по первым словам при делении по ответственным ", compare_results(true_clusters_by_assignee, first_words_baseline(lines, 1)), "%")