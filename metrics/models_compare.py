import hdbscan
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score


def hdbscan_clustering_fixed(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        max_cluster_size=14,
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.1,
        allow_single_cluster=True
    )
    labels = clusterer.fit_predict(embeddings)

    print(f"Исходные метки HDBSCAN: {np.unique(labels, return_counts=True)}")
    print(f"Количество шумовых точек: {np.sum(labels == -1)}")

    # ВАЖНО: Сохраняем ВСЕ точки, включая шум
    # Шумовые точки превращаем в отдельные кластеры
    noise_mask = labels == -1
    if noise_mask.any():
        # Создаем новые уникальные номера для шумовых точек
        max_cluster = labels.max()
        noise_indices = np.where(noise_mask)[0]

        # Каждой шумовой точке даем уникальный номер кластера
        for i, idx in enumerate(noise_indices):
            labels[idx] = max_cluster + 1 + i

    print(f"Метки после обработки шума: {np.unique(labels, return_counts=True)}")
    print(f"Размер pred_labels после обработки: {len(labels)}")
    return labels


def calculate_all_metrics(embeddings, true_labels, pred_labels):
    # Проверка совпадения размеров
    if len(true_labels) != len(pred_labels):
        print(f"ОШИБКА: Размеры не совпадают! true_labels: {len(true_labels)}, pred_labels: {len(pred_labels)}")
        return None

    metrics = {}

    # Adjusted Rand Index (ARI)
    metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)

    # Silhouette Score
    unique_clusters = len(np.unique(pred_labels))
    if unique_clusters > 1 and unique_clusters < len(pred_labels):
        metrics['Silhouette'] = silhouette_score(embeddings, pred_labels, metric='euclidean')
    else:
        metrics['Silhouette'] = np.nan

    # Davies-Bouldin Index
    if unique_clusters > 1:
        metrics['Davies-Bouldin'] = davies_bouldin_score(embeddings, pred_labels)
    else:
        metrics['Davies-Bouldin'] = np.nan

    # Дополнительная информация
    metrics['n_true_clusters'] = len(np.unique(true_labels))
    metrics['n_pred_clusters'] = unique_clusters
    metrics['n_samples'] = len(true_labels)

    return metrics


# Ваши данные
text_file = open("test-data.txt", "r")
lines = text_file.read().split(';')
text_file.close()

true_clusters_by_type = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                         7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10,
                         10, 10, 10, 10, 10, 8, 11, 11, 11, 12, 12, 13, 13, 13, 14]

print(f"Количество описаний: {len(lines)}")
print(f"Количество истинных меток: {len(true_clusters_by_type)}")

# Проверка данных
assert len(lines) == len(
    true_clusters_by_type), f"Несовпадение: lines={len(lines)}, true_labels={len(true_clusters_by_type)}"

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
print("Получение эмбеддингов...")
embeddings = model.encode(lines, show_progress_bar=True)
print(f"Размерность эмбеддингов: {embeddings.shape}")

print("\n=== Метрики HDBSCAN кластеризации (исправленная версия) ===")
pred_clusters = hdbscan_clustering_fixed(embeddings)
metrics = calculate_all_metrics(embeddings, true_clusters_by_type, pred_clusters)

if metrics:
    print("\nРезультаты HDBSCAN:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# Альтернативный вариант: использовать HDBSCAN без обработки шума для сравнения
print("\n=== HDBSCAN без обработки шума (для сравнения) ===")
clusterer_raw = hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=1,
    metric='euclidean'
)
pred_clusters_raw = clusterer_raw.fit_predict(embeddings)
print(f"Метки без обработки: {np.unique(pred_clusters_raw, return_counts=True)}")

# Рассчитываем метрики только для не-шумовых точек
non_noise_mask = pred_clusters_raw != -1
if non_noise_mask.any():
    non_noise_embeddings = embeddings[non_noise_mask]
    non_noise_true_labels = np.array(true_clusters_by_type)[non_noise_mask]
    non_noise_pred_labels = pred_clusters_raw[non_noise_mask]

    metrics_non_noise = calculate_all_metrics(non_noise_embeddings, non_noise_true_labels, non_noise_pred_labels)
    if metrics_non_noise:
        print(f"\nМетрики только для не-шумовых точек ({non_noise_mask.sum()} точек):")
        for key, value in metrics_non_noise.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

# Сравнение с K-Means
print("\n=== Сравнение с K-Means ===")
kmeans = KMeans(n_clusters=len(np.unique(true_clusters_by_type)), random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(embeddings)
kmeans_metrics = calculate_all_metrics(embeddings, true_clusters_by_type, kmeans_labels)

if kmeans_metrics:
    print("K-Means результаты:")
    for key, value in kmeans_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")