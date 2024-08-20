# -*- coding: shift_jis -*-
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def compute_distance_matrix(X):
    n = len(X)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = euclidean_distance(X[i], X[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # �Ώ̍s��
    return distance_matrix

def find_closest_clusters(distance_matrix):
    min_dist = np.inf
    cluster_pair = (-1, -1)
    n = len(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < min_dist:
                min_dist = distance_matrix[i, j]
                cluster_pair = (i, j)
    return cluster_pair

def ward_clustering(X, num_clusters=3):
    clusters = [[i] for i in range(len(X))]  # �e�f�[�^�|�C���g���ʂ̃N���X�^��
    distance_matrix = compute_distance_matrix(X)

    while len(clusters) > num_clusters:
        # �����s�񂩂�ł��߂��N���X�^�̃y�A��������
        i, j = find_closest_clusters(distance_matrix)

        # �N���X�^������
        clusters[i].extend(clusters[j])
        del clusters[j]

        # �V���������s��̍X�V
        for k in range(len(clusters)):
            if k != i:
                dist_ik = np.mean([euclidean_distance(X[p], X[q]) for p in clusters[i] for q in clusters[k]])
                distance_matrix[i, k] = distance_matrix[k, i] = dist_ik

        # �������ꂽ�N���X�^�̍s�Ɨ���폜
        distance_matrix = np.delete(distance_matrix, j, axis=0)
        distance_matrix = np.delete(distance_matrix, j, axis=1)

    return clusters




wine = load_wine()

# �f�[�^�ƃ^�[�Q�b�g�̎擾
X = wine.data  # �����ʃf�[�^
y = wine.target  # ���x���f�[�^

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clusters = ward_clustering(X, num_clusters=3)
print(clusters)