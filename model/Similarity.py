from model.ImageProcessing import process_images
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


__all__ = ['calculate_similarity']


def calculate_similarity(directory, target_image_path):
    features, img_paths = process_images(directory)
    img = Image.open(target_image_path)
    query = features[0]

    # 유클리드 거리
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)
    euclidean_scores = [(dists[id], img_paths[id]) for id in ids]

    # 코사인 거리
    query_feature = query.reshape(1, -1)
    cosine_similarities = cosine_similarity(features, query_feature).flatten()
    ids = np.argsort(cosine_similarities)[::-1]
    cosine_scores = [(cosine_similarities[id], img_paths[id]) for id in ids]

    return euclidean_scores, cosine_scores