from model.model import FeatureExtractor
from PIL import Image
import numpy as np
import os


__all__ = ["process_images"]

def process_images(directory):
    fe = FeatureExtractor()
    features = []
    img_paths = []

    # 폴더 내 모든 파일 이름을 가져옵니다.
    img_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    for img_name in img_files:
        try:
            image_path = os.path.join(directory, img_name)
            
            img_paths.append(image_path)


            feature = fe.extract(img=Image.open(image_path))

            features.append(feature)

            feature_path = "./" + os.path.splitext(img_name)[0] + ".npy"
            np.save(feature_path, feature)
        except Exception as e:
            print('예외가 발생했습니다.', e)

    return features, img_paths