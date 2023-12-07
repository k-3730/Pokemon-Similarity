from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


__all__ = ['FeatureExtractor']


class FeatureExtractor:
    # 이미 학습된 VGG16 모델을 불러온다.
    def __init__(self):

        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    def extract(self, img):
        # VGG 모델에서 들어올 이미지 사이즈가 224 x 224이다.
        img = img.resize((224, 224))

        img = img.convert('RGB')

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
 
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)