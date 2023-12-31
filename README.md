# Pokemon-Similarity

포켓몬 데이터 : https://kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types

![Python 3.11.7](https://img.shields.io/badge/python-3.11.7-blue.svg)
![TensorFlow 2.15.0](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)


Kaggle에서 포켓몬 데이터를 다운로드 받고, 타겟 포켓몬을 정해주면 나머지 포켓몬에 대해서 유사도를 계산합니다.

유사도 계산은 코사인 유사도, 유클리드 거리 두 가지 방법을 사용하였으며 각각의 csv 파일로 저장됩니다.

CNN 모델은 각 이미지의 Feature를 Extract 하게되고 여기서 Softmax를 사용하면 가장 일치하는 라벨을 찾게 됩니다.

하지만 여기서는 Softmax를 사용하지 않고 Fully connected layer만 통과시켜 Feature Vector를 출력합니다.

그리고 타겟 이미지의 Feature Vector과 나머지 이미지의 Feature Vector에서 유사도를 계산합니다.




## 모델
VGG16 : https://arxiv.org/abs/1409.1556

![vgg16](https://github.com/k-3730/Pokemon-Similarity/assets/45035923/850b65c6-427a-40ca-b0ff-4b30bca9d250)

이미지 출처 : https://neurohive.io/en/popular-networks/vgg16/





## 결과 예시
첫 번째 이미지가 타겟 이미지로 나머지 포켓몬과 유클리드 거리를 계산한 결과입니다.

0에 가까울수록 타겟 이미지와 유사하며 가까운 이미지부터 정렬되어 있습니다.

![output](https://github.com/k-3730/Pokemon-Similarity/assets/45035923/0a5b0f81-b16e-4c32-abb8-f3b2badfdd09)
