import pandas as pd
from model import calculate_similarity

# 포켓몬 데이터 다운로드 : kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types 
directory = "./Pokemon/pokemon/images/images"
target_image_path = "./Pokemon/pokemon/images/images/abomasnow.png"

euclidean_scores, cosine_scores = calculate_similarity(directory, target_image_path)


# 유클리드 거리 CSV 파일 저장
euclidean_df = pd.DataFrame(euclidean_scores, columns=['Euclidean_Distance', 'ImagePath'])
euclidean_df.to_csv('euclidean_scores.csv', index=False)

# 코사인 유사도 CSV 파일 저장
cosine_df = pd.DataFrame(cosine_scores, columns=['Cosine_Similarity', 'ImagePath'])
cosine_df.to_csv('cosine_scores.csv', index=False)