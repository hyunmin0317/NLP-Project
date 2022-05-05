import pandas as pd
import numpy as np
import urllib.request


# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
# total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
# print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
# print(total_data[:5])
#
# # 2. 훈련 데이터와 테스트 데이터 분리
# # 평점 4, 5 - 레이블 1, 평점 1, 2 - 레이블 0
# total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)
# print(total_data[:5])

data = pd.read_csv('dataset.csv')

print(data[:5])