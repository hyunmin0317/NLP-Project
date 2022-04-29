import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters


# 1. 로이터 뉴스 데이터에 대한 이해

# 데이터 로드 후 뉴스 기사 데이터를 훈련용과 테스트용으로 나누기
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
print('훈련용 뉴스 기사 : {}'.format(len(X_train)))    # 훈련용 뉴스 기사 : 8982
print('테스트용 뉴스 기사 : {}'.format(len(X_test)))    # 테스트용 뉴스 기사 : 2246
num_classes = len(set(y_train))
print('카테고리 : {}'.format(num_classes))            # 카테고리 : 46

# 훈련용 뉴스 기사 데이터 구성 확인
print('첫번째 훈련용 뉴스 기사 :',X_train[0])           # 토큰화와 정수 인코딩이 끝난 데이터
print('첫번째 훈련용 뉴스 기사의 레이블 :',y_train[0])    # 첫번째 훈련용 뉴스 기사의 레이블 : 3

# 훈련용 뉴스 기사의 길이 확인
print('뉴스 기사의 최대 길이 :{}'.format(max(len(sample) for sample in X_train)))    # 뉴스 기사의 최대 길이 : 2376
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))       # 뉴스 기사의 평균 길이 : 145.5398574927633
plt.hist([len(sample) for sample in X_train], bins=50)  # 대부분의 뉴스 길이 100~200 사이
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 레이블 값의 분포 확인
fig, axe = plt.subplots(ncols=1)    # 3, 4가 가장 많은 레이블을 차지
fig.set_size_inches(12,5)
sns.countplot(y_train)
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))   # 3 레이블: 3159개, 4 레이블: 1949개

# 각 단어와 그 단어에 부여된 인덱스 확인
word_to_index = reuters.get_word_index()
print(word_to_index)

# word_to_index에서 key와 value를 반대로 저장한 index_to_word 생성
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key
print('빈도수 상위 1번 단어 : {}'.format(index_to_word[4]))     # the
print('빈도수 상위 128등 단어 : {}'.format(index_to_word[131])) # tax

# 로이터 뉴스 데이터셋의 규칙에 맞게 index_to_word 생성
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
  index_to_word[index] = token
print(' '.join([index_to_word[index] for index in X_train[0]]))