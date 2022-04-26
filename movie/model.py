import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 데이터 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print('훈련용 리뷰 개수 :',len(train_data))    # 훈련용 리뷰 개수 출력
print(train_data[:5])                       # 상위 5개 출력
print('테스트용 리뷰 개수 :',len(test_data))   # 테스트용 리뷰 개수 출력
print(test_data[:5])                        # 상위 5개 출력

# 2. 데이터 정제
print(train_data['document'].nunique(), train_data['label'].nunique())  # document 열과 label 열의 중복을 제외한 값의 개수
train_data.drop_duplicates(subset=['document'], inplace=True)           # document 열의 중복 제거
print('총 샘플의 수 :',len(train_data))
train_data['label'].value_counts().plot(kind = 'bar')                   # train_data 레이블 값의 분포 확인
print(train_data.groupby('label').size().reset_index(name = 'count'))   # train_data 레이블 개수 확인

# NULL 값을 가진 샘플 제거
print(train_data.isnull().values.any())             # NULL 값을 가진 샘플이 있는 지 확인
print(train_data.isnull().sum())                    # NULL 값을 가진 열 확인
print(train_data.loc[train_data.document.isnull()]) # NULL 값 인덱스 확인
train_data = train_data.dropna(how = 'any')         # NULL 값이 존재하는 행 제거
print(train_data.isnull().values.any())             # NULL 값이 존재하는지 확인
print(len(train_data))

# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(train_data[:5])
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
print(train_data.loc[train_data.document.isnull()][:5])
train_data = train_data.dropna(how = 'any')
print(len(train_data))

# 데이터 전처리
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

# 3. 토큰화
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

X_train = []    # train_data
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)

X_test = []     # test_data
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)

print(X_train[:3])
print(X_test[:3])

# 4. 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)

# 등장 빈도수가 3회 미만인 단어들 비중 확인
threshold = 3
total_cnt = len(tokenizer.word_index)   # 단어의 수
rare_cnt = 0                            # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0                          # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0                           # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
for key, value in tokenizer.word_counts.items():    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    total_freq = total_freq + value
    if(value < threshold):      # 단어의 등장 빈도수가 threshold보다 작으면
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt + 1   # 0번 패딩 토큰을 고려하여 + 1
print('단어 집합의 크기 :',vocab_size)
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])

# 별도 저장
json = json.dumps(tokenizer.word_index)
f1 = open("word_index.json", "w")
f1.write(json)
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 5. 빈 샘플 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

# 6. 패딩 (서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 과정)
print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# max_len 이하인 샘플의 비율을 구하는 함수
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)
# 전체 샘플 중 길이가 30 이하인 샘플의 비율: 94.31944999380003

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 7. LSTM으로 네이버 영화 리뷰 감성 분류하기
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))  # 테스트 정확도: 0.8544