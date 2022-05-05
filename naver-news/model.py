import json
import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# 1. 데이터 로드
total_data = pd.read_csv('dataset.csv')
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
print(total_data[:5])


# 2. 훈련 데이터와 테스트 데이터 분리
train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('훈련용 리뷰의 개수 :', len(train_data))   # 훈련용 리뷰의 개수 : 90
print('테스트용 리뷰의 개수 :', len(test_data))  # 테스트용 리뷰의 개수 : 30


# 3. 레이블 분포 확인
train_data['label'].value_counts().plot(kind = 'bar')
# 레이블 분포: 0 - 13, 1 - 17, 2 - 14, 3 - 14, 4 - 16, 5 - 16
print(train_data.groupby('label').size().reset_index(name = 'count'))


# 4. 토큰화
mecab = Mecab('C:/mecab/mecab-ko-dic')
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
# 훈련 데이터와 테스트 데이터 토큰화
train_data['tokenized'] = train_data['text'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['text'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test= test_data['tokenized'].values
y_test = test_data['label'].values


# 5. 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
threshold = 2
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

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)

# 정수 인코딩 과정에서 큰 숫자가 부여된 단어들 OOV로 변환
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])
print(X_test[:3])
json = json.dumps(tokenizer.word_index)
f1 = open("word_index.json", "w")
f1.write(json)


# 6. 패딩
print('뉴스기사의 최대 길이 :',max(len(review) for review in X_train))
print('뉴스기사의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 2000
below_threshold_len(max_len, X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
y_train = to_categorical(y_train)                   # 원-핫 인코딩
y_test = to_categorical(y_test)                     # 원-핫 인코딩


# 7. 다중 클래스 분류 문제를 수행할 LSTM 모델 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


embedding_dim = 128  # 하이퍼파라미터인 임베딩 벡터의 차원
hidden_units = 128   # 은닉 상태의 크기
num_classes = 6      # 선택지 개수
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(num_classes, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)   # 데이터 손실이 4회 증가할 경우 과적합으로 학습 조기 종료
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)    # 검증 데이터의 정확도가 전보다 좋아질 경우만 모델 저장
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))  # 과적합 판단을 위해 validation_data로 X_test와 y_test 사용

# 저장한 모델 로드 후 성능 평가
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))  # 테스트 정확도: 0.7195

# 에포크마다 변화하는 손실 시각화
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()