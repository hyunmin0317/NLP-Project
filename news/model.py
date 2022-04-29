from keras.datasets import reuters
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# 2. LSTM으로 로이터 뉴스 분류하기
vocab_size = 1000   # 단어 집합의 크기
max_len = 100       # 뉴스 기사의 길이

# 데이터 로드 후 뉴스 기사 데이터를 훈련용과 테스트용으로 나누기
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)
X_train = pad_sequences(X_train, maxlen=max_len)    # 모든 뉴스 기사의 길이 100으로 패딩
X_test = pad_sequences(X_test, maxlen=max_len)      # 모든 뉴스 기사의 길이 100으로 패딩
y_train = to_categorical(y_train)                   # 원-핫 인코딩
y_test = to_categorical(y_test)                     # 원-핫 인코딩

# 다중 클래스 분류 문제를 수행할 LSTM 모델 생성
embedding_dim = 128  # 하이퍼파라미터인 임베딩 벡터의 차원
hidden_units = 128   # 은닉 상태의 크기
num_classes = 46     # 선택지 개수
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