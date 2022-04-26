import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, save_model


# 리뷰에 반응을 예측하는 함수
def sentiment_predict(new_sentence):
  max_len = 30
  okt = Okt()
  tokenizer = Tokenizer()
  stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
  loaded_model = load_model('best_model.h5')
  tokenizer.word_index = json.load(open('word_index.json'))

  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측

  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


if __name__ == '__main__':
  # 임의의 리뷰에 대해서 예측
  sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
  sentiment_predict('이 영화 핵노잼 ㅠㅠ')
  sentiment_predict('이딴게 영화냐 ㅉㅉ')
  sentiment_predict('감독 뭐하는 놈이냐?')
  sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')