import re
import json
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, save_model


# 리뷰에 반응을 예측하는 함수
def sentiment_predict(new_sentence):
  max_len = 80
  tokenizer = Tokenizer()
  mecab = Mecab('C:/mecab/mecab-ko-dic')
  stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
  loaded_model = load_model('best_model.h5')
  tokenizer.word_index = json.load(open('word_index.json'))

  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = mecab.morphs(new_sentence)
  new_sentence = [word for word in new_sentence if not word in stopwords]
  encoded = tokenizer.texts_to_sequences([new_sentence])
  pad_new = pad_sequences(encoded, maxlen = max_len)

  score = float(loaded_model.predict(pad_new))
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))


if __name__ == '__main__':
  # 임의의 리뷰에 대해서 예측
  sentiment_predict('이 상품 진짜 좋아요... 저는 강추합니다. 대박')
  sentiment_predict('진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?')
  sentiment_predict('판매자님... 너무 짱이에요.. 대박나삼')
  sentiment_predict('ㅁㄴㅇㄻㄴㅇㄻㄴㅇ리뷰쓰기도 귀찮아')