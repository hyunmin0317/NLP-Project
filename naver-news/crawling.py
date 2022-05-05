import json
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs


def news_list():
    dic = {}
    url = 'https://news.naver.com/main/mainNews.naver?date=%2000:00:00&page=1'
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
    data = json.loads(res['airsResult'])['result']

    for i in range(100, 106):
        sid = str(i)
        list = []
        for news in data[sid]:
            list.append(news['officeId']+'/'+news['articleId'])
        dic[sid] = list
    return dic


def news_text(articleId):
    url = 'https://n.news.naver.com/mnews/article/'+articleId
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    soup = bs(res, 'html.parser')
    div = soup.select_one('#dic_area').text.replace('\n','').replace('\t','')
    return div


list = news_list()
text_list = []
label_list = []
for i in range(0, 6):
    sid = '10'+str(i)
    for data in list[sid]:
        label_list.append(i)
        text_list.append(news_text(data))

news_data = pd.DataFrame({
    'label': label_list,
    'text': text_list,
})

news_data.to_csv('dataset.csv', encoding='utf-8-sig', index=False)
print(news_data)