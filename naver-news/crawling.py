import json
import requests


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

print(news_list())