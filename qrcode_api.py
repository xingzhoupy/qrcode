import requests
import json

def decode_qr(qr_url):
    # 使用jiema.wwei.cn解码二维码, 返回解码结果。

    headers = {
    'Host': 'jiema.wwei.cn',
    'Content-Length': '0',
    'Origin': 'http://jiema.wwei.cn',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
    'Accept': '*/*',
    'Referer': 'http://jiema.wwei.cn/',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.8',
    }

    cookies = {
    'cookies': 'PHPSESSID=52fhfnjsi31ghhds9kd66ts0u7'
    }
    '20160702128962'

    params = {
        'data': '{0}'.format(qr_url),
        'apikey': '20160702128962'
    }

    try:
        html = requests.get('http://api.wwei.cn/dewwei.html', params=params)
        return json.loads(html.text)['data']['raw_text']
    except:
        return 0