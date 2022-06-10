import hashlib
import json
import random
import time

import pandas as pd
# df = pd.read_csv(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\HeFeiallData.csv')
# def createAreaName(areaid):
#     d = {199:"蜀山区",198:"庐阳区",197:"瑶海区",200:"包河区",5729:"经开区",5733:"政务区",
#         5731:"新站区",5732:"高新区",7607:"巢湖市",2853:"肥东县",2854:"肥西县",2852:"长丰县",7205:"庐江县"}
#     return d

# test = [{'id': 701905196, 'title': '大釜烤肉双人餐，有赠品', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 128.0, 'value': 210.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 701904284, 'title': '大釜烤肉三人餐，有赠品', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 178.0, 'value': 259.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 701895719, 'title': '100元代金券1张，可叠加', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 88.0, 'value': 100.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 701900668, 'title': '大釜烤肉4人餐，有赠品', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 228.0, 'value': 338.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 733158776, 'title': '100元代金券1张，仅限使用1张', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 59.9, 'value': 100.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 701904843, 'title': '6人烤肉套餐，有赠品', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 318.0, 'value': 504.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 736779240, 'title': '秘制鸡腿肉1份，包间免费', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 18.8, 'value': 28.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 701905737, 'title': '5人烤肉套餐，有赠品', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 278.0, 'value': 417.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}},
#       {'id': 736809342, 'title': '缤纷水果烤面包组合，建议1-2人使用，包间免费', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 10.9, 'value': 21.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}]
#
# str = "[{'1':'id'},{'2':'name'}]"
# list1 = eval(str)
# print(len(list1))
# print(list1[0].get('1'),list1[1].get('2'))

# str1 = str(1)+ ";先马甲："+'2'
# print(str1)

# data = '["非阿佛·"]'
# if data[-1] == ']':
#       print(data)
# else:
#       print("缺乏]")
#       print(eval(data))

# data = "[{1:'filsr + fi" \
#        "\nr\ns'}]"
# print(data)
# data = data.replace("\n"," ")
# print(data)
# eval(data)

# def getdeals(data):
#       data = data.replace("\n"," ")
#       data = data.replace("\r"," ")
#       data = data.replace("\t"," ")
#
#       if data[-1] == ']':
#             data = eval(data)
#             if data != None:
#                   length = len(data)
#                   for i in range(length):
#                         try:
#                               str1 = data[i].get('title')+",现价:"+str(data[i].get('price'))+",原价:"+str(data[i].get('value'))+";\n"
#                               return str1
#                         except:
#                               print("失败")
#                               return "信息录入问题"
#
#       else:
#             print("信息不完全")
#             return "信息太长，未导入"
#
# df = df.drop_duplicates()
# df['套餐/代金券(筛选)'] = df["套餐/代金券"].apply(getdeals)
# df.to_csv('HfData.csv',encoding='utf_8_sig')
# str ="[{'id': 702792115, 'title': '【小食】北海道风味华夫筒_25507_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 3.0, 'value': 4.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 738452120, 'title': '【双人餐】小皇堡+炫辣鸡腿堡+小薯*2+洋葱圈5个+小可*2_40769_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 37.9, 'value': 76.5, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 692047406, 'title': '【会员】【会员】KING暴风阿华田_25487_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 6.6, 'value': 13.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 738443270, 'title': '【3件套】炫辣鸡腿堡+王道嫩香鸡块+可口可乐（中）_40768_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 19.9, 'value': 40.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 676509979, 'title': '【3件套】安格斯厚牛堡+王道嫩香鸡块+可口可乐（小）_10613_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 37.5, 'value': 56.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 676517335, 'title': '【4件套】果木风味鸡腿堡+王道椒香鸡腿+小薯+中可_10612" \
#      "_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 26.0, 'value': 52.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 657399889, 'title': '【双堡】果木风味鸡腿堡+小皇堡_10155_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 21.0, 'value': 36.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 661479060, 'title': '【会员专享】小皇堡+王道嫩香鸡块+薯霸王（小）+可口可乐（小）_10289_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 22.5, 'value': 45.5, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 750165607, 'title': '【新品单堡】咔咔脆鸡堡（会员专享）_25957_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 9.9, 'value': 16.0, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}, {'id': 653199641, 'title': '【小食】洋葱圈（5个）_25264_汉堡王1份', 'subtitleA': None, 'subtitleB': None, 'subtitleC': None, 'price': 5.5, 'value': 7.5, 'sales': 0.0, 'iUrl': '', 'stid': None, 'trace': None, 'tag': {'promotion': []}}]"
# str = eval(str)

# df = pd.read_csv(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\HfData.csv')
# def createAreaName(areaid):
#     d = {199:"蜀山区",198:"庐阳区",197:"瑶海区",200:"包河区",5729:"经开区",5733:"政务区",
#         5731:"新站区",5732:"高新区",7607:"巢湖市",2853:"肥东县",2854:"肥西县",2852:"长丰县",7205:"庐江县"}
#     return d.get(areaid)
# df['地区'] = df['地区id'].apply(createAreaName)
# df.to_csv("HfData.csv")
# import requests
# host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=hkMudQMShdmHUwSRwCen0su4&client_secret=E5RINX272VqNIhMW1r1kU9V12Ov39yo0'
# response = requests.get(host).json()
# token = response.get('access_token')
# print(token)
# text = "苹果你吃吗?"
# from_lang = 'zh'
# to_lang = 'en'
# headers = {'Content-Type': 'application/json'}
# term_ids = ''
# payload = {'q': text, 'from': from_lang, 'to': to_lang, 'termIds' : term_ids}
# url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token
# response2 = requests.get(url,params=payload,headers=headers).json()
#
# print(response2)

