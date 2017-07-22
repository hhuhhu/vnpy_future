# coding: utf-8

import os
import xlrd
from glob import glob
import re
from urllib import request as urllib2
from bs4 import BeautifulSoup
import datetime
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import jieba


def word_feats(text, feature):
    words_t = cut(text)
    words = []
    for w in words_t:

        if feature in (w.decode('utf-8')):
            words.append(w)

    return dict([(word, True) for word in words])


def deal_corpus():
    l = os.listdir('corpus')
    for f in l:
        corpus = open('corpus/' + f, 'r')
        text = corpus.read()
        p = re.compile(r'[\x80-\xff]+')
        chlist = p.findall(text)

        t = " ".join(chlist)
        corpus.close()
        corpus = open('corpus/' + f, 'w')
        corpus.write(t)


def cut(text):
    seg_list = jieba.cut(text, cut_all=False)
    t = " ".join(seg_list).encode('utf-8')
    p = re.compile(r'[\x80-\xff]+')
    chlist = p.findall(t)

    return chlist


def prodict(freq):
    l = os.listdir('corpus')
    for f in l:
        corpus = open('corpus/' + f, 'r')
        text = corpus.read()
        seg_list = jieba.cut(text, cut_all=False)
        for d in seg_list:
            # print d.encode('utf-8')
            if freq.has_key(d):
                freq[d] += 1
            else:
                freq[d] = 1


# sentiment
def method1():
    freq = {}

    feature = {}
    f = open('feature1.txt', 'r')
    while 1:
        line = f.readline()
        if not line:
            break
        feature[line[:-1].decode('utf-8')] = None

    f.close()
    l = os.listdir('pos')
    posfeats = []
    negfeats = []
    for f in l:
        pos = open('pos/' + f, 'r')
        text = pos.read()
        ret = word_feats(text, feature)
        if ret != {}:
            posfeats.append((word_feats(text, feature), 'pos'))

    for t in posfeats:
        for k in t[0]:
            print(k)

    l = os.listdir('neg')
    for f in l:
        neg = open('neg/' + f, 'r')
        text = neg.read()
        ret = word_feats(text, feature)
        if ret != {}:
            negfeats.append((word_feats(text, feature), 'neg'))

    for t in negfeats:
        for k in t[0]:
            print(k)

    negcutoff = len(negfeats) * 3 / 4
    poscutoff = len(posfeats) * 3 / 4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
    classifier = NaiveBayesClassifier.train(trainfeats)
    test_f = open('test.txt', 'r')
    test = test_f.read()
    # tt = []
    tt = word_feats(test, feature)
    # tt.append((word_feats(test, feature)))
    for t in tt:
        # for k in t:
        print(t)

    print(classifier.classify(word_feats(test, feature)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))


# classifier.show_most_informative_features(5)
def get_price(code, cur_date, length):
    date = {}
    today = datetime.strptime(cur_date, "%Y-%m-%d")
    # print today
    # return
    i = 0
    while len(date) < length:

        y = today - timedelta(days=i)
        t = datetime.strftime(y, "%Y-%m-%d")

        d = t.split('-')

        # 统计日是周末
        weekday = datetime(int(d[0]), int(d[1]), int(d[2])).strftime("%w")
        if weekday == '0' or weekday == '6':
            i += 1
            continue
        i += 1

        date[t] = -1
    url = 'http://quotes.money.163.com/trade/lsjysj_' + str(code) + '.html'

    headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
    req = urllib2.Request(url, headers=headers)
    content = ''
    try:
        content = urllib2.urlopen(req).read()
    except Exception as  e:
        print(e)
    soup = BeautifulSoup(content)

    table = soup.find('table', class_='table_bg001 border_box limit_sale')
    tr = table.findAll('tr')

    web = {}
    for i in range(1, len(tr)):
        td = tr[i].findAll('td')

        web[td[0].contents[0]] = (td[4].contents[0])

    url = 'http://quotes.money.163.com/trade/lsjysj_' + str(code) + '.html?year=2014&season=4'
    print(url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
    req = urllib2.Request(url, headers=headers)
    content = ''
    try:
        content = urllib2.urlopen(req).read()
    except Exception as e:
        print(e)
    soup = BeautifulSoup(content)

    table = soup.find('table', class_='table_bg001 border_box limit_sale')
    tr = table.findAll('tr')

    for i in range(1, len(tr)):
        td = tr[i].findAll('td')

        web[td[0].contents[0]] = (td[4].contents[0])

    price = []
    for key in date:
        if web.has_key(key):
            price.append(float(web[key]))
            # date[key] = float(td[4].contents[0])
    return price


def method2():
    file_names = glob('score*')
    # file_names = glob('industry*')
    file_names.sort()
    wb = xlrd.open_workbook('stock.xls')
    sh = wb.sheet_by_name('stock')
    code = {}
    for rownum in range(sh.nrows):
        if rownum < 2:
            continue
        code[sh.cell(rownum, 1).value] = sh.cell(rownum, 0).value

    length = 5
    pro = {}
    for j in range(len(file_names) - length + 1):
        if file_names[j].find('2014-11') != -1:
            continue

        stock = {}
        i = j
        date = {}
        end = file_names[j + length - 1]
        end_date = end[5:-4]
        d = end[5:-4].split('-')

        # 统计日是周末
        weekday = datetime(int(d[0]), int(d[1]), int(d[2])).strftime("%w")
        if weekday == '0' or weekday == '6':
            i += 1
            continue

        while len(date) < length and i < len(file_names):
            fname = file_names[i]
            date[fname[5:-4]] = -1

            f = open(fname, 'r')

            stock_t = {}
            summ = 0
            while 1:
                line = f.readline()
                # print line
                if not line:
                    break
                array = line[:-1].split('%')
                a = array[0].decode('utf-8')
                stock_t[a] = int(array[2])

                # print line
                summ += int(array[2])

            for key in stock_t:
                if stock in key:
                    stock[key].append(stock_t[key] / float(summ))
                else:
                    stock[key] = [stock_t[key] / float(summ)]

            i += 1

        for key in stock:
            if len(stock[key]) != length:
                continue

            price = get_price(code[key], end_date, length)
            if len(price) != length:
                continue

            pattern = ''
            for i in range(1, len(stock[key]) - 1):
                if stock[key][i] < stock[key][i - 1]:
                    pattern += '1'
                elif stock[key][i] == stock[key][i - 1]:
                    pattern += '2'
                elif stock[key][i] > stock[key][i - 1]:

                    pattern += '0'

            for i in range(1, len(stock[key]) - 1):
                if price[i] < price[i - 1]:
                    pattern += '1'
                elif price[i] == price[i - 1]:
                    pattern += '2'
                elif price[i] > price[i - 1]:

                    pattern += '0'

            if price[length - 1] < price[length - 2]:
                p = 1
            elif price[length - 1] == price[length - 2]:
                p = 2
            elif price[length - 1] > price[length - 2]:
                p = 0

            if pro.has_key(pattern):
                pro[pattern].append(p)
            else:
                pro[pattern] = [p]

    for key in pro:
        up = 0
        down = 0
        draw = 0
        for t in pro[key]:
            if t == 0:
                up += 1
            elif t == 1:
                down += 1
            elif t == 2:
                draw += 1
        summ = up + down + draw

        print(key, up * 1.0 / (summ), down * 1.0 / summ, draw * 1.0 / summ)


def method3():
    file_names = glob('score*')
    file_names.sort()

    wb = xlrd.open_workbook('stock.xls')
    sh = wb.sheet_by_name('stock')
    code = {}
    for rownum in range(sh.nrows):
        if rownum < 2:
            continue
        code[sh.cell(rownum, 1).value] = sh.cell(rownum, 0).value

    price = {}
    stock = {}
    days = 2
    for i in range(days):
        file_name = file_names[len(file_names) - i - 1]
        f = open(file_name, 'r')

        stock_t = {}
        summ = 0
        while 1:

            line = f.readline()
            # print line
            if not line:
                break
            array = line[:-1].split('%')
            a = array[0].decode('utf-8')
            date = file_names[len(file_names) - 1][5:-4]

            today = datetime.strptime(date, "%Y-%m-%d")

            summ += int(array[2])
            # stock.append(int(array[2]))
            if i == 0:
                y = today + timedelta(days=1)
                t = datetime.strftime(y, "%Y-%m-%d")
                # price.append(get_price(code[a], t, 2))
                price[a] = get_price(code[a], t, 2)

            stock_t[a] = int(array[2])

            # print line
            summ += int(array[2])

        for key in stock_t:
            if stock.has_key(key):
                stock[key].append(stock_t[key] / float(summ))
            else:
                stock[key] = [stock_t[key] / float(summ)]

    y = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []

    for t in price:
        if len(price[t]) == 2 and len(stock[t]) == days:
            if price[t][1] > price[t][0]:
                # y.append(0)
                x1.append(stock[t][0])
                y1.append(stock[t][1])

            elif price[t][1] < price[t][0]:
                x2.append(stock[t][0])
                y2.append(stock[t][1])

            # y.append(1)
            else:
                x3.append(stock[t][0])
                y3.append(stock[t][1])
                # y.append(2)
        else:
            y.append(-1)

    fig = plt.figure()
    f = fig.add_subplot(111)
    f.plot(x1, y1, '*', color='red')
    f.plot(x2, y2, '*', color='green')
    f.plot(x3, y3, '*', color='blue')

    plt.show()


def get_pattern(stock_name):
    wb = xlrd.open_workbook('stock.xls')
    sh = wb.sheet_by_name('stock')
    code = {}
    for rownum in range(sh.nrows):
        if rownum < 2:
            continue
        code[sh.cell(rownum, 1).value] = sh.cell(rownum, 0).value

    file_names_t = glob('score*')
    # file_names = glob('industry*')
    file_names_t.sort()

    length = 4
    pro = {}
    stock = {}
    date = {}
    file_names = []
    for fname in file_names_t:
        # d = fname[5:-4].split('-')
        # weekday = datetime.datetime( int(d[0]), int(d[1]), int(d[2])).strftime("%w")

        # if weekday == '0' or weekday == '6' :

        #	continue
        file_names.append(fname)

    for i in range(len(file_names) - length, len(file_names)):

        fname = file_names[i]

        date[fname[5:-4]] = -1

        f = open(fname, 'r')
        stock_t = {}
        summ = 0
        while 1:
            line = f.readline()
            # print line
            if not line:
                break
            array = line[:-1].split('%')
            a = array[0].decode('utf-8')

            stock_t[a] = int(array[2])

            # print line
            summ += int(array[2])

        for key in stock_t:
            if stock.has_key(key):
                stock[key].append(stock_t[key] / float(summ))
            else:
                stock[key] = [stock_t[key] / float(summ)]

    if stock in (stock_name):
        key = stock_name
        # for key in stock:
        # if len(stock[key]) != len(file_names):
        if len(stock[key]) != length:
            print
            'data not enougth'
            return ''

        price = get_price(code[key], file_names[len(file_names) - 1][5:-4], length)

        if len(price) != length:
            print
            'price data not enougth'
            return ''

        pattern = ''
        for i in range(1, len(stock[key])):
            if stock[key][i] < stock[key][i - 1]:
                pattern += '1'
            elif stock[key][i] == stock[key][i - 1]:
                pattern += '2'
            elif stock[key][i] > stock[key][i - 1]:

                pattern += '0'
        # print pattern
        for i in range(1, len(stock[key])):
            if price[i] < price[i - 1]:
                pattern += '1'
            elif price[i] == price[i - 1]:
                pattern += '2'
            elif price[i] > price[i - 1]:

                pattern += '0'

        return pattern


def predict():
    f = open('pattern.txt', 'r')
    pattern = {}
    length = 3
    while 1:
        line = f.readline()
        if not line:
            break
        array = line[:-1].split(' ')
        pattern[array[0]] = int(array[1])

    wb = xlrd.open_workbook('stock.xls')
    sh = wb.sheet_by_name('stock')
    code = {}

    wrong = 0
    right = 0
    for rownum in range(sh.nrows):
        if rownum < 2:
            continue
        print(sh.cell(rownum, 1).value.encode('utf-8'))
        p = get_pattern(sh.cell(rownum, 1).value)
        if pattern.has_key(p):
            ddd = {}
            ddd['2015-01-13'] = -1
            ddd['2015-01-14'] = -1
            price = get_price(sh.cell(rownum, 0).value, '2015-01-13', length)
            t = -1
            if price[1] > price[0]:
                t = 0
            elif price[1] < price[0]:
                t = 1
            else:
                t = 2
            if pattern[p] != t:
                wrong += 1
            else:
                right += 1
            print
            '--------------------------', sh.cell(rownum, 1).value.encode('utf-8'), pattern[p], t

    print
    wrong, right, right * 1.0 / (wrong + right)


# code[sh.cell(rownum, 1).value] = sh.cell(rownum, 0).value

if __name__ == "__main__":
    # method2()
    # get_pattern(u'中信证券')
    # predict()
    method3()
