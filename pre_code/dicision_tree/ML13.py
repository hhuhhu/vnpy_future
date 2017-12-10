__author__ = 'junchenlu'

import numpy as np
from sklearn import svm, preprocessing, tree
from sklearn.neural_network import BernoulliRBM
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation
import matplotlib.pyplot as plt
import math
import timeit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
features_selected = ['DE Ratio',
             'Trailing P/E',
             'Price/Sales']
'''

# features_selected = ['DE Ratio',
#              'Trailing P/E',
#              'Price/Sales',
#              'Price/Book',
#              'Profit Margin',
#              'Operating Margin',
#              'Return on Assets',
#              'Return on Equity',
#              'Revenue Per Share',
#              'Market Cap',
#              'Enterprise Value',
#              'Forward P/E',
#              'PEG Ratio',
#              'Enterprise Value/Revenue',
#              'Enterprise Value/EBITDA',
#              'Revenue',
#              'Gross Profit',
#              'EBITDA',
#              'Net Income Avl to Common ',
#              'Diluted EPS',
#              'Earnings Growth',
#              'Revenue Growth',
#              'Total Cash',
#              'Total Cash Per Share',
#              'Total Debt',
#              'Current Ratio',
#              'Book Value Per Share',
#              'Cash Flow',
#              'Beta',
#              'Held by Insiders',
#              'Held by Institutions',
#              'Shares Short (as of',
#              'Short Ratio',
#              'Short % of Float',
#              'Shares Short (prior ']
#
#
# def Build_Data_Set():
#     data_df = pd.DataFrame.from_csv("finance_stat.csv")
#     print(data_df.keys())
#     #data_df = data_df[:100]
#     data_df = data_df.reindex(np.random.permutation(data_df.index))
#     data_df = data_df.replace("NaN", 0).replace("N/A", 0)
#
#     X = np.array(data_df[features_selected].values)#.tolist())
#     X_norm = X
#
#     y = (data_df["Status"]
#          .replace("underperform",0)
#          .replace("outperform",1)
#          .values.tolist())
#
#     print("Before preprocessing, feature number is " + str(len(X[0])))
#     X = preprocessing.scale(X)
#     X = SelectKBest(k=20).fit_transform(X, y)
#     print("after preprocessing, feature number is " + str(len(X[0])))
#     #X = preprocessing.normalize(X, norm = "l1")
#     #X.reshape(-1, 1)
#
#     return X,X_norm,y
#
#
# def getSet(original, indeces):
#     ret = []
#
#     for ind in indeces:
#         ret.append(original[ind])
#
#     return ret
#
# X, X_norm, y = Build_Data_Set()
# num_sample = len(X)
# print("The number of samples in the original dataset is " + str(num_sample))
#
#
# def Analysis(classifier_info):
#     # perform up to 50 times of the classification and get the mean prediction
#     global X, y
#     test_size = num_sample / 6
#     analysis_times = 1
#     ss = cross_validation.ShuffleSplit(num_sample, n_iter=analysis_times, test_size=0.05, random_state=0)
#
#     total_accuracy = .0
#     total_precision = .0
#     total_recall = .0
#     total_f1 = .0
#     total_times = analysis_times
#
#     for train_indices, test_indices in ss:
#         #print "train" + str(len(train_indices))
#         #print "test" + str(len(test_indices))
#         train_set = getSet(X, train_indices)
#         test_set = getSet(X, test_indices)
#
#         train_lables = getSet(y, train_indices)
#         test_lables = getSet(y, test_indices)
#
#         clf = classifier_info[0]
#         clf.fit(train_set, train_lables)
#
#         total_num = 0
#         tp = 0
#         tn = 0
#         fn = 0
#         fp = 0
#
#         for j in range(len(test_set)):
#             test_val = test_set[j]
#             test_lable = test_lables[j]
#             p_val = clf.predict(test_val)[0]
#             total_num += 1
#
#             if p_val == test_lable:
#                 if test_lable == 1:
#                     tp += 1
#                 else:
#                     tn += 1
#             else:
#                 if test_lable == 0:
#                     fp += 1
#                 else:
#                     fn += 1
#
#         accuracy = float(tp + tn) / total_num if tp != 0 else 0
#         precision = float(tp) / (tp + fp) if tp != 0 else 0
#         recall = float(tp) / (tp + fn) if tp != 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
#
#         total_accuracy += accuracy
#         total_precision += precision
#         total_recall += recall
#         total_f1 += f1
#
#     final_accuracy = total_accuracy / total_times
#     final_precision = total_precision / total_times
#     final_recall = total_recall / total_times
#     final_f1 = total_f1 / total_times
#
#     classifier_name = classifier_info[1]
#     print("Classifier is " + classifier_name + "    accuracy:" + str(final_accuracy) + \
#         "   precision:" + str(final_precision) + " recall:" + str(final_recall) + \
#         "   f1:" + str(final_f1))
#
#     return final_accuracy, final_precision, final_recall, final_f1
#
# degree_ = len(features_selected)
# infos = []
# infos.append([svm.SVC(kernel="rbf", C=1.), "rbf"])
# infos.append([svm.SVC(kernel="poly", C=1., degree=3), "poly"])
# infos.append([svm.SVC(kernel="linear", C=1.), "linear"])
# infos.append([svm.SVC(kernel="sigmoid", C=1.), "sigm"])
# infos.append([tree.DecisionTreeClassifier(), "DecTree"])
# infos.append([RandomForestClassifier(n_estimators=100), "Ranforest"])
# infos.append([GaussianNB(), "NaiBayes"])
# infos.append([KNeighborsClassifier(n_neighbors=math.sqrt(num_sample) / 2), "KNN"])
#
#
# accs = []
# precs = []
# recs = []
# f1s = []
# names = []
# # now let's try something different. change the parameters of each of those classifiers
# start = timeit.default_timer()
# for classifier_info in infos:
#      acc, prec, rec, f1 = Analysis(classifier_info)
#      accs.append(acc)
#      precs.append(prec)
#      recs.append(rec)
#      f1s.append(f1)
#      names.append(classifier_info[1])
#
# end = timeit.default_timer()
# print("time used is " + str(end - start) + " seconds")
#
# #Now plot the accs precs recs f1s
# N = len(infos)
# ind = np.arange(N)  # the x locations for the groups
# width = 0.2       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, accs, width, color='r')
# rects2 = ax.bar(ind + width, precs, width, color='y')
# rects3 = ax.bar(ind + width * 2, recs, width, color='g')
# rects4 = ax.bar(ind + width * 3, f1s, width, color='b')
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('rates')
# ax.set_xticks(ind + width)
# ax.set_xticklabels(tuple(names))
#
# ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
#           ('Accuracy', 'Precision', 'Recall', 'f1'), loc=2,prop={'size':8})
#
# plt.show()
# plt.savefig('barChart')

if __name__ == '__main__':
    data_df = pd.DataFrame.from_csv("finance_stat.csv")
    print(data_df)
    for key in data_df.keys():
        if key != 'Date':
            plt.plot(data_df[key])
            plt.legend()
            plt.show()
