# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
from mpl_toolkits.mplot3d import axes3d, Axes3D
import copy
import itertools
import numpy as np
import scipy.linalg as linalg
from membership import mfDerivs


class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS

    Attributes:
        X
        Y
        XLen
        memClass
        memFuncs
        memFuncsByVariable
        rules
        consequents
        errors
        memFuncsHomo
        trainingType

    """

    def __init__(self, X, Y, memFunction, andfunc='mamdini'):
        self.and_func = andfunc
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]

        # 共m1*m2*...*mn条rules，mi为第i个输入变量的mf数量。
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))

        # X.shape[1] + 1是因为加入一个常数项。
        self.consequents = np.empty(self.Y.shape[1] * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.memFuncsHomo = all(len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'

    # 重写的LSE，能处理多维的Y变量。
    def LSE(self, mat_a, mat_b):
        x, res, r, s = linalg.lstsq(mat_a, mat_b)
        if np.ndim(x) == 1:
            x = x.reshape(len(x), 1)
        return x

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, eta=0.001):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            # layer four: forward pass
            [layerFour, wSum, w] = forwardHalfPass(self, self.X)

            # layer five: least squares estimate
            # layer4是 m = number_of_input_case行，n = len(rule_number) * len(1+input_dimension)列的矩阵
            # 其中，每行是rule和（1+input_variable）的笛卡尔乘积
            self.consequents = np.array(self.LSE(layerFour, self.Y))

            # 第四层到第五层（即结果）的系数矩阵
            layerFive = np.dot(layerFour, self.consequents)

            #error
            error = np.sum((self.Y - layerFive) ** 2)
            print('current error: ', error)
            self.errors = np.append(self.errors, error)

            if error < tolerance:
                convergence = True

            # back propagation
            if convergence is not True:
                cols = range(len(self.X[0, :]))
                dE_dAlpha = list(backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.X.shape[1]))

            # handling of variables with a different number of MFs
            dAlpha = copy.deepcopy(dE_dAlpha)
            if not(self.memFuncsHomo):
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)

            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = sorted(self.memFuncs[varsWithMemFuncs][MFs][1])
                    for param in range(len(paramList)):
                        self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] += dAlpha[varsWithMemFuncs][MFs][param]
            epoch += 1

        self.fittedValues = predict(self, self.X)
        self.residuals = self.Y - self.fittedValues

        return self.fittedValues

    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.errors)), self.errors, 'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, inputVar):
        import matplotlib.pyplot as plt
        from skfuzzy import gaussmf, gbellmf, sigmf

        for mf in range(len(self.memFuncs[inputVar])):
            if self.memFuncs[inputVar][mf][0] == 'gaussmf':
                y = gaussmf(x, **self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'gbellmf':
                y = gbellmf(x, **self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'sigmf':
                y = sigmf(x, **self.memClass.MFList[inputVar][mf][1])

            plt.plot(x, y, 'r')

        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            # plt.plot(range(len(self.fittedValues)),self.fittedValues,'r', label='trained')
            # plt.plot(range(len(self.Y)),self.Y,'b', label='original')
            # plt.legend(loc='upper left')
            # plt.show()

            # 搞成3D的好看点
            fig = plt.figure()
            ax1 = axes3d.Axes3D(fig)
            # ax2 = axes3d.Axes3D(fig)

            z = np.asarray(self.Y[:, 0]).reshape(11, 11)
            znew = np.asarray(self.fittedValues[:, 0]).reshape(11, 11)

            ax1.plot_surface(X=np.asarray(self.X[:, 0]).reshape(11, 11), Y=np.asarray(self.X[:, 1]).reshape(11, 11),
                             Z=znew, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.8)
            # ax2.plot_surface(X=np.asarray(self.X[:, 0]).reshape(11, 11), Y=np.asarray(self.X[:, 1]).reshape(11, 11),
            #                  Z=z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.8)
            plt.show()


def forwardHalfPass(ANFISObj, Xs):
    """
    Parameters
    ----------
    ANFISObj： anfis对象
    Xs： 输入变量

    Returns
    -------
    layerFour：第四层节点，n*m形矩阵。每行一个inputcase，每列分别对应（variable，1）和rules的笛卡尔乘积。
    wSum：第二层节点的和的列表，第三层为归一化的第二层节点，即w/wSum。
    w：第二层节点，n*m形矩阵。每行一个inputcase，每列分别对应一个rule的结果。
    """
    layerFour = np.empty(0,)
    wSum = []

    for pattern in range(len(Xs[:, 0])):
        # pattern 为一行输入
        # 该输入为一个列表，共n个值，n = 输入变量个数

        # layer one
        # 该pattern输入对应的MF输出，每个变量X对应M个MF，共N个X，则layerOne形如
        # | mf1 mf2 mf3...mfm1 |
        # | mf1 mf2 mf3...mfm2 |
        # |....................|
        # | mf1 mf2 mf3...mfmn |
        layerOne = ANFISObj.memClass.evaluateMF(Xs[pattern, :])

        # layer two
        # rule个数为r = m1*m2*...*mn
        # 每个rule为一个列表，形如：
        # [[l1],[l2],...,[lr]]
        # 其中[li]形如：
        # [a1, a2, ..., an], n为X个数，ai值为Xi中某个mf编号
        # print 'rules:\n', ANFISObj.rules
        miAlloc = [[layerOne[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in range(len(ANFISObj.rules))]
        # 第一层不同X的不同mf间互作乘积（‘并’运算）
        if ANFISObj.and_func == 'mamdani':
            layerTwo = np.array([np.min(x) for x in miAlloc]).T
        else:
            layerTwo = np.array([np.product(x) for x in miAlloc]).T

        if pattern == 0:
            w = layerTwo
        else:
            w = np.vstack((w, layerTwo))

        # layer three
        # 归一化
        wSum.append(np.sum(layerTwo))
        if pattern == 0:
            wNormalized = layerTwo/wSum[pattern]
        else:
            wNormalized = np.vstack((wNormalized, layerTwo/wSum[pattern]))

        # prep for layer four (bit of a hack)
        layerThree = layerTwo/wSum[pattern]
        # np.append(Xs,1)是因为引入常数项1
        # np.conatenate()将原来的列数组（rules）摊成行数组：
        # | [x1,   x2,...,   xn,   1]|          //rule1
        # | [x1',  x2',...,  xn',  1]|          //rule2
        # |..........................|          //...
        # | [x1'', x2'',..., xn'', 1]|          //ruleR
        # ------------------------------>
        # | [x1, x2,..., xn, 1, x1', x2',..., xn', 1, .........., xn'', 1]|
        # 这样的一行对应一个inputcase
        rowHolder = np.concatenate([x*np.append(Xs[pattern, :], 1) for x in layerThree])
        layerFour = np.append(layerFour, rowHolder)

    wNormalized = wNormalized.T

    # 按输入的X个数展开layer4，每行对应一个testcase，为一个rowHolder。
    layerFour = np.array(np.array_split(layerFour, pattern + 1))

    return layerFour, wSum, w


def backprop(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):
    """
    Parameters
    ----------
    ANFISObj: anfis对象
    columnX： 第X个变量id
    columns：列表[0, 1, 2,...,len(n)-1]，共n个变量
    theWSum：第二层的和
    theW： 第二层
    theLayerFive：第五层

    Returns
    -------
    paramGrp：bp后的参数Group，[[p1],[p2],[p3],...,[pn]]。共n个mf，[pi]为第i个mf的参数列表。
    """
    paramGrp = [0] * len(ANFISObj.memFuncs[columnX])
    for MF in range(len(ANFISObj.memFuncs[columnX])):

        parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF][1]))
        timesThru = 0

        # alpha为每个mf的参数。
        for alpha in sorted(ANFISObj.memFuncs[columnX][MF][1].keys()):

            # bucket3用于存放每个inputcase对当前MF-alpha参数的修正值
            bucket3 = np.empty(len(ANFISObj.X))
            for rowX in range(len(ANFISObj.X)):

                # 这些是错误语句所用到的变量，现已失效
                # varToTest = ANFISObj.X[rowX, columnX]
                # tmpRow = np.empty(len(ANFISObj.memFuncs))
                # tmpRow.fill(varToTest)

                # bucket2用于存放当前inputcase（即第rowX个input）下，对应output中每个colY值的MF-alpha参数修正值。
                bucket2 = np.empty(ANFISObj.Y.shape[1])
                for colY in range(ANFISObj.Y.shape[1]):

                    # 选出rules中包含MF的那些rule的编号。
                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:, columnX] == MF))[0]

                    # 删除本列后的其他列的编号
                    adjCols = np.delete(columns, columnX)

                    # MF关于alpha的偏导数
                    # 这相当于第一层的delta
                    senSit = mfDerivs.partial_dMF(ANFISObj.X[rowX, columnX], ANFISObj.memFuncs[columnX][MF], alpha)

                    # produces d_ruleOutput/d_parameterWithinMF
                    # 每个rule in rulesWithAlpha为形如(3, 4, 1)的元组（该例为有三个输入变量，该rule由三个输入变量中的第3， 4， 1个
                    # MF的并组成。所以alpha在该rule对应的偏导dW_dAlpha为senSit(alpha所在的MF的偏导)*其他两个输入变量对应的MF值的积。
                    # dW_dAlpha其实就是前两层的对应rulesWithAopha的delta
                    # 一个错误的计算，正确的在下一行
                    # dW_dAplha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])

                    if ANFISObj.and_func == 'T-S':
                        dW_dAlpha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(ANFISObj.X[rowX])[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])
                    else:
                        dW_dAlpha = senSit * np.array([1. for r in rulesWithAlpha])

                    bucket1 = np.empty(len(ANFISObj.rules[:, 0]))
                    for consequent in range(len(ANFISObj.rules[:, 0])):

                        # fConsequent为第4层的delta
                        # 因为第五层与consequent对应的rule的值L5为：rule*x1*w1 + rule*x2*w2 + ...+ rule*xn*wn, n为x的维数
                        # = rule*(x1*w1 + x2*w2 + ...+ xn*wn)
                        # 故第四层delta=(x1*w1 + x2*w2 + ...+ xn*wn)
                        # 上式中rule为第三层与consequent对应的输出
                        l4_rule_gap = ANFISObj.X.shape[1] + 1
                        l4_rule_id = l4_rule_gap * consequent
                        fConsequent = np.dot(np.append(ANFISObj.X[rowX, :], 1.), ANFISObj.consequents[l4_rule_id: l4_rule_id + l4_rule_gap, colY])

                        # acum为第三层及之前各层的delta
                        # 第三层归一化后的rule(L3) = rule(L2)/ Total(rule_l2)
                        # 后Total(rule_l2)记为Total
                        # 故在第三层求alpha的偏导为：
                        # ...................................................
                        # 记rule共n条，第二层偏导数结果为d_rule(对应程序中的dW_dAlpha)
                        # 1、若当前rulei与alpha相关：
                        # d_rule(L3) = (d_rulei * Total - d_Total * rulei) / (Total * Total)
                        # 其中d_rulei不为0，且d_rulei = dW_dAlpha
                        # 2、若当前rulei与alpha无关：
                        # 则上面的d_rulei = 0
                        acum = 0
                        if consequent in rulesWithAlpha:
                            acum = dW_dAlpha[np.where(rulesWithAlpha == consequent)] * theWSum[rowX]

                        acum = acum - theW[rowX, consequent] * np.sum(dW_dAlpha)
                        acum /= theWSum[rowX] ** 2

                        # 该consequent的delta为前四层的delta的积
                        bucket1[consequent] = fConsequent * acum

                    # 加总所有consequent的结果
                    sum1 = np.sum(bucket1)
                    # sum1*最后一层的误差的delta
                    bucket2[colY] = sum1 * (ANFISObj.Y[rowX, colY]-theLayerFive[rowX, colY])*(-2)

                # 加总所有的colY的结果
                sum2 = np.sum(bucket2)
                # MF-alpha中第rowX个input的参数修正delta
                bucket3[rowX] = sum2

            # 加总所有的inputcase的结果
            sum3 = np.sum(bucket3)
            # MF的第timesThru个参数的修正delta
            parameters[timesThru] = sum3
            # 计算MF的下一个参数
            timesThru = timesThru + 1

        # MF的参数修正delta
        paramGrp[MF] = parameters

    return paramGrp


def predict(ANFISObj, varsToTest):

    [layerFour, wSum, w] = forwardHalfPass(ANFISObj, varsToTest)

    #layer five
    layerFive = np.dot(layerFour, ANFISObj.consequents)

    return layerFive


if __name__ == "__main__":
    print("I am main!")
