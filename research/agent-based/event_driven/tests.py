import numpy
from anfis import anfis
from membership import membershipfunction

ts = numpy.loadtxt("trainingSet.txt", usecols=[1, 2, 3])
X = ts[:, 0:2]
Y = ts[:, 2:3]
mf = [[['gaussmf',{'mean':-11.,'sigma':5.}],['gaussmf',{'mean':-8.,'sigma':5.}],['gaussmf',{'mean':-14.,'sigma':20.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':-10.,'sigma':20.}],['gaussmf',{'mean':-20.,'sigma':11.}],['gaussmf',{'mean':-9.,'sigma':30.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]

and_func = ['mamdani', 'T-S']

mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc, andfunc=and_func[1])
anf.trainHybridJangOffLine(epochs=50)

print(anf.fittedValues)
print('mf list:')
print(anf.memFuncs)

anf.plotErrors()
anf.plotResults()
anf.plotMF(X[:, 0], 0)
# let's see ...