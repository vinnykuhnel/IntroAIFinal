from numpy import ndarray
from sklearn import datasets, neighbors

def kNearest(irisData: ndarray):
    kn = neighbors.KNeighborsClassifier(n_neighbors=1)
    kn.fit(irisData.data, irisData.target)





irisData = datasets.load_iris()
kNearest(irisData)

