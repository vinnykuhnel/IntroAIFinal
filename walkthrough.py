from numpy import ndarray
from sklearn import datasets, neighbors
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans


#Supervised K Nearest Neighbor Algorithm that checks 3 neighbors
def kNearest(trainData: ndarray, trainTarget: ndarray, testData: ndarray, testTarget: ndarray):
    kn = neighbors.KNeighborsClassifier(n_neighbors=3)
    kn.fit(trainData, trainTarget)   

    
    result = kn.predict(testData)
    correctCounter = 0
    for prediction, correct in zip(result, testTarget):
        if prediction == correct:
            correctCounter += 1

    return (correctCounter / len(testTarget))

#Supervised Decision tree Algorithm
def DecisionTree(trainData: ndarray, trainTarget: ndarray, testData: ndarray, testTarget: ndarray):
    decTree = DecisionTreeClassifier()
    decTree.fit(trainData, trainTarget)
    return decTree.score(testData, testTarget)
    
#Supervised Backpropogation Network
def BackProp(trainData: ndarray, trainTarget: ndarray, testData: ndarray, testTarget: ndarray):
    scaler = StandardScaler()
    scaler.fit(trainData)
    trainData = scaler.transform(trainData)
    testData = scaler.transform(testData)
    BackMLP = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    BackMLP.fit(trainData, trainTarget)
    result = BackMLP.predict(testData)
    correctCounter = 0
    for prediction, correct in zip(result, testTarget):
        if prediction == correct:
            correctCounter += 1

    return (correctCounter / len(testTarget))

#Unsupervised K means clustering algorithm
def KCluster(trainData: ndarray, testData: ndarray, testTarget: ndarray, clusterNum: int):
    kmc = KMeans(n_clusters = clusterNum, init = 'k-means++', random_state = 21)
    kmc.fit(trainData)
    result = kmc.predict(testData)
    correctCounter = 0
    for prediction, correct in zip(result, testTarget):
        if prediction == correct:
            correctCounter += 1

    return (correctCounter / len(testTarget))
    

#Load data set
irisData = datasets.load_iris()
x = irisData.data
y = irisData.target

#Randomly split set into training and test data 
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

print("Iris dataset: ")
print("KNearest " + str(kNearest(x_train, y_train, x_test, y_test)))
print("Decision Tree " + str(DecisionTree(x_train, y_train, x_test, y_test)))
print("Back Propogation " + str(BackProp(x_train, y_train, x_test, y_test)))
print("K Means Cluster " + str(KCluster(x_train, x_test, y_test, 4)))


digits = load_digits()
x = digits.data
y = digits.target

#Randomly split set into training and test data 
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

print("Digits dataset: ")
#run algorithms again with digits dataset
print("KNearest " + str(kNearest(x_train, y_train, x_test, y_test)))
print("Decision Tree " + str(DecisionTree(x_train, y_train, x_test, y_test)))
print("Back Propogation " + str(BackProp(x_train, y_train, x_test, y_test)))
print("K Means Cluster " + str(KCluster(x_train, x_test, y_test, 10)))