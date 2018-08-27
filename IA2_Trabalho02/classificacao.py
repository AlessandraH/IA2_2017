import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


#  54 atributes + class (15120 amostras)
dataset_name = 'forest_cover.csv'
titles = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'class']


dataset = pd.read_csv(dataset_name, header=None, names=titles)


X = np.array(dataset.ix[:, 0:53])
y = np.array(dataset['class'])


def knn(test_size, n_neighbors):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))


def knn_neighbors(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    knn_scores = []
    for k in range(1, 50):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        scores = accuracy_score(y_test, pred)
        knn_scores.append(scores.mean())
    plt.plot(knn_scores)
    plt.title("K-Nearest Neighbors (test_size = %.2f)" % test_size)
    plt.grid(True)
    plt.xlabel('Number of neighbors K = x + 1')
    plt.ylabel('Accuracy score')
    plt.show()


def knn_testsize(n_neighbors):
    knn_scores = []
    for test_size in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        scores = accuracy_score(y_test, pred)
        knn_scores.append(scores.mean())
    plt.plot(knn_scores)
    plt.title("K-Nearest Neighbors (n_neighbors = %d)" % n_neighbors)
    plt.grid(True)
    plt.xlabel('Test size = (x + 1)/10')
    plt.ylabel('Accuracy score')
    plt.show()


def mlp(test_size, neurons):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # mlp = MLPClassifier(hidden_layer_sizes=neurons)
    # mlp = MLPClassifier(hidden_layer_sizes=(neurons, neurons))
    mlp = MLPClassifier(hidden_layer_sizes=(neurons, neurons, neurons))
    # mlp = MLPClassifier(hidden_layer_sizes=(neurons, neurons, neurons, neurons))
    mlp.fit(x_train, y_train)
    pred = mlp.predict(x_test)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))


def mlp_1layer(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlp_scores = []
    for i in range(1, 30):
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp = MLPClassifier(hidden_layer_sizes=i, random_state=1)
        mlp.fit(x_train, y_train)
        pred = mlp.predict(x_test)
        scores = accuracy_score(y_test, pred)
        mlp_scores.append(scores.mean())
    plt.plot(mlp_scores)
    plt.title("Multiple layer perceptron (2 layers w/ test_size = %.2f)" % test_size)
    plt.grid(True)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy score')
    plt.show()


def mlp_2layers(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlp_scores = []
    for i in range(1, 30):
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp = MLPClassifier(hidden_layer_sizes=(i, i), random_state=1)
        mlp.fit(x_train, y_train)
        pred = mlp.predict(x_test)
        scores = accuracy_score(y_test, pred)
        mlp_scores.append(scores.mean())
    plt.plot(mlp_scores)
    plt.title("Multiple layer perceptron (2 layers w/ test_size = %.2f)" % test_size)
    plt.grid(True)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy score')
    plt.show()


def mlp_3layers(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlp_scores = []
    for i in range(1, 30):
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp = MLPClassifier(hidden_layer_sizes=(i, i, i), random_state=1)
        mlp.fit(x_train, y_train)
        pred = mlp.predict(x_test)
        scores = accuracy_score(y_test, pred)
        mlp_scores.append(scores.mean())
    plt.plot(mlp_scores)
    plt.title("Multiple layer perceptron (3 layers w/ test_size = %.2f)" % test_size)
    plt.grid(True)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy score')
    plt.show()


def mlp_testsize(neurons):
    mlp_1l = []
    mlp_2l = []
    mlp_3l = []
    for test_size in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp1 = MLPClassifier(hidden_layer_sizes=neurons, random_state=1)
        mlp1.fit(x_train, y_train)
        pred1 = mlp1.predict(x_test)
        scores1 = accuracy_score(y_test, pred1)
        mlp_1l.append(scores1.mean())

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp2 = MLPClassifier(hidden_layer_sizes=(neurons, neurons), random_state=1)
        mlp2.fit(x_train, y_train)
        pred2 = mlp2.predict(x_test)
        scores2 = accuracy_score(y_test, pred2)
        mlp_2l.append(scores2.mean())

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp3 = MLPClassifier(hidden_layer_sizes=(neurons, neurons, neurons), random_state=1)
        mlp3.fit(x_train, y_train)
        pred3 = mlp3.predict(x_test)
        scores3 = accuracy_score(y_test, pred3)
        mlp_3l.append(scores3.mean())

    plt.plot(mlp_1l, color='m', label=u"1 layer", linestyle="-")
    plt.plot(mlp_2l, color='b', label="2 layers", linestyle='--')
    plt.plot(mlp_3l, color='r', label="3 layers", linestyle=':')
    plt.legend(loc='lower right')
    plt.title("Multiple layer perceptron w/ %d neurons" % neurons)
    plt.grid(True)
    plt.xlabel('Test size = (x + 1)/10')
    plt.ylabel('Accuracy score')
    plt.show()


def mlp_neurons(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mlp_1l = []
    mlp_2l = []
    mlp_3l = []
    for i in range(1, 30):

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp1 = MLPClassifier(hidden_layer_sizes=i, random_state=1)
        mlp1.fit(x_train, y_train)
        pred1 = mlp1.predict(x_test)
        scores1 = accuracy_score(y_test, pred1)
        mlp_1l.append(scores1.mean())

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp2 = MLPClassifier(hidden_layer_sizes=(i, i), random_state=1)
        mlp2.fit(x_train, y_train)
        pred2 = mlp2.predict(x_test)
        scores2 = accuracy_score(y_test, pred2)
        mlp_2l.append(scores2.mean())

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mlp3 = MLPClassifier(hidden_layer_sizes=(i, i, i), random_state=1)
        mlp3.fit(x_train, y_train)
        pred3 = mlp3.predict(x_test)
        scores3 = accuracy_score(y_test, pred3)
        mlp_3l.append(scores3.mean())

    plt.plot(mlp_1l, color='m', label=u"1 layer", linestyle="-")
    plt.plot(mlp_2l, color='b', label="2 layers", linestyle='--')
    plt.plot(mlp_3l, color='r', label="3 layers", linestyle=':')
    plt.legend(loc='lower right')
    plt.title("Multiple layer perceptron (test_size = %.2f)" % test_size)
    plt.grid(True)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy score')
    plt.show()


# knn_testsize(1)
# knn_neighbors(0.10)
# knn(0.10, 1)
# mlp_testsize(5)
# mlp_neurons(0.20)
mlp(0.20, 30)
# mlp_1layer(0.20)
# mlp_2layers(0.20)
# mlp_3layers(0.20)
