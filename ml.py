from sklearn.datasets import load_iris  # Imports the standard Iris dataset from scikit
from sklearn import tree


iris = load_iris()
print(list(iris.target_names))  # Lists the target label names
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)
print(classifier.predict([[5.1, 3.5, 1.4, 1.5]]))
