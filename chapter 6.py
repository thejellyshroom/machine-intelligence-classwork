from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#Start of example 1
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
plot_tree(tree_clf,
           feature_names=iris.get('feature_names'),
           class_names=iris.get('target_names'))
plt.show()
#End of example 1

#Start of example 2
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
#End of example 2

'''
#Start of example 3
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
plot_tree(tree_reg,
           feature_names=iris.get('feature_names'),
           class_names=iris.get('target_names'))
plt.show()
#End of example 3
'''


