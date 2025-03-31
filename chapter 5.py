import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC



#start of first example
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))
#end of first example


#Start of second example
X, y = make_moons(n_samples=500, noise=0.1)
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge",max_iter=5000))
    ])

polynomial_svm_clf.fit(X, y)
print(polynomial_svm_clf.predict([[2.0,-0.5]]))
#end of second example


#start of third example
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
print(poly_kernel_svm_clf.predict([[2.0,-0.5]]))
#end of third example


#start of fourth example
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=1000.0))
    ])
rbf_kernel_svm_clf.fit(X, y)
print(rbf_kernel_svm_clf.predict([[0,0]]))
#end of fourth example
