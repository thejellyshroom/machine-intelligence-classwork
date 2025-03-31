import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


#Generate 200 random numbers. Since np.random.seed has a value 42, we get the same numbers
np.random.seed(42)
X = np.random.rand(200, 1) - 0.5  # a single random input feature
y = X ** 2 + 0.025 * np.random.randn(200, 1)


tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)

plot_tree(tree_reg)
plt.show()


#Start of scatter plot example
plt.scatter(X, y, c=y, cmap='viridis')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Number Scatter Plot')
plt.show()
#end of scatter plot example
