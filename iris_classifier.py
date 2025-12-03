"""
Basic test script to load the Iris dataset and print the first five rows.
"""

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.head())
