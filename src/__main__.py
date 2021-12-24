# Imports
from sklearn.datasets import load_iris  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore

irus_data = load_iris()

x = irus_data.data
y = irus_data.target

# stupid mistake
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=42)


knn_test = KNeighborsClassifier(n_neighbors=7)

knn_test.fit(x1, y1)

# resulta

print(knn_test.predict(x2))
