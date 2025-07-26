from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to model/iris_model.pkl")