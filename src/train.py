"""
Train a model and save outputs.
"""

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

"""
Load data (replace this with your own)
"""
data = load_iris()
X, y = data.data, data.target

"""
Split the data
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Train model (replace with your model)
"""
model = RandomForestClassifier()
model.fit(X_train, y_train)

"""
Predict
"""
y_pred = model.predict(X_test)

"""
Save confusion matrix in outputs/
"""
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig("C:/Users/HP/Desktop/Python CLass/Project/outputs/confusion_matrix.png")

"""
Save trained model in outputs/
"""
joblib.dump(model, "C:/Users/HP/Desktop/Python CLass/Project/outputs/model.joblib")

print("Training complete. Files saved in outputs/")
