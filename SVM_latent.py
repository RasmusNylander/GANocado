import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([2, 3, 1, 3, 4])
y = np.array([0, 0, 0, 1, 1])


svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(np.column_stack((X1, X2)), y)


plt.scatter(X1[y==0], X2[y==0], color='red', label='Class 0')
plt.scatter(X1[y==1], X2[y==1], color='blue', label='Class 1')


w = svm_classifier.coef_[0]
b = svm_classifier.intercept_[0]
x = np.linspace(0, 6, 100)  # x轴范围
y = -(w[0] * x + b) / w[1]  # 决策边界的方程
plt.plot(x, y, color='green', label='Decision Boundary')


plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classifier')
plt.show()


predictions = svm_classifier.predict(np.column_stack((X1, X2)))
accuracy = accuracy_score(y, predictions)
print("Accuracy:", accuracy)
