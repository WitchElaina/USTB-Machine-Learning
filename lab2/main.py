
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 加载手写数字识别图片数据集
digits = load_digits()

# 显示所有图片
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')

plt.show()

# 将数据集分成训练集和测试集，其中20%作为测试数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构造不同参数的SVM模型进行训练和预测
svm_params = [
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1, 'kernel': 'linear'},
    {'C': 10, 'kernel': 'linear'},
    {'C': 0.1, 'kernel': 'rbf', 'gamma': 0.001},
    {'C': 1, 'kernel': 'rbf', 'gamma': 0.001},
    {'C': 10, 'kernel': 'rbf', 'gamma': 0.001},
    {'C': 0.1, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 1, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 10, 'kernel': 'rbf', 'gamma': 0.01},
]

best_accuracy = 0
best_svm = None

for params in svm_params:
    svm = SVC(**params)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(params, "Accuracy:", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_svm = svm

print("Best SVM:", best_svm, "Accuracy:", best_accuracy)

