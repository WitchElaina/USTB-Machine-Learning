import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CSV_PATH = 'peng.csv'

# 读取csv文件
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print('文件未找到')
    exit(0)

# 删除“island”和“sex”列
df = df.drop(['island', 'sex'], axis=1)

# 绘制散点图矩阵
sns.pairplot(df, hue='species')
plt.show()

# 分离目标属性
target = df['species']

# 分离数值型属性
# 从第二列开始选择所有列，即选择除目标属性外的所有数值型属性
features = df.iloc[:, 1:]

# 输出目标属性和数值型属性的形状
print('目标属性的形状：', target.shape)
print('数值型属性的形状：', features.shape)

# 使用LinearDiscriminantAnalysis将4维数据降维至2维
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(features, target)

# 输出降维后数据的形状
print('降维后数据的形状：', X_lda.shape)

# 将降维后的数据与目标属性合并
df_lda = pd.DataFrame(data=X_lda, columns=['lda1', 'lda2'])
df_lda['species'] = target

# 使用seaborn绘制降维后的散点图
sns.scatterplot(x='lda1', y='lda2', hue='species', data=df_lda)
plt.show()

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.7, random_state=42)

# 使用LinearDiscriminantAnalysis进行分类预测
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# 使用classification_report方法显示分类结果
print(classification_report(y_test, y_pred))
