import matplotlib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# 读取数据
df = pd.read_csv(r"D:\GraduationProject\demo1\output\merged_pose_eeg_feature_files.csv")
# 删除包含空值的行
df = df.dropna()
# 去除列名中的空格，并选择从 gaze_0_x 到 p_33 之间的列
df.columns = df.columns.str.strip()  # 去除列名首尾空格
start_column = 'x0'
end_column = 'y24'

if start_column in df.columns and end_column in df.columns:
    all_cols = df.columns.tolist()
    start_idx = all_cols.index(start_column)
    end_idx = all_cols.index(end_column)
    feature_columns = all_cols[start_idx:end_idx+1]
    # 过滤掉以 'conf' 开头的列
    feature_columns = [col for col in feature_columns if not col.lower().startswith('conf')]
    print(f"特征个数: {len(feature_columns)}")
else:
    print(f"警告: 找不到列 {start_column} 或 {end_column}")


target_column = 'attention'  # 请根据实际列名修改
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查目标变量分布
# plt.figure(figsize=(5, 5))
# plt.subplot(1, 1, 1)
# ax1=sns.histplot(df[target_column], kde=True)
# plt.title('注意力值分布')
# # 添加数量标注
# for bar in ax1.patches:
#     height = bar.get_height()
#     if height > 0:  # 只标注高度大于0的柱子
#         ax1.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{int(height)}',
#                 ha='center', va='bottom', fontsize=10)
# plt.tight_layout()
# plt.show()

# 分离特征和目标
X = df[feature_columns]
y = df[target_column]
label_mapping={
    '低':0,
    '稍低':1,
    '中性':2,
    '稍高':3,
    '高':4
}
# 如果注意力是分类变量，需要先编码（如果是字符串标签）
if y.dtype == 'object':
    y=y.map(label_mapping)

# 重新划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 分层采样
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 分类模型
class_models = {
    '逻辑回归': LogisticRegression(max_iter=10000, random_state=42),
    '贝叶斯': GaussianNB() ,
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

class_results = {}

for name, model in class_models.items():
    model.fit(X_train_scaled, y_train)
    # -----------------------------------------------------------------------------------------------------
    # # 获取特征重要性
    # feature_importance = model.feature_importances_
    # feature_names = X.columns
    # importance_df = pd.DataFrame({
    #     'feature': feature_names,
    #     'importance': feature_importance
    # }).sort_values('importance', ascending=False)
    #
    # print("前10个最重要特征:")
    # print(importance_df.head(10))
    # --------------------------------------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # y_pred = model.predict(X_train_scaled)  # 改为使用训练集
    #
    # accuracy = accuracy_score(y_train, y_pred)  # 改为使用训练集真实标签
    # precision = precision_score(y_train, y_pred, average='weighted')
    # recall = recall_score(y_train, y_pred, average='weighted')
    # f1 = f1_score(y_train, y_pred, average='weighted')

    class_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(f"{name}:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print("-" * 40)

    # 显示分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    # print(classification_report(y_train, y_pred))


