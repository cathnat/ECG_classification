import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }
# 设置随机种子
seed_value = config['seed']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# X_train,y_train is the training set
# X_test,y_test is the test set
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])
# 定义集成学习参数
num_classifiers = 5
subspace_size = int(np.sqrt(X_train.shape[1]))

# 创建子空间判别器集合
subspace_classifiers = []

for i in range(num_classifiers):
    # 随机选择子空间特征
    subspace_features = np.random.choice(X_train.shape[1], size=subspace_size, replace=False)

    # 创建基分类器（决策树）并在子空间上训练
    base_classifier = DecisionTreeClassifier(random_state=42)
    base_classifier.fit(X_train[:, subspace_features], y_train)

    # 将子空间特征和训练好的分类器保存到集合中
    subspace_classifiers.append((subspace_features, base_classifier))


# 集成学习预测
def ensemble_predict(subspace_classifiers, X):
    predictions = np.zeros((X.shape[0], len(subspace_classifiers)))

    for i, (subspace_features, classifier) in enumerate(subspace_classifiers):
        X_subspace = X[:, subspace_features]
        predictions[:, i] = classifier.predict(X_subspace)

    # 取投票结果作为最终预测
    final_predictions = np.argmax(predictions, axis=1)
    return final_predictions

# 计算ESD算法的参数量和MFLOPS
def count_esd_params(classifiers):
    total_params = 0
    for subspace_features, classifier in classifiers:
        total_params += classifier.tree_.node_count
    return total_params

def estimate_esd_flops(classifiers, X):
    total_flops = 0
    for subspace_features, classifier in classifiers:
        num_nodes = classifier.tree_.node_count
        flops_per_tree = num_nodes * len(X)  # 每个节点假设执行一次浮点运算
        total_flops += flops_per_tree
    return total_flops / 1e6  # 转换为MFLOPS

# 在测试集上进行预测
start_time = time.time()
y_pred_ensemble = ensemble_predict(subspace_classifiers, X_test)
end_time = time.time()
inference_time=end_time-start_time
print(f"平均运行时间: {inference_time:.4f} 秒")
# 计算准确度
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"测试集准确度 (Ensemble Subspace Discriminant): {accuracy_ensemble}")

# 计算参数量和MFLOPS
total_params = count_esd_params(subspace_classifiers)
mflops = estimate_esd_flops(subspace_classifiers, X_test)

print(f"ESD算法的参数量: {total_params}")
print(f"ESD算法的MFLOPS: {mflops:.2f} MFLOPS")