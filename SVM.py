import random
import time

import numpy as np
import torch
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_data

# 计算SVM的参数量和MFLOPS
def count_svm_params(svm):
    num_support_vectors = len(svm.support_vectors_)
    num_features = svm.support_vectors_.shape[1]
    num_params = num_support_vectors * num_features + num_support_vectors  # 支持向量参数和对应的alpha参数
    return num_params

def estimate_svm_flops(svm, X):
    num_support_vectors = len(svm.support_vectors_)
    num_features = svm.support_vectors_.shape[1]
    flops_per_sample = num_support_vectors * num_features  # 每个支持向量和每个特征的内积运算
    total_flops = flops_per_sample * len(X)  # 总浮点运算次数
    return total_flops / 1e6  # 转换为MFLOPS


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
# 加载数据
X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])

# 创建线性核 SVM 模型
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# 在测试集上进行预测
start_time = time.time()
y_pred_linear = linear_svm.predict(X_test)
end_time = time.time()
inference_time=end_time-start_time
print(f"平均运行时间: {inference_time:.4f} 秒")
# 计算准确度
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"准确度 (Linear Kernel): {accuracy_linear}")
num_params = count_svm_params(linear_svm)
mflops = estimate_svm_flops(linear_svm, X_test)

print(f"SVM的参数量: {num_params}")
print(f"SVM的MFLOPS: {mflops:.4f} MFLOPS")



# 创建二次核 SVM 模型
quadratic_svm = svm.SVC(kernel='poly', degree=2)
quadratic_svm.fit(X_train, y_train)

# 在测试集上进行预测
start_time = time.time()

y_pred_quadratic = quadratic_svm.predict(X_test)
end_time = time.time()
inference_time=end_time-start_time
print(f"平均运行时间: {inference_time:.4f} 秒")

# 计算准确度
accuracy_quadratic = accuracy_score(y_test, y_pred_quadratic)
print(f"准确度 (Quadratic Kernel): {accuracy_quadratic}")
num_params = count_svm_params(quadratic_svm)
mflops = estimate_svm_flops(quadratic_svm, X_test)

print(f"SVM的参数量: {num_params}")
print(f"SVM的MFLOPS: {mflops:.4f} MFLOPS")


# 创建三次核 SVM 模型
cubic_svm = svm.SVC(kernel='poly', degree=3)
cubic_svm.fit(X_train, y_train)

# 在测试集上进行预测
start_time = time.time()
y_pred_cubic = cubic_svm.predict(X_test)
end_time = time.time()
inference_time=end_time-start_time
print(f"平均运行时间: {inference_time:.4f} 秒")
# 计算准确度
accuracy_cubic = accuracy_score(y_test, y_pred_cubic)
print(f"准确度 (Cubic Kernel): {accuracy_cubic}")
num_params = count_svm_params(cubic_svm)
mflops = estimate_svm_flops(cubic_svm, X_test)

print(f"SVM的参数量: {num_params}")
print(f"SVM的MFLOPS: {mflops:.4f} MFLOPS")