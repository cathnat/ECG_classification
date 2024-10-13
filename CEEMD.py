import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
import time
# 生成示例信号
t = np.linspace(0, 1, 512)
n = 1
signal = np.random.rand(128, 512)
# 创建CEEMDAN对象并进行信号分解
ceemdan = CEEMDAN()
high_freq = np.zeros((128, 512))
low_freq = np.zeros((128, 512))
residual = np.zeros((128, 512))
start_time = time.time()
for i in range(signal.shape[0]):
    imfs = ceemdan(signal[i])


    # 提取高频分量、低频分量和残差分量
    high_freq[i] = np.sum(imfs[:2], axis=0)
    low_freq[i] = np.sum(imfs[4:], axis=0)
    residual[i] = imfs[3]

end_time = time.time()  # 记录结束时间
inference_time = (end_time - start_time)  # 计算推理时间
print(f'Inference Time: {inference_time} seconds')
# 绘制原始信号和分解后的分量
plt.figure(figsize=(10, 3))

plt.subplot(3, 1, 1)
plt.plot(t, high_freq)
plt.title('high_freq')

plt.subplot(3, 1, 2)
plt.plot(t, low_freq)
plt.title('low_freq')

plt.subplot(3, 1, 3)
plt.plot(t, residual)
plt.title('residual')

plt.tight_layout()
plt.show()