from data import test_loader
import torch
from net import Net
import matplotlib.pyplot as plt



# 实例化模型
network = Net()
network_state_dict = torch.load('model.pth')
# 加载参数
network.load_state_dict(network_state_dict)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  # 第i个样本第0个通道的像素[28,28]
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()