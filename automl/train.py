

from common_import import *


dataset = ImageDataset(data=makeImagePathList_Celeb())
print(dataset)
data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
exit()

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.explorer import Explorer
from nni.nas.pytorch import enas
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner


# ネットワークの定義
class ENASNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(ENASNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = LayerChoice([
                nn.Conv2d(self.input_size, 64, 3, padding=1),
                nn.Conv2d(self.input_size, 128, 3, padding=1),
                nn.Conv2d(self.input_size, 256, 3, padding=1)
            ])
            self.layers.append(layer)
        
        self.fc = nn.Linear(256, self.output_size)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# モデルの探索空間の定義
def get_model(input_size, output_size, num_layers):
    model = ENASNetwork(input_size, output_size, num_layers)
    return model

# 学習の設定
input_size = 3
output_size = 10
num_layers = 3
batch_size = 64
lr = 0.001
epochs = 50
train_loader, valid_loader, test_loader = load_data()  # データローダーの読み込み

# ENASの探索スペースの定義
input_node = InputChoice(n_candidates=3, n_chosen=1)
prev_node = input_node
for i in range(num_layers):
    conv_node = LayerChoice([
        nn.Conv2d(input_size, 64, 3, padding=1),
        nn.Conv2d(input_size, 128, 3, padding=1),
        nn.Conv2d(input_size, 256, 3, padding=1)
    ])
    prev_node = conv_node

# ENASのExplorerの設定
explorer = Explorer.from_spec([input_node, prev_node])
model = get_model(input_size, output_size, num_layers)
fixed_model = apply_fixed_architecture(model, explorer.get_arc())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fixed_model.parameters(), lr=lr)

# モデルの学習
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = fixed_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, epochs, batch_idx+1, len(train_loader), loss.item()))

    # バリデーションの評価
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = fixed_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_acc = 100 * correct / total
    print('Validation Accuracy: {:.2f}%'.format(val_acc))

    # NNIによるハイパーパラメータ探索
    nni.report_intermediate_result(val_acc)  # 中間結果を報告
    if nni.check_trial_budget():  # 予算をチェック
        break

# テストの評価
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = fixed_model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

test_acc = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(test_acc))

# 最終結果を報告
nni.report_final_result(test_acc)
