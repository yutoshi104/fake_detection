
from common_import import *



# 前処理
transform = transforms.Compose([
    # 画像をTensorに変換してくれる
    # チャネルラストをチャネルファーストに
    # 0〜255の整数値を0.0〜1.0の浮動少数点に変換してくれる
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)



num_batches = 100
train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
train_iter = iter(train_dataloader)
# 100個だけミニバッチからデータをロードする
imgs, labels = train_iter.next()

# 100個のデータ, グレースケール, 28px, 28px
print(imgs.size())




img = imgs[0]
# 画像データを表示するために、チャネルファーストのデータをチャネルラストに変換する
img_permute = img.permute(1, 2, 0)
# tensorから2次元のarrayに変換する
hm = sns.heatmap(img_permute.numpy()[:, :, 0])
print(hm)











class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第1引数：input
            # 第2引数：output
            nn.Linear(28 * 28, 400),
            # メモリを節約出来る
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        output = self.classifier(x)
        return output





model = MLP()

model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


###学習###
num_epochs = 15
losses = []
accs = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.view(num_batches, -1)
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        running_loss += loss.item()
        # dim=1 => 0-9の分類方向のMax値を返す
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    # 600回分で割る
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))



loss_plt = plt.plot(losses)
plt.savefig("loss.png")
plt.clf()
acc_plt = plt.plot(accs)
plt.savefig("acc.png")
plt.clf()




imgs_gpu = imgs.view(100, -1).to(device)
output= model(imgs_gpu)
pred = torch.argmax(output, dim=1)
print(pred)
print(label)
print()



###パラメータ保存###
# 重み、バイアスを抜き出す
params = model.state_dict()
# 抜き出した重み、バイアスをprmファイルに保存する。
torch.save(params, "model.prm")


###パラメータ再開###
param_load = torch.load("model.prm")
model.load_state_dict(param_load)




###再学習###
num_epochs = 10
losses = []
accs = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.view(num_batches, -1)
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        running_loss += loss.item()
        # dim=1 => 0-9の分類方向のMax値を返す
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    # 600回分で割る
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))
