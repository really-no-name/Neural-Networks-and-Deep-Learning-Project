import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 数据加载器
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

tudui = Tudui().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(tudui.parameters(), lr=0.001)

# TensorBoard SummaryWriter初始化
writer = SummaryWriter("./logs")

# 训练和测试
epochs = 10
for epoch in range(epochs):
    tudui.train()
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == targets).sum().item()

        # 记录每个批次的损失
        writer.add_scalar('Training loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

    # 每个epoch的训练精度
    train_accuracy = running_correct / len(train_data)

    # 测试阶段
    tudui.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

    # 每个epoch的测试精度
    test_accuracy = correct / len(test_data)

    # 在TensorBoard中记录训练和测试的精度
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_dataloader):.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 关闭TensorBoard的SummaryWriter
writer.close()
