import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from datetime import datetime


# --- 数据准备 ---

# 指定数据目录
data_dir = "./data"

# 进行数据增强和预处理
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomRotation(10),  # 随机旋转±10度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # 随机平移
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 创建训练集 Dataset 对象
train_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,  # 如果 data_dir 中没有数据，则下载；如果已有，则使用
    transform=train_transform  # 在这里应用转换
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建测试集 Dataset 对象
test_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 检查设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"计算设备是{device}")


# --- 构建模型 ---
class CNN(nn.Module):  # 定义CNN类，继承自 nn.module
    def __init__(self):  # 初始化方法，定义网络层
        super(CNN, self).__init__()  # 调用父类的初始化方法

        # --- 卷积层和池化层 ---
        # 第一个卷积块（conv1）
        """
        输入通道：1（灰度图，MNIST图像是单通道的）
        输出通道：32（尝试通过卷积核提取32种不同的特征）
        卷积核大小：3*3（一个3*3的滑动窗口）
        padding：1（在图像边缘填充一圈0，目的是保证卷积后的特征图尺寸不变）
        公式: Output_size = (Input_size - Kernel_size + 2*Padding) / Stride + 1
        对于28x28输入, 3x3核, padding=1, stride=1 (默认): (28 - 3 + 2*1)/1 + 1 = 28
        输入 (N, 1, 28, 28) -> conv1 -> 输出 (N, 32, 28, 28)
        """
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 第二个卷积块（conv2）
        """
        输入通道：32
        输出通道：64（进一步提取64种特征）
        卷积核大小：3*3
        padding：1
        输入 (N,32,14,14)（经过了第一次池化） -> conv2 -> 输出 (N,64,14,14)
        """
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

       # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层 (pool)
        """
        2x2 最大池化 (kernel_size=2)
        步长: 2 (stride=2)
        这会使特征图的高度和宽度都减半。
        例如: 28x28 -> MaxPool(2,2) -> 14x14
        14x14 -> MaxPool(2,2) -> 7x7
        7x7 -> MaxPool(2,2) -> 3x3
        """
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dropout层 ---
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # --- 全连接层部分 ---
        # 全连接层（fc1）
        """
        经过三次池化后，原始 28*28 的图像尺寸变为 3*3
        此时有 self.conv3 输出的 128 个特征图
        展平 (flatten) 后的特征向量维度是 128 (通道数) * 3 (高度) * 3 (宽度) = 1152。
        这个全连接层将 1152 维的特征向量映射到 512 维。
        """
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 修复：使用正确的维度计算

        # 输出层 (fc2)
        """
        将 512 维的特征向量映射到 128 维
        """
        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):  # 前向传播方法，定义数据如何流经网络层
        # x 的初始形状: (batch_size, 1, 28, 28)

        # --- 第一个卷积块 ---
        # x 通过 conv1: (N, 1, 28, 28) -> (N, 32, 28, 28)
        # 应用 ReLU 激活函数 (引入非线性): 形状不变 (N, 32, 28, 28)
        # 应用池化层 pool: (N, 32, 28, 28) -> (N, 32, 14, 14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # --- 第二个卷积块 ---
        # x 通过 conv2: (N, 32, 14, 14) -> (N, 64, 14, 14)
        # 应用 ReLU 激活函数: 形状不变 (N, 64, 14, 14)
        # 应用池化层 pool: (N, 64, 14, 14) -> (N, 64, 7, 7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三个卷积块: (N, 64, 7, 7) -> (N, 128, 7, 7)
        # 应用池化层 pool: (N, 128, 7, 7) -> (N, 128, 3, 3)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # --- 展平 ---
        # 展平 (Flatten) 操作，为全连接层做准备。
        # 将形状 (N, 128, 3, 3) 的张量展平成 (N, 128*3*3) = (N, 1152) 的二维张量。
        # -1 表示 PyTorch 会自动计算该维度的大小 (即 batch_size)。
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout1(x)

        # --- 进入全连接层 ---
        # 通过第一个全连接层 fc1: (N, 1152) -> (N, 512)
        # 应用 ReLU 激活函数: 形状不变 (N, 512)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        # 通过第二个全连接层 fc2: (N, 512) -> (N, 128)
        # 应用 ReLU 激活函数: 形状不变 (N, 128)
        x = F.relu(self.fc2(x))

        # 通过第三个全连接层 fc3: (N, 128) -> (N, 10)
        x = self.fc3(x)
        return x  # 返回模型的最终输出 (logits)


# --- 早停类 ---


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_accuracy = 0
        self.counter = 0
        self.best_weights = None

    def __call__(self, accuracy, model):
        if accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

# --- 评估函数 ---


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# --- 保存训练信息函数 ---


def save_training_info(epoch, train_loss, test_accuracy, test_loss, model_path, info_path):
    training_info = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'device': str(device)
    }

    # 如果文件存在，读取现有数据
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            all_info = json.load(f)
    else:
        all_info = []

    # 添加新的训练信息
    all_info.append(training_info)

    # 保存更新后的信息
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)


# --- 实例化CNN模型，并将其参数和缓冲区移动到指定的device (CPU或GPU) ---
model = CNN().to(device)
print(model)  # 打印模型结构


# --- 损失函数 (Loss Function) 和优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# --- 创建保存目录 ---
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)

# --- 初始化早停 ---
early_stopping = EarlyStopping(patience=10, min_delta=0.01)

# --- 训练过程 ---
num_epochs = 50  # 增加最大轮次，但会通过早停控制
training_info_path = os.path.join(save_dir, "training_history.json")
print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(
                f'轮次 [{epoch+1}/{num_epochs}], 批次 [{i+1}/{len(train_loader)}], 损失值: {loss.item():.4f}')

    # 计算平均训练损失
    avg_train_loss = running_loss / len(train_loader)

    # 评估模型
    test_accuracy, test_loss = evaluate_model(model, test_loader, device)

    print(f'轮次 {epoch+1} 完成:')
    print(f'  训练损失: {avg_train_loss:.4f}')
    print(f'  测试准确率: {test_accuracy:.2f}%')
    print(f'  测试损失: {test_loss:.4f}')
    print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')

    # 保存模型权重
    model_filename = f"mnist_cnn_epoch_{epoch+1}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # 保存训练信息
    save_training_info(epoch, avg_train_loss, test_accuracy,
                       test_loss, model_path, training_info_path)

    # 更新学习率
    scheduler.step()

    # 检查早停条件
    if early_stopping(test_accuracy, model):
        print(f"早停触发！在第 {epoch+1} 轮次停止训练")
        print(f"最佳准确率: {early_stopping.best_accuracy:.2f}%")
        break

print("\n训练完成！")

# --- 最终评估 ---
final_accuracy, final_loss = evaluate_model(model, test_loader, device)
print(f'\n最终模型在 {len(test_dataset)} 张测试图片上的准确率: {final_accuracy:.2f}%')

# --- 保存最终最佳模型 ---
best_model_path = os.path.join(save_dir, "best_mnist_cnn_model.pth")
torch.save(model.state_dict(), best_model_path)
print(f"最佳模型已保存到: {best_model_path}")
print(f"训练历史已保存到: {training_info_path}")

# --- 打印训练总结 ---
print(f"\n训练总结:")
print(f"- 最佳测试准确率: {early_stopping.best_accuracy:.2f}%")
print(f"- 模型文件保存在: {save_dir}")
print(f"- 训练信息保存在: {training_info_path}")
