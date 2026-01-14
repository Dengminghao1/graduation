import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
from feature_extract.resnet.face_feature import *
warnings.filterwarnings('ignore')


# 设置随机种子确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ==================== 1. 自定义数据集类 ====================
class AttentionDataset(Dataset):
    """注意力等级数据集"""

    def __init__(self, root_dir, transform=None, phase='train'):
        """
        参数:
            root_dir: 数据集根目录
            transform: 数据增强变换
            phase: 数据集阶段 ('train', 'val', 'test')
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.phase = phase

        # 获取所有图片路径和标签
        self.image_paths = []
        self.labels = []

        # 假设目录结构为: root_dir/phase/label/*.jpg
        for label_dir in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label_dir)
            if not os.path.isdir(label_path):
                continue

            label = int(label_dir)  # 假设目录名为标签编号 (0, 1, 2, 3)
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(label_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"{phase}数据集: {len(self.image_paths)} 张图片, {len(set(self.labels))} 个类别")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 2. 数据预处理和增强 ====================
def get_transforms(phase='train', input_size=224):
    """获取数据预处理和增强变换"""

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # val/test
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ==================== 3. ResNet模型定义 ====================
class AttentionResNet(nn.Module):
    """基于ResNet的注意力等级分类模型"""

    def __init__(self, num_classes=4, model_name='resnet50', pretrained=True):
        super(AttentionResNet, self).__init__()

        # 加载预训练的ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 冻结部分层（可选）
        if pretrained:
            # 只训练最后几层
            for param in self.backbone.parameters():
                param.requires_grad = False

            # 解冻最后两个block
            for param in list(self.backbone.parameters())[-30:]:
                param.requires_grad = True

        # 替换最后的全连接层
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        """解冻所有层用于微调"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ==================== 4. 训练函数 ====================
class Trainer:
    """训练器类"""

    def __init__(self, model, device, num_classes=4):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

    def train_epoch(self, dataloader, criterion, optimizer, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        if scheduler:
            scheduler.step()

        return epoch_loss, epoch_acc

    def validate(self, dataloader, criterion):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validating', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def test(self, dataloader):
        """测试"""
        return self.validate(dataloader, None)


# ==================== 5. 训练主函数 ====================
def main():
    # 超参数设置
    config = {
        'data_dir': 'path/to/your/dataset',  # 修改为你的数据集路径
        'input_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 50,
        'num_classes': 4,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'model_name': 'resnet50',  # resnet18, resnet34, resnet50, resnet101
        'pretrained': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"使用设备: {config['device']}")
    print(f"模型: {config['model_name']}")

    # 创建数据加载器
    train_transform = get_transforms('train', config['input_size'])
    val_transform = get_transforms('val', config['input_size'])

    train_dataset = AttentionDataset(config['data_dir'], train_transform, 'train')
    val_dataset = AttentionDataset(config['data_dir'], val_transform, 'val')
    test_dataset = AttentionDataset(config['data_dir'], val_transform, 'test')

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True)

    # 创建模型
    model = AttentionResNet(num_classes=config['num_classes'],
                            model_name=config['model_name'],
                            pretrained=config['pretrained'])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=config['num_epochs'])

    # 训练器
    trainer = Trainer(model, config['device'], config['num_classes'])

    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    # 最佳模型保存
    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    # 训练循环
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader, criterion, optimizer, scheduler)

        # 验证
        val_loss, val_acc, _, _ = trainer.validate(val_loader, criterion)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {history['lr'][-1]:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, f'checkpoints/best_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    # ==================== 6. 模型评估 ====================
    print("\n" + "=" * 50)
    print("训练完成，开始评估...")
    print("=" * 50)

    # 加载最佳模型
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上评估
    test_loss, test_acc, test_preds, test_labels = trainer.validate(test_loader, criterion)
    print(f"\n测试集结果:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")

    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds,
                                target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))

    # 绘制混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Level 0', 'Level 1', 'Level 2', 'Level 3'],
                yticklabels=['Level 0', 'Level 1', 'Level 2', 'Level 3'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    # ==================== 7. 可视化训练过程 ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 训练/验证准确率
    axes[0].plot(history['train_acc'], label='Train Accuracy')
    axes[0].plot(history['val_acc'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 训练/验证损失
    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 学习率变化
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

    # ==================== 8. 保存完整模型 ====================
    # 保存完整模型（用于推理）
    torch.save(model, 'attention_resnet_final.pth')

    # 保存ONNX格式（可选）
    try:
        dummy_input = torch.randn(1, 3, config['input_size'], config['input_size']).to(config['device'])
        torch.onnx.export(model, dummy_input, "attention_resnet.onnx",
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        print("模型已保存为ONNX格式: attention_resnet.onnx")
    except Exception as e:
        print(f"保存ONNX格式失败: {e}")

    print("\n训练完成！模型已保存到:")
    print("  - checkpoints/best_model.pth (最佳模型)")
    print("  - attention_resnet_final.pth (完整模型)")
    print("  - training_history.png (训练历史)")
    print("  - confusion_matrix.png (混淆矩阵)")


# ==================== 9. 推理函数 ====================
def predict_single_image(image_path, model_path='attention_resnet_final.pth', device='cuda'):
    """预测单张图片的注意力等级"""

    # 加载模型
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用，使用CPU")

    model = torch.load(model_path, map_location=device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # 注意力等级描述
    attention_levels = {
        0: "完全不关注 (Level 0)",
        1: "分心状态 (Level 1)",
        2: "一般关注 (Level 2)",
        3: "高度集中 (Level 3)"
    }

    print(f"\n预测结果:")
    print(f"  图片: {os.path.basename(image_path)}")
    print(f"  注意力等级: {attention_levels[predicted_class]}")
    print(f"  置信度: {confidence * 100:.2f}%")

    # 显示所有类别的概率
    print(f"\n各等级概率:")
    for i in range(4):
        prob = probabilities[0][i].item() * 100
        print(f"  {attention_levels[i]}: {prob:.2f}%")

    return predicted_class, confidence


# ==================== 10. 批量预测函数 ====================
def predict_batch(image_dir, model_path='attention_resnet_final.pth', device='cuda'):
    """批量预测目录中的图片"""

    # 加载模型
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = torch.load(model_path, map_location=device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 获取图片文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = []
    print(f"\n批量预测 {len(image_files)} 张图片:")
    print("-" * 50)

    for img_file in tqdm(image_files, desc="预测中"):
        img_path = os.path.join(image_dir, img_file)

        try:
            # 加载和预处理图片
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            results.append({
                'filename': img_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0]
            })

        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            results.append({
                'filename': img_file,
                'error': str(e)
            })

    # 统计结果
    if results and 'predicted_class' in results[0]:
        class_counts = {}
        for result in results:
            if 'predicted_class' in result:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\n预测结果统计:")
        print("-" * 30)
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            percentage = (count / len(image_files)) * 100
            attention_levels = ['完全不关注', '分心状态', '一般关注', '高度集中']
            print(f"  {attention_levels[cls]} (Level {cls}): {count} 张 ({percentage:.1f}%)")

    return results


# ==================== 11. 迁移学习微调 ====================
def fine_tune_model(model_path, new_data_dir, num_new_classes=4, num_epochs=20):
    """
    微调预训练模型

    参数:
        model_path: 预训练模型路径
        new_data_dir: 新数据集目录
        num_new_classes: 新数据的类别数
        num_epochs: 微调epoch数
    """

    print("开始微调模型...")

    # 加载预训练模型
    model = torch.load(model_path)

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 修改最后的分类层
    if hasattr(model, 'backbone'):
        num_ftrs = model.backbone.fc[6].in_features
        model.backbone.fc[6] = nn.Linear(128, num_new_classes)
    else:
        # 如果是普通ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_new_classes)

    # 重新训练
    # ... (使用与main()类似的训练流程)

    return model


if __name__ == "__main__":
    # 执行训练
    main()

    # 使用示例 - 预测单张图片
    # predict_single_image("test_image.jpg")

    # 使用示例 - 批量预测
    # results = predict_batch("test_images/")