"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/16
@Desc    : 使用 PyTorch 实现 IMDB 电影评论分类
@License : MIT License (MIT)
@Version : 1.0

"""

#### 导入库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#### 数据处理
def data_processing():
    # ------------------------------
    # 1. 加载 IMDB 数据集（Hugging Face datasets 方法）
    # ------------------------------
    dataset = load_dataset("imdb", cache_dir="./data")
    print("数据集结构：", dataset)
    # 检查一个样本的结构
    print(dataset["train"][20000])

    # ------------------------------
    # 2. 数据预处理：使用预训练的 BERT 分词器进行分词
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir="./data"  # 分词器文件会下载到这个路径下
    )
    test_text = "This movie is absolutely fantastic! I love it so much."
    # 2. 用分词器处理文本
    tokens = tokenizer(test_text)
    # 3. 打印结果，查看是否有正常输出
    print("===== 分词器加载成功验证 =====")
    print("1. 原始文本：", test_text)
    print("2. 分词后的token IDs（数字编码）：", tokens["input_ids"])
    print("3. 注意力掩码（attention_mask）：", tokens["attention_mask"])
    print("4. 转换回文本（反向验证）：", tokenizer.decode(tokens["input_ids"]))

    def preprocess_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokens["label"] = examples["label"]  # 确保 label 被保留
        return tokens

    # 对整个数据集进行分词处理
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    print("分词后的数据集示例：", tokenized_datasets["train"][0])

    # ------------------------------
    # 3. 转换为 PyTorch 数据格式，并创建 DataLoader
    # ------------------------------
    # 指定返回格式为 PyTorch tensor
    train_dataset = tokenized_datasets["train"].with_format("torch")
    test_dataset = tokenized_datasets["test"].with_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader,tokenizer

#### 模型构建
class SimpleRNN(nn.Module):
    def __init__(self,
                 num_classes=2,          # IMDB二分类，默认设为2
                 vocab_size=30522,       # BERT-base的词汇表大小，对应bert-base-uncased
                 embedding_dim=128,      # 词嵌入维度
                 rnn_hidden_dim=256,     # RNN隐藏层维度
                 rnn_layers=2,           # RNN层数，至少2层（和CNN保持一致要求）
                 rnn_type="LSTM",        # RNN类型：RNN/LSTM/GRU
                 dropout_rate=0.5,       # Dropout概率
                 bidirectional=True,     # 是否使用双向RNN
                 max_seq_len=512):       # 序列最大长度，对应预处理的max_length=512
        super(SimpleRNN, self).__init__()

        # 检查RNN层数是否满足要求
        if rnn_layers < 2:
            raise ValueError(f"RNN层数需至少为2层，当前为{rnn_layers}层")

        # 1. 词嵌入层：将token IDs转换为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0对应BERT的PAD token

        # 2. 动态构建RNN层（支持RNN/LSTM/GRU、双向、多层）
        rnn_kwargs = {
            "input_size": embedding_dim,
            "hidden_size": rnn_hidden_dim,
            "num_layers": rnn_layers,
            "bidirectional": bidirectional,
            "batch_first": True,  # 输入格式为(batch, seq_len, feature)，符合文本数据习惯
            "dropout": dropout_rate if rnn_layers > 1 else 0  # 多层时才用dropout
        }

        # 根据类型创建RNN层
        if rnn_type == "RNN":
            self.rnn = nn.RNN(**rnn_kwargs)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"暂不支持RNN类型：{rnn_type}，可选'RNN'/'LSTM'/'GRU'")

        # 3. 计算RNN输出的维度（双向则维度翻倍）
        self.rnn_output_dim = rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim

        # 4. 分类器（全连接层，和CNN的classifier结构对齐）
        classifier_layers = []
        classifier_layers.append(nn.Linear(self.rnn_output_dim, self.rnn_output_dim // 2))  # 隐藏层
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(self.rnn_output_dim // 2, num_classes))  # 输出层
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        # x: 输入的token IDs，形状为(batch_size, max_seq_len)

        # 1. 词嵌入：(batch_size, max_seq_len) → (batch_size, max_seq_len, embedding_dim)
        x_emb = self.embedding(x)

        # 2. RNN层：输出为(output, hidden)，其中output是所有时间步的输出，hidden是最后一层的隐藏状态
        if isinstance(self.rnn, nn.LSTM):
            rnn_output, (hidden, cell) = self.rnn(x_emb)
            # 取最后一层的最后一个时间步的隐藏状态（双向则拼接两个方向的最后状态）
            if self.rnn.bidirectional:
                final_hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
            else:
                final_hidden = hidden[-1]
        else:  # RNN/GRU
            rnn_output, hidden = self.rnn(x_emb)
            if self.rnn.bidirectional:
                final_hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
            else:
                final_hidden = hidden[-1]

        # 3. 分类器：(batch_size, rnn_output_dim) → (batch_size, num_classes)
        output = self.classifier(final_hidden)

        return output

#### 模型训练
def train(model, device ,train_loader, test_loader , optimizer_choice ,epochs=20,lr=0.001):
    ## 采用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    ## 使用 SGD 或 Adam 进行优化
    # 移除冗余的初始赋值，直接根据选择创建优化器
    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)           # Adam 优化器
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)            # SGD 优化器
    else:
        raise ValueError("优化器选择无效！请选择 'Adam' 或 'SGD' .")

    ## 训练至少 10 轮（Epochs）
    # 校验epochs并抛出异常
    if epochs < 10:
        # 使用raise抛出ValueError，附带指定错误信息
        raise ValueError("训练至少 10 轮（Epochs）")

    ## 训练多个 epoch，并在测试集上评估准确率
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # 修正：遍历批次时提取input_ids和label（适配文本数据），替换原images、labels
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(input_ids)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        ## 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # 修正：测试阶段同样提取input_ids和label
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return train_losses, test_accuracies

#### 结果可视化
### 绘制训练损失曲线
def plot_training_loss(train_losses ,epochs=10):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

### 绘制测试集准确率曲线
def plot_test_accuracy(test_accuracies ,epochs=10):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.show()

### 计算并绘制混淆矩阵（适配IMDB二分类、文本数据）
def plot_confusion_matrix(model, test_loader, device, classes):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 从批次中提取input_ids和label（IMDB数据集的张量名）
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    # 适配二分类，标注更清晰
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('IMDB Movie Review Classification Confusion Matrix')
    plt.show()


def plot_test_review_results(test_loader, model, device, classes, tokenizer):
    # 取一个批次的测试数据
    dataiter = iter(test_loader)
    batch = next(dataiter)

    # 提取数据并限制展示数量（选前5个，避免内容过多）
    input_ids = batch["input_ids"][:5].to(device)
    labels = batch["label"][:5].to(device)
    # 保留原始文本（从数据集里取对应位置的text，这里需注意loader的数据集是否保留text，若没有则从input_ids解码）
    # 方式：从input_ids解码回文本（适配分词后的张量）
    reviews = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"][:5]]

    # 模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)

    # 展示结果
    for i in range(len(reviews)):
        print(f"\n=== 评论 {i + 1} ===")
        print(f"原始评论：{reviews[i][:200]}..." if len(reviews[i]) > 200 else f"原始评论：{reviews[i]}")
        print(f"真实标签：{classes[labels[i].item()]}")
        print(f"预测标签：{classes[predicted[i].item()]}")

#### 主函数
def main():
    classes = ["negative", "positive"]
    epochs = 15
    optimizer_choice = 'Adam'
    lr = 0.0005
    # 定义SimpleRNN的参数
    rnn_params = {
        "num_classes": 2,  # IMDB二分类
        "vocab_size": 30522,  # bert-base-uncased词汇表大小
        "embedding_dim": 256,  # 词嵌入维度
        "rnn_hidden_dim": 512,  # RNN隐藏层维度
        "rnn_layers": 2,  # RNN层数（至少2层）
        "rnn_type": "LSTM",  # 使用LSTM（可选RNN/GRU）
        "dropout_rate": 0.3,  # Dropout概率
        "bidirectional": True,  # 双向LSTM
        "max_seq_len": 256  # 序列最大长度，和预处理一致
    }

    ## 主体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备（如果有 GPU 可用则使用 GPU）
    train_loader, test_loader,tokenizer = data_processing()
    model = SimpleRNN(**rnn_params).to(device)  # 模型构建
    train_losses, test_accuracies = train(model, device,
                                          train_loader, test_loader,
                                          optimizer_choice, epochs, lr)  # 模型训练
    plot_training_loss(train_losses=train_losses, epochs=epochs)  # 绘制训练损失曲线
    plot_test_accuracy(test_accuracies=test_accuracies, epochs=epochs)  # 绘制测试集准确率曲线
    plot_confusion_matrix(model=model, test_loader=test_loader,
                          device=device, classes=classes)  # 绘制混淆矩阵
    # 修改后的评论结果展示调用语句（新增tokenizer参数，适配新函数）
    plot_test_review_results(test_loader=test_loader, model=model,
                             device=device, classes=classes, tokenizer=tokenizer)  # 显示部分测试评论及其预测结果

#### 运行
if __name__ == "__main__":
    main()