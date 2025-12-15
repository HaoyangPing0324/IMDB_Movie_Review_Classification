# 使用 PyTorch 实现 IMDB 电影评论分类

## 项目来源
苏州大学未来科学与工程学院《人工智能》课程
任课老师：吴洪状

## 任务描述
使用 PyTorch 训练一个 RNN（或 LSTM/GRU），对 IMDB 数据集的电影评论进行分类（正面 / 负面）。

## 要求
1. **加载数据**：使用 Hugging Face datasets 库加载 IMDB 数据集，并进行文本预处理（分词、填充、词向量）。注意：需要安装datasets和transformers 库
2. **构建模型**：使用 PyTorch 实现一个 RNN（或LSTM / GRU），输入电影评论文本，输出分类结果（0：负面，1：正面）。
3. **训练模型**：用交叉熵损失函数 torch.nn.CrossEntropyLoss()，优化器可选 Adam 或 SGD。
4. **测试模型**：在测试集上评估准确率。
5. **可视化示例**：选取若干个测试样本，展示原始评论文本、真实标签、预测标签。

## 提示
1. 使用预训练的 BERT 分词器进行分词：`tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")`。
2. RNN 结构可以用 `nn.LSTM` 或 `nn.GRU` 实现。
3. 训练时需使用 padding 处理不同长度的输入序列。
4. 评估时使用 `torch.no_grad()` 进行推理，避免梯度计算。
