# conformer&squeezeformer自动语音识别模型代码复现

**项目结构：**

```
conformer(squeezeformer)（各版本结构相似）/
│
├── conf/                          # 配置文件
│   └── config.yaml                # 模型与训练配置
│
├── data/                          # 数据处理脚本
│   ├── aishell_data_process.py    # 生成数据清单
│   ├── extract_data.py            # 解压数据集
│   └── tokenizer.py               # 生成词汇表
│
├── model/                         # 模型代码
│   ├── __init__.py                # 模型入口（ConformerASR）
│   ├── attention.py               # 相对位置注意力机制
│   ├── conformer.py               # Conformer/Squeezeformer编码器
│   ├── convolution.py             # 卷积模块
│   ├── decoder.py                 # Transformer解码器
│   └── feed_forward.py            # 前馈网络
│
├── utils/                         # 工具模块
│   ├── __init__.py                # 工具入口
│   ├── checkpoint.py              # 模型保存与加载
│   ├── dataset.py                 # 数据集加载器
│   ├── logger.py                  # 日志记录
│   └── metrics.py                 # 评估指标
│
├── data_aishell/                  # AIShell数据集（需自行准备）
│   ├── wav/                       # 音频文件
│   └── transcript/                # 标注文件
│
├── checkpoints/                   # 模型保存目录（自动创建）
├── logs/                          # 日志目录（自动创建）
│
├── train.py                       # 训练脚本
├── inference.py                   # 推理脚本
├── test_dataset.py                # 测试集评估
└── test_train_set.py              # 训练集评估
```

## pytorch框架下运行注意事项

### **环境配置**

conformer2.0，3.0需要pytorch3.9以上环境，包括os pandas numpy torch等深度学习常用工具包和tarfile tqdm math torchaudio argparse等辅助工具包。

### **数据集准备**

1. 下载aishell数据集并解压，如果还未解压可借助项目文件夹中的data/extract_data.py脚本解压。
2. 将aishell数据集里的dev,train,test文件夹放到项目文件夹的data_aishell文件夹下，
  或者进入conf/config.yaml，更改data部分的data_path路径为自己的路径。
3. 如果data文件夹下没有dev, test, train的csv文件，需先运行data/aishell_data_process.py文件生成。

### **模型测试**

1. 确保checkpoints文件夹下有best_model.pth和latest_model.pth的权重文件，这是已训练好的模型。
2. 试单个音频文件：
  运行inference.py: python inference.py --wav "data_aishell/wav/test/S0764/BAC009S0764W0121.wav"
  直接在终端运行即可。如果aishell数据集不在项目文件夹内，需把双引号内的路径改为自己的路径，测试音频可随意替换。
3. 测试训练集：
  运行test_train_set.py：python test_train_set.py --checkpoint checkpoints/best_model.pth --sample_count 1000
  确保conf/config.yaml内data_path路径正确即可。
  本命令只测试前一千条样本，全训练集共约12万条样本，可修改--sample_count 1000决定测试数目。
4. 测试验证集：
  运行test_dataset.py：python test_dataset.py --checkpoint checkpoints/best_model.pth --sample_count 50
  确保conf/config.yaml内data_path路径正确即可。
  本命令只测试前50条样本，全验证集约七千条样本，去掉--sample_count 50可全部测试。
