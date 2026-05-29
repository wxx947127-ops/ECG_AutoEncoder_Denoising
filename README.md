# 基于自编码器的ECG心电信号去噪系统

## 项目简介

本项目实现了一个面向 ECG 心电信号的去噪系统，使用一维卷积自编码器学习噪声残差，并通过 `denoised = noisy - predicted_noise` 的方式恢复干净心电波形。系统支持 CSV 心电数据读取、模型训练、去噪推理、传统滤波对比、ECG 特征波形标注以及 Streamlit 网页展示。

数据默认格式为无表头 CSV，每个文件 12 列，每列对应一个 ECG 导联。当前系统默认读取第 2 列，也就是 B 列，作为单导联 ECG 信号，并按长度 1000 进行处理。

## 功能介绍

- 自动遍历 `data/raw` 下的 ECG CSV 文件并构建 PyTorch Dataset
- 默认读取 CSV 第 2 列作为单导联 ECG 信号
- 对 ECG 信号进行 NaN 去除、归一化和 1000 点切片
- 基于一维卷积自编码器进行 ECG 去噪
- 采用残差去噪结构，模型输出预测噪声 `predicted_noise`
- 支持 Early Stopping 和最佳模型保存
- 保存训练结果：
  - `results/best_model.pth`
  - `results/loss.png`
  - `results/denoise_result.png`
  - `results/metrics.csv`
  - `results/training_history.csv`
- 支持传统滤波器对比：
  - Butterworth 低通滤波
  - 50Hz Notch 陷波滤波
  - Median 中值滤波
  - Combined Filter 组合滤波
- 支持 Tkinter 桌面 GUI
- 支持 Streamlit 网页应用
- 支持 ECG 特征显示：
  - P 波
  - QRS 波群
  - T 波
  - R 峰
- 支持评价指标展示：
  - MSE
  - MAE
  - Pearson 相关系数
  - SNR Improvement

## 技术栈

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-signal 相关算法：`scipy.signal`
- Streamlit
- Tkinter

## 项目结构

```text
ECG_AutoEncoder_Denoising/
├── app.py                         # Streamlit 网页应用入口
├── README.md                      # 项目说明文档
├── requirements.txt               # Python 依赖
├── data/
│   └── raw/                       # 原始 ECG CSV 数据
├── results/
│   ├── best_model.pth             # 训练得到的最佳模型
│   ├── model.pth                  # 历史模型文件
│   ├── loss.png                   # 损失曲线
│   ├── denoise_result.png         # 去噪效果图
│   ├── metrics.csv                # 随机样本评价指标
│   ├── training_history.csv       # 训练历史
│   ├── filter_comparison.csv      # 自编码器与传统滤波指标对比
│   └── filter_comparison.png      # 自编码器与传统滤波效果对比图
└── src/
    ├── dataset.py                 # ECG 数据集构建
    ├── model.py                   # 自编码器模型
    ├── noise.py                   # 噪声生成
    ├── train.py                   # 模型训练脚本
    ├── gui.py                     # Tkinter 桌面 GUI
    ├── traditional_filters.py     # 传统滤波方法
    └── evaluate.py                # 预留评估脚本
```

## 安装方法

建议使用 Anaconda 或 Miniconda 创建独立环境。

```powershell
conda create -n xxwecg python=3.13
conda activate xxwecg
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果环境中缺少 PyTorch，可根据本机 CUDA 或 CPU 情况安装。例如 CPU 版本：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

安装 Streamlit：

```powershell
pip install streamlit
```

本项目当前使用的解释器示例：

```powershell
D:\Anaconda\envs\xxwecg\python.exe
```

## 使用方法

### 1. 准备数据

将原始 ECG CSV 文件放入：

```text
data/raw/
```

CSV 要求：

- 无表头
- 12 列
- 每列为一个 ECG 导联
- 默认使用第 2 列作为单导联 ECG 信号

### 2. 训练模型

运行训练脚本：

```powershell
D:\Anaconda\envs\xxwecg\python.exe src\train.py
```

训练过程会自动：

- 读取全部 CSV
- 构建训练集和验证集
- 使用 Early Stopping
- 保存最佳模型到 `results/best_model.pth`
- 保存损失曲线和评价指标

### 3. 启动 Tkinter 桌面 GUI

```powershell
D:\Anaconda\envs\xxwecg\python.exe src\gui.py
```

桌面 GUI 支持：

- 选择本地 ECG CSV 文件
- 添加模拟噪声
- 自编码器去噪
- 传统滤波对比
- P 波、QRS 波群、T 波、R 峰标注
- 保存滤波对比结果

### 4. 启动 Streamlit 网页应用

```powershell
D:\Anaconda\envs\xxwecg\python.exe -m streamlit run app.py
```

启动后浏览器会打开本地网页，通常地址为：

```text
http://localhost:8501
```

网页应用支持：

- 上传 CSV 文件
- 自动读取第 2 列 ECG 信号
- 加载 `results/best_model.pth`
- 显示原始信号、带噪信号和去噪信号
- 显示 MSE、MAE、Pearson、SNR Improvement
- 显示 P 波、QRS 波群、T 波和 R 峰

## 评价指标

### MSE

MSE 是均方误差，用于衡量去噪信号与干净信号之间的平方误差。

```text
MSE = mean((denoised - clean)^2)
```

MSE 越小，说明去噪结果与原始干净信号越接近。

### MAE

MAE 是平均绝对误差，用于衡量去噪信号与干净信号之间的平均绝对偏差。

```text
MAE = mean(abs(denoised - clean))
```

MAE 越小，说明整体幅值误差越小。

### Pearson

Pearson 相关系数用于衡量去噪信号与干净信号的波形相关性。

```text
Pearson = corr(clean, denoised)
```

Pearson 越接近 1，说明去噪后波形与原始 ECG 形态越一致。

### SNR Improvement

SNR Improvement 表示去噪前后信噪比的提升量。

```text
SNR Improvement = SNR_after - SNR_before
```

其中：

```text
SNR = 10 * log10(signal_power / noise_power)
```

SNR Improvement 越大，说明去噪效果越明显。

## Streamlit在线部署说明

可以将项目部署到 Streamlit Community Cloud 或其他支持 Python 的云平台。

### 部署准备

确保项目包含：

```text
app.py
requirements.txt
src/
results/best_model.pth
```

### requirements.txt 示例

```text
torch
numpy
pandas
matplotlib
scipy
streamlit
```

### Streamlit Community Cloud 部署步骤

1. 将项目上传到 GitHub 仓库
2. 登录 Streamlit Community Cloud
3. 选择 GitHub 仓库
4. 设置入口文件为：

```text
app.py
```

5. 点击 Deploy 开始部署

### 注意事项

- `results/best_model.pth` 必须随项目一起上传，否则网页无法加载模型
- 如果模型文件较大，建议使用 Git LFS 或云存储下载机制
- 在线部署环境通常使用 CPU 推理，首次加载模型可能需要等待
- 如果部署平台不支持本地大文件，应将模型文件放到可访问的下载地址，并在 `app.py` 中添加自动下载逻辑

## 项目截图占位符

### Streamlit 网页应用

![Streamlit ECG去噪系统截图](docs/images/streamlit_app_placeholder.png)

### Tkinter 桌面 GUI

![Tkinter ECG去噪系统截图](docs/images/tkinter_gui_placeholder.png)

### 去噪效果图

![ECG去噪效果图](results/denoise_result.png)

## 作者信息

作者：熊相文

学校：华东交通大学

专业：通信工程ICT

邮箱：wxx947127@gmail.com

GitHub：https://github.com/wxx947127-ops/ECG_AutoEncoder_Denoising?utm_source=chatgpt.com

## 许可证

本项目仅用于学习、课程设计、科研实验和原型验证。若用于医学诊断、临床辅助或商业系统，请进行充分的数据验证、模型评估和合规审查。
