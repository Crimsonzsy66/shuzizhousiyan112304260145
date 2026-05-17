# 🔢 手写数字识别器

基于CNN的手写数字识别Web应用，使用PyTorch和Gradio构建。

## 在线访问

**HuggingFace Spaces部署地址**: `https://chuxihappy-dukejin112304260147-02.hf.space`

## 功能特性

- 📝 **手写输入**: 在画布上直接手写数字
- 🖼️ **图片上传**: 支持上传手写数字图片
- 📊 **概率展示**: 显示各数字的识别置信度
- 🎨 **美观界面**: 简洁易用的Web界面

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python app_hf.py
```

访问 `http://localhost:7860` 查看应用。

## 技术栈

- **PyTorch**: 深度学习框架
- **Gradio**: Web界面构建
- **CNN**: 卷积神经网络

## 模型信息

- 数据集: MNIST
- 验证准确率: 98.7%
- 网络结构: Conv-BN-ReLU-Pool × 2 + FC

## 项目结构

```
├── app_hf.py          # HuggingFace Spaces主程序
├── app.py             # Flask版本主程序
├── mnist_cnn.pth      # 训练好的模型
├── requirements.txt   # 依赖列表
└── README.md          # 说明文档
```