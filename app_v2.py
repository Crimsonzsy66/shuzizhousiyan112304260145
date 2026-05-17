"""
Kaggle数字识别竞赛 - Web识别系统 v2.0
基于新训练的CNN模型
"""

from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import socket

original_getfqdn = socket.getfqdn
def patched_getfqdn(name=''):
    try:
        return original_getfqdn(name)
    except UnicodeDecodeError:
        return 'localhost'
socket.getfqdn = patched_getfqdn

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("加载模型...")
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()
print("模型加载成功！")

def preprocess_image(image):
    image = image.convert('L')
    img_array = np.array(image)
    
    img_array = img_array.astype(np.float32) / 255.0
    
    img_array = 1.0 - img_array
    
    contours = np.where(img_array > 0.05)
    if len(contours[0]) > 0:
        min_row, max_row = contours[0].min(), contours[0].max()
        min_col, max_col = contours[1].min(), contours[1].max()
        
        padding = 4
        min_row = max(0, min_row - padding)
        max_row = min(img_array.shape[0], max_row + padding)
        min_col = max(0, min_col - padding)
        max_col = min(img_array.shape[1], max_col + padding)
        
        img_array = img_array[min_row:max_row+1, min_col:max_col+1]
    
    height, width = img_array.shape
    if height == 0 or width == 0:
        return torch.zeros(1, 1, 28, 28).float()
    
    max_dim = max(height, width)
    scale = 24 / max_dim
    
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
    img_resized = np.array(img_pil).astype(np.float32) / 255.0
    
    img_final = np.zeros((28, 28))
    
    start_row = (28 - new_height) // 2
    start_col = (28 - new_width) // 2
    
    img_final[start_row:start_row+new_height, start_col:start_col+new_width] = img_resized
    
    img_final = (img_final - np.min(img_final)) / (np.max(img_final) - np.min(img_final) + 1e-6)
    
    img_tensor = torch.from_numpy(img_final).view(1, 1, 28, 28).float()
    return img_tensor

@app.route('/')
def index():
    return render_template('index_new.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file.stream)
        img_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            prob = F.softmax(output, dim=1)[0].tolist()
        
        return jsonify({
            'prediction': int(predicted.item()),
            'probabilities': {str(i): round(p * 100, 2) for i, p in enumerate(prob)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()
        base64_image = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        img_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            prob = F.softmax(output, dim=1)[0].tolist()
        
        return jsonify({
            'prediction': int(predicted.item()),
            'probabilities': {str(i): round(p * 100, 2) for i, p in enumerate(prob)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动服务器: http://127.0.0.1:5001/")
    app.run(host='127.0.0.1', port=5001, debug=False)