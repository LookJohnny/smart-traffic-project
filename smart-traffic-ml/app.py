from flask import Flask, request, jsonify
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# 定义 CNN_LSTMModel 类
class CNN_LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(CNN_LSTMModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 调整回 LSTM 所需的维度
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        return output


# 加载预训练的模型
model = torch.load('model.pth')
model.eval()

# 定义标准化器，用于特征预处理
scaler = StandardScaler()


# 定义预测函数
def predict(data):
    # 处理数据，转换为 PyTorch tensor
    data = np.array(data).reshape(1, -1)  # 假设数据为一维数组
    data = scaler.transform(data)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # 使用模型预测
    with torch.no_grad():
        output = model(data_tensor)
        _, predicted = torch.max(output.data, 1)

    return int(predicted.item())


# 定义 API 路由
@app.route('/predict', methods=['POST'])
def predict_api():
    # 从请求中获取数据
    if request.method == 'POST':
        try:
            data = request.json['data']  # 假设输入数据为 JSON 格式，键名为 'data'
            prediction = predict(data)
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})


# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
