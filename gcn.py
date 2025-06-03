# transformers==4.40.0
# 
# torch==1.11.0+cu113
# 
# torch-geometric==2.5.2
# 
# torch-scatter==2.0.9
# 
# torch-sparse==0.6.13

# 训练模型


import pickle
with open("/root/autodl-tmp/graph_data/hc3_train.pkl", "rb") as f:
    hc3_train = pickle.load(f)
with open("/root/autodl-tmp/graph_data/hc3_val.pkl", "rb") as f:
    hc3_val = pickle.load(f)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time

# 构建 GCN 模型
class GCN2(nn.Module):
    def __init__(self,  input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)  
        return torch.sigmoid(x)  

class GCN4(nn.Module):
    def __init__(self,  input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim):
        super(GCN4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, output_dim)
        self.fc = nn.Linear(output_dim, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)  
        return torch.sigmoid(x)

seed = 2024
dataset_name = 'hc3'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 768  # 输入维度
hidden_dim = 512  # 隐藏层维度
hidden_dim2 = 256  # 隐藏层维度
hidden_dim3 = 128  # 隐藏层维度
output_dim = 64  # 输出类别数
gcnmodel = GCN2(input_dim, hidden_dim2, output_dim).to(device)
# gcnmodel = GCN4(input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim)
optimizer = optim.Adam(gcnmodel.parameters(), lr=0.0001)
criterion = nn.BCELoss()

train_len = len(hc3_train['y'])
val_len = len(hc3_val['y'])
epochs = 40
train_loss = []
val_loss = []
train_acc = []
val_acc = []
val_max_acc = -1
writer = SummaryWriter(f'logs/{dataset_name}_{seed}'+ datetime.now().strftime("%Y%m%d-%H%M%S"))
start_time = time.time()
for epoch in range(epochs):
    # 训练集
    gcnmodel.train()
    epoch_loss = 0.0
    correct_predictions = 0
    for i in tqdm(range(train_len),  f"epoch: {epoch+1}, Training"):
        data = Data(x=hc3_train['all_token_embeddings'][i], edge_index=hc3_train['all_edge_index'][i], y=hc3_train['y'][i]).to(device)
        optimizer.zero_grad()
        outputs = gcnmodel(data)
        loss = criterion(outputs, data.y.float().view(-1, 1))
        # print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predictions = (outputs >= 0.5).long()  
        correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()
    epoch_loss /= train_len
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    epoch_acc = correct_predictions / train_len
    writer.add_scalar('Acc/train', epoch_acc, epoch)
    print(f"epoch: {epoch+1}, train_loss: {epoch_loss}, train_acc: {epoch_acc}")
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    # 验证集
    gcnmodel.eval()
    epoch_loss = 0.0
    correct_predictions = 0
    all_predictions = []
    with torch.no_grad():
        for i in tqdm(range(val_len),  f"epoch: {epoch+1}, Validation"):
            data = Data(x=hc3_val['all_token_embeddings'][i], edge_index=hc3_val['all_edge_index'][i], y=hc3_val['y'][i]).to(device)
            outputs = gcnmodel(data)
            loss = criterion(outputs, data.y.float().view(-1, 1))
            epoch_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            all_predictions.append(predictions)
            correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()
    epoch_loss /= val_len
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    epoch_acc = correct_predictions / val_len
    writer.add_scalar('Acc/val', epoch_acc, epoch)
    print(f"epoch: {epoch+1}, val_loss: {epoch_loss}, val_acc: {epoch_acc}")
    val_loss.append(epoch_loss)
    val_acc.append(epoch_acc)
    
    tag = 3
    if epoch_acc >= val_max_acc:
        val_max_acc = epoch_acc
        tag = 3
        torch.save(gcnmodel.state_dict(), f'./model/{dataset_name}_gcn_model_{seed}.pth')
    else:
        tag -= 1
        if tag == 0:
            break
end_time = time.time()
elapsed_time = end_time - start_time
print(f"运行时间: {elapsed_time} 秒")
writer.close()

# 测试

import pickle
test_file = "hc3_test"
with open(f"/root/autodl-tmp/graph_data/{test_file}.pkl", "rb") as f:
    hc3_test = pickle.load(f)
test_len = len(hc3_test['y'])

from sklearn.metrics import roc_auc_score, f1_score

# test_gcnmodel = gcnmodel
test_gcnmodel = GCN2(input_dim, hidden_dim2, output_dim).to(device)
test_gcnmodel.load_state_dict(torch.load(f'./model/{dataset_name}_gcn_model_{seed}.pth'))
test_gcnmodel.eval()
test_loss = 0.0
correct_predictions = 0
test_pres = list()
start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(test_len),  f"Test"):
        data = Data(x=hc3_test['all_token_embeddings'][i], edge_index=hc3_test['all_edge_index'][i], y=hc3_test['y'][i]).to(device)
        outputs = test_gcnmodel(data)
        test_pres.append(outputs.item())
        loss = criterion(outputs, data.y.float().view(-1, 1))
        test_loss += loss.item()
        predictions = (outputs >= 0.5).long()
        correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"运行时间: {elapsed_time} 秒")
y_pred = [1 if prob >= 0.5 else 0 for prob in test_pres]
y_true = hc3_test['y'].view(-1, 1)
test_loss /= test_len
test_acc = correct_predictions / test_len
test_f1 = f1_score(y_true, y_pred)
print(f"test_loss: {test_loss}, test_acc: {test_acc}, test_f1: {test_f1}")

auc = roc_auc_score(hc3_test['y'], test_pres)
print(auc)

with open(f"test_result.txt", "a", encoding="utf-8") as w:
    w.write(f"{test_file}\t acc: {test_acc}\t auc: {auc}\t f1: {test_f1}\t seed: {seed}\t model: {dataset_name}\t{datetime.now()}\n")



