import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from datetime import datetime
from tqdm import tqdm
import time
import pickle
import argparse
from sklearn.metrics import roc_auc_score, f1_score

parser = argparse.ArgumentParser()

dataset = parser.add_argument('--dataset', choices = ['hc3', 'gpt3.5'])
seed = parser.add_argument('--seed', choices = ['2021', '2022', '2023', '2024', '2025'])
test_file = parser.add_argument('--file', type = str)
do_save = parser.add_argument('-s', '--save', action = 'store_true')

args = parser.parse_args()


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
    

def test(test_file, dataset_name, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768  # 输入维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = 64  # 输出类别数
    gcnmodel = GCN2(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(gcnmodel.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    
    with open(f"/root/autodl-tmp/graph_data/{test_file}.pkl", "rb") as f:
        hc3_test = pickle.load(f)
    test_len = len(hc3_test['y'])
    test_gcnmodel = GCN2(input_dim, hidden_dim, output_dim).to(device)
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
    if args.do_save:
        with open(f"/root/autodl-tmp/result/test_result.txt", "a", encoding="utf-8") as w:
            w.write(f"{test_file}\t acc: {test_acc}\t auc: {auc}\t f1: {test_f1}\t seed: {seed}\t{datetime.now()}\n")
    return y_pred


def main():
    if __name__ == "__main__":
        print(args.test_file, args.dataset, args.seed)
        test(args.test_file, args.dataset, args.seed)