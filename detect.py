import argparse
import json
import torch
from building_graph import build_graph
from torch_geometric.data import Data
from model.GCN2 import GCN2

parser = argparse.ArgumentParser()

text = parser.add_argument('--text', type=str, required=True)
dataset = parser.add_argument('--dataset', choices = ['hc3', 'gpt3.5'], default='hc3')
seed = parser.add_argument('--seed', choices = ['2021', '2022', '2023', '2024', '2025'], default='2024')

args = parser.parse_args()

def detect(text, dataset, seed):
    json_texts = json.dumps({"text": text, "label": ""})
    all_token_embeddings, edge_index, y = build_graph(json_texts)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768  # 输入维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = 64  # 输出类别数
    gcnmodel = GCN2(input_dim, hidden_dim, output_dim).to(device)
    gcnmodel.load_state_dict(torch.load(f'./model/{dataset}_gcn_model_{seed}.pth'))
    gcnmodel.eval()
    data = Data(x=all_token_embeddings, edge_index=edge_index, y=y).to(device)
    outputs = gcnmodel(data)
    probability = outputs.item()
    prediction = (outputs >= 0.5).long()
    return probability, prediction

print(detect(args.text,args.dataset,args.seed))







