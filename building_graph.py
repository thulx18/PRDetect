import spacy
import torch
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from scipy.sparse import csr_matrix

# 加载英语模型
nlp = spacy.load("en_core_web_sm")
# model = RobertaModel.from_pretrained("roberta-base")
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
model = RobertaModel.from_pretrained("./roberta-base/")
tokenizer = RobertaTokenizer("./roberta-base/vocab.json", "./roberta-base/merges.txt", use_fast=False)
vocab_size = len(tokenizer)

def build_graph(json_texts):
    start_time = time.time()
    texts = list()
    y = list()
    for json_text in json_texts:
        texts.append(json.loads(json_text)['text'])
        label = 1 if "human" in json.loads(json_text)['label'] else 0
        y.append(label)
    y = torch.tensor(y, dtype=torch.float32)
    tokenized_sentences = list()
    all_token_embeddings = list()
    all_edge_index = list()
    all_sparse_adj_matrix = list()
    for text in tqdm(texts):
        try:
            doc = nlp(text)
            tokenized_sentence = [token.text for token in doc]
            tokenized_sentences.append(tokenized_sentence)
            # print(tokenized_sentence)
            
            max_length = 512
            chunks = [tokenized_sentence[i:i+max_length] for i in range(0, len(tokenized_sentence), max_length)]
            chunk_outputs = []
            for chunk in chunks:
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_ids)

                last_hidden_states = output.last_hidden_state
                token_embeddings = last_hidden_states[0]
                chunk_outputs.append(token_embeddings)
            token_embeddings = torch.cat(chunk_outputs, dim=0)
            all_token_embeddings.append(token_embeddings)
            # print(len(tokenized_sentence))
            # print(token_embeddings.shape)
            node_relations = list()
            for word in doc:        
                node_relations.append([word.i,word.head.i])
                # 加上自环
                # if word.i != word.head.i:
                #     node_relations.append([word.i,word.i])
            edge0 = list()
            edge1 = list()
            for edge in node_relations:
                edge0.append(edge[0])
                edge1.append(edge[1])
            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            all_edge_index.append(edge_index)
            # sparse_adj_matrix = csr_matrix((np.ones(len(edge0)),(np.array(edge0), np.array(edge1))),shape=(len(tokenized_sentence),len(tokenized_sentence)))
            # dependency_matrix = sparse_adj_matrix
            # print(sparse_adj_matrix)
            # all_sparse_adj_matrix.append(sparse_adj_matrix)
        except Exception as e:
            print(text)
            print(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time} 秒")
    return all_token_embeddings, all_edge_index, y

def read_json(file_name):
    texts = list()
    with open(f"original_text/{file_name}.json", "r", encoding="utf-8") as f:
        for line in f.readlines():
            texts.append(line)
    return texts

def save_pkl(file_name, all_token_embeddings, all_edge_index, y):
    with open(f"/root/autodl-tmp/graph_data/{file_name}.pkl", "wb") as f:
        pickle.dump({"all_token_embeddings": all_token_embeddings,
                     "all_edge_index": all_edge_index,
                     "y": y}, f)

if __name__ == '__main__':
    files = [
        "hc3_train",
        "hc3_val",
        "hc3_test"
    ]
    for file in files:
        all_token_embeddings, all_edge_index, y = build_graph(read_json(file))
        save_pkl(file, all_token_embeddings, all_edge_index, y)