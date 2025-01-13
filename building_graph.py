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
    texts = list()
    y = list()
    for json_text in json_texts:
        texts.append(json.loads(json_text['text']))
        y.append(1 if json.loads(json_text['label'])=="hc3human" else 0)
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
                input_ids = torch.tensor(token_ids).unsqueeze(0)
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
            for i,word in enumerate(doc):        
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
    return all_token_embeddings, edge_index, y