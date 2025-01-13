import spacy
import torch
import json
import pickle
import time
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from scipy.sparse import csr_matrix
from graphviz import Digraph

def drawtree(text, output="tree"):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    node_relations = list()
    nodes = list()
    for _, word in enumerate(doc):
        nodes.append(str(word))
        if word.i == word.head.i or str(word).isspace():
            continue
        node_relations.append([word.i,word.head.i])

    dot = Digraph()
    for i in range(len(nodes)):
        if nodes[i].isspace():
            continue
        dot.node(str(i),nodes[i])
    for edge in node_relations:
        dot.edge(str(edge[1]),str(edge[0]))
    dot.render(output, view=False)

if __name__ == "__main__":
    drawtree("Do you use any other online features of Quicken?  How many unique ticker symbols do you have?")