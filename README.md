## PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree
[Finding of NAACL 2025](https://aclanthology.org/2025.findings-naacl.464/)

### Installation
```shell
pip install -r requirements.txt
pip -m spacy download en_core_web_sm
```

### Train
```shell
python build_graph.py
python gcn.py
```

### Detect
```shell
python detect --text "This is an example."
```

### Citation
```bibtex
@inproceedings{li2025prdetect,
  title={PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree},
  author={Li, Xiang and Yin, Zhiyi and Tan, Hexiang and Jing, Shaoling and Su, Du and Cheng, Yi and Shen, Huawei and Sun, Fei},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={8290--8301},
  year={2025}
}
```