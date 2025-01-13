import requests
import json
from tqdm import tqdm
import time


files = ["hc3_test_delete_15.json", "hc3_test_repeat_15.json", "hc3_test_insert_15.json", "hc3_test_replace_15.json"]
for file_name in files:
    with open(f"gpt2-detector/gpt2-detector-{file_name}", "w", encoding = "utf-8") as w:
        with open(f"perturbed_text/{file_name}", "r", encoding = "utf-8") as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                try:
                    line = json.loads(lines[i])
                    text = line['text']
                    r = requests.get('http://localhost:8080/?'+text)
                    t = json.loads(r.text)
                    t['id'] = i
                    t['label'] = line['label']
                    w.write(json.dumps(t)) 
                    # w.write(r.text)
                    w.write("\n")
                    # time.sleep(0.01)
                except Exception as e:
                    w.write(json.dumps({"id": i}))
                    print(i, e)
                    w.write("\n")