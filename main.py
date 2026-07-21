import os
import sys
from pydantic import BaseModel
import re
from rank_bm25 import BM25Okapi
import ast
import pickle
import json
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

[]


PY_EXT = [".py"]
DOCS_TXT_EXT = [".txt", ".md"]




class Chunk(BaseModel):
    file_path: str
    start: int
    end: int
    content: str
    typee: str



def offsetts(lst):
    res = [0]
    for ele in lst:
        res.append(len(ele) + res[-1] + 1)
    return res


def chunking_code(path, content):
    s = 0
    res = []

    roots = []
    tree = ast.parse(content)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            roots.append(node)
    
    lines = content.split("\n")
    offst = offsetts(lines)

    for node in roots:
        start_line = node.lineno
        end_line = node.end_lineno - 1

        s_idx = offst[start_line]
        e_idx = offst[end_line]

        data = content[s_idx: e_idx]
        
        obj = Chunk(file_path=path, start=s, end=s+len(data), content=data, typee="code")
        res.append(obj)
        s += len(content)
    
    return res


def chunking_docs(path, contnt):
    res = []

    if len(contnt) < 2000:
        obj = Chunk(file_path=path, start=0, end=len(contnt), content=contnt, typee="docs")
        res.append(obj)
        return res

    strt = 0
    chunk = ""

    lines = contnt.split("\n")

    for line in lines:
        line += "\n"

        if len(line) + len(chunk) <= 2000:
            chunk += line
        
        else:
            if line.strip():
                obj = Chunk(file_path=path, start=strt, end=strt+len(chunk), content=chunk, typee="docs")
                res.append(obj)
                strt += len(chunk)
                chunk = line
    
    if line.strip():
        obj = Chunk(file_path=path, start=strt, end=strt+len(chunk), content=chunk, typee="docs")
        res.append(obj)

    return res



def chunking_data(repo):
    res = []

    if not os.path.exists(repo):
        print("Error")
        sys.exit(1)
    
    for root, folder, files in os.walk(repo):

        for file in files:
            ext = os.path.splitext(file)[1]
            
            if ext in PY_EXT:
                current_path = os.path.join(root, file)
                with open(current_path, "r") as f:
                    content = f.read()
                lst = chunking_code(current_path, content)
                res.extend(lst)

            elif ext in DOCS_TXT_EXT:
                current_path = os.path.join(root, file)
                with open(current_path, "r") as f:
                    content = f.read()
                lst = chunking_docs(current_path, content)
                res.extend(lst)
    
    return res




def my_tokenizer(s):
    s = s.lower()
    return [e for e in s.split() if len(e) > 1]




def build_the_indexed_stock(list_of_chunks):

    data_to_index_it = []

    for chunk_obj in list_of_chunks:
        data_to_index_it.append(my_tokenizer(chunk_obj.content))
    
    bm25_obj = BM25Okapi(data_to_index_it)
    binaries = pickle.dumps(bm25_obj)
    with open("BM25.pkl", "wb") as f:
        f.write(binaries)
    
    lst_dct_objs = []
    for obj in list_of_chunks:
        lst_dct_objs.append(obj.model_dump())
    data = json.dumps(lst_dct_objs, indent=2)
    with open("chnks.json", "w") as f:
        f.write(data)
    
    print("\nDATA ARE SAVED SUCCESSFULLY.\n")


import sys

def get_retrive_data(query, k):
    tok_query = my_tokenizer(query)

    with open("/home/hel-achh/goinfre/rcd2/BM25.pkl", "rb") as f:
        data = f.read()
    bm25 = pickle.loads(data)

    with open("/home/hel-achh/goinfre/rcd2/chnks.json", "r") as f:
        data = f.read()
    list_of_dicts = json.loads(data)

    scores = bm25.get_scores(tok_query)


    top_k_indexes = numpy.argsort(scores)
    print(top_k_indexes)
    sys.exit(1)



    res = []
    for idx in top_k_indexes:
        res.append(list_of_dicts[idx])

    return res


def merge_data(lst):
    res = ""
    for l in lst:
        res += l["content"]
        res += "\n"
    return res



def generate_answer(fully_prompt, new_max):
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    inputs = TOKENIZER(fully_prompt, return_tensors="pt")
    outputs = MODEL.generate(**inputs, max_new_tokens=new_max)

    ids = outputs[0].tolist()
    len_of_prompt = len(inputs["input_ids"][0].tolist())
    ids = ids[len_of_prompt:]
    answer = TOKENIZER.decode(ids)
    return answer










question = "What is this function (test_vllm_gc_ed) takes in parameters??"

chunks_objs = chunking_data("/home/hel-achh/goinfre/rcd2/vllm-0.10.1")
build_the_indexed_stock(chunks_objs)

sttrr_merged = merge_data(get_retrive_data(question, 10))


fully_prompt = f"""
THE CONTENT:
{sttrr_merged}


THE QUESTION IS : {question}

- answer from the content on this question.

ANSWER:

"""

answer = generate_answer(fully_prompt, 90)
print()
print(answer)

