import os
import sys
from pydantic import BaseModel
import re
import ast



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
                print(ext)
    
    return res












res = chunking_data("/home/hel-achh/goinfre/rcd2/vllm-0.10.1")


for o in res:
    print(o.content)
    print(len(o.content))
    print("=" * 20)

