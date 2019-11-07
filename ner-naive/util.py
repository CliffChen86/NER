from __future__ import print_function
import sys
import time
import json
from multiprocessing import Pool
import pandas as pd
import numpy as np
#from tqdm import tqdm

def get_label(query,label):
    res = []
    for i in range(len(query)):
        que = query[i]
        lab = label[i]
        lab = lab.split('#')
        labres = []
        for i in range(len(que)):
            labres.append('O')
        assert(len(labres)==len(que))
        for l in lab:
            if l != '':
                l = l.split('=')
                if len(l) == 2:
                    key = l[0]
                    val = l[1]
                    if val in que:
                        inx = que.index(val)
#                         print(key,val,que,inx,len(que),len(labres))
                        labres[inx] = 'B_' + key.upper()
                        for i in range(inx+1,inx+len(val),1):
                            labres[i] = 'I_' + key.upper()
#                         print(key,val,que,inx,len(que),len(labres))
#         print(que,labres)
        res.append(labres)
    return res

def get_tag2id(label):
    dic = []
    for i in label:
        dic.extend(i)
    dic = list(set(dic))
    dic.append("<START>")
    dic.append("<STOP>")
    tag2id = {}
    id2tag = {}
    for i in range(len(dic)):
        tag2id[dic[i]] = i
        id2tag[i] = dic[i]
    return tag2id,id2tag

def get_char2id(query):
    dic = []
    for i in query:
        dic.extend(i)
    dic = list(set(dic))
    char2id = {}
    id2char = {}
    for i in range(len(dic)):
        char2id[dic[i]] = i
        id2char[i] = dic[i]
    return char2id,id2char