
import asyncio
import enum


#from nermodel.data_loader import DataLoader
#from .data import read_corpus
#from .globals import tag2label



__all__ = ['RespType']

START = 7
STOP = 8
PAD = 0

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "START": 7, "STOP": 8
             }


label2tag = {}
for tag, lb in tag2label.items():
    label2tag[lb] = tag



class RespType(enum.Enum):
    JSON = 0
    MSGPACK = 1

