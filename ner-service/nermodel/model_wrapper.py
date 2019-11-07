from .model import Model
from .globals import tag2label,label2tag
import pickle
import asyncio
import numpy as np
import string
import torch
import os

__all__ = ['ModelWrapper']


class ModelWrapper:
    def __init__(self,
                args):
        self.word2id = self.read_dictionary(args.vocab_path)
        args.word_size = len(self.word2id)
        args.label_size = len(tag2label)

        self._use_cuda = args.use_cuda
        if self._use_cuda and not torch.cuda.is_available():
            self._use_cuda = False
        self._device = torch.device('cuda' if self._use_cuda else 'cpu')

        self._model = Model(args)
        if self._use_cuda:
            self._model.load_state_dict(torch.load(args.save))
        else:
            self._model.load_state_dict(torch.load(args.save,map_location='cpu'))


        if self._use_cuda:
            self._model.to(self._device)

        self._model.eval()
    
    async def startup(self,app):
        pass
    
    def read_dictionary(self,vocab_path):
        """

        :param vocab_path:
        :return:
        """
        vocab_path = os.path.join(vocab_path)
        with open(vocab_path, 'rb') as fr:
            word2id = pickle.load(fr)
        print('vocab_size:', len(word2id))
        return word2id
    
    def encode_query(self,query):
        sents = []
        
        sent_ = list(query.strip())
        # print(sent_)
        sentence_id = []
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            #     word = '<ENG>'
            if word not in self.word2id:
                word = '<UNK>'
            sentence_id.append(self.word2id[word])
        
        sents.append(sentence_id)
        # print(sents)
        seq_len_list = [len(inst) for inst in sents]
        sents = np.array(sents)

        with torch.no_grad():
            inst_data_tensor = torch.from_numpy(sents)
            seq_len = torch.LongTensor(seq_len_list)
        
        if self._use_cuda:

            inst_data_tensor = inst_data_tensor.cuda()
            seq_len = seq_len.cuda()

        return inst_data_tensor,seq_len

    def decode_pre(self,pred):
        _tags = [label2tag[tag] for tag in pred]
        res = {}
        for i in range(len(_tags)):
            tag = _tags[i].split('_')
            if 'B' in tag:
                label = tag[1].lower()
                if label not in res:
                    res[label] = []
                res[label].append([i,i+1])
                for j in range(i+1,len(_tags)):
                    itag = _tags[j].split('_')
                    if itag[0] == 'I' and itag[1] == tag[1]:
                        res[label][-1][1] = j + 1
                        i = j + 1
                    else:
                        res[label][-1][1] = j
                        break
        return res


    
    async def predict(self,query):
        word,seq_len = self.encode_query(query)
        with torch.no_grad():
            pred = self._model.predict(word,seq_len)
        decode = self.decode_pre(pred.numpy()[0])   
        res = {}
        for k in decode:
            res[k] = []
            for name in decode[k]:
                res[k].append(query[name[0]:name[1]])
        print('res',pred.numpy()[0],res)
        return res
