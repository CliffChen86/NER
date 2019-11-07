from __future__ import print_function
import sys
import time
import json
from multiprocessing import Pool
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import argparse
import torch
import pickle
from torch.autograd import Variable

from model import *
from util import *
from Config import *

parser = argparse.ArgumentParser(description='LSTM_CRF')
parser.add_argument('--epoch', type=int, default=EPOCH,
                    help='number of epochs for train')
parser.add_argument('--lr', type=float, default=LR,
                    help='learning rate')
parser.add_argument('--lrdecay', type=float, default=LR_DECAY,
                    help='learning rate decay')                    
parser.add_argument('--save', type=str, default=PATH,
                    help='path to save the final model')
parser.add_argument('--wordemdim', type=int, default=EMBEDDING_DIM,
                    help='number of word embedding dimension')
parser.add_argument('--hsz', type=int, default=HIDDEN_DIM ,
                    help='BiLSTM hidden size')
parser.add_argument('--evalepoch', type=int, default=DEV_EPOCH,
                    help='number of epochs for dev')

args = parser.parse_args()

def load_data(data_dir):
    lines = open(data_dir,encoding = 'utf-8').readlines()
    query = []
    label = []
    q = []
    l = []
    for line in lines:
        line = line.strip().split('\t')
        if len(line) == 2:
            q.append(line[0])
            l.append(line[1])
        else:
            query.append(q)
            label.append(l)
            q = []
            l = []
    return query,label
            

if __name__ == "__main__":
    train_X,train_Y = load_data('./data/train.txt')
    dev_X,dev_Y = load_data('./data/dev.txt')

    print('train: ',len(train_X))
    print('dev: ',len(dev_X))

    # char_to_ix,ix_to_char = get_char2id(train_X)
    # tag_to_ix,ix_to_tag = get_tag2id(train_Y)
    with open('./data/id.pkl','rb') as f:
        char_to_ix,ix_to_char,tag_to_ix,ix_to_tag = pickle.load(f)
    print(tag_to_ix.keys())
    print('char dic',len(char_to_ix))
    print('tag dic',len(tag_to_ix))
    
    model = BiLSTM_CRF(len(char_to_ix), tag_to_ix, args.wordemdim, args.hsz)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lrdecay)
    if torch.cuda.is_available():
        print('cuda available')
        torch.cuda.set_device(1)
        model = model.cuda()

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(train_X[0], char_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in train_Y[0]], dtype=torch.long)
        if torch.cuda.is_available():
            precheck_sent = precheck_sent.cuda()
            precheck_tags = precheck_sent.cuda()
        print(model(precheck_sent))
    best_loss = float('inf')
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(args.epoch+1):  # again, normally you would NOT do 300 epochs, it is toy data
        t1 = time.time()
        train_loss = 0
        for i in range(len(train_X))[:2]:
            sentence = train_X[i]
            tags = train_Y[i]

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # model.train()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, char_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # targets = torch.LongTensor([tag_to_ix[t] for t in tags])
            if torch.cuda.is_available():
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            train_loss += loss
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

        # Check predictions after training
        if epoch % args.evalepoch == 0:
            # model.eval()
            eval_loss = 0
            y_pred = []
            y_true = []
            for i in range(len(dev_X))[:2]:
                sentence = dev_X[i]
                tags = dev_Y[i]
                sentence_in = prepare_sequence(sentence, char_to_ix)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                # targets = torch.LongTensor([tag_to_ix[t] for t in tags])
                if torch.cuda.is_available():
                    sentence_in = sentence_in.cuda()
                    targets = targets.cuda()
                loss = model.neg_log_likelihood(sentence_in, targets)
                eval_loss += loss
                score,tags = model(sentence_in)
                y_true.extend([tag_to_ix[t] for t in dev_Y[i]])
                y_pred.extend(tags)
            
            
            f1 = f1_score(y_true,y_pred,average='micro')
            t2 = time.time()
            print("Epoch:{},train_loss:{},eval_loss:{},f1_score:{},time:{} s".format(epoch,train_loss.item()/len(train_X),eval_loss.item()/len(dev_X),f1,t2-t1)) 
            if eval_loss.item()<best_loss:
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(model.state_dict(),args.save)
                best_loss = eval_loss


    



