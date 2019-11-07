import argparse
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='LSTM_CRF')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--use-cuda', action='store_true',
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--use-crf', action='store_true',
                    help='use crf')

parser.add_argument('--mode', type=str, default='train',
                    help='train mode or test mode')

parser.add_argument('--save', type=str, default='./checkpoints/lstm_crf.pth',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='dataset',
                    help='location of the data corpus')

parser.add_argument('--word-ebd-dim', type=int, default=128,
                    help='number of word embedding dimension')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--lstm-hsz', type=int, default=128,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='biLSTM layer numbers')
parser.add_argument('--l2', type=float, default=0.005,
                    help='l2 regularization')
parser.add_argument('--clip', type=float, default=.5,
                    help='gradient clipping')
parser.add_argument('--result-path', type=str, default='./result',
                    help='result-path')
parser.add_argument('--cudaD', type=int, default=1,
                    help='cuda-device-id')

args = parser.parse_args()

import torch 
torch.manual_seed(args.seed)
args.use_cuda = True



# load data
from data_loader import DataLoader
from data import read_corpus, tag2label
import os 
# from eval import conlleval

print('loading data...')
sents_train, labels_train, args.word_size, _ = read_corpus(os.path.join('.', args.data, 'train_data.txt'), os.path.join('.', args.data, 'train_label.txt'),is_train=args.mode)
sents_test, labels_test, _, data_origin = read_corpus(os.path.join('.', args.data, 'smalldata.txt'), os.path.join('.', args.data, 'smalllabel.txt'), is_train=False)
args.label_size = len(tag2label)

train_data = DataLoader(sents_train, labels_train, cuda=args.use_cuda, batch_size=args.batch_size)
test_data = DataLoader(sents_test, labels_test, cuda=args.use_cuda, shuffle=False, evaluation=True, batch_size=args.batch_size)

label2tag = {}
for tag, lb in tag2label.items():
    label2tag[lb] = tag if lb != 0 else lb

from model import Model 
model = Model(args)



if args.use_cuda:
    print('use cuda...')
    torch.cuda.set_device(args.cudaD)
    model = model.cuda()
    

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.l2)

def train():
    print('model training...')
    model.train()
    total_loss = 0
    for word, label, seq_lengths, _  in train_data:
        optimizer.zero_grad()
        loss, _ = model(word, label, seq_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    return total_loss / train_data._stop_step

def evaluate(epoch):
    print('model eval...')
    model.eval()
    eval_loss = 0
    


    label_list = []

    for word, label, seq_lengths, unsort_idx in test_data:
        loss, _ = model(word, label, seq_lengths)
        pred = model.predict(word, seq_lengths)
        pred = pred[unsort_idx]
        seq_lengths = seq_lengths[unsort_idx]
        for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
            pred_ = list(pred[i][:seq_len].cpu().numpy())
            label_list.append(pred_) 
        eval_loss += loss.detach().item()


    


    y_true = []
    y_pred = []
    for label_, (sent, tag) in zip(label_list, data_origin):
        if  len(label_) != len(sent):
            print(len(sent))
            print(len(label_))
        for i in range(len(sent)):
            y_true.append(tag2label[tag[i]])
            y_pred.append(label_[i])

    
    f1 = f1_score(y_true,y_pred,average='micro')
    return eval_loss / test_data._stop_step,f1

import time 
train_loss = []
best_loss = float('inf')

total_start_time = time.time()

print('-' * 90)
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    loss = train()
    train_loss.append(loss * 1000.)

    print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
            epoch, time.time() - epoch_start_time, loss))
    if epoch % 10 == 0:
        eval_loss,f1 = evaluate(epoch)
        if eval_loss < best_loss:
            torch.save(model.state_dict(), args.save)
            best_loss = eval_loss
        print('| Eval Epoch {:3d} | time: {:2.2f}s | eval loss {:5.6f} | f1 score {:5.6f} '.format(
                epoch, time.time() - epoch_start_time, eval_loss,f1))



