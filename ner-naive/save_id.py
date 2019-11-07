import pickle
from util import *

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
    char_to_ix,ix_to_char = get_char2id(train_X)
    tag_to_ix,ix_to_tag = get_tag2id(train_Y)
    print(tag_to_ix.keys())
    print('char dic',len(char_to_ix))
    print('tag dic',len(tag_to_ix))

    with open('./data/id.pkl','wb') as f:
        pickle.dump([char_to_ix,ix_to_char,tag_to_ix,ix_to_tag],f)

