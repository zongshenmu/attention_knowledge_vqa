#encoding=utf-8

#创建问题答案单词的字典

#read:
# v2_OpenEnded_mscoco_train2014_questions.json
# v2_OpenEnded_mscoco_val2014_questions.json
# v2_OpenEnded_mscoco_test2015_questions.json
# v2_OpenEnded_mscoco_test-dev2015_questions.json
# glove.6B.300d.txt

#write:
# dictionary.pkl
# glove6b_init_300d.npy

import os
import sys
import json
import numpy as np

# os.path.abspath(__file__)可以获得当前模块的绝对路径
# os.path.dirname可以获取到除文件名以外的路径
dir_name=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_name)

#工程下的模块要在添加到系统变量后才能导入
from Dictionary import Dictionary

# 创建单词字典
def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        # 文件名和目录用/连接
        question_path = os.path.join(dataroot, path)
        # dump将python对象转化为json load将json对象转换为python dic
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


# 创建question的word embedding
def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        # map() 会根据提供的函数对指定序列做映射。
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        # 未定义的单词为0向量
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary('rawdata')
    d.dump_to_file('maturedata/dictionary.pkl')
    #print(len(d.idx2word))
    print('dictionary has been saved')

    #词向量的维数
    emb_dim = 300
    glove_file = 'rawdata/glove.6B.%dd.txt' % emb_dim
    print('embedding dim is %d' % emb_dim)
    weights, _ = create_glove_embedding_init(d.idx2word, glove_file)
    weights_path='maturedata/glove6b_init_%dd.npy' % emb_dim
    np.save(weights_path,weights)
    print('weights have been saved')
