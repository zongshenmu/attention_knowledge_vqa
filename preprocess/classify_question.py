#encoding=utf-8

#按照训练集合测试集中每个问题的顺序
#给出问题所属的三种类型 yes/no number other

#read:
# train_target.pkl
# val_target.pkl

#write:
# train_qtype.pkl
# val_qtype.pkl


import pickle
import os

names=['train','val']
for name in names:
    name = 'val'
    dataroot = 'maturedata'
    answer_path = os.path.join(dataroot, '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    q_type = []
    for answer in answers:
        q_type.append(answer['type'])
    # print(q_type)
    q_type_path = os.path.join(dataroot, '%s_qtype.pkl' % name)
    with open(q_type_path, 'wb') as file:
        pickle.dump(q_type, file)