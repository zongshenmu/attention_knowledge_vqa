#encoding=utf-8

#按照网络输入的格式整理原始数据

#read:
# v2_OpenEnded_mscoco_train2014_questions.json v2_OpenEnded_mscoco_val2014_questions.json
# train_target.pkl vla_target.pkl
# dictionary.pkl
# trainval_ans2label.pkl
# train36_imgid2idx.pkl val36_imgid2idx.pkl

#write:
# train_imgs.hdf5 val_imgs.hdf5
# train_questions.hdf5 val_questions.hdf5
# train_targets.hdf5 val_targets.hdf5

import numpy as np
import json
import h5py
import pickle
import utils
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dictionary import Dictionary
import config

cfg=config.Config()
MAX_LEN = cfg.MAX_LEN #每句话最长的单词数
feature_length = cfg.FEATURE_LEN
num_fixed_boxes = cfg.NUM_BOXES

#加载数据
def write_dataset(dataroot, name, imgid2idx, dictionary, num_ans_candidates, max_length=14):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        'rawdata', 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    nums=len(questions)
    utils.assert_eq(nums, len(answers))

    train_imgs="maturedata/%s_imgs.hdf5"%name
    train_questions="maturedata/%s_questions.hdf5"%name
    train_targets="maturedata/%s_targets.hdf5"%name

    h_imgs = h5py.File(train_imgs, "w")
    h_questions=h5py.File(train_questions, "w")
    h_targets=h5py.File(train_targets, "w")

    train_imgs = h_imgs.create_dataset(
         'imgs', (nums,), 'f')
    train_questions = h_questions.create_dataset(
        'questions', (nums, max_length), 'f')
    train_targets = h_targets.create_dataset(
        'targets', (nums, num_ans_candidates), 'f')

    count=0
    #zip将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    for question, answer in list(zip(questions, answers)):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']

        # 设置数据input的每一项
        img=imgid2idx[img_id]
        # token每句话单词所在位置，最长14个单词
        tokens = dictionary.tokenize(question['question'], False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [dictionary.padding_idx] * (max_length - len(tokens))
            tokens = padding + tokens
        utils.assert_eq(len(tokens), max_length)
        train_questions[count,:]=tokens
        train_imgs[count]=img
        labels = np.asarray(answer['labels'],dtype=np.int32)
        scores = np.asarray(answer['scores'])
        target = np.zeros(num_ans_candidates)
        if labels is not None:
            target[labels]=scores
        train_targets[count,:]=target
        count+=1

if __name__=='__main__':
    # 加载单词字典
    dictionary = Dictionary.load_from_file('maturedata/dictionary.pkl')

    # 加载答案的长度
    dataroot = 'maturedata'
    ans2label_path = os.path.join(dataroot, 'trainval_ans2label.pkl')
    ans2label = pickle.load(open(ans2label_path, 'rb'))
    num_ans_candidates = len(ans2label)

    # 加载图片对应的下标
    names =['train','val']
    for name in names:
        img_id2idx = pickle.load(open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
        # 图片问题和答案相结合
        print("start")
        write_dataset(dataroot, name, img_id2idx, dictionary, num_ans_candidates, MAX_LEN)
        print("done")