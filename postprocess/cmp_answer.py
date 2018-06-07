#encoding=utf-8

#将验证集中加入知识库前后的答案进行对比
#筛选知识库前回答错误以及知识库后回答正确的问题
#记录下图片index 问题描述 图片id 预测正误的答案对 标准答案

#read：
# v2_OpenEnded_mscoco_val2014_questions.json
# val_target.pkl
# result{}kn{}_pos2score.pkl
# trainval_label2ans.pkl

#write:
# cmp_answer.json

import numpy as np
import json
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#获取验证集问题的id和详细描述
def get_questiones():
    question_path = os.path.join(
        'rawdata', 'v2_OpenEnded_mscoco_%s2014_questions.json' % 'val')
    questions = json.load(open(question_path))['questions']
    qdata={}
    for item in questions:
        qid=int(item['question_id'])
        qtext=item['question']
        qdata[qid]=qtext
    return qdata

def cmp_predict_real_answer(model,kn):
    # 读question和answer
    dataroot = 'maturedata'
    name = 'val'
    answer_path = os.path.join(dataroot, '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    # 读取预测答案
    # 候选答案n个
    # 最大分数的位置->预测的soft分数
    # model1kn0_pos2score 无knowledge
    # model1kn1_pos2score 有knowledge
    with open('result/model{}kn{}_pos2score.pkl'.format(model,kn), 'rb') as f0:
        base_predict_answers = pickle.load(f0)
    with open('result/model{}kn{}_pos2score.pkl'.format(model,kn+1), 'rb') as f1:
        kn_predict_answers = pickle.load(f1)

    # print(len(base_predict_answers))
    # print(len(kn_predict_answers))
    # 获取答案位置对应的字符串
    # index->answer word
    with open('maturedata/trainval_label2ans.pkl', 'rb') as f2:
        label2ans = pickle.load(f2)

    # 匹配预测答案和问题、标准答案
    result_dict = {}
    for id in range(len(base_predict_answers)):
        base_predict = base_predict_answers[id]
        kn_predict = kn_predict_answers[id]
        # print(base_predict)
        # print(kn_predict)
        if base_predict[1] == 0 and kn_predict[1] == 1:
            #print(1)
            result_dict[id] = {
                't_pos': kn_predict[0],
                'f_pos': base_predict[0]
            }
    #print(len(result_dict))
    # dict:
    #   imgindex question imageid std_answer predict_true predict_false
    dicts = []

    with open('maturedata/val_img2categories.json','r') as file:
        img2ctg=json.load(file)
    with open('maturedata/knowledge.json','r') as file:
        kn=json.load(file)

    questiones=get_questiones()
    for key in result_dict.keys():
        answer = answers[key]
        qid = answer['question_id']
        imgid = answer['image_id']
        labels = answer['labels']
        scores = answer['scores']
        # print(key)
        # ctg=img2ctg[str(key)]
        # kns=kn[ctg]
        if len(labels) > 0:
            max_label = labels[np.argmax(scores)]
            real_answer = label2ans[max_label]
            pt_answer = label2ans[int(result_dict[key]['t_pos'])]
            pf_answer = label2ans[int(result_dict[key]['f_pos'])]
            dicts.append({
                'imgindex': key,
                'question':questiones[qid],
                'imgid': imgid,
                't_predict': pt_answer,
                'f_predict': pf_answer,
                'r_answer': real_answer
                # 'ctg':ctg,
                # 'knowledge':kns
            })
    with open('result/cmp_answer.json','w') as file:
        json.dump(dicts,file)

if __name__=='__main__':
    model=1
    kn=0
    cmp_predict_real_answer(model,kn)
