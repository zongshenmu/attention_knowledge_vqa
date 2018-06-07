#encoding=utf-8

#本测试结果以attention的第一种机制为例，不加入知识库的引导
#在获得最优的测试集结果后，将注意力保存
#并和每一张图片与之对应问题答案的正确

#read:
# feature_att.pkl
# val_bbox.hdf5
# val_imgs.hdf5
# val_target.pkl
# model1kn0_pos2score.pkl
# trainval_label2ans.pkl

#write:
# feature_att.pkl
# attention_answer.pkl
# test_visualize.pkl
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess.cmp_answer import *

from postprocess.test import test


#写测试集的attention
def write_attention(model,kn):
    feature_attention=test(model,kn)
    with open('result/feature_att.pkl','wb') as file:
        pickle.dump(feature_attention,file)

def answer_attention(model,kn):
    #读取特征
    with open('result/feature_att.pkl','rb') as file:
        fea_att=pickle.load(file)
        att=[]
        for item in fea_att:
            #print(item.shape) (36,1)
            att.append(item.reshape(-1).tolist())

    #读取bbox
    with h5py.File('maturedata/val_bbox.hdf5', 'r') as hf:
        bbox = np.array(hf.get('image_bb'),dtype=np.float32)

    #读取valimg的index2imgid
    with h5py.File('maturedata/val_imgs.hdf5', 'r') as hf:
        val_imgs = np.array(hf.get('imgs'), dtype=np.int32)

    # 读question和answer
    answer_path = os.path.join('maturedata', 'val_target.pkl')
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    # 读取预测答案
    # 候选答案n个 最大分数的位置->预测的soft分数
    with open('result/model{}kn{}_pos2score.pkl'.format(model,kn), 'rb') as f0:
        predict_answers = pickle.load(f0)

    # 获取答案位置对应的字符串
    with open('maturedata/trainval_label2ans.pkl', 'rb') as f1:
        label2ans = pickle.load(f1)

    # 匹配预测答案和问题、标准答案
    # dict questionid: imageid: answer: predict:
    answer_dict = {}
    for id,base_predict in enumerate(predict_answers):
        if base_predict[1] == 1:
            answer_dict[id] = label2ans[int(base_predict[0])]
    #print(len(answer_dict))
    dicts=[]
    questiones=get_questiones()
    for key in answer_dict.keys():
        answer = answers[key]
        qid = answer['question_id']
        imgid = answer['image_id']
        labels = answer['labels']
        scores = answer['scores']
        if len(labels) > 0:
            max_label = labels[np.argmax(scores)]
            real_answer = label2ans[max_label]
            pred_answer = answer_dict[key]
            dicts.append({
                'imgindex': key,
                'bbox':bbox[val_imgs[key]].tolist(),
                'question':questiones[qid],
                'imgid': imgid,
                'p_predict': pred_answer,
                'r_answer': real_answer,
                'att':att[key]
            })
    with open('result/attention_answer.pkl','wb') as file:
        pickle.dump(dicts,file)
    #print(dicts[0])
    with open('result/test_visualize.json', 'w') as file:
        json.dump(dicts[0:1000], file)

if __name__=='__main__':
    model=1
    kn=0
    #write_attention(model,kn)
    answer_attention(model,kn)