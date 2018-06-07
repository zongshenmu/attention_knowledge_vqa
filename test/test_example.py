#encoding=utf-8

#保存的模型最优结果，单独的测试

#read:
# val36.hdf5
# val_imgs.hdf5
# val_questions.hdf5
# val_targets.hdf5
# valimg_knowledge.pkl
# val_qtype.pkl
# checkpoint

#write:
# result{}kn{}_pos2score.pkl

import h5py
import math
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forward_network import *
import config
from utils import *

cfg=config.Config()

MAX_LEN=cfg.MAX_LEN
SEED=cfg.SEED
BATCH_SIZE=cfg.BATCH_SIZE
EMB_DIM=cfg.EMB_DIM
MAX_CTG=cfg.MAX_CTG

#设置随机种子
tf.set_random_seed(SEED)


def test(model,kn):
    preroot = "test"
    # train加载v、q、a数据
    val_name = "test"
    print('loading %s data' % val_name)
    with open('test/test_questions.json','r') as file:
        questions=json.load(file)
        questions_id=[]
        for i in questions.keys():
            questions_id.append(i)
        questions_id=np.asarray(questions_id,np.int32)
    # val加载v、q、a数据name = 'val'
    path0 = os.path.join(preroot, '%s_img_region_vector.json' % val_name)
    with open(path0, 'r') as f0:
        features = np.array(json.load(f0), dtype=np.float32)[questions_id]
    path2 = os.path.join(preroot, '%s_questions_vector.json' % val_name)
    with open(path2, 'r') as f2:
        questions = np.array(json.load(f2), dtype=np.int32)[questions_id]
    if kn==0:
        knowleges=None
    else:
        val_knowledge_path = os.path.join(preroot, '%simg_knowledge.json' % val_name)
        with open(val_knowledge_path, 'rb') as file:
            knowleges = np.array(json.load(file), dtype=np.int32)
    print('loading %s data over' % val_name)

    size = features.shape[0]
    # 加载答案的长度
    dataroot = 'maturedata'
    ans2label_path = os.path.join(dataroot, 'trainval_ans2label.pkl')
    ans2label = pickle.load(open(ans2label_path, 'rb'))
    num_ans_candidates = len(ans2label)
    num_objs = features.shape[1]
    v_dim = features.shape[2]

    # 加载模型
    input_path = "result/saved_model{}kn{}/".format(model,kn)
    with tf.Graph().as_default() as g:
        q_tokens = tf.placeholder(dtype=tf.int32, shape=[None, MAX_LEN], name="question_tokens")
        v_features = tf.placeholder(dtype=tf.float32, shape=[None, num_objs, v_dim], name='visual_features')
        answer = tf.placeholder(dtype=tf.float32, shape=[None, num_ans_candidates], name="answer")
        if kn == 0:
            attention, logits = inference(q_tokens, v_features, None, None, num_objs, num_ans_candidates, v_dim, model, kn)
        else:
            extren_knowleges = tf.placeholder(dtype=tf.int32, shape=[None, MAX_CTG], name='knowledge')
            attention, logits = inference(q_tokens, v_features, extren_knowleges, None, num_objs, num_ans_candidates, v_dim,
                                  model, kn)

        # 计算答案位置
        answer_pos = tf.argmax(logits, axis=1, name="answer_pos")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 模型循环和批处理
            num_complete_minibatches = math.floor(size / BATCH_SIZE)

            answerpos = []
            ids= []
            # 数据并行化处理
            for k in range(num_complete_minibatches + 1):

                batch_id=questions_id[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                #print(batch_id)
                batch_vfeatures = features[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_questions = questions[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                if kn == 1:
                    batch_knowldeges = knowleges[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                # batch大小数据送入网络
                # forward network
                ckpt = tf.train.get_checkpoint_state(input_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    if kn == 0:
                        batchpos = sess.run(
                            [answer_pos],
                            feed_dict={q_tokens: batch_questions,
                                       v_features: batch_vfeatures})
                    else:
                        batchpos = sess.run([answer_pos],
                            feed_dict={q_tokens: batch_questions,
                                       v_features: batch_vfeatures,
                                       extren_knowleges: batch_knowldeges})
                    answerpos.extend(batchpos)
                    ids.extend(batch_id)
                else:
                    print("No checkpoint file found")
                    break
            answerpos = np.asarray(answerpos).reshape(-1, 1)
            ids = np.asarray(ids).reshape(-1, 1)
            result = np.concatenate((answerpos, ids), axis=1)

    with open('test/model{}kn{}_pos2score.pkl'.format(model,kn),'wb') as file:
        pickle.dump(result,file)

    print('done')

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

def test_vqa(model,kn):
    # 读取答案
    with open('test/model{}kn{}_pos2score.pkl'.format(model,kn),'rb') as file:
        predict_answers=pickle.load(file)
    #print(predict_answers)
    # 获取答案位置对应的字符串
    with open('maturedata/trainval_label2ans.pkl', 'rb') as f1:
        label2ans = pickle.load(f1)
        answer_dict = {}
        for id, base_predict in enumerate(predict_answers):
            answer_dict[id] = label2ans[int(base_predict[0])]
    #print(answer_dict)
    # 读question
    with open('test/test_questions.json','r') as file:
        questions=json.load(file)
    with open('test/test_imgs_id.json','r') as file:
        imgids=json.load(file)
    dicts = []
    #print(questions)
    for i,q in enumerate(predict_answers):
        #print(i)
        #print(q[1])
        dicts.append({
            'question': questions[str(q[1])],
            'imgid': imgids[q[1]],
            'p_predict': answer_dict[i],
        })
    # answer_path = os.path.join('maturedata', 'val_target.pkl')
    # answers = pickle.load(open(answer_path, 'rb'))
    # answers = sorted(answers, key=lambda x: x['question_id'])
    # dicts = []
    # q=[]
    # img=[]
    # questiones = get_questiones()
    # for key in answer_dict.keys():
    #     answer = answers[key]
    #     qid = answer['question_id']
    #     imgid = answer['image_id']
    #     pred_answer = answer_dict[key]
    #     dicts.append({
    #         'question': questiones[qid],
    #         'imgid': imgid,
    #         'p_predict': pred_answer,
    #     })
    #     q.append(questiones[qid])
    #     img.append(imgid)
    # with open('test_questions.json','w') as file:
    #     json.dump(q,file)
    # with open('test_imgs_id.json','w') as file:
    #     json.dump(img,file)
    with open('test/test_vqa.json', 'w') as file:
        json.dump(dicts, file)

if __name__=='__main__':
    model = 1
    kn = 0
    print("model: {}\tkn: {}".format(model, kn))
    test(model,kn)
    test_vqa(model,kn)
    # with open('../test/test_questions.json', 'r') as file:
    #     questions = json.load(file)
    #     q={}
    #     for k,v in enumerate(questions):
    #         q[k]=v
    # with open('../test/test_questions.json', 'w') as file:
    #     json.dump(q,file)



