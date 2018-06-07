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
    preroot = "maturedata"
    # train加载v、q、a数据
    val_name = "val"
    print('loading %s rawdata from h5 file' % val_name)
    # val加载v、q、a数据name = 'val'
    val_h5_path0 = os.path.join(preroot, '%s36.hdf5' % val_name)
    with h5py.File(val_h5_path0, 'r') as hf:
        features = np.array(hf.get('image_features'), dtype=np.float32)
    val_h5_path1 = os.path.join(preroot, '%s_imgs.hdf5' % val_name)
    with h5py.File(val_h5_path1, 'r') as hf:
        imgs = np.array(hf.get('imgs'), dtype=np.int32)
    val_h5_path2 = os.path.join(preroot, '%s_questions.hdf5' % val_name)
    with h5py.File(val_h5_path2, 'r') as hf:
        questions = np.array(hf.get('questions'), dtype=np.int32)
    val_h5_path3 = os.path.join(preroot, '%s_targets.hdf5' % val_name)
    with h5py.File(val_h5_path3, 'r') as hf:
        targets = np.array(hf.get('targets'), dtype=np.float32)
    val_type_path = os.path.join(preroot, '{}_qtype.pkl'.format(val_name))
    with open(val_type_path, 'rb') as qf:
        types = np.asarray(pickle.load(qf)).reshape(-1, 1)
    if kn==0:
        knowleges=None
    else:
        val_knowledge_path = os.path.join(preroot, '%simg_knowledge.pkl' % val_name)
        with open(val_knowledge_path, 'rb') as file:
            knowleges = pickle.load(file)
    print('loading %s rawdata over' % val_name)

    size = imgs.shape[0]
    num_ans_candidates = targets.shape[1]
    num_objs = features.shape[1]
    v_dim = features.shape[2]

    # 加载模型
    input_path = "result/saved_model{}kn{}/".format(model,kn)
    with tf.Graph().as_default() as g:
        q_tokens = tf.placeholder(dtype=tf.int32, shape=[None, MAX_LEN], name="question_tokens")
        v_features = tf.placeholder(dtype=tf.float32, shape=[None, num_objs, v_dim], name='visual_features')
        answer = tf.placeholder(dtype=tf.float32, shape=[None, num_ans_candidates], name="answer")
        q_types = tf.placeholder(dtype=tf.string, shape=[None, 1], name='types')
        if kn == 0:
            attention, logits = inference(q_tokens, v_features, None, None, num_objs, num_ans_candidates, v_dim, model, kn)
        else:
            extren_knowleges = tf.placeholder(dtype=tf.int32, shape=[None, MAX_CTG], name='knowledge')
            attention, logits = inference(q_tokens, v_features, extren_knowleges, None, num_objs, num_ans_candidates, v_dim,
                                  model, kn)

        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=answer))

        # 计算分数
        answer_pos = tf.argmax(logits, axis=1, name="answer_pos")
        pos = tf.one_hot(answer_pos, num_ans_candidates, 1., 0., dtype=tf.float32)
        v_scores = tf.reduce_sum(tf.multiply(pos, answer), axis=1, keep_dims=True)
        yes_type = tf.cast(tf.equal(q_types, 'yes/no'), tf.float32)
        num_type = tf.cast(tf.equal(q_types, 'number'), tf.float32)
        other_type = tf.cast(tf.equal(q_types, 'other'), tf.float32)
        yes_score = tf.reduce_sum(tf.multiply(yes_type, v_scores))
        num_score = tf.reduce_sum(tf.multiply(num_type, v_scores))
        other_score = tf.reduce_sum(tf.multiply(other_type, v_scores))
        scores = tf.reduce_sum(v_scores)

        # 计算分数上限
        standard_answer = tf.reduce_max(answer, axis=1, keep_dims=True)
        standard_yes = tf.reduce_sum(tf.multiply(standard_answer, yes_type))
        standard_num = tf.reduce_sum(tf.multiply(standard_answer, num_type))
        standard_other = tf.reduce_sum(tf.multiply(standard_answer, other_type))
        upper_bound = tf.reduce_sum(standard_answer)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 模型循环和批处理
            num_complete_minibatches = math.floor(size / BATCH_SIZE)

            val_score = 0
            val_yes = 0
            val_num = 0
            val_other = 0
            val_bound = 0
            std_yes = 0
            std_num = 0
            std_other = 0
            val_loss = 0
            feature_attention = []
            answerpos = []
            answerscore = []
            # 数据并行化处理
            for k in range(num_complete_minibatches + 1):
                batch_imgs = imgs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_vfeatures = features[batch_imgs]
                batch_types = types[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_questions = questions[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                batch_targets = targets[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                if kn == 1:
                    batch_knowldeges = knowleges[batch_imgs]
                # batch大小数据送入网络
                # forward network
                ckpt = tf.train.get_checkpoint_state(input_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    if kn == 0:
                        batch_loss, \
                        batch_score, batch_yes, batch_num, batch_other, \
                        batch_std_yes, batch_std_num, batch_std_other, batch_bound, \
                        batchatt, \
                        batchpos, batchscore = sess.run(
                            [loss, scores, yes_score, num_score, other_score,
                             standard_yes, standard_num,standard_other,upper_bound,
                             attention,
                             answer_pos,v_scores],
                            feed_dict={q_tokens: batch_questions,
                                       v_features: batch_vfeatures,
                                       answer: batch_targets,
                                       q_types: batch_types})
                    else:
                        batch_loss, batch_score, batch_yes, batch_num, batch_other, batch_std_yes, batch_std_num, batch_std_other, batch_bound = sess.run(
                            [loss, scores, yes_score, num_score, other_score, standard_yes, standard_num,
                             standard_other,
                             upper_bound],
                            feed_dict={q_tokens: batch_questions,
                                       v_features: batch_vfeatures,
                                       answer: batch_targets,
                                       q_types: batch_types,
                                       extren_knowleges: batch_knowldeges})
                    val_loss += batch_loss
                    val_score += batch_score
                    val_yes += batch_yes
                    val_num += batch_num
                    val_other += batch_other
                    val_bound += batch_bound
                    std_yes += batch_std_yes
                    std_num += batch_std_num
                    std_other += batch_std_other
                    feature_attention.extend(batchatt)
                    answerpos.extend(batchpos)
                    answerscore.extend(batchscore)
                else:
                    print("No checkpoint file found")
                    break
            val_loss /= size
            val_score = 100 * val_score / size
            val_bound = 100 * val_bound / size
            val_yes = 100 * val_yes / std_yes
            val_num = 100 * val_num / std_num
            val_other = 100 * val_other / std_other
            print('\ttest_loss: %.2f, test_score: %.2f (%.2f)' % (val_loss, val_score, val_bound))
            print('\tyes_score: %.2f, num_score: %.2f other_score:%.2f' % (val_yes, val_num, val_other))
            answerpos=np.asarray(answerpos).reshape(-1,1)
            answerscore=np.asarray(answerscore).reshape(-1,1)
            result=np.concatenate((answerpos,answerscore),axis=1)

    with open('result/model{}kn{}_pos2score.pkl'.format(model,kn),'wb') as file:
        pickle.dump(result,file)

    print('done')
    return feature_attention

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
    with open('result/model{}kn{}_pos2score.pkl'.format(model,kn),'rb') as file:
        predict_answers=pickle.load(file)
    # 获取答案位置对应的字符串
    with open('maturedata/trainval_label2ans.pkl', 'rb') as f1:
        label2ans = pickle.load(f1)
        answer_dict = {}
        for id, base_predict in enumerate(predict_answers):
            answer_dict[id] = label2ans[int(base_predict[0])]
    # 读question和answer
    answer_path = os.path.join('maturedata', 'val_target.pkl')
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    dicts = []
    questiones = get_questiones()
    for key in answer_dict.keys():
        answer = answers[key]
        qid = answer['question_id']
        imgid = answer['image_id']
        pred_answer = answer_dict[key]
        dicts.append({
            'imgindex': key,
            'question': questiones[qid],
            'imgid': imgid,
            'p_predict': pred_answer,
        })
    with open('result/test_vqa.json', 'w') as file:
        json.dump(dicts, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_choice', type=int, default=0)
    parser.add_argument('--kn_choice', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # args = parse_args()
    # model = args.model_choice
    # kn = args.kn_choice
    # print("model: {}\tkn: {}".format(model, kn))
    # test(model,kn)
    a=np.asarray([1,2])
    b=np.asarray([3,4])
    c=[a,b]
    print(c)

