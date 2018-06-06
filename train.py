#encoding=utf-8

#训练整个数据集
#通过命令行控制参数选择model和kn
#每轮训练和测试选择最优的分数

#read:
# train36.hdf5 val36.hdf5
# train_imgs.hdf5 val_imgs.hdf5
# train_questions.hdf5 val_questions.hdf5
# train_targets.hdf5 val_targets.hdf5
# trainimg_knowledge.pkl valimg_knowledge.pkl
# val_qtype.pkl
#write:
# log.txt
# checkpoint

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forward_network import *
from val import *
import h5py
import time
import argparse
import config

cfg=config.Config()

EPOCHS=cfg.EPOCHS
MAX_LEN=cfg.MAX_LEN
SEED=cfg.SEED
BATCH_SIZE=cfg.BATCH_SIZE
EMB_DIM=cfg.EMB_DIM
MAX_CTG=cfg.MAX_CTG

#设置随机种子
tf.set_random_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_choice', type=int, default=0)
    parser.add_argument('--kn_choice', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    model=args.model_choice
    kn=args.kn_choice
    print("model: {}\tkn: {}".format(model,kn))

    print('loading rawdata from h5 file')
    preroot = "maturedata"
    # train加载v、q、a数据
    train_name = 'train'
    h5_path0 = os.path.join(preroot, '%s36.hdf5' % train_name)
    with h5py.File(h5_path0, 'r') as hf:
        features = np.array(hf.get('image_features'), dtype=np.float32)
    h5_path1 = os.path.join(preroot, '%s_imgs.hdf5' % train_name)
    with h5py.File(h5_path1, 'r') as hf:
        imgs = np.array(hf.get('imgs'), dtype=np.int32)
    h5_path2 = os.path.join(preroot, '%s_questions.hdf5' % train_name)
    with h5py.File(h5_path2, 'r') as hf:
        questions = np.array(hf.get('questions'), dtype=np.int32)
    h5_path3 = os.path.join(preroot, '%s_targets.hdf5' % train_name)
    with h5py.File(h5_path3, 'r') as hf:
        targets = np.array(hf.get('targets'), dtype=np.float32)
    # train模型参数
    size = imgs.shape[0]
    num_ans_candidates = targets.shape[1]
    num_objs = features.shape[1]
    v_dim = features.shape[2]

    # val加载v、q、a数据name = 'val'
    val_name = "val"
    val_h5_path0 = os.path.join(preroot, '%s36.hdf5' % val_name)
    with h5py.File(val_h5_path0, 'r') as hf:
        val_features = np.array(hf.get('image_features'), dtype=np.float32)
    val_h5_path1 = os.path.join(preroot, '%s_imgs.hdf5' % val_name)
    with h5py.File(val_h5_path1, 'r') as hf:
        val_imgs = np.array(hf.get('imgs'), dtype=np.int32)
    val_h5_path2 = os.path.join(preroot, '%s_questions.hdf5' % val_name)
    with h5py.File(val_h5_path2, 'r') as hf:
        val_questions = np.array(hf.get('questions'), dtype=np.int32)
    val_h5_path3 = os.path.join(preroot, '%s_targets.hdf5' % val_name)
    with h5py.File(val_h5_path3, 'r') as hf:
        val_targets = np.array(hf.get('targets'), dtype=np.float32)
    val_type_path=os.path.join(preroot, '{}_qtype.pkl'.format(val_name))
    with open(val_type_path, 'rb') as qf:
        val_types = np.asarray(pickle.load(qf)).reshape(-1, 1)

    if kn==0:
        knowleges=None
        val_knowleges=None
    else:
        train_kn_path=os.path.join(preroot, '%simg_knowledge.pkl' % train_name)
        with open(train_kn_path, 'rb') as file:
            knowleges = pickle.load(file)
        val_kn_path=os.path.join(preroot, '%simg_knowledge.pkl' % val_name)
        with open(val_kn_path, 'rb') as file:
            val_knowleges = pickle.load(file)

    print('loading rawdata over')

    # 保存模型
    output = "result/saved_model{}kn{}/".format(model,kn)
    temp="temp/temp_model{}kn{}/".format(model,kn)
    utils.create_dir(output)
    utils.create_dir(temp)
    logger = utils.Logger(os.path.join(output, 'log.txt'))

    # 定义输入网络的变量
    q_tokens = tf.placeholder(dtype=tf.int32, shape=[None, MAX_LEN], name="question_tokens")
    v_features = tf.placeholder(dtype=tf.float32, shape=[None, num_objs, v_dim], name='visual_features')
    answer = tf.placeholder(dtype=tf.float32, shape=[None, num_ans_candidates], name="answer")
    regularizer = tf.contrib.layers.l2_regularizer(0.0008)
    if kn==0:
        _,logits = inference(q_tokens, v_features, None, regularizer, num_objs, num_ans_candidates, v_dim, model, kn)
    else:
        extren_knowleges = tf.placeholder(dtype=tf.int32, shape=[None, MAX_CTG], name='knowledge')
        _,logits = inference(q_tokens, v_features, extren_knowleges, regularizer, num_objs, num_ans_candidates,v_dim, model, kn)
    # 计算损失
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=answer))
    # 计算分数
    answer_pos = tf.argmax(logits, axis=1, name="answer_pos")
    pos = tf.one_hot(answer_pos, num_ans_candidates, 1., 0., dtype=tf.float32)
    scores = tf.reduce_sum(tf.multiply(pos, answer))
    # 正则化损失
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # 总损失
    cost = loss + regularization_loss
    train_step = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.8).minimize(cost)

    saver = tf.train.Saver()
    print("start train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        best_score = 0
        # 模型循环和批处理
        for i in range(EPOCHS):
            permutation = np.random.permutation(size)
            shuffled_imgs = imgs[permutation]
            shuffled_questions = questions[permutation, :]
            shuffled_targets = targets[permutation, :]
            num_complete_minibatches = math.floor(size / BATCH_SIZE)
            total_loss = 0
            train_score = 0
            t = time.time()
            # 数据并行化处理
            for k in range(num_complete_minibatches + 1):
                batch_imgs = shuffled_imgs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_vfeatures = features[batch_imgs]
                batch_questions = shuffled_questions[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                batch_targets = shuffled_targets[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                if kn==1:
                    batch_knowldeges = knowleges[batch_imgs]

                # batch大小数据送入网络
                # forward network
                if kn==0:
                    _, batch_loss, batch_score = sess.run([train_step, cost, scores],
                                                          feed_dict={q_tokens: batch_questions,
                                                                     v_features: batch_vfeatures,
                                                                     answer: batch_targets})
                else:
                    _, batch_loss, batch_score = sess.run([train_step, cost, scores],
                                                          feed_dict={q_tokens: batch_questions,
                                                                     v_features: batch_vfeatures,
                                                                     answer: batch_targets,
                                                                     extren_knowleges: batch_knowldeges})
                total_loss += batch_loss
                train_score += batch_score
            total_loss /= size
            train_score = 100 * train_score / size
            temp_path = os.path.join(temp, 'result.pth')
            saver.save(sess, temp_path)
            if kn==0:
                val_loss, val_score, yes_score, num_score, other_score, val_bound = evaluate(val_features, val_imgs,
                                                                                             val_questions, val_types,
                                                                                             val_targets,None,
                                                                                             model,kn)
            else:
                val_loss, val_score, yes_score, num_score, other_score, val_bound = evaluate(val_features, val_imgs,
                                                                                             val_questions, val_types,
                                                                                             val_targets, val_knowleges,
                                                                                             model, kn)
            logger.write('epoch %d, time: %.2f' % (i, (time.time() - t)))
            logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
            logger.write('\tval_loss: %.2f, val_score: %.2f (%.2f)' % (val_loss, val_score, val_bound))
            logger.write('\tyes_score: %.2f, num_score: %.2f other_score:%.2f' % (yes_score, num_score, other_score))
            if val_score > best_score:
                model_path = os.path.join(output, 'model.pth')
                saver.save(sess, model_path)
                best_score = val_score
        logger.write('\tbest val_score: %.2f' % best_score)
        print("end!")

