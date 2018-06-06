#encoding=utf-8

#每轮训练之后都在测试集上跑一遍进行分数的评估

import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forward_network import *

cfg=config.Config()
MAX_LEN=cfg.MAX_LEN
SEED=cfg.SEED
BATCH_SIZE=cfg.BATCH_SIZE
EMB_DIM=cfg.EMB_DIM
MAX_CTG=cfg.MAX_CTG

def evaluate(features,imgs,questions,types,targets,knowleges,model,kn):
    input_path = "temp/temp_model{}kn{}/".format(model,kn)
    # 模型参数
    size = imgs.shape[0]
    num_ans_candidates = targets.shape[1]
    num_objs = features.shape[1]
    v_dim = features.shape[2]

    with tf.Graph().as_default() as g:
        q_tokens = tf.placeholder(dtype=tf.int32, shape=[None, MAX_LEN], name="question_tokens")
        v_features = tf.placeholder(dtype=tf.float32, shape=[None, num_objs, v_dim], name='visual_features')
        answer = tf.placeholder(dtype=tf.float32, shape=[None, num_ans_candidates], name="answer")
        q_types = tf.placeholder(dtype=tf.string, shape=[None, 1], name='types')
        if kn == 0:
            _, logits = inference(q_tokens, v_features, None, None, num_objs, num_ans_candidates, v_dim, model, kn)
        else:
            extren_knowleges = tf.placeholder(dtype=tf.int32, shape=[None, MAX_CTG], name='knowledge')
            _, logits = inference(q_tokens, v_features, extren_knowleges, None, num_objs, num_ans_candidates, v_dim,
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
            permutation = np.random.permutation(size)
            shuffled_imgs = imgs[permutation]
            shuffled_types = types[permutation]
            shuffled_questions = questions[permutation, :]
            shuffled_targets = targets[permutation, :]
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

            # 数据并行化处理
            for k in range(num_complete_minibatches + 1):
                batch_imgs = shuffled_imgs[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_vfeatures = features[batch_imgs]
                batch_types = shuffled_types[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                batch_questions = shuffled_questions[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                batch_targets = shuffled_targets[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :]
                if kn == 1:
                    batch_knowldeges = knowleges[batch_imgs]
                # batch大小数据送入网络
                # forward network
                ckpt = tf.train.get_checkpoint_state(input_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    if kn == 0:
                        batch_loss, batch_score, batch_yes, batch_num, batch_other, batch_std_yes, batch_std_num, batch_std_other, batch_bound = sess.run(
                            [loss, scores, yes_score, num_score, other_score, standard_yes, standard_num,
                             standard_other,
                             upper_bound],
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
                else:
                    print("No checkpoint file found")
                    break
            val_loss /= size
            val_score = 100 * val_score / size
            val_bound = 100 * val_bound / size
            val_yes = 100 * val_yes / std_yes
            val_num = 100 * val_num / std_num
            val_other = 100 * val_other / std_other

    return val_loss,val_score,val_yes,val_num,val_other,val_bound