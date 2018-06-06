#encoding=utf-8

#前向神经网络的定义

#read:
# glove6b_init_300d.npy
# doc_embeddings.pkl

import tensorflow as tf
import numpy as np
import pickle
import config

cfg=config.Config()

EMB_DIM=cfg.EMB_DIM
HIDEN_NODE=cfg.HIDEN_NODE
DOC_DIM=cfg.DOC_DIM

def inference(q_tokens, v_features, knowledges, regularizer, num_objs, num_ans_candidates, v_dim, model=0, kn=0):
    with tf.name_scope('WordEmbedding'):
        weight_init = np.load('maturedata/glove6b_init_300d.npy')
        true_weight=tf.get_variable(name="true_weight_embedding",initializer=weight_init)
        padding_index=np.zeros([1,EMB_DIM],dtype=np.float32)
        weight=tf.concat([true_weight,padding_index],axis=0)
        w_embs=tf.nn.embedding_lookup(weight,q_tokens,name="words_embedding")

    with tf.name_scope('QuestionEmbedding'):
        gru_cell=tf.nn.rnn_cell.BasicRNNCell(HIDEN_NODE)
        q_features,_=tf.nn.dynamic_rnn(gru_cell,w_embs,dtype=tf.float32,scope="GRU")
        q_features=q_features[:,-1,:]

    #模型的选择 1:attention result 2:new attention mode 0:baseline
    if model==1:
        # batch_norm加快收敛和最优解寻找
        with tf.name_scope('Attention'):
            q_new = tf.expand_dims(q_features, 1, name="expand_dims")
            q_new = tf.tile(q_new, (1, num_objs, 1), name="tile")
            joint_emb = tf.concat([v_features, q_new], 2)
            W_joint1 = tf.get_variable(name="weight_joint_embedding1",
                                       shape=[v_dim + HIDEN_NODE, HIDEN_NODE],
                                       dtype=tf.float32,
                                       regularizer=regularizer,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            B_joint1 = tf.get_variable(name="bais_joint_embedding1",
                                       shape=[HIDEN_NODE],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            Logits_joint1 = tf.einsum('ijk,kl->ijl', joint_emb, W_joint1) + B_joint1
            Hjoint1 = tf.nn.relu(Logits_joint1, name='joint_fc')
            W_joint2 = tf.get_variable(name="weight_joint_embedding2",
                                       shape=[HIDEN_NODE, 1],
                                       dtype=tf.float32,
                                       regularizer=regularizer,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            B_joint2 = tf.get_variable(name="bais_joint_embedding2",
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            Logits_joint2 = tf.einsum('ijk,kl->ijl', Hjoint1, W_joint2) + B_joint2
            Logits_joint2_norm = tf.contrib.layers.batch_norm(Logits_joint2, center=True)
            v_attention = tf.nn.softmax(Logits_joint2_norm, dim=1, name="visual_attention")
            v_att_features = tf.reduce_sum(tf.multiply(v_attention, v_features, name="visaul_att_features"), 1)
    elif model==2:
        # batch_norm加快收敛和最优解寻找
        with tf.name_scope("New_Attention"):
            Wqa = tf.get_variable(name="w_qafc",
                                  shape=[HIDEN_NODE, HIDEN_NODE],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bqa = tf.get_variable(name="b_qafc",
                                  shape=[HIDEN_NODE],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.0))
            Logits_qa = tf.matmul(q_features, Wqa, name='qafc_logits') + Bqa
            q_repra = tf.nn.relu(Logits_qa, name="qa_fc")
            q_repra = tf.tile(tf.expand_dims(q_repra, 1, name="expand_dims"), (1, num_objs, 1), name="tile")
            Wva = tf.get_variable(name="w_vafc",
                                  shape=[v_dim, HIDEN_NODE],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bva = tf.get_variable(name="b_vafc",
                                  shape=[HIDEN_NODE],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.0))
            Logits_va = tf.einsum('ijk,kl->ijl', v_features, Wva) + Bva
            v_repra = tf.nn.relu(Logits_va, name="va_fc")
            joint_repr = v_repra * q_repra
            joint_repr_drop = tf.nn.dropout(joint_repr, keep_prob=0.25, name="joint_repr_dropout")
            W_joint = tf.get_variable(name="weight_joint_embedding",
                                      shape=[HIDEN_NODE, 1],
                                      dtype=tf.float32,
                                      regularizer=regularizer,
                                      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            B_joint = tf.get_variable(name="bais_joint_embedding",
                                      shape=[1],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
            Logits_joint = tf.einsum('ijk,kl->ijl', joint_repr_drop, W_joint) + B_joint
            Logits_joint_norm = tf.contrib.layers.batch_norm(Logits_joint, center=True)
            v_attention = tf.nn.softmax(Logits_joint_norm, dim=1, name="visual_attention")
            v_att_features = tf.reduce_sum(tf.multiply(v_attention, v_features, name="visaul_att_features"), 1)
    else:
        with tf.name_scope('Attention'):
            v_attention = tf.ones([1,num_objs], dtype=tf.float32, name="equal_attention")
            v_att_features = tf.reduce_sum(v_features, 1)


    with tf.name_scope('question_net'):
        Wq=tf.get_variable(name="w_qfc",
                           shape=[HIDEN_NODE,HIDEN_NODE],
                           dtype=tf.float32,
                           regularizer=regularizer,
                           initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02))
        Bq=tf.get_variable(name="b_qfc",
                           shape=[HIDEN_NODE],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0))
        Logits_q=tf.matmul(q_features,Wq,name='qfc_logits')+Bq
        q_repr=tf.nn.relu(Logits_q,name="q_fc")

    with tf.name_scope('visual_net'):
        Wv = tf.get_variable(name="w_vfc",
                             shape=[v_dim, HIDEN_NODE],
                             dtype=tf.float32,
                             regularizer=regularizer,
                             initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02))
        Bv = tf.get_variable(name="b_vfc",
                             shape=[HIDEN_NODE],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        Logits_v = tf.matmul(v_att_features, Wv, name='vfc_logits') + Bv
        v_repr = tf.nn.relu(Logits_v, name="v_fc")

    if kn==0:
        with tf.name_scope("classifier"):
            joint_repr = tf.multiply(q_repr, v_repr, name="joint_representation")
            Wjr = tf.get_variable(name="w_joint_repr_fc",
                                  shape=[HIDEN_NODE, 2 * HIDEN_NODE],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bjr = tf.get_variable(name="b_joint_repr_fc",
                                  shape=[2 * HIDEN_NODE],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.0))
            hjr = tf.nn.relu(tf.matmul(joint_repr, Wjr, name="joint_repr_logits") + Bjr, name="joint_repr_fc")
            hdrop = tf.nn.dropout(hjr, keep_prob=0.5, name="classifier_dropout")
            Wc = tf.get_variable(name="w_classifier",
                                 shape=[2 * HIDEN_NODE, num_ans_candidates],
                                 dtype=tf.float32,
                                 regularizer=regularizer,
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bc = tf.get_variable(name="b_classifier",
                                 shape=[num_ans_candidates],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            Logits_c = tf.matmul(hdrop, Wc, name='classifier_logits') + Bc
    else:
        # 知识库的引导
        with tf.name_scope("extern_knowldge"):
            weight_init = pickle.load(open('maturedata/doc_embeddings.pkl', 'rb'))
            true_weight = tf.get_variable(name="doc_embedding", initializer=weight_init)
            padding_index = np.zeros([1, DOC_DIM], dtype=np.float32)
            unknow_index = np.ones([1, DOC_DIM], dtype=np.float32) * 0.001
            weight = tf.concat([true_weight, padding_index, unknow_index], axis=0)
            doc_embs = tf.nn.embedding_lookup(weight, knowledges, name="doc_embedding")
            doc_embs = tf.reduce_mean(doc_embs, axis=1)
            knowledge_dense = tf.contrib.layers.fully_connected(doc_embs,
                                                                HIDEN_NODE,
                                                                activation_fn=None,
                                                                weights_initializer=tf.random_normal_initializer(
                                                                    mean=0.0, stddev=0.02),
                                                                scope="dense_expanddim_knowldge")
            knowledge_bn = tf.contrib.layers.batch_norm(knowledge_dense,
                                                        center=True, scale=True,
                                                        scope='knowldge_bn')
            knowledge_repr = tf.nn.relu(knowledge_bn, 'relu')

        with tf.name_scope("classifier"):
            kv_joint_repr = tf.multiply(knowledge_repr, v_repr, name="kv_joint_representation")
            joint_repr = tf.multiply(kv_joint_repr, q_repr, name="joint_representation")
            Wjr = tf.get_variable(name="w_joint_repr_fc",
                                  shape=[HIDEN_NODE, 2 * HIDEN_NODE],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bjr = tf.get_variable(name="b_joint_repr_fc",
                                  shape=[2 * HIDEN_NODE],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.0))
            hjr = tf.nn.relu(tf.matmul(joint_repr, Wjr, name="joint_repr_logits") + Bjr, name="joint_repr_fc")
            hdrop = tf.nn.dropout(hjr, keep_prob=0.5, name="classifier_dropout")
            Wc = tf.get_variable(name="w_classifier",
                                 shape=[2 * HIDEN_NODE, num_ans_candidates],
                                 dtype=tf.float32,
                                 regularizer=regularizer,
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Bc = tf.get_variable(name="b_classifier",
                                 shape=[num_ans_candidates],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            Logits_c = tf.matmul(hdrop, Wc, name='classifier_logits') + Bc

    return v_attention, Logits_c