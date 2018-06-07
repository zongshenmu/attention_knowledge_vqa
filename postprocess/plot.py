#endoing=utf-8

#读取模型训练后得到的loss和score
#绘制成曲线展示epcoh中的变化

#read:
# base_log.txt
# att_log.txt
# newatt_log.txt
# kn_base_log.txt
# kn_att_log.txt
# kn_newatt_log.txt

import matplotlib.pyplot as plt

epoches = range(30)

# ====================基本模型的loss score=======================

base_log = open('result/result_log/base_log.txt')
att_log = open('result/result_log/att_log.txt')
newatt_log = open('result/result_log/newatt_log.txt')

base_log_train_loss=[]
base_log_val_loss=[]
base_log_train_score=[]
base_log_val_score=[]

att_log_train_loss=[]
att_log_train_score=[]
att_log_val_loss=[]
att_log_val_score=[]

newatt_log_train_loss=[]
newatt_log_train_score=[]
newatt_log_val_loss=[]
newatt_log_val_score=[]
try:
    for line in base_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            base_log_train_loss.append(strs[1][:-1])
            base_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            base_log_val_loss.append(strs[1][:-1])
            base_log_val_score.append(strs[-2])
finally:
    base_log.close( )

try:
    for line in att_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            att_log_train_loss.append(strs[1][:-1])
            att_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            att_log_val_loss.append(strs[1][:-1])
            att_log_val_score.append(strs[-2])
finally:
    att_log.close( )

try:
    for line in newatt_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            newatt_log_train_loss.append(strs[1][:-1])
            newatt_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            newatt_log_val_loss.append(strs[1][:-1])
            newatt_log_val_score.append(strs[-2])
finally:
    newatt_log.close( )

# ==================knowledge模型的loss score=====================

kn_base_log = open('result/result_log/kn_base_log.txt')
kn_att_log = open('result/result_log/kn_att_log.txt')
kn_newatt_log = open('result/result_log/kn_newatt_log.txt')

kn_base_log_train_loss=[]
kn_base_log_val_loss=[]
kn_base_log_train_score=[]
kn_base_log_val_score=[]

kn_att_log_train_loss=[]
kn_att_log_train_score=[]
kn_att_log_val_loss=[]
kn_att_log_val_score=[]

kn_newatt_log_train_loss=[]
kn_newatt_log_train_score=[]
kn_newatt_log_val_loss=[]
kn_newatt_log_val_score=[]

try:
    for line in kn_base_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            kn_base_log_train_loss.append(strs[1][:-1])
            kn_base_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            kn_base_log_val_loss.append(strs[1][:-1])
            kn_base_log_val_score.append(strs[-2])
finally:
    kn_base_log.close()

try:
    for line in kn_att_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            kn_att_log_train_loss.append(strs[1][:-1])
            kn_att_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            kn_att_log_val_loss.append(strs[1][:-1])
            kn_att_log_val_score.append(strs[-2])
finally:
    kn_att_log.close()

try:
    for line in kn_newatt_log:
        strs = line.split(' ')
        if strs[0]=='\ttrain_loss:':
            kn_newatt_log_train_loss.append(strs[1][:-1])
            kn_newatt_log_train_score.append(strs[-1][:-1])
        elif strs[0]=='\tval_loss:':
            kn_newatt_log_val_loss.append(strs[1][:-1])
            kn_newatt_log_val_score.append(strs[-2])
finally:
    kn_newatt_log.close()

# =======================绘图======================

#基本模型绘制在一起
plt.figure(1)
plt.title('model loss without knowledge')
plt.plot(epoches, base_log_train_loss,label="base train loss")
plt.plot(epoches, base_log_val_loss,label="base val loss")
plt.plot(epoches, att_log_train_loss,label="att train loss")
plt.plot(epoches, att_log_val_loss,label="att val loss")
plt.plot(epoches, newatt_log_train_loss,label="newatt train loss")
plt.plot(epoches, newatt_log_val_loss,label="newatt val loss")
plt.legend()

plt.figure(2)
plt.title('model score without knowledge')
plt.plot(epoches, base_log_train_score,label="base train score")
plt.plot(epoches, base_log_val_score,label="base val score")
plt.plot(epoches, att_log_train_score,label="att train score")
plt.plot(epoches, att_log_val_score,label="att val score")
plt.plot(epoches, newatt_log_train_score,label="newatt train score")
plt.plot(epoches, newatt_log_val_score,label="newatt val score")
plt.legend()

#knowledge模型绘制在一起
plt.figure(3)
plt.title('model loss included knowledge')
plt.plot(epoches, kn_base_log_train_loss, label="kn_base train loss")
plt.plot(epoches, kn_base_log_val_loss, label="kn_base val loss")
plt.plot(epoches, kn_att_log_train_loss, label="kn_att train loss")
plt.plot(epoches, kn_att_log_val_loss, label="kn_att val loss")
plt.plot(epoches, kn_newatt_log_train_loss, label="kn_newatt train loss")
plt.plot(epoches, kn_newatt_log_val_loss, label="kn_newatt val loss")
plt.legend()

plt.figure(4)
plt.title('model score included knowledge')
plt.plot(epoches, kn_base_log_train_score, label="kn_base train score")
plt.plot(epoches, kn_base_log_val_score, label="kn_base val score")
plt.plot(epoches, kn_att_log_train_score, label="kn_att train score")
plt.plot(epoches, kn_att_log_val_score, label="kn_att val score")
plt.plot(epoches, kn_newatt_log_train_score, label="kn_newatt train score")
plt.plot(epoches, kn_newatt_log_val_score, label="kn_newatt val score")
plt.legend()
plt.show()

#单独绘制有无knowledge模型的score
plt.figure(5)
plt.title('base model score with(out) knowledge')
plt.plot(epoches, base_log_val_score, label="base val score")
plt.plot(epoches, kn_base_log_val_score, label="kn_base val score")
plt.legend()

plt.figure(6)
plt.title('attention model score with(out) knowledge')
plt.plot(epoches, att_log_val_score, label="att val score")
plt.plot(epoches, kn_att_log_val_score, label="kn_att val score")
plt.legend()

plt.figure(7)
plt.title('newatt result score with(out) knowledge')
plt.plot(epoches, newatt_log_val_score, label="newatt val score")
plt.plot(epoches, kn_newatt_log_val_score, label="kn_newatt val score")
plt.legend()

plt.figure(8)
plt.title('two attention models score without knowledge')
plt.plot(epoches, att_log_val_score, label="att val score")
plt.plot(epoches, newatt_log_val_score, label="newatt val score")
plt.legend()

plt.figure(9)
plt.title('two attention models score with knowledge')
plt.plot(epoches, kn_att_log_val_score, label="kn_att val score")
plt.plot(epoches, kn_newatt_log_val_score, label="kn_newatt val score")
plt.legend()

plt.show()

