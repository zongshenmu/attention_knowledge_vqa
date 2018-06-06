#encoding=utf-8

#训练和测试网络的配置
class Config(object):
    def __init__(self):
        self.SEED = 100  # 固定随机种子
        self.EPOCHS=30 #epoch大小
        self.BATCH_SIZE = 512  # batch大小
        self.MAX_LEN=14 #每句话最长的单词数
        self.EMB_DIM=300 #词向量维数
        self.HIDEN_NODE = 1024 #隐藏层个数
        self.DOC_DIM = 300 #文本向量维数
        self.MAX_CTG=18 #每张图像最多的类别
        self.FEATURE_LEN=2048 #每个区域的向量长度
        self.NUM_BOXES=36 #每张图片边界盒子的个数