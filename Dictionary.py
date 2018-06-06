#encoding=utf-8

#定义字典,构建两个表
# word2idx 单词索引下标
# idx2word 下标索引单词
# 序列化句子，将句子的单词转换成在字典中所对应的位置

#read:
# dictionary.pkl

#write：
# dictionary.pkl


from __future__ import print_function
import pickle

# 继承object类
class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            # 单词所在位置key：value
            word2idx = {}
        if idx2word is None:
            # 非重复单词的列表，计数总共单词
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    # 包装器，setter和getter
    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    # 序列化单词位置
    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        # '\'s'表示类似于xxx's的单词，去掉非字母字符
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        # 句子单词以空格区分，分解句子
        words = sentence.split()
        # 这句话的单词所在字典位置序列
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx.keys():
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.padding_idx)
        return tokens

    # dump倾倒,序列化对象到文件
    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    # 类方法，类调用
    @classmethod
    # cls表示类等同于self
    def load_from_file(cls, path):
        #print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    # 添加单词
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    #获取类的长度 重定义，通过len调用
    def __len__(self):
        return len(self.idx2word)