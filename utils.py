#encoding=utf-8

#配置文件

#read:
# train36_imgid2idx.pkl val36_imgid2idx.pkl

import errno
import os
import numpy as np

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)

#创建目录
def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

#logger输出打印在控制台和文件
class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

#重命名文件
import pickle
def renameImages(mode):
    with open("maturedata/%s36_imgid2idx.pkl"%mode, 'rb') as f:
        imgid2idx = pickle.load(f)
    # for file in os.listdir('./val_2014'):
    dir_name = 'image/%s2014'%mode
    for file in os.listdir(dir_name):
        print(file)
        end = file.find('.jpg')
        if end != -1:
            start = len('COCO_val2014_')
            imgid = int(file[start:end])
            idx = imgid2idx[imgid]
            new_name = '{}.jpg'.format(idx)
            os.rename(os.path.join(dir_name, file), os.path.join(dir_name, new_name))
    print("done")
