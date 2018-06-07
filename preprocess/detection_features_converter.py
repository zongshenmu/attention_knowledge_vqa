#encoding=utf-8

#读取从Faster-R-CNN保存的区域特征向量
#根据保存的tsv表对应顺序写每张图片对应的边界盒子位置及其特征

#read:
# trainval_resnet101_faster_rcnn_genome_36.tsv
# train_ids.pkl val_ids.pkl

#write:
# train36_imgid2idx.pkl val36_imgid2idx.pkl
# train36.hdf5 val36.hdf5
# val_bbox.hdf5

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import pickle
import numpy as np

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'rawdata/trainval_resnet101_faster_rcnn_genome_36.tsv'
train_ids_file = 'rawdata/train_ids.pkl'
val_ids_file = 'rawdata/val_ids.pkl'

train_data_file = 'maturedata/train36.hdf5'
val_data_file = 'maturedata/val36.hdf5'
train_indices_file = 'maturedata/train36_imgid2idx.pkl'
val_indices_file = 'maturedata/val36_imgid2idx.pkl'
val_bbox_file = 'maturedata/val_bbox.hdf5'


feature_length = 2048
num_fixed_boxes = 36

#文件夹加载后缀的文件
def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs

#返回图像id
def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids

#pickle2.7到3.6的转换
class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

#h5py库保存键值格式的数据
if __name__ == '__main__':
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    # 加载id
    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        with open(train_ids_file) as data_file:
            train_imgids = pickle.load(StrToBytes(data_file))
        with open(val_ids_file) as data_file:
            val_imgids = pickle.load(StrToBytes(data_file))
    else:
        train_imgids = load_imageid('image/train2014')
        val_imgids = load_imageid('image/val2014')
        pickle.dump(train_imgids, open(train_ids_file, 'wb'))
        pickle.dump(val_imgids, open(val_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}

    # 图像特征 f代表float
    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    # 边界
    #train_img_bb = h_train.create_dataset(
    #    'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    # 类的分数和边界比例
    #train_spatial_img_features = h_train.create_dataset(
    #    'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    #val_img_bb = h_val.create_dataset(
    #    'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    #val_spatial_img_features = h_val.create_dataset(
    #    'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')
    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')


    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(infile, "r") as tsv_in_file:
        # delimiter为逗号是分割csv、制表符分割tsv
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            # 图像的边界个数
            item['num_boxes'] = int(item['num_boxes'])
            # 图像的id
            image_id = int(item['image_id'])
            # base64编码规则
            # bbox:x1,y1,x2,y2左上和右下坐标
            bboxes = np.frombuffer(
                base64.b64decode(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            # 图像的长宽
            #image_w = float(item['image_w'])
            #image_h = float(item['image_h'])
            # base64编码规则
            #bboxes = np.frombuffer(
            #    base64.b64decode(item['boxes']),
            #    dtype=np.float32).reshape((item['num_boxes'], -1))

            #box_width = bboxes[:, 2] - bboxes[:, 0]
            #box_height = bboxes[:, 3] - bboxes[:, 1]

            #scaled_width = box_width / image_w
            #scaled_height = box_height / image_h
            #scaled_x = bboxes[:, 0] / image_w
            #scaled_y = bboxes[:, 1] / image_h
            # ...表示省略所有的维度，none和newaxis是增加一个维度，:表示一个维度
            #box_width = box_width[..., np.newaxis]
            #box_height = box_height[..., np.newaxis]
            #scaled_width = scaled_width[..., np.newaxis]
            #scaled_height = scaled_height[..., np.newaxis]
            #scaled_x = scaled_x[..., np.newaxis]
            #scaled_y = scaled_y[..., np.newaxis]
            # 组合成三维数组，axis=0表示数量，1表示类别，2表示值
            #spatial_features = np.concatenate(
            #    (scaled_x,
            #     scaled_y,
            #     scaled_x + scaled_width,
            #     scaled_y + scaled_height,
            #     scaled_width,
            #     scaled_height),
            #    axis=1)

            # remove参数是元素，pop参数是下标
            if image_id in list(train_imgids):
                train_imgids.remove(image_id)
                #图片下标
                train_indices[image_id] = train_counter
                #box图片中坐标
                #train_img_bb[train_counter, :, :] = bboxes
                #box特征
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                #空间相对位置
                #train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in list(val_imgids):
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                #val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_img_bb[val_counter, :, :] = bboxes
                #val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    pickle.dump(train_indices, open(train_indices_file, 'wb'))
    pickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")

