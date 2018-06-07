#encoding=utf-8

#将每张图片对应的物体和知识做相应的映射，得到行为图片id到列为对应知识库文档的矩阵

#read:
# instances_train2014.json instances_val2014.json
# train36_imgid2idx.pkl val36_imgid2idx.pkl
# train_imgs.hdf5 val_imgs.hdf5

#write:
# docid.json
# train_img2categories.json val_img2categories.json
# trainimg_knowledge.pkl valimg_knowledge.pkl

import json
import pickle
import h5py
import numpy as np

def img2knoeldge():
    modes = ['train', 'val']
    #从数据集提取的特征
    for mode in modes:
        instances_path = 'rawdata/instances_{}2014.json'.format(mode)
        img2idx_path = 'maturedata/{}36_imgid2idx.pkl'.format(mode)
        with open(instances_path, 'r') as file:
            print('dump start')
            instances = json.load(file)
            categories = instances['categories']
            annotations = instances['annotations']

            # 物体类别
            ctgs = {}
            if mode=='train':
                for item in categories:
                    ctg = item['id']
                    name = item['name']
                    ctgs[ctg] = name
                with open("maturedata/docid.json".format(mode), 'w') as f:
                    json.dump(ctgs, f)

            # 原始图片id对应的物体类别
            objects = {}
            for value in annotations:
                img_id = value['image_id']
                ctg_id = value['category_id']
                if img_id in objects.keys():
                    # print('already have')
                    if ctgs[ctg_id] not in objects[img_id]:
                        objects[img_id].append(ctgs[ctg_id])
                else:
                    objects[img_id] = [ctgs[ctg_id]]

            img2ctg = {}
            #print(len(objects))
            with open(img2idx_path, 'rb') as f:
                img2idx = pickle.load(f)
                #print(len(img2idx))
                for k in objects.keys():
                    #print(k, img2idx[k])
                    img2ctg[img2idx[k]] = objects[k]
                with open('maturedata/{}_img2categories.json'.format(mode), 'w') as wf:
                    json.dump(img2ctg, wf)
            print('dump over')

            #找到每个对象最多的类别
            maxv = 0
            for v in img2ctg.values():
                if len(v) > maxv:
                    maxv = len(v)
            #类和id反转
            reverse_ctg = dict(zip(ctg.values(), ctg.keys()))

            # 从知识库提取的物体类别对应图片id
            with h5py.File('maturedata/{}_imgs.hdf5'.format(mode), 'r') as hf:
                imgs = np.asarray(hf.get('imgs'), dtype=np.int32)
            img2id = {}
            # 并不是所有的图片都有object
            for i, v in enumerate(imgs):
                if v in img2id.keys():
                    img2id[v].append(i)
                else:
                    img2id[v] = [i]
            print(len(img2id))
            # 每张图片最大的类别
            maxctgs = maxv #18
            maximg = imgs.size
            lookups = np.ones((maximg, maxctgs)) * (maximg + 1)
            print(lookups)
            # print(reverse_ctg)
            for k, v in img2ctg.items():
                new_value = []
                # print(v)
                for i in v:
                    # print(i)
                    # print(reverse_ctg[i])
                    new_value.append(int(reverse_ctg[i]))
                new_value = new_value + [maximg] * (maxctgs - len(new_value))
                index = img2id[int(k)]
                for id in index:
                    lookups[id] = new_value
            with open('maturedata/{}img_knowledge.pkl'.format(mode), 'wb') as f:
                pickle.dump(lookups, f)

if __name__=='__main__':
    img2knoeldge()
