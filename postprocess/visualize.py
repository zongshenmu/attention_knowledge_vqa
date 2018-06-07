import cv2
import colorsys
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import time

#绘制attention的heatmap以及36个物体
#选择六张图片的attention作为实验的结果
#visual.json是通过人工筛选出六张适合作图展示的结果

#read:
# val2014
# test_visualize.pkl
# visual.json
# att_imgaes/{}.jpg

#write:
# val_images/{}.jpg
# bbox_imgaes/{}.jpg
# mask_imgaes/{}.jpg


#产生不同的颜色
def _create_unique_color(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id or class in detection (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(255 * r), int(255 * g), int(255 * b)

#画出图像中物体的边界
#bboxes: [[box1, box2..], [..]] in every box is [x1, y1, x2, y2]
def draw_boxes_and_label_on_image_cv2(img, bboxes):
    for c, box in enumerate(bboxes):
        assert len(box) == 4, 'boxes_map every item must be [x1, y1, x2, y2]'
        # checking box order is bb_left, bb_top, bb_width, bb_height
        # make sure all box should be int for OpenCV
        x1=int(box[0])
        y1=int(box[1])
        x2=int(box[2])
        y2=int(box[3])
        unique_color = _create_unique_color(c)
        cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, 2)
    return img

def draw_bbox_image():
    # 修改valimage名字
    # path = os.listdir('image/val2014')
    # for img in path:
    #     if not os.path.isdir(img):  # 判断是否为文件夹，如果是输出所有文件就改成： os.path.isfile(p)
    #         str = 'COCO_val2014_'
    #         start = len(str)
    #         end = img.find('.jpg')
    #         idx = img[start:end]
    #         os.rename('image/val2014/' + img, 'image/val2014/{}.jpg'.format(int(idx)))

    with open('result/test_visualize.pkl','rb') as file:
        attentions=pickle.load(file)
    img_path='image/val2014/{}.jpg'
    new_img_path='image/val_images/{}.jpg'
    save_path='result/bbox_imgaes/{}.jpg'
    i=1
    print('start')
    for item in attentions:
        img=img_path.format(i)
        bbox=item['bbox']
        image=cv2.imread(img)
        cv2.imwrite(new_img_path.format(i), image)
        bbox_image=draw_boxes_and_label_on_image_cv2(image,bbox)
        cv2.imwrite(save_path.format(i), bbox_image)
        i+=1
    print('done')

def draw_mask_image():
    # 底板图案
    bottom_pic = 'result/bbox_images/{}.jpg'
    save_path = 'result/mask_images/{}.jpg'
    with open('result/test_visualize.pkl', 'rb') as file:
        data = pickle.load(file)
    for i in range(1000):
        index = i + 1
        img = cv2.imread(bottom_pic.format(index))
        bboxs = data[i]['bbox']
        attention = data[i]['att']
        maxpos = np.argmax(attention)
        # print(bbox)
        # print(remain)
        mask = np.ones(img.shape, dtype="uint8")

        bbox = bboxs[maxpos]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        # print(x1,x2)
        mask[y1:y2, x1:x2, :] = 0

        mask = mask * 255
        newimg = cv2.add(img, mask)
        cv2.imwrite(save_path.format(index), newimg)

def draw_heatmap_image():
    with open('result/visual.json', 'r') as file:
        visual = json.load(file)
    #print(len(visual))
    img_path = 'result/selected_images/{}.jpg'
    save_heat_path= 'result/att_images/{}.jpg'
    sns.set()
    for i in range(6):
        img = cv2.imread(img_path.format(i + 1))
        att_mat = np.zeros(img.shape[0:2])
        print(att_mat.shape)
        bboxs = visual[i]['bbox']
        att_w = visual[i]['att']
        for idx, bbox in enumerate(bboxs):
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            for x in range(x2 - x1):
                for y in range(y2 - y1):
                    curr = att_w[idx]
                    if curr > att_mat[y1 + y][x1 + x]:
                        att_mat[y1 + y][x1 + x] = curr
        ax = sns.heatmap(att_mat, cbar=False, xticklabels=False, yticklabels=False, square=True, cmap='Blues')
        # 子图填满整个画布
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        fig = ax.get_figure()
        fig.savefig(save_heat_path.format(i+1))
        plt.close()
        #plt.show()

def draw_mix_image():
    bg_path = 'result/selected_images/{}.jpg'
    fg_path = 'result/att_images/{}.png'
    save_path = 'result/att_images/new{}.jpg'
    for i in range(6):
        bg_img = cv2.imread(bg_path.format(i + 1))
        fg_img = cv2.imread(fg_path.format(i + 1))
        h, w=bg_img.shape[0:2]
        fg_img = cv2.resize(fg_img,(w,h))
        print(bg_img.shape)
        print(fg_img.shape)
        overlapping = cv2.addWeighted(bg_img, 0.3, fg_img, 0.7, 0)
        cv2.imwrite(save_path.format(i+1), overlapping)


if __name__=='__main__':
    #draw_bbox_image()
    #draw_mask_image()
    draw_heatmap_image()
    draw_mix_image()