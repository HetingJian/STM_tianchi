import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob


class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)
        # 这个K是什么东西？下面的All_to_OneHot等函数里的Ms,Fs我猜就是MaskTensor和FrameTensor的意思
        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        # np.empty()返回一个随机元素的矩阵,正如文档所说"without initializing entries"
        # N_frames 返回一个高维Tensor,参数为{num_frames[0],shape[0],3),同理 N_mask返回一个二维Tensor
        # frames就是一个视频给的图片帧的数量，shape就是分辨率那个参数，至于这个3是三通道RGB吗？
        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
        # '{：05d}.jpg'.format(x)是字符串的format格式化输出，冒号相当于C里的%，:05也就是输出五位数嘛
        # 我们需要改成：06因为我们的视频有六位数的帧数
        # Image模块的convert()函数,使用不同的参数，将当前的图像转换为新的模式，并产生新的图像作为返回值
        # frames用的是'RGB'格式，蒙版用的是'P'格式
        # 模式"P"为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的(?)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB')) / 255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                N_masks[f] = 255

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


if __name__ == '__main__':
    pass
