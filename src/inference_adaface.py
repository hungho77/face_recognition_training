import argparse
import os

import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
import time

from backbones import get_model

@torch.no_grad()
def inference(weight, name, img_dir):
    dict_feat = {}       
    net = get_model(name)
#     net.load_state_dict(torch.load(weight))
#     net.eval()
    statedict = torch.load(weight)
    model_statedict = {key.replace('model.', ''):val for key,val in statedict['state_dict'].items() if 'model.' in key}
    net.load_state_dict(model_statedict)
    net.eval()
    for img_name in tqdm(os.listdir(img_dir)):
        try:
            img_path = os.path.join(img_dir, img_name)
            if img_path.split(".")[-1] in ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG"]:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (112, 112))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float()
                img.div_(255).sub_(0.5).div_(0.5)
                feat, norm = net(img)
                feat = feat.numpy()
                feat = normalize(feat, axis=1, norm='l2')
                dict_feat[img_name] = feat
        except Exception as e:
            print(e)
    with open('./save_feat/feature_adaface_ir101_webface12m_test.pickle', 'wb') as handle:
        pickle.dump(dict_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done save feature")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch AdaFace Training')
    parser.add_argument('--network', type=str, default='ir_101', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/adaface_ir101_webface12m.ckpt')
    parser.add_argument('--img_dir', type=str, default='/home/hunght21/data/tinyface/Testing_Set/Probe')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img_dir)
    