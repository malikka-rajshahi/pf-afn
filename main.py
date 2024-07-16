# from firebase_helper import FirebaseHelper
from os import stat
# from flask import Flask,request,send_file
from models.helper import Helper_Model
from models.networks import load_checkpoint
from models.networks import ResUnetGenerator
from models.afwm import AFWM
from data.cloth_edge import GenerateEdge
from data.aligned_dataset_test import AlignedDataset
from data.base_dataset import BaseDataset, get_params, get_transform
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
import base64
import io
import os
import json


class Tryon:
    def __init__(self):
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        load_checkpoint(self.gen_model, 'checkpoints/PFAFN/gen_model_final.pth')

        self.warp_model = AFWM(3)
        load_checkpoint(self.warp_model, 'checkpoints/PFAFN/warp_model_final.pth')

        self.model = Helper_Model(self.gen_model, self.warp_model)
        self.model.eval()

    def send_here(self,image,clothImage,edgeImage,cloth_blob=None,edge_blob=None):
        opt = TestOptions().parse()

        I = image.convert('RGB')
        params = get_params(opt, I.size)
        transform = get_transform(opt, params)
        transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

        print('transforming I')
        I_tensor = transform(I)
        I_tensor = I_tensor.unsqueeze(0)
        print(f"I Shape: {I_tensor.shape}")

        C = clothImage.convert('RGB')
        print('transforming C')
        C_tensor = transform(C)
        C_tensor = C_tensor.unsqueeze(0)
        print(f"C Shape: {C_tensor.shape}")

        E = edgeImage.convert('L')
        print('transforming E')
        E_tensor = transform_E(E)
        E_tensor = E_tensor.unsqueeze(0)
        print(f"E Shape: {E_tensor.shape}")
        print('all transforms done')

        data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor} 
        print('defined data')

        real_image = data['image']
        clothes = data['clothes']
        edge = data['edge']

        p_tryon = self.model(real_image, clothes, edge)
    
        cv_img = p_tryon
        rgb = (cv_img*255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('dataset/out.jpg', bgr)
        with open(r'dataset/out.jpg', 'rb') as f:
            im_b64 = base64.b64encode(f.read())

        return im_b64


    def base(self):
        return Tryon.app_health(message="tryon model accessible at `send_here` method")


    def health(self):
        return Tryon.app_health()



def get_img_path(img_name):
    extensions = ['jpg', 'jpeg', 'png']
    img_directory = 'dataset/test_img'
    for ext in extensions:
        img_path = os.path.join(img_directory, f"{img_name}.{ext}")
        if os.path.exists(img_path): return img_path


def main():
    application = Tryon()
    inp_path = get_img_path('in_2')

    # set input image
    try:
        input_img = Image.open(inp_path)
        print(f"Input image opened: {inp_path}")
    except IOError:
        print("Error in opening input image")

    # set cloth image
    cloth_path = 'dataset/test_clothes/00067_00.jpg'
    try:
        cloth_img = Image.open(cloth_path)
        print(f"Cloth image opened: {cloth_path}")
    except IOError:
        print("Error in opening cloth image")

    # produce cloth edge image
    edge_path = 'dataset/test_edge/cloth_edge.jpg'
    edge_gen = GenerateEdge(cloth_path)
    edge_gen.process_images(edge_path)
    try:
        edge_img = Image.open(edge_path)
        print(f"Edge image opened: {edge_path}")
    except IOError:
        print("Error in opening edge image")

    # set cloth edge image 
    # edge_path = 'dataset/test_edge/clothes_test_1.jpg'
    # try:
    #     edge_img = Image.open(edge_path)
    #     print(f"Edge image opened: {edge_path}")
    # except IOError:
    #     print("Error in opening edge image")

    # run model
    application.send_here(input_img, cloth_img, edge_img)

if __name__=="__main__":
    main()
    # application = Main()
    
