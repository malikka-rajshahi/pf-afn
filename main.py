# from firebase_helper import FirebaseHelper
from os import stat
# from flask import Flask,request,send_file
from models.helper import Helper_Model
from models.networks import load_checkpoint
from models.networks import ResUnetGenerator
from models.afwm import AFWM
from data.cloth_edge import GenerateEdge
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
        # self.gen_model = torch.load('pt_models/generator_new.pt')
        # self.warp_model = torch.load('pt_models/warp_pure_corr_model.pt')
        
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        load_checkpoint(self.gen_model, 'checkpoints/PFAFN/gen_model_final.pth')

        self.warp_model = AFWM(3)
        load_checkpoint(self.warp_model, 'checkpoints/PFAFN/warp_model_final.pth')

        self.model = Helper_Model(self.gen_model, self.warp_model)
        self.model.eval()
        # self.firebase = FirebaseHelper()

    def send_here(self,image,clothImage,edgeImage,cloth_blob=None,edge_blob=None):

        #payload = request.form.to_dict(flat=False)
        #im_b64 = request['real_image']
        #im_binary = base64.b64decode(im_b64)
        #im_binary=im_binary.decode("utf-8")
        #buf = io.BytesIO(im_binary)
        #img = Image.open(buf)

        try:
            os.mkdir('dataset/test_clothes')
            os.mkdir('dataset/test_img')
            os.mkdir('dataset/test_edge')
        except:
            pass


        image.save('/tmp/test_img/clothes_1.jpg')
        clothImage.save('/tmp/test_clothes/clothes_test_1.jpg')
        edgeImage.save('/tmp/test_edge/clothes_test_1.jpg')
        print(os.listdir('/tmp'))
        print(os.listdir('/tmp/test_img'))

        #cloth_blob = payload['cloth_image'][0]
        #edge_blob = payload['edge_image'][0]
        # self.firebase.getImageromGSUrl(blobPath=cloth_blob)
        # self.firebase.getImageromGSUrl(blobPath=edge_blob,imgType=1)
        print(os.listdir('/tmp/test_clothes'))
        print(os.listdir('/tmp/test_edge'))
        opt = TestOptions().parse()
        start_epoch, epoch_iter = 1, 0
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        total_steps = (start_epoch-1) * dataset_size + epoch_iter   

        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image']
            # print(real_image)
            clothes = data['clothes']
            # print(clothes)
            edge = data['edge']
            # print(edge)

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
    
