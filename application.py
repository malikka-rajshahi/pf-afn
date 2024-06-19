
from flask import Flask,request,send_file
from models.helper import Helper_Model
import cv2
import numpy as np
import torch
from PIL import Image
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
import json

application = app= Flask(__name__)

gen_model = torch.load('pt_models/generator_new.pt')
warp_model = torch.load('pt_models/warp_pure_corr_model.pt')
model = Helper_Model(gen_model,warp_model)    

@app.route('/send_here',methods=['GET','POST'])
def send_here():
    
    # real_image_file = Image.open(request.files['real_image'].stream)
    # clothes_file = Image.open(request.files['clothes'].stream)
    # edge_file = Image.open(request.files['edge'].stream)

    # with open("dataset/test_img/clothes_1.jpg", 'w') as f:
    #     real_image_file.save("dataset/test_img/clothes_1.jpg")
    # with open("dataset/test_clothes/clothes_test_1.jpg", 'w') as f:
    #     clothes_file.save("dataset/test_clothes/clothes_test_1.jpg")
    # with open("dataset/test_edge/clothes_test_1.jpg", 'w') as f:
    #     edge_file.save("dataset/test_edge/clothes_test_1.jpg")
    
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
        clothes = data['clothes']
        edge = data['edge']

        p_tryon=model(real_image,clothes,edge)
        cv_img=p_tryon
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite('out_new_21_12_8.jpg',bgr)
        return send_file('out.jpg',as_attachment=True)


def app_health(message="health OK"):
    reply = {
        "message": message,
        "code": 200
    }

    return json.dumps(reply,indent=4)


@app.route('/',methods=['GET'])
def base():
    return app_health()


@app.route('/health',methods=['GET'])
def health():
    return app_health()


if __name__=="__main__":
    application.run(host="0.0.0.0",port=9000)