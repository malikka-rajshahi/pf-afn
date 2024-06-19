# torch version == 1.1.0 torchvision == 0.3.0
#requires GPU

import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F


class Helper_Model(nn.Module):
  def __init__(self,):
         super(Helper_Model,self).__init__()
         self.warp_model= warp_model.cuda()
         self.gen_model= gen_model.cuda()
         self.grid_sample= F.grid_sample           
  def forward(self,real_image,clothes,edge ):

        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int)).cuda()
        clothes=clothes.cuda()
        clothes = clothes * edge        

        flow_out = self.warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = self.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros').cuda()

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1).cuda()
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered=p_rendered.cuda()
        m_composite=m_composite.cuda()
        p_rendered = torch.tanh(p_rendered).cuda()
        m_composite = torch.sigmoid(m_composite).cuda()
        warped_edge=warped_edge.cuda()
        m_composite = m_composite * warped_edge
        warped_cloth=warped_cloth.cuda()
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite) 
        a = real_image.float().cuda()
        b= clothes.cuda()
        c = p_tryon.cuda()
         
        return p_tryon
opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)

gen_model = torch.load('pt_models/generator_new.pt')
warp_model = torch.load('pt_models/warp_model.pt')

model= Helper_Model()
total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

for epoch in range(1,2):

    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']

        p_tryon = model(real_image,clothes,edge)
        print(p_tryon.shape)
        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        sub_path = path + '/PFAFN'
        os.makedirs(sub_path,exist_ok=True)

        if step % 1 == 0:
            a = real_image.float().cuda()
            b= clothes.cuda()
            c = p_tryon
            combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

            cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)

        step += 1
        if epoch_iter >= dataset_size:
            break


