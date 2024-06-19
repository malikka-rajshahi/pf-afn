import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Helper_Model(nn.Module):
  def __init__(self,g_model,w_model):
         super(Helper_Model,self).__init__()
         self.warp_model= w_model
         self.gen_model= g_model
         self.grid_sample= F.grid_sample  
            
  def forward(self,real_image,clothes,edge ):

        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes=clothes
        clothes = clothes * edge        

        flow_out = self.warp_model(real_image, clothes)
        warped_cloth, last_flow, = flow_out
        warped_edge = self.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered=p_rendered
        m_composite=m_composite
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        warped_edge=warped_edge
        m_composite = m_composite * warped_edge
        warped_cloth=warped_cloth
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite) 
        a = real_image.float()
        b= clothes
        c = p_tryon
        combine = c[0].squeeze()
        cv_img=(combine.permute(1,2,0).detach().numpy()+1)/2     
        return cv_img