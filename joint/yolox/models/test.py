import torch
x=torch.zeros(4,5,480,640)

patch_top_left = x[...,::2, ::2,...]    
patch_top_right = x[...,::2, 1::2,...]      
patch_bot_left = x[...,1::2, ::2,...]        
patch_bot_right = x[...,1::2, 1::2,...]       
print(patch_top_left.shape)        
print(patch_top_right.shape)      
print(patch_bot_left.shape)     
print(patch_bot_right.shape)
