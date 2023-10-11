# import _init_paths
import torch
import torch.nn as nn
# import pose.models as models
from .models.hourglass import *
from .model.modulated_gcn import ModulatedGCN


def soft_argmax(voxels):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W,D = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W,D))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	z = indices%D
	y = (indices/D).floor()%W
	x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([y,x,z],dim=2)
	return coords[:,:,:2]/64.0

# def get_state_dict():
#     state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/human36_gaus_1/model_2.pth') #16kps checkpoint pre trained on mpii
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#     # import pdb;pdb.set_trace()
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     return new_state_dict

class MyNet(nn.Module):
    def __init__(self, adj,num_stacks, block) -> None:
        super(MyNet, self).__init__()
        self.sthg = hg(num_stacks=num_stacks, num_blocks=1, num_classes=16)
        # self.sthg.load_state_dict(get_state_dict())
        self.MGCN = ModulatedGCN(adj, 384, num_layers=block, p_dropout=0, nodes_group=None)
        # self.MGCN.load_state_dict(torch.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results_2layers/model_module_gcn_20_eva_xyz_5236.pth'))

    def forward(self, x ,left_top=[0, 0], ratio_x=1.0, ratio_y=1.0):
    # def forward(self, x ): # for onnx
        left_top=[0, 0] 
        ratio_x=1.0
        ratio_y=1.0
        N = x.shape[0]
        heatmaps = self.sthg(x)
        output = heatmaps[-1].unsqueeze(4)
        key_points = soft_argmax(output)
        # import pdb;pdb.set_trace()
        for i in range(N):
            key_points[i,:,0] /= ratio_x
            key_points[i,:,1] /= ratio_y
        for i in range(N):
            key_points[i,:,0] += left_top[0]
            key_points[i,:,1] += left_top[1]
        for i in range(N):
            key_points[i,:,0] *= ratio_x
            key_points[i,:,1] *= ratio_y

        output = output.view(N, 16, -1)
        # import pdb;pdb.set_trace()
        key_points = key_points.view(N, -1, 16, 2, 1).permute(0, 3, 1, 2, 4)
        out_3d = self.MGCN(key_points)

        # return out_3d, heatmaps
        return out_3d, heatmaps[-1] # for mobile 