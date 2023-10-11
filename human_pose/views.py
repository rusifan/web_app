import base64
import cv2
from django.shortcuts import render

# Create your views here.
import matplotlib.pyplot as plt
import html
from pathlib import Path
import os
from collections import OrderedDict

import urllib.parse 
from .infer_mynet import MyNet
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
from .forms import ImageUploadForm
from django.shortcuts import render

#skeleton
skeleton = [
[0, 1], [1, 2], [2, 3],
[0, 4], [4, 5], [5, 6],
[0, 7], [7, 8], [8, 9],
[7, 10], [10, 11], [11, 12],
[7, 13], [13, 14], [14, 15]
] 

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x
#create the model
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
adj = np.load('human_pose/model_values/adj_4_16.npy')
# human_pose\model_values\adj_4_16.npy
adj = torch.from_numpy(adj)
model = MyNet(adj,num_stacks =2, block=2)
state_dict_path = "human_pose/model_values/model_5.pth"
new_state_dict = OrderedDict()
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

def transform_image(image_bytes):

    # with opencv
    image_np_array = np.frombuffer(image_bytes, np.uint8)
    # Decode the image using OpenCV
    img = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) /255
    img = torch.from_numpy(img).float()
    img = color_normalize(img, DATASET_MEANS, DATASET_STDS)

    return img.unsqueeze(0)

def predict(image_bytes):
    image_bytes = transform_image(image_bytes)
    out_3d, heatmap = model(image_bytes)
    out_3d = out_3d.squeeze()
    out_3d = out_3d.permute(1, 0)
    out_3d = out_3d.detach().cpu().numpy()
    out_3d [1:] += out_3d [:1]
    return out_3d

def draw_keypoints(keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
    for connection in skeleton:
        x = [keypoints[connection[0], 0], keypoints[connection[1], 0]]
        y = [keypoints[connection[0], 1], keypoints[connection[1], 1]]
        z = [keypoints[connection[0], 2], keypoints[connection[1], 2]]
        ax.plot(x, y, z, c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=-79, azim=-90)

    #return the image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  'data:image/png;base64,' + urllib.parse.quote(string)
    return uri

def index(request):
    
    image_uri = None
    out_3d = None
    plot_image = None
    if request.method == 'POST':
    # in case of POST: get the uploaded image from the form and process it
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label with previously implemented PyTorch function
            try:
                out_3d = predict(image_bytes)
                plot_image = draw_keypoints(out_3d)
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading images
        form = ImageUploadForm()
    
    context = {'form': form, 'image_uri': image_uri, 'out_3d': out_3d, 'plot_image': plot_image}
    return render(request, 'human_pose/index.html', context)