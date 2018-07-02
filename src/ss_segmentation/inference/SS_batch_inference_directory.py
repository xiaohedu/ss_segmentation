# -----------------------------------------------------------------------
# 	       Code to produce Segmentation Output in PyTorch
#             Originall written by Eduardo Romera (Sept 2017)
# 		Modified and Adapted by Shreyas Shivakumar
# -----------------------------------------------------------------------

import numpy as np
import torch
import os
import importlib
import pdb
import time

import sys
sys.path.append('../')

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from skimage.color import label2rgb
from skimage.io import imsave

from config.SS_config_batch_inference import *
from SS_data_definition_inference import SemanticSegmentationInference
from train.SS_network_design import Net

class InferenceTransform(object):
    def __init__(self, enc, augment=True, height=IMG_HEIGHT):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input):
        input =  Resize(self.height, Image.BILINEAR)(input)
        input = ToTensor()(input)
        return input

def main():
    modelpath = ARGS_LOAD_DIR + ARGS_LOAD_MODEL
    weightspath = ARGS_LOAD_DIR + ARGS_LOAD_WEIGHTS
    print ("---------- DATA PATHS: ----------")
    print ("Model File: " + modelpath)
    print ("Weight File: " + weightspath)
    model = Net(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not ARGS_CPU):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights loaded..")
    print ("---------------------------------")

    model.eval()

    if(not os.path.exists(ARGS_INFERENCE_DIR)):
        print ("Problem finding Inference Directory. Check path and try again.")

    co_transform = InferenceTransform(False, augment=False, height=IMG_HEIGHT)
    dataset = SemanticSegmentationInference(ARGS_INFERENCE_DIR, co_transform)
    loader = DataLoader(dataset, num_workers=ARGS_NUM_WORKERS, batch_size=ARGS_BATCH_SIZE, shuffle=False)

    inf_ctr = 0;
    start_time = time.time()
    for step, (images) in enumerate(loader):
        if (not ARGS_CPU):
            images = images.cuda()
        inputs = Variable(images, volatile=True)
        outputs = model(inputs)
        label = outputs[0].max(0)[1].byte().cpu().data
        label_color = label.unsqueeze(0)
        filenameSave = ARGS_SAVE_DIR + "inference_folder/" + "inference_" + str(inf_ctr).zfill(6) + ".png"
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

        if (ARGS_SAVE_COLOR == 1):
            numpy_labelcolour = label_color.numpy()
            rgb_label = label2rgb(numpy_labelcolour, colors=['blue','red'])
            imsave(filenameSave, rgb_label[0,:,:,:])
        else:
            label_save = ToPILImage()(label_color)
            label_save.save(filenameSave)
        inf_ctr = inf_ctr + 1
        print (step, filenameSave)

    time_val = time.time() - start_time
    print ("Total time taken : [{ttime} seconds] | Inference time per image : [{atime} ms]".format(ttime=time_val, atime=(time_val*1000)/inf_ctr))

if __name__ == '__main__':
    main()



