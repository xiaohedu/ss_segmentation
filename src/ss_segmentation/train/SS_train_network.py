###############################################################
# Main code for training ERFNet-based Semantic Segmentation CNN
# 			April 2018
#   Shreyas Skandan Shivakumar | University of Pennsylvania
# 		Adapted from Eduardo Romera
###############################################################

import os
import random
import time
import numpy as np
import torch
import math
import pdb
import importlib

import sys
sys.path.append('../')

from config.SS_config_train import *

from PIL import Image, ImageOps
from argparse import ArgumentParser
from shutil import copyfile

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from SS_data_definition import SemanticSegmentation
from utils.util_transform import ToLabel, Colorize
from utils.util_visualize import Dashboard
from utils.util_iouEVAL import iouEval, getColorEntry

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

class ImageTransform(object):
    def __init__(self, enc, height=IMG_HEIGHT):
        self.enc=enc
        self.height = height
        
    def __call__(self, input, target):
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/ENC_SCALEDOWN), Image.NEAREST)(target)
        target = ToLabel()(target)
        return input, target

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

def train(model, enc=False):
    if (enc == True):
        save_prefix = 'encoder'
    else:
        save_prefix = 'decoder'
    best_acc = 0
    
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = CLASS_0_WEIGHT	
    else:
        weight[0] = CLASS_0_WEIGHT
    weight[1] = CLASS_1_WEIGHT

    co_transform = ImageTransform(enc, height=IMG_HEIGHT) 
    co_transform_val = ImageTransform(enc, height=IMG_HEIGHT)
    dataset_train = SemanticSegmentation(ARGS_TRAIN_DIR, co_transform)
    dataset_val = SemanticSegmentation(ARGS_VAL_DIR, co_transform_val)

    loader = DataLoader(dataset_train, num_workers=ARGS_NUM_WORKERS, batch_size=ARGS_BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=ARGS_NUM_WORKERS, batch_size=ARGS_BATCH_SIZE, shuffle=False)

    if ARGS_CUDA:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
    savedir = ARGS_SAVE_DIR

    if (enc):
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        modeltxtpath = savedir + "/model.txt"    
    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), OPT_LEARNING_RATE_INIT, OPT_BETAS, eps=OPT_EPS_LOW, weight_decay=OPT_WEIGHT_DECAY)   

    start_epoch = 1
    if ARGS_RESUME:
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    lambda1 = lambda epoch: pow((1-((epoch-1)/ARGS_NUM_EPOCHS)),0.9)  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  

    for epoch in range(start_epoch, ARGS_NUM_EPOCHS + 1):
        print("--------------- [TRAINING] Epoch #", epoch, "---------------")
        scheduler.step(epoch) 
        epoch_loss = []
        time_train = []
        doIouTrain = ARGS_IOU_TRAIN
        doIouVal =  ARGS_IOU_VAL  

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("Learning rate: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()
            if ARGS_CUDA:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, only_encode=enc)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)      

            if ARGS_STEPS_LOSS > 0 and step % ARGS_STEPS_LOSS == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print("Loss: {average} (epoch: {epoch}, step: {step})// Avg time/img: {avgtime} s".format(average=average, epoch=epoch, step=step, avgtime=(sum(time_train) / len(time_train) / ARGS_BATCH_SIZE)))
            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print ("Average loss after epoch : {avgloss}".format(avgloss=average_epoch_loss_train))
 
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("IoU on training data after EPOCH: ", iouStr, "%")

        print("\n---------- [VALIDATING] Epoch #", epoch, "----------")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if ARGS_CUDA:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True) 
            targets = Variable(labels, volatile=True)
            outputs = model(inputs, only_encode=enc) 

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)
            
            if (doIouVal):
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if ARGS_STEPS_LOSS > 0 and step % ARGS_STEPS_LOSS == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print("Testing loss: {average} (epoch: {epoch}, step: {step}) // Avg time/img: {avgstep} s".format(average=average, epoch=epoch, step=step, avgstep=(sum(time_val) / len(time_val) / ARGS_BATCH_SIZE)))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_val_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("IoU on test data after epoch: ", iouStr, "%")

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        if (enc):
            filename = savedir + "/model_encoder-{epoch}.pth".format(epoch=epoch)
            filenamebest = savedir + "/model_encoder_best.pth"
        else:
            filename = savedir + "/model-{epoch}.pth".format(epoch=epoch)
            filenamebest = savedir + "/model_best.pth"
        if ARGS_EPOCHS_SAVE > 0 and step > 0 and step % ARGS_EPOCHS_SAVE == 0:
            torch.save(model.state_dict(), filename)
            print("Saving to: {filename} (epoch: {epoch})".format(filename=filename, epoch=epoch))
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print("Saving to: {filenamebest} (epoch: {epoch})".format(filenamebest=filenamebest, epoch=epoch))
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           
        print ('\n\n')   
    return(model)  

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model with best IoU Score..")
        torch.save(state, filenameBest)


def main():
    savedir = ARGS_SAVE_DIR
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    os.makedirs(savedir + '/plotdata')

    model_file = importlib.import_module(ARGS_MODEL)
    model = model_file.Net(NUM_CLASSES)
    copyfile(ARGS_MODEL + ".py", savedir + '/' + ARGS_MODEL + ".py")
    if ARGS_CUDA:
        model = torch.nn.DataParallel(model).cuda()
    if ARGS_STATE:
        def load_my_state_dict(model, state_dict):  
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model
        model = load_my_state_dict(model, torch.load(ARGS_STATE))

    if (not ARGS_DECODER):
        print("#################### ENCODER TRAINING ####################")
        model = train(model, True)
    print("#################### DECODER TRAINING ####################")
    if (not ARGS_STATE):
        if ARGS_PRETRAINED_ENCODER:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(ARGS_PRETRAINED_ENCODER)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not ARGS_CUDA):
                pretrainedEnc = pretrainedEnc.cpu()     
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)
        if ARGS_CUDA:
            model = torch.nn.DataParallel(model).cuda()
    model = train(model, False)   
    print("#################### TRAINING FINISHED ####################")

if __name__ == '__main__':
    main()
