import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import argparse

import numpy as np

from modelv import netGv
from modelv import GRFFNetv
from utils import AverageMeter, accuracy
from utils import setup_seed
from utils import Logger
from data_loader import get_dataloader


# -------- fix data type ----------------
torch.set_default_tensor_type(torch.FloatTensor)

# ======== parameter settings =======
parser = argparse.ArgumentParser(description='Kernel Learning via multi-layer GRFF')
# -------- file param. --------------
parser.add_argument('--data_folder',type=str,help='data folder')
parser.add_argument('--dataset',type=str,help='data set name')
parser.add_argument('--log_dir',type=str,default='./runs/',help='tensorboard')
parser.add_argument('--model_dir',type=str,default='./save/',help='model save folder')
parser.add_argument('--output_dir',type=str,default='./output/',help='terminal output')
# -------- exp. settings ------------
parser.add_argument('--batch_size',type=int,default=512,help='batch-size')
parser.add_argument('--num_train',type=int,default=-1,help='number of training set')
parser.add_argument('--num_test',type=int,default=-1,help='number of test data')
parser.add_argument('--num_val',type=int,default=-1,help='number of validation set')
parser.add_argument('--num_dim',type=int,default=-1,help='number of dimension')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
# -------- exp. settings ------------
parser.add_argument('--val',type=int,help='enable the adversarial training')
parser.add_argument('--ratio_val',type=float,default=0.2,help='ratio of validation')
parser.add_argument('--ratio_test',type=float,default=0.5,help='ratio of test')
# -------- model settings -----------
parser.add_argument('--num_layers',type=int,default=2,help='number of layers in GRFF')
parser.add_argument('--D',nargs='+',type=int,default=[256,64])
args = parser.parse_args()

args.data_folder='/home/dev/fangkun/data/MNIST/'
if not os.path.exists(os.path.join(args.model_dir,args.dataset)):
    os.makedirs(os.path.join(args.model_dir,args.dataset))
args.save_path=os.path.join(args.model_dir,args.dataset)
if not os.path.exists(os.path.join(args.output_dir,args.dataset)):
    os.makedirs(os.path.join(args.output_dir,args.dataset))
args.output_path=os.path.join(args.output_dir,args.dataset,'attack.log')
sys.stdout = Logger(filename=args.output_path,stream=sys.stdout)

# -------- main function
def main():
    setup_seed(666)

    # -------- Evaluation on the whole training set
    args.val=0
    args.batch_size=1
    trainloader, testloader, _ = get_dataloader(args)
    print("-------- --------")
    print("---- data set: ", args.dataset)
    print("---- # train/test/val. = %d/%d/%d"%(args.num_train, args.num_test, args.num_val))
    
    # -------- Preparing model
    net = GRFFNetv(netGv) 
    net = net.cuda()    
    ckpt = torch.load(os.path.join(args.save_path,"best-0.pth"),map_location=torch.device("cpu"))
    net.load_state_dict(ckpt['state_dict'])
    print("-------- --------")
    print(net)

    # -------- Evaluate on clean
    print("---- Evaluating on clean...")
    acctr=val(net, trainloader, None)
    accte=val(net, testloader, None)
    print("---- CLEAN: Train/Test acc.=%.2f/%.2f"%(acctr,accte))

    # -------- Perform Iter.L.L. attack
    print("---- Iter.L.L. attacking...")
    ill_eps = [0,2,4,6,8,10,12]
    for eps in ill_eps:
        print("---- Current eps = %d"%eps)
        acc_original, acc_resample = attack(net, testloader, eps, alpha=1, iters=0)
        print("---- Original/Resample acc.=%.2f/%.2f"%(acc_original, acc_resample))
        

def val(net, dataloader, epoch):

    net.eval()
    acc = AverageMeter()

    with torch.no_grad():
        for batch_idx, (X,y) in enumerate(dataloader):
            X,y = X.cuda(), y.cuda()
            b_size = X.size(0)

            noise = []
            for _, value in enumerate(args.D):
                noise.append(torch.randn(value, 100, 1, 1).cuda())
            logits, _, _ = net.forward(X, noise)
            prec1 = accuracy(logits.data, y)[0]
            acc.update(prec1.item(), b_size)
    return acc.avg

# Iteratively least-likely class method
def ill_attack(model, image, target, noise_ori, epsilon, alpha=1, iters=0):
       
    # forward passing the image through the model one time to get the leastly likely labels
    output, _, _ = model(image, noise_ori)
    ll_label = torch.min(output, 1)[1] # get the index of the min log-probability
    
    if iters == 0:
        iters = int(min(epsilon+4, 1.25*epsilon))
    
    epsilon = epsilon / 255
    
    for i in range(iters):
        
        # set requires_grad attribute of tensor
        image.requires_grad = True
        
        # forward pass the data
        output, _, _ = model(image, noise_ori)
        init_pred = output.max(1, keepdim=True)[1]
        
        # if the current prediction is wrong, dont bother to continue
        if init_pred.item() != target.item():
            return image
        
        # caculate the loss
        loss = F.cross_entropy(output, ll_label)
        model.zero_grad()
        loss.backward()
        
        # collect data gradient
        data_grad = image.grad.data
        
        # collect the element-wise sign of data gradient
        sign_data_grad = data_grad.sign()
        # create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image - alpha*sign_data_grad
        
        # update the image for next iteration
        a = torch.clamp(image - epsilon, min=0)  
        b = (perturbed_image>=a).float()*perturbed_image + (a>perturbed_image).float()*a
        c = (b > image+epsilon).float()*(image+epsilon) + (image+epsilon >= b).float()*b
        image = torch.clamp(c, max=1).detach_()

    return image

# attack
def attack(net, testloader, eps, alpha=1, iters=0):
    
    net.eval()
    acc_resample = AverageMeter()
    acc_original = AverageMeter()

    for batch_idx, (image, label) in enumerate(testloader):
        image, label = image.cuda(), label.cuda()

        # generate noise
        noise_original = []
        for _, value in enumerate(args.D):
            noise_original.append(torch.randn(value, 100, 1, 1).cuda())

        # evaluate clean
        output, _, _ = net(image, noise_original)
        init_pred = output.max(1, keepdim=True)[1]
        
        # obtain the perturbed sample
        perturbed_image = ill_attack(net, image, label, noise_original, eps, alpha, iters)

        # re-classify the perturbed images
        noise_resample = []
        for _, value in enumerate(args.D):
            noise_resample.append(torch.randn(value, 100, 1, 1).cuda())
        logits_resample, _, _ = net(perturbed_image, noise_resample)
        logits_original, _, _ = net(perturbed_image, noise_original)
        prec_resample = accuracy(logits_resample.data, label)[0]
        acc_resample.update(prec_resample.item(), image.size(0))
        prec_original = accuracy(logits_original.data, label)[0]
        acc_original.update(prec_original.item(), image.size(0))
    
    return acc_original.avg, acc_resample.avg


# ======== startpoint
if __name__ == '__main__':
    main()