import enum
from sched import scheduler
import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse

from model import net_g_shallow
from model import net_g_deep
from model import GRFFNet
from utils import AverageMeter, accuracy, setup_seed
from data_loader import get_dataloader

# -------- fix data type ----------------
torch.set_default_tensor_type(torch.FloatTensor)

# ======== parameter settings =======
parser = argparse.ArgumentParser(description='Kernel Learning via multi-layer GRFF')
# -------- file param. --------------
parser.add_argument('--data_folder',type=str,help='data folder')
parser.add_argument('--dataset',type=str,help='data set name')
# -------- exp. settings ------------
parser.add_argument('--batch_size',type=int,default=512,help='batch-size')
parser.add_argument('--num_train',type=int,default=0,help='number of training set')
parser.add_argument('--num_test',type=int,default=0,help='number of test data')
parser.add_argument('--num_val',type=int,default=0,help='number of validation set')
parser.add_argument('--num_dim',type=int,default=0,help='number of dimension')
parser.add_argument('--num_classes',type=int,default=2,help='number of classes')
# -------- exp. settings ------------
parser.add_argument('--val',type=int,help='enable the adversarial training')
parser.add_argument('--ratio_val',type=float,default=0.2,help='ratio of validation')
parser.add_argument('--ratio_test',type=float,default=0.5,help='ratio of test')
# -------- exp. settings ------------
parser.add_argument('--num_repeat',type=int,default=5,help='number of repeat')
# -------- model settings -----------
parser.add_argument('--num_layers',type=int,default=2,help='number of layers in GRFF')
parser.add_argument('--D',nargs='+',type=int,default=[256, 64])
parser.add_argument('--epochs',nargs='+',type=int,default=[1000,5000])
args = parser.parse_args()

# -------- main function
def main():
    setup_seed(666)

    # -------- Repeating 5 Times
    for repeat_idx in range(args.num_repeat):

        print("======== ========")
        print("---- Repeat %d/%d..."%(repeat_idx+1,args.num_repeat))
        # -------- Preparing data sets
        args.val=1
        trainloader, testloader, valloader = get_dataloader(args)
        print("-------- --------")
        print("---- Data information:")
        print("---- data set: ", args.dataset)
        print("---- # dimension = %d."%args.num_dim)
        print("---- # train/test/val. = %d/%d/%d"%(args.num_train, args.num_test, args.num_val))

        # -------- Preparing model
        assert args.num_layers==len(args.D), "Mismatch between the number of noises and the number of layers."
        assert args.num_layers==len(args.epochs), "Mismatch between the number of layers and the number of training phases."
        for layer_idx in range(args.num_layers):
            if layer_idx == 0:
                net = GRFFNet(GRFFBlock=net_g_shallow, num_classes=args.num_classes, num_layers=1,d=args.num_dim,D=args.D)
            else:
                net._add_layer(GRFFBlock=net_g_shallow, layer_index=layer_idx)
        net = net.cuda()
        print("-------- --------")
        print("---- Net information:")
        print(net)

        # -------- Progressively Training Freeze Inverse
        print("---- Start Progressively Training...")
        prog_train(net)
        # -------- Perform Training on the whole training set
        args.val=0
        trainloader, testloader, _ = get_dataloader(args)
        print("-------- --------")
        print("---- Data information:")
        print("---- data set: ", args.dataset)
        print("---- # dimension = %d."%args.num_dim)
        print("---- # train/test/val. = %d/%d/%d"%(args.num_train, args.num_test, args.num_val))

    return

# -------- progressively training Freeze & Inverse
def prog_train(net, trainloader):
    for layer_idx in range(args.num_layers):
        print('----------------')
        print('Training phase %d/%d...' % (layer_idx+1, args.num_layers))
        opt_params, opt_params_idx = get_opt_params(layer_idx, net)
        print('To-be-optimized params: ', opt_params_idx)

        # -------- preparing optimizer and scheduler
        optimizer = optim.Adam(opt_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.sche_step], gamma=args.sche_gamma)

        for epoch in range(args.epochs[layer_idx]):
            prog_train_epoch(net, trainloader, optimizer, epoch)
            scheduler.step()


    return

def get_opt_params(layer_idx, net):
    if layer_idx == (args.num_layers-1):
        opt_params = net.parameters()
    else:
        opt_params = []
        for idx in range(layer_idx+1):
            opt_params.append({'params':net.GRFF[args.num_layers-idx-1].generator.parameters()})
        opt_params.append({'params':net.fc.parameters()})
    opt_params_idx = []
    for idx in range(layer_idx+1):
        opt_params_idx.append(args.num_layers-idx-1)
    return opt_params, opt_params_idx

def prog_train_epoch(net, trainloader, optimizer, epoch):

    net.train()
    losses = AverageMeter()
    
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()
        b_size = b_data.size(0)

        # -------- generate random noise
        noise = []
        for _, value in enumerate(args.D):
            noise.append(torch.randn(value, 100).cuda())
        
        # -------- forward
        logits = net.forward(b_data.view(b_size, -1), noise)
        loss = F.cross_entropy(logits, b_label)
        losses.update(loss.float().item(), b_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return

def val(net, dataloader, epoch):

    net.eval()
    acc = AverageMeter()

    with torch.no_grad():
        for batch_idx, (X,y) in enumerate(dataloader):
            X,y = X.cuda(), y.cuda()
            b_size = X.size(0)

            noise = []
            for _, value in enumerate(args.D):
                noise.append(torch.randn(value, 100).cuda())
            logits = net.forward(X.view(b_size,-1), noise)
            prec1 = accuracy(logits.data, y)[0]
            acc.update(prec1.item(), b_size)
    return acc.avg

# ======== startpoint
if __name__ == '__main__':
    main()