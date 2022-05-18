import torch

import argparse


from utils import setup_seed
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
# -------- exp. settings ------------
parser.add_argument('--val',type=int,help='enable the adversarial training')
parser.add_argument('--ratio_val',type=float,default=0.2,help='ratio of validation')
parser.add_argument('--ratio_test',type=float,default=0.5,help='ratio of test')
# -------- exp. settings ------------
parser.add_argument('--num_repeat',type=int,default=5,help='number of repeat')
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
    
        # -------- Perform Training on the whole training set
        args.val=0
        trainloader, testloader, _ = get_dataloader(args)
        print("-------- --------")
        print("---- Data information:")
        print("---- data set: ", args.dataset)
        print("---- # dimension = %d."%args.num_dim)
        print("---- # train/test/val. = %d/%d/%d"%(args.num_train, args.num_test, args.num_val))

    return

# ======== startpoint
if __name__ == '__main__':
    main()