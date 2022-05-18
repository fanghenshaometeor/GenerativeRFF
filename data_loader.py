import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import scipy.io as scio

import os

# -------- get data sets --------
def get_dataloader(args):
    if args.dataset=='monks1' or args.dataset=='monks2' or args.dataset=='monks3' \
        or args.dataset=='adult':

        if args.dataset=='monks1' or args.dataset=='monks2' or args.dataset=='monks3':
            args.num_dim=6
            trainset = Monks(args.data_folder, train=True, 
                            transform=transforms.Compose([transforms.ToTensor()]))
            testset = Monks(args.data_folder, train=False,
                            transform=transforms.Compose([transforms.ToTensor()]))
        elif args.dataset=='adult':
            args.num_dim=123
            trainset = Adult(args.data_folder, train=True, 
                            transform=transforms.Compose([transforms.ToTensor()]))
            testset = Adult(args.data_folder, train=False,
                            transform=transforms.Compose([transforms.ToTensor()]))
        if args.val:
            args.num_test = len(testset)
            args.num_val = round(len(trainset)*args.ratio_val)
            args.num_train = len(trainset)-args.num_val

            train_val = data.random_split(trainset, (args.num_train, args.num_val))
            trainset, valset = train_val[0], train_val[1]
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        else:
            args.num_val = 0
            args.num_test = len(testset)
            args.num_train = len(trainset)

            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            valloader = None
    elif args.dataset == 'australian' or args.dataset == 'climate' \
        or args.dataset == 'diabetic' or args.dataset == 'sonar' or args.dataset == 'phishing':
        
        if args.dataset == 'australian':
            args.num_dim=14
            alldataset = Australian(args.data_folder, transform=transforms.Compose([transforms.ToTensor()]))
        elif args.dataset == 'climate':
            args.num_dim=18
            alldataset = Climate(args.data_folder)
        elif args.dataset == 'diabetic':
            args.num_dim=19
            alldataset = Diabetic(args.data_folder)
        elif args.dataset == 'sonar':
            args.num_dim=60
            alldataset = Sonar(args.data_folder)
        elif args.dataset == 'phishing':
            args.num_dim=68
            alldataset = Phishing(args.data_folder)
        
        if args.val:
            args.num_test = round(len(alldataset)*args.ratio_test)
            args.num_val = round((len(alldataset)-args.num_test)*args.ratio_val)
            args.num_train = len(alldataset)-args.num_test-args.num_val

            train_test_val = data.random_split(alldataset, (args.num_train, args.num_test, args.num_val))
            trainset, testset, valset = train_test_val[0], train_test_val[1], train_test_val[2]
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        else:
            args.num_val = 0
            args.num_test = round(len(alldataset)*args.ratio_test)
            args.num_train = len(trainset)-args.num_test

            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
            valloader = None
    elif args.dataset == 'ijcnn':
        args.num_dim=22
        trainset = Ijcnn(args.data_folder, datatype='train', 
                        transform=transforms.Compose([transforms.ToTensor()]))
        testset = Ijcnn(args.data_folder, datatype='test',
                        transform=transforms.Compose([transforms.ToTensor()]))
        valset = Ijcnn(args.data_folder, datatype='val',
                        transform=transforms.Compose([transforms.ToTensor()]))
        args.num_val = len(valset)
        args.num_test = len(testset)
        args.num_train = len(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)
    return trainloader, testloader, valloader


"""
Load UCI-Monks1,2,3 Data Set 
ATTRIBUTE:
    6           dimensions
    432         instances total
    2           labels (0, 1)
    FLOAT32
"""
class Monks(data.Dataset):
    def __init__(self, data_folder, train=True, transform=None):
        if train == True:
            data_path = os.path.join(data_folder, 'train.dat')
        else:
            data_path = os.path.join(data_folder, 'test.dat')
        
        read_data = np.loadtxt(data_path, delimiter=" ", dtype=str, usecols=(1,2,3,4,5,6,7))
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        y[y==2] = 0

        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Australian Data Set
ATTRIBUTE:
    14          dimensions
    690         instances total
    2           labels (0, 1)
    FLOAT32
"""
class Australian(data.Dataset):
    def __init__(self, data_folder, transform=None):
        data_path = os.path.join(data_folder, 'australian.dat')
        read_data = np.loadtxt(data_path, delimiter=" ")
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
            
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Climate Data Set
ATTRIBUTE:
    18          dimensions
    540         instances total
    2           labels (0, 1)
    FLOAT32
"""
class Climate(data.Dataset):
    def __init__(self, data_folder, transform=None):
        data_path = os.path.join(data_folder, 'pop_failures.dat')
        read_data = np.loadtxt(data_path,delimiter=None,dtype=str,skiprows=1,usecols=range(2,21))

        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
            
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Diabetic Data Set
ATTRIBUTE:
    19          dimensions
    1151        instances total
    2           labels (0, 1)
    FLOAT32
"""
class Diabetic(data.Dataset):
    def __init__(self, data_folder, transform=None):
        data_path = os.path.join(data_folder, 'messidor_features1.arff')
        read_data = np.loadtxt(data_path, delimiter=",")
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
            
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Sonar Data Set
ATTRIBUTE:
    60          dimensions
    208         instances total
    2           labels (0, 1)
    FLOAT32
"""
class Sonar(data.Dataset):
    def __init__(self, data_folder, transform=None):
        data_path = os.path.join(data_folder, 'sonar.dat')
        read_data = np.loadtxt(data_path, delimiter=",", dtype=str)
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y[y=='R']=1
        y[y=='M']=0
        y = y.astype(np.int64)
        
        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
            
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Adult Data Set 
ATTRIBUTE:
    123         binary dimensions
    48842       instances total
    2           labels (0, 1)
    FLOAT32
"""
class Adult(data.Dataset):
    def __init__(self, data_folder, train=True, transform=None):
        if train == True:
            data_path = os.path.join(data_folder, 'train.mat')
            read_data = scio.loadmat(data_path)['train_data']
        else:
            data_path = os.path.join(data_folder, 'test.mat')
            read_data = scio.loadmat(data_path)['test_data']
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        y[y!=1] = 0

        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Ijcnn Data Set 
ATTRIBUTE:
    22          dimensions
    48842       instances total
    2           labels (0, 1)
    FLOAT32
"""
class Ijcnn(data.Dataset):
    def __init__(self, data_folder, datatype='train', transform=None):
        if datatype == 'train':
            data_path = os.path.join(data_folder, 'train.mat')
            read_data = scio.loadmat(data_path)['train_data']
        elif datatype == 'test':
            data_path = os.path.join(data_folder, 'test.mat')
            read_data = scio.loadmat(data_path)['test_data']
        elif datatype == 'val':
            data_path = os.path.join(data_folder, 'val.mat')
            read_data = scio.loadmat(data_path)['val_data']
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        y[y!=1] = 0

        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]

"""
Load UCI-Phishing Data Set 
ATTRIBUTE:
    68          dimensions
    11055       instances total
    2           labels (0, 1)
    FLOAT32
"""
class Phishing(data.Dataset):
    def __init__(self, data_folder,  transform=None):
        data_path = os.path.join(data_folder, 'phishing.mat')
        read_data = scio.loadmat(data_path)['data']
        y = np.array(read_data[:,-1])
        X = np.array(read_data[:,0:-1])
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        y[y!=1] = 0
        
        # normalize
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].min()) / (X[:,i].max()-X[:,i].min())
            
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
    def __len__(self):
        return self.X.shape[0]