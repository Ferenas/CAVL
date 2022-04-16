import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.special import comb
from utils.gen_index_dataset import gen_index_dataset
import torch.nn.functional as F

def generate_uniform_cv_candidate_labels(dataname, train_labels):
    # firstly choose a candidate number T for each instance, then make a random list
    # [5,4,3,2,1,9,7,8] for example (using the class number), and select T-1 labels in them as the candidates.
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
        
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2**K - 2).float()
    number = torch.tensor([comb(K, i+1) for i in range(K-1)]).float() # 1 to K-1 because cannot be empty or full label set, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K-1) # tensor of K-1
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float() # tensor: n
    mask_n = torch.ones(n) # n is the number of train_data
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    
    temp_num_partial_train_labels = 0 # save temp number of partial train_labels
    
    for j in range(n): # for each instance
        for jj in range(K-1): # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj+1 # decide the number of partial train_labels
                mask_n[j] = 0
                
        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(np.random.permutation(K.item())).long() # because K is tensor type
        candidates = candidates[candidates!=train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
        
        partialY[j, temp_fp_train_labels] = 1.0 # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def prepare_cv_datasets(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='./data/KMNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.KMNIST(root='./data/KMNIST', train=False, transform=transforms.ToTensor())
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(root='./data/FashionMnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.FashionMNIST(root='./data/FashionMnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(), # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ordinary_train_dataset = dsets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True, num_workers=0)
    num_classes = 10
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes

def prepare_train_loaders_for_uniform_cv_candidate_labels(dataname, full_train_loader, batch_size):
    for i, (data, labels) in enumerate(full_train_loader):
        K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    partialY = generate_uniform_cv_candidate_labels(data, labels)
    partial_matrix_dataset = gen_index_dataset(data, partialY.float(), labels.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dim = int(data.reshape(-1).shape[0]/data.shape[0])
    return partial_matrix_train_loader, data, partialY, dim