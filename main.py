import argparse
from utils.models import mlp_model, linear_model, LeNet
import torch
from utils.utils_data import prepare_cv_datasets, prepare_train_loaders_for_uniform_cv_candidate_labels
from utils.utils_algo import accuracy_check, confidence_update,accuracy_check_train
from utils.utils_loss import rc_loss
from cifar_models import densenet, resnet, convnet
import numpy as np

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=64, type=int)
parser.add_argument('-ds', help='specify a dataset', default="mnist", type=str, required=False) # mnist, kmnist, fashion, cifar10
parser.add_argument('-mo', help='model name', default='mlp', choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet','convnet'], type=str, required=False)
parser.add_argument('-ep', help='number of epochs', type=int, default=250)
parser.add_argument('-wd', help='weight decay', default=1e-3, type=float)
parser.add_argument('-seed', help = 'Random seed', default=0, type=int, required=False)
parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)
parser.add_argument('-op', help = 'optimizer', default='adam', type=str, required=False)


args = parser.parse_args()
print(set_ep)
print("NOW is dataset: {} with model {} weight_decay {} learning rate {} batch_size {} op {}".format(args.ds,args.mo,args.wd,args.lr,args.bs,args.op,args.ca))

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
    partial_matrix_train_loader, train_data, train_givenY, dim = prepare_train_loaders_for_uniform_cv_candidate_labels(dataname=args.ds, full_train_loader=full_train_loader, batch_size=args.bs) #full_train_loader.batch_size


    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.to(device)

    if args.mo == 'mlp':
        model = mlp_model(input_dim=dim, hidden_dim=500, output_dim=K)
    elif args.mo == 'linear':
        model = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        model = LeNet(output_dim=K) #  linear,mlp,lenet are for MNIST-type datasets.
    elif args.mo == 'densenet':
        model = densenet(num_classes=K)
    elif args.mo == 'resnet':
        model = resnet(depth=32, num_classes=K)
    elif args.mo == 'convnet':
        model = convnet.Cnn(input_channels=3,n_outputs= K,dropout_rate=0.25)  # densenet,resnet are for CIFAR-10.

        # densenet,resnet are for CIFAR-10.

    model = model.to(device)

    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wd,momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
    test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)



    print('Epoch: 0. Tr Acc: {}. Te Acc: {} '.format(train_accuracy, test_accuracy))

    test_acc_list = []
    train_acc_list = []
    best = 0
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(partial_matrix_train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            average_loss = rc_loss(outputs, confidence, index)
            average_loss.backward()
            optimizer.step()
            confidence = confidence_update(model, confidence, X, Y, index)

        model.eval()

        train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
        test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch+1, train_accuracy, test_accuracy))


        if test_accuracy > best:
            best = test_accuracy
        if epoch >= (args.ep-10):
            test_acc_list.extend([test_accuracy])
            train_acc_list.extend([train_accuracy])

    avg_test_acc = np.mean(test_acc_list)
    avg_train_acc = np.mean(train_acc_list)

    print("Learning Rate:", args.lr, "Weight Decay:", args.wd)
    print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
    print("Best Test Accuracy:", best)
    print("Average Training Accuracy over Last 10 Epochs:", avg_train_acc,"\n\n\n")
    print("NOW is dataset: {} with model {} weight_decay {} learning rate {} batch_size {} op {}".format(args.ds,args.mo,args.wd,args.lr,args.bs,args.op))
