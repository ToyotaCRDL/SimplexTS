import argparse
import sys
import os
import torch

from src.utils import torch_fix_seed
from src.dataset import get_dataset
from src.nets import get_net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    
    print('Training...')
    
    torch_fix_seed(args.seed)

    trainset = get_dataset(args.dataset_name, args.dataset_dir, phase='train')

    model = get_net(args.dataset_name).to(device)
    
    train_for_classification(model, trainset)
    
    torch.save(model.state_dict(), args.model_dir + '/trained_model.pth')
    
    print('\n')
    
    
def train_for_classification(model, dataset):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    params_to_update = get_params_to_update(model)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
    
    model.train()
    
    for epoch in range(args.epoch_num):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            sys.stdout.write('\r')
            sys.stdout.write(
                ' Epoch [%3d/%3d] Iter [%3d/%3d]: Loss: %.4f' %
                (epoch+1, args.epoch_num, batch_idx+1,
                len(dataloader), loss.item()))
            sys.stdout.flush()
        
        scheduler.step()
    
    
def get_params_to_update(model):
    
    params_to_update = []
    for name, param in model.named_parameters():
        if 'branch' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_update.append(param)

    return params_to_update
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training before calibration')
    parser.add_argument('--dataset_name', default='FMNIST', type=str, help='Name of dataset', choices=['FMNIST', 'CIFAR10', 'CIFAR100', 'STL10'])
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epoch_num', default=200, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for dataloader')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generator')
    parser.add_argument('--dataset_dir', default='./data', type=str, help='Directory for dataset')
    parser.add_argument('--model_dir', default='./model', type=str, help='Directory for model')
    args = parser.parse_args()
    
    try:
        os.makedirs(args.model_dir)
    except FileExistsError:
        raise FileExistsError('Use another directory name')
    
    main()