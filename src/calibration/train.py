import argparse
import sys
import os
import csv
import torch

from src.utils import torch_fix_seed
from src.dataset import get_dataset
from src.nets import get_net

from src.calibration.sampler import get_labels_set, BalancedBatchSampler
from src.calibration.sts import multi_mixup, ConcreteDistributionLoss
from src.calibration.eval import compute_ece

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    print('Calibration...')
    
    torch_fix_seed(args.seed)
    
    valset = get_dataset(args.dataset_name, args.dataset_dir, phase='val')
    
    model = get_net(args.dataset_name).to(device)
    model.calibration = True
    try:
        model.load_state_dict(torch.load(args.model_dir + '/trained_model.pth'))
    except FileNotFoundError:
        raise FileNotFoundError('You must train a model before calibration.')
    
    
    train_for_calibration(model, valset)
    
    torch.save(model.state_dict(), args.model_dir + f'/calibrated_model_beta={args.beta}.pth')
    
    _, val_ece = compute_ece(model, valset, args.n_bins, args.num_workers, args.gumbel_softmax_sample_size)
    
    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.results_dir + '/val_ece.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.beta, val_ece])
        
    print('\n')
    

def train_for_calibration(model, dataset):
    
    labels, labels_set = get_labels_set(dataset)
    num_classes = len(labels_set)
    
    batch_sampler = BalancedBatchSampler(labels, labels_set, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
    
    params_to_update = get_params_to_update(model)
    
    criterion = ConcreteDistributionLoss().to(device)
    optimizer = torch.optim.Adam(params_to_update, lr=args.lr, weight_decay=5e-4)
    
    model.train()
    
    for epoch in range(args.epoch_num):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)

            inputs, targets = multi_mixup(inputs, targets, num_classes, args.beta)
            assert torch.all(torch.isfinite(inputs)).item()
            assert torch.all(torch.isfinite(targets)).item()
            
            optimizer.zero_grad()
            
            logits, temps = model(inputs)
            assert torch.all(torch.isfinite(logits)).item()
            assert torch.all(torch.isfinite(temps)).item()

            loss = criterion(logits, temps, targets)
            assert torch.all(torch.isfinite(loss)).item()

            loss.backward()
            optimizer.step()
            
            sys.stdout.write('\r')
            sys.stdout.write(
                ' Epoch [%3d/%3d] Iter [%3d/%3d]: Loss: %.4f' %
                (epoch+1, args.epoch_num, batch_idx+1,
                len(dataloader), loss.item()))
            sys.stdout.flush()


def get_params_to_update(model):
    
    params_to_update = []
    for name, param in model.named_parameters():
        if 'branch' in name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    return params_to_update


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Accuracy-Preserving Calivration via Simplex Temperature Scaling')
    parser.add_argument('--dataset_name', default='FMNIST', type=str, help='Name of dataset', choices=['FMNIST', 'CIFAR10', 'CIFAR100', 'STL10'])
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for calibration')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size for calibration')
    parser.add_argument('--epoch_num', default=5000, type=int, help='Number of epochs for calibration')
    parser.add_argument('--beta', default=0.7, type=float, help='Balance of interpolation weights in Multi-Mixup')
    parser.add_argument('--n_bins', default=10, type=int, help='Number of bins for ECE')
    parser.add_argument('--gumbel_softmax_sample_size', default=30, type=int, help='Number of samples for Gumbel Softmax')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for dataloader')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generator')
    parser.add_argument('--dataset_dir', default='./data', type=str, help='Directory for dataset')
    parser.add_argument('--model_dir', default='./model', type=str, help='Directory for model')
    parser.add_argument('--results_dir', default='./results', type=str, help='Directory for results')
    args = parser.parse_args()
    
    
    main()