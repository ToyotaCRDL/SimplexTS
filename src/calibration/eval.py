import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset
from src.nets import get_net
from src.utils import ECELoss

from src.calibration.sts import gumbel_softmax_sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    
    print('Evaluation...')
    
    testset = get_dataset(args.dataset_name, args.dataset_dir, phase='test')
    
    model = get_net(args.dataset_name).to(device)
    model.calibration = True
    try:
        model.load_state_dict(torch.load(args.model_dir + f'/calibrated_model_beta={args.beta}.pth'))
    except FileNotFoundError:
        raise FileNotFoundError('You must calibrate a model before evaluation.')
    
    ece, calibrated_ece = compute_ece(model, testset, args.n_bins, args.num_workers, args.gumbel_softmax_sample_size)
    
    print(f'ECE: {ece:.4f}, Calibrated ECE: {calibrated_ece:.4f}')
    
    print('\n')

    
def compute_ece(model, dataset, n_bins, num_workers, sample_size):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)
    
    labels_list = []
    probs_list = []
    calibrated_probs_list = []
    predictions_list = []
    
    with torch.no_grad():
        model.eval()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, temps = model(inputs)
            
            # labels
            labels_list.append(targets)
            
            # predictions
            predictions = torch.argmax(logits, dim=1)
            predictions_list.append(predictions)
            
            # non-calibrated probabilities
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)    
            
            # calibrated probabilities
            alphas = torch.exp(logits).clamp(min=1e-8)
            lam = F.softplus(temps).clamp(min=1e-8)
            sampled_data = gumbel_softmax_sample(alphas, lamb=lam, sample_size=sample_size)
            calibrated_probs = torch.mean(sampled_data, dim=2)
            calibrated_probs_list.append(calibrated_probs)
            
        probs = torch.cat(probs_list, dim=0).to(device)
        calibrated_probs = torch.cat(calibrated_probs_list, dim=0).to(device)
        predictions = torch.cat(predictions_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)
        
    criterion = ECELoss(n_bins).to(device)
    ece = criterion(probs, predictions, labels).item()
    calibrated_ece = criterion(calibrated_probs, predictions, labels).item()
    
    return ece, calibrated_ece


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Accuracy-Preserving Calivration via Simplex Temperature Scaling')
    parser.add_argument('--dataset_name', default='FMNIST', type=str, help='Name of dataset', choices=['FMNIST', 'CIFAR10', 'CIFAR100', 'STL10'])
    parser.add_argument('--beta', type=float, help='Balance of interpolation weights in Multi-Mixup')
    parser.add_argument('--n_bins', default=10, type=int, help='Number of bins for ECE')
    parser.add_argument('--gumbel_softmax_sample_size', default=30, type=int, help='Number of samples for Gumbel Softmax')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for dataloader')
    parser.add_argument('--dataset_dir', default='./data', type=str, help='Directory for dataset')
    parser.add_argument('--model_dir', default='./model', type=str, help='Directory for model')
    parser.add_argument('--results_dir', default='./results', type=str, help='Directory for results')
    args = parser.parse_args()
    
    if args.beta == None: 
        try:
            args.beta = np.load(args.results_dir + '/best_beta.npy').item()
        except FileNotFoundError:
            raise FileNotFoundError('If tuning is not performed, beta must be specified.')
    
    main()