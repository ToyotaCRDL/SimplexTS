import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import get_dataset
from src.nets import get_net
from src.utils import compute_acc, ECELoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    
    print('Evaluation...')
    
    testset = get_dataset(args.dataset_name, args.dataset_dir, phase='test')
    
    model = get_net(args.dataset_name).to(device)
    try:
        model.load_state_dict(torch.load(args.model_dir + f'/trained_model.pth'))
    except FileNotFoundError:
        raise FileNotFoundError('You must train a model before evaluation.')
    
    acc = compute_acc(model, testset, args.num_workers)
    ece = compute_ece(model, testset, args.n_bins, args.num_workers)
    
    print(f'Accuracy: {acc:.4f}, ECE: {ece:.4f}')
    
    print('\n')


def compute_ece(model, dataset, n_bins, num_workers):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)
    
    labels_list = []
    probs_list = []
    predictions_list = []
    
    with torch.no_grad():
        model.eval()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            labels_list.append(targets)
            probs_list.append(probs)
            predictions_list.append(predictions)
            
    labels = torch.cat(labels_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    predictions = torch.cat(predictions_list, dim=0)
    
    criterion = ECELoss(n_bins).to(device)
    ece = criterion(probs, predictions, labels).item()
    
    return ece


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Accuracy-Preserving Calivration via Simplex Temperature Scaling')
    parser.add_argument('--dataset_name', default='FMNIST', type=str, help='Name of dataset', choices=['FMNIST', 'CIFAR10', 'CIFAR100', 'STL10'])
    parser.add_argument('--n_bins', default=10, type=int, help='Number of bins for ECE')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for dataloader')
    parser.add_argument('--dataset_dir', default='./data', type=str, help='Directory for dataset')
    parser.add_argument('--model_dir', default='./model', type=str, help='Directory for model')
    args = parser.parse_args()
    
    main()