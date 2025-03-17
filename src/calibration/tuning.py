import argparse
import csv
import numpy as np    
    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Accuracy-Preserving Calivration via Simplex Temperature Scaling')
    parser.add_argument('--results_dir', default='./results', type=str, help='Directory for model')
    args = parser.parse_args()
    
    try:
        with open(args.results_dir + '/val_ece.csv', 'r') as f:
            reader = csv.reader(f)
            val_ece_list = [row for row in reader]
    except FileNotFoundError:
        raise FileNotFoundError('You must calibrate a model before tuning.')
    
    print('Tuning...')
    
    beta_array = np.array(val_ece_list, dtype=float)[:, 0]
    val_ece_array = np.array(val_ece_list, dtype=float)[:, 1]
    
    best_beta = beta_array[np.argmin(val_ece_array)]
    
    print('The best beta value: ', best_beta)
    
    np.save(args.results_dir + '/best_beta.npy', best_beta)
    
    print('\n')