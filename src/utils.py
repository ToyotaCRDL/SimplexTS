import random
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    

def compute_acc(model, dataset, num_workers):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)

    total = 0
    success = 0
    with torch.no_grad():
        model.eval()
        for images, labels in dataloader:
            total += images.shape[0]
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            success += torch.sum(predicted == labels).item()
    return success / total
    

class ECELoss(nn.Module):
    '''
    We refered to the open source code https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78-L121 to implement the expected calibration error (ECE).
    We updated this code for Simplex Temperature Scaling.
    '''
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, predicted_prob, predictions, labels):
        confidences, _ = torch.max(predicted_prob, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=predicted_prob.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece