import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(dataset_name, dataset_dir, phase='train'):
    
    gen = torch.Generator().manual_seed(42)
    
    if  dataset_name == 'FMNIST':

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859,), (0.3530,))
        ])

        trainset = torchvision.datasets.FashionMNIST(root=dataset_dir, train=True, download=True, transform=transform)

        testset_org = torchvision.datasets.FashionMNIST(root=dataset_dir, train=False, download=True, transform=transform)
        valset, testset = torch.utils.data.random_split(dataset=testset_org, lengths=[5000, 5000], generator=gen)

    elif dataset_name == 'CIFAR10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)

        testset_org = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
        valset, testset = torch.utils.data.random_split(dataset=testset_org, lengths=[5000, 5000], generator=gen)

    elif dataset_name == 'CIFAR100':
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform_train)
        
        testset_org = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform_test)
        valset, testset = torch.utils.data.random_split(dataset=testset_org, lengths=[5000, 5000], generator=gen)
        
    elif dataset_name == 'STL10':
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        trainset = torchvision.datasets.STL10(root=dataset_dir, split='train', download=True, transform=transform_train)

        testset_org = torchvision.datasets.STL10(root=dataset_dir, split='test', download=True, transform=transform_test)
        valset, testset = torch.utils.data.random_split(dataset=testset_org, lengths=[4000, 4000], generator=gen)
    
    else:
        raise ValueError('no support dataset name')
    
    if phase == 'train':
        return trainset
    elif phase == 'val':   
        return valset
    elif phase == 'test':
        return testset
    else:
        raise ValueError('no support phase name')