def get_net(dataset_name):
    
    if dataset_name == 'FMNIST':
        from src.nets.resnet.for_fmnist import ResNet18
        model =  ResNet18(num_classes=10)
        return model
       
    elif dataset_name == 'CIFAR10':
        from src.nets.resnet.for_cifar import ResNet18
        model =  ResNet18(num_classes=10)
        return model
        
    elif dataset_name == 'CIFAR100':
        from src.nets.resnet.for_cifar import ResNet18
        model =  ResNet18(num_classes=100)
        return model
    
    elif dataset_name == 'STL10':
        from src.nets.resnet.for_stl import ResNet18
        model =  ResNet18(num_classes=10)
        return model
        
    else:
        raise ValueError('no support dataset name')
        