import torch

def check_hardware():
    print('\nHardware Detection:')
    print('-' * 50)
    cuda_available = torch.cuda.is_available()
    print(f'CUDA Available: {cuda_available}')
    if cuda_available:
        print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    rocm_available = hasattr(torch.version, 'hip') and torch.version.hip is not None
    print(f'ROCm Available: {rocm_available}')
    mkl_available = torch.backends.mkl.is_available()
    print(f'MKL Available: {mkl_available}')
    
    if cuda_available:
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f'\nUsing device: {device}')
    return device

if __name__ == '__main__':
    device = check_hardware()
    print(f'\nReady for training on {device}!')
