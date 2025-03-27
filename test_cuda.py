import torch

def check_cuda():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        x = torch.rand(5, 3).cuda()
        print("Tensor on GPU:", x)

if __name__ == "__main__":
    check_cuda()