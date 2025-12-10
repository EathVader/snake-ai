#!/usr/bin/env python3
"""
CUDA Status Checker for Snake AI Training
检查CUDA状态用于贪吃蛇AI训练
"""

import sys

def check_cuda_status():
    """Check CUDA availability and provide recommendations"""
    print("="*60)
    print("CUDA Status Check / CUDA状态检查")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"{'✅' if cuda_available else '❌'} CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA device count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                memory_gb = device_props.total_memory / 1e9
                
                print(f"   Device {i}: {device_name}")
                print(f"   Memory: {memory_gb:.1f}GB")
                print(f"   Compute Capability: {device_props.major}.{device_props.minor}")
            
            # Test CUDA functionality
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor)
                print("✅ CUDA functionality test: PASSED")
                
                # Memory info
                allocated = torch.cuda.memory_allocated(0) / 1e6
                cached = torch.cuda.memory_reserved(0) / 1e6
                print(f"   Memory allocated: {allocated:.1f}MB")
                print(f"   Memory cached: {cached:.1f}MB")
                
            except Exception as e:
                print(f"❌ CUDA functionality test: FAILED - {e}")
                
        else:
            print("❌ CUDA not available")
            
            # Check for common issues
            print("\nTroubleshooting / 故障排除:")
            print("1. Install CUDA-enabled PyTorch using conda:")
            print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            print("2. Or use environment file:")
            print("   conda env create -f environment.yml")
            print("3. Check NVIDIA driver:")
            print("   nvidia-smi")
            print("4. Verify CUDA installation:")
            print("   nvcc --version")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            print(f"{'✅' if mps_available else '❌'} MPS (Apple Silicon) available: {mps_available}")
            
            if mps_available:
                try:
                    test_tensor = torch.randn(100, 100).to('mps')
                    result = torch.matmul(test_tensor, test_tensor)
                    print("✅ MPS functionality test: PASSED")
                except Exception as e:
                    print(f"❌ MPS functionality test: FAILED - {e}")
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("Install with: pip install torch torchvision torchaudio")
        return False
    
    print("="*60)
    
    # Training recommendations
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("Training Recommendations / 训练建议:")
        print("="*60)
        
        if memory_gb >= 20:
            print("🚀 High-end GPU detected!")
            print("   Recommended environments: 64-128")
            print("   Recommended batch size: 1024-2048")
            print("   Expected training time: 4-6 hours")
        elif memory_gb >= 10:
            print("⚡ Mid-range GPU detected!")
            print("   Recommended environments: 32-64")
            print("   Recommended batch size: 512-1024")
            print("   Expected training time: 6-10 hours")
        else:
            print("💻 Entry-level GPU detected!")
            print("   Recommended environments: 16-32")
            print("   Recommended batch size: 256-512")
            print("   Expected training time: 10-16 hours")
            
        print(f"\nOptimal training command:")
        print(f"   python train_cnn_anti_loop.py")
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected!")
        print("   Recommended environments: 32")
        print("   Recommended batch size: 512-1024")
        print("   Expected training time: 8-12 hours")
        print(f"\nOptimal training command:")
        print(f"   python train_cnn_simple.py")
        
    else:
        print("💻 CPU training detected!")
        print("   Recommended environments: 8-16")
        print("   Recommended batch size: 256")
        print("   Expected training time: 24-48 hours")
        print("   ⚠️  Consider using cloud GPU for faster training")
    
    print("="*60)
    return cuda_available

if __name__ == "__main__":
    check_cuda_status()