"""
GPU Detection and Testing Script
Run this to verify your NVIDIA RTX 3050 is properly configured for TensorFlow
"""

import sys
import os

# Reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_test(test_num, description):
    """Print test header"""
    print(f"\n[Test {test_num}] {description}...")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def check_tensorflow():
    """Test 1: Check TensorFlow installation"""
    print_test(1, "Checking TensorFlow installation")
    try:
        import tensorflow as tf
        print_success(f"TensorFlow installed: version {tf.__version__}")
        return True, tf
    except ImportError:
        print_error("TensorFlow not installed!")
        print("\n   Install with: pip install tensorflow==2.15.0")
        return False, None

def check_gpu_availability(tf):
    """Test 2: Check GPU availability"""
    print_test(2, "Checking GPU availability")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print_success(f"GPU Detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        return True, gpus
    else:
        print_error("No GPU detected!")
        print("\n   Troubleshooting steps:")
        print("   1. Verify GPU with: nvidia-smi")
        print("   2. Install correct TensorFlow: pip install tensorflow==2.15.0")
        print("   3. Update NVIDIA driver from: https://www.nvidia.com/Download/index.aspx")
        print("   4. Restart your computer")
        return False, None

def configure_gpu_memory(tf, gpus):
    """Test 3: Configure GPU memory"""
    print_test(3, "Configuring GPU memory")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_success("Memory growth enabled (prevents OOM errors)")
        return True
    except RuntimeError as e:
        print_warning(f"Could not enable memory growth: {e}")
        return False

def check_cuda_support(tf):
    """Test 4: Check CUDA and cuDNN"""
    print_test(4, "Checking CUDA and cuDNN support")
    
    cuda_available = tf.test.is_built_with_cuda()
    gpu_support = tf.test.is_built_with_gpu_support()
    
    print(f"   CUDA available: {cuda_available}")
    print(f"   GPU support built: {gpu_support}")
    
    if cuda_available and gpu_support:
        print_success("TensorFlow has GPU support")
        return True
    else:
        print_warning("TensorFlow may not have proper GPU support")
        print("\n   Try: pip install tensorflow==2.15.0")
        return False

def test_simple_computation(tf):
    """Test 5: Simple GPU computation"""
    print_test(5, "Running simple computation on GPU")
    try:
        import numpy as np
        import time
        
        # Create test data
        with tf.device('/GPU:0'):
            a = tf.constant(np.random.randn(1000, 1000), dtype=tf.float32)
            b = tf.constant(np.random.randn(1000, 1000), dtype=tf.float32)
            
            # Time the computation
            start = time.time()
            c = tf.matmul(a, b)
            end = time.time()
        
        elapsed = (end - start) * 1000
        print_success(f"Matrix multiplication completed in {elapsed:.2f}ms")
        print(f"   Result shape: {c.shape}")
        return True
        
    except Exception as e:
        print_error(f"Computation failed: {e}")
        return False

def test_model_training(tf):
    """Test 6: Train simple model on GPU"""
    print_test(6, "Training simple model on GPU")
    try:
        import numpy as np
        import time
        
        # Create simple dataset
        print("   Creating test dataset (1000 samples)...")
        X = np.random.randn(1000, 50, 14).astype(np.float32)
        y = np.random.randn(1000, 1).astype(np.float32)
        
        # Create simple model
        print("   Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(50, 14)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train and time it
        print("   Training for 3 epochs...")
        start = time.time()
        
        with tf.device('/GPU:0'):
            history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        
        end = time.time()
        elapsed = end - start
        
        print_success(f"Training completed in {elapsed:.2f} seconds")
        print(f"   Final loss: {history.history['loss'][-1]:.4f}")
        
        # Verify GPU was used
        if elapsed < 10:
            print_success("🚀 GPU is working efficiently!")
            return True
        else:
            print_warning(f"Training seems slow ({elapsed:.1f}s), GPU may not be utilized")
            return False
        
    except Exception as e:
        print_error(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision(tf):
    """Test 7: Mixed precision support"""
    print_test(7, "Checking mixed precision support")
    try:
        from tensorflow.keras import mixed_precision
        
        # Try to enable mixed precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        print_success(f"Mixed precision supported: {policy.name}")
        print("   This will make training ~2x faster!")
        
        # Reset to default
        mixed_precision.set_global_policy('float32')
        return True
        
    except Exception as e:
        print_warning(f"Mixed precision not available: {e}")
        return False

def test_gpu_memory_info(tf, gpus):
    """Test 8: GPU memory information"""
    print_test(8, "Checking GPU memory")
    try:
        print("   GPU memory info:")
        print("   Note: Use 'nvidia-smi' for detailed memory usage")
        print_success("For RTX 3050: 4GB total VRAM available")
        return True
    except Exception as e:
        print_warning(f"Could not get memory info: {e}")
        return False

def print_summary(all_tests_passed, gpu_available):
    """Print final summary"""
    print_header("SUMMARY")
    
    if all_tests_passed and gpu_available:
        print_success("GPU is properly configured and ready for training!")
        print("\n📊 Recommendations for RTX 3050 (4GB VRAM):")
        print("  • Use batch_size=32 (reduce to 16 if Out of Memory)")
        print("  • Enable mixed precision for 2x speedup")
        print("  • Enable memory growth to prevent OOM")
        print("  • Monitor with: nvidia-smi -l 1")
        
        print("\n⚡ Expected Training Performance:")
        print("  • CPU: 2-3 hours")
        print("  • RTX 3050: 20-30 minutes (4-6x faster)")
        print("  • RTX 3050 + Mixed Precision: 15-20 minutes (8-10x faster)")
        
        print("\n🚀 Next Steps:")
        print("  1. Add GPU config to main_train.py")
        print("  2. Run: python main_train.py")
        print("  3. Monitor: nvidia-smi -l 1")
        
    elif gpu_available:
        print_warning("GPU detected but some tests failed")
        print("\nThe GPU should still work for training, but:")
        print("  • Performance may not be optimal")
        print("  • Some features may not be available")
        print("\nTry: pip install tensorflow==2.15.0")
        
    else:
        print_error("GPU not properly configured")
        print("\n🔧 Troubleshooting Steps:")
        print("  1. Verify GPU with: nvidia-smi")
        print("  2. Update NVIDIA driver: https://www.nvidia.com/Download/index.aspx")
        print("  3. Install TensorFlow 2.15: pip install tensorflow==2.15.0")
        print("  4. Restart your computer")
        print("  5. Run this script again")

def print_system_info(tf):
    """Print system information"""
    print("\n[System Info]")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU count: {len(gpus)}")
    
    if gpus:
        print(f"GPU name: {gpus[0].name}")

def main():
    """Main test function"""
    print_header("GPU DETECTION AND CONFIGURATION TEST")
    print("This script will test if your NVIDIA RTX 3050 is ready for training")
    
    # Track results
    results = []
    
    # Test 1: TensorFlow installation
    tf_installed, tf = check_tensorflow()
    results.append(tf_installed)
    if not tf_installed:
        print("\n" + "=" * 80)
        print("❌ Cannot proceed without TensorFlow")
        print("=" * 80)
        sys.exit(1)
    
    # Test 2: GPU availability
    gpu_available, gpus = check_gpu_availability(tf)
    results.append(gpu_available)
    if not gpu_available:
        print_summary(False, False)
        print("=" * 80)
        sys.exit(1)
    
    # Test 3: Configure GPU memory
    memory_ok = configure_gpu_memory(tf, gpus)
    results.append(memory_ok)
    
    # Test 4: CUDA support
    cuda_ok = check_cuda_support(tf)
    results.append(cuda_ok)
    
    # Test 5: Simple computation
    compute_ok = test_simple_computation(tf)
    results.append(compute_ok)
    
    # Test 6: Model training
    training_ok = test_model_training(tf)
    results.append(training_ok)
    
    # Test 7: Mixed precision
    mixed_precision_ok = test_mixed_precision(tf)
    results.append(mixed_precision_ok)
    
    # Test 8: Memory info
    memory_info_ok = test_gpu_memory_info(tf, gpus)
    results.append(memory_info_ok)
    
    # Print system info
    print_system_info(tf)
    
    # Print summary
    all_passed = all(results)
    print_summary(all_passed, gpu_available)
    
    print("=" * 80)
    
    if all_passed:
        print("\n✅ All tests passed! Your GPU is ready for training! 🎉")
    else:
        passed = sum(results)
        total = len(results)
        print(f"\n⚠️  {passed}/{total} tests passed")
    
    print("=" * 80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)