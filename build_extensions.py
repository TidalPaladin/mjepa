#!/usr/bin/env python3
"""Build script for CUDA extensions."""

import os
import sys
from pathlib import Path


def build_cuda_extensions():
    """Build CUDA extensions and place them in the package directory."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping CUDA extensions")
        return False
        
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA extensions")
        return False
        
    from torch.utils.cpp_extension import load
    
    # Get the package directory
    package_dir = Path(__file__).parent / "mjepa"
    package_dir.mkdir(exist_ok=True)
    
    # Clean up any existing compiled extensions
    for ext_file in package_dir.glob("*_cuda.so"):
        ext_file.unlink()
        print(f"Removed existing {ext_file.name}")
    
    # Define extensions
    extensions = [
        ("invert_cuda", "csrc/invert.cu"),
        ("mixup_cuda", "csrc/mixup.cu"),
        ("posterize_cuda", "csrc/posterize.cu"),
        ("noise_cuda", "csrc/noise.cu"),
    ]
    
    # CUDA compilation flags (using only widely supported architectures)
    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",  # A100, RTX 30xx
        "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
        "-gencode=arch=compute_90,code=sm_90",  # H100
    ]
    
    compiled_count = 0
    
    for name, source_file in extensions:
        try:
            print(f"Compiling {name}...")
            
            # Compile the extension
            module = load(
                name=name,
                sources=[source_file],
                extra_cuda_cflags=cuda_flags,
                extra_cflags=["-O3"],
                verbose=True,
            )
            
            # Copy the compiled .so file to the package directory
            import torch.utils.cpp_extension
            cache_dir = torch.utils.cpp_extension._get_build_directory(name, verbose=False)
            
            # Find the compiled .so file
            import glob
            so_pattern = os.path.join(cache_dir, f"{name}*.so")
            so_files = glob.glob(so_pattern)
            
            if so_files:
                import shutil
                dest_path = package_dir / f"{name}.so"
                shutil.copy2(so_files[0], dest_path)
                print(f"Copied {name}.so to package directory")
            
            compiled_count += 1
            print(f"Successfully compiled {name}")
            
        except Exception as e:
            print(f"Failed to compile {name}: {e}")
    
    print(f"Compiled {compiled_count}/{len(extensions)} CUDA extensions")
    return compiled_count > 0


if __name__ == "__main__":
    success = build_cuda_extensions()
    sys.exit(0 if success else 1)
