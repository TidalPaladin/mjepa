"""Hatchling build hook for automatic CUDA extension compilation."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore


class CudaBuildHook(BuildHookInterface):
    """Build hook that automatically compiles CUDA extensions during package installation."""
    
    PLUGIN_NAME = "custom"
    
    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Initialize and compile CUDA extensions."""
        self.app.display_info("Checking for CUDA extensions...")
        
        # Check if we should skip CUDA compilation
        if os.environ.get("MJEPA_SKIP_CUDA_BUILD", "").lower() in ("1", "true", "yes"):
            self.app.display_info("CUDA compilation skipped via MJEPA_SKIP_CUDA_BUILD")
            return
        
        try:
            import torch
        except ImportError:
            self.app.display_warning("PyTorch not available during build, CUDA extensions will be compiled at runtime")
            return
            
        # Check for CUDA availability with better error handling
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as e:
            self.app.display_warning(f"Could not check CUDA availability: {e}")
            cuda_available = False
            
        if not cuda_available:
            self.app.display_info("CUDA not available during build, extensions will be compiled at runtime if CUDA becomes available")
            return
            
        # Check for required tools
        if not self._check_build_requirements():
            return
            
        self.app.display_info("Compiling CUDA extensions...")
        try:
            self._compile_extensions(build_data)
        except Exception as e:
            self.app.display_warning(f"CUDA compilation failed: {e}")
            self.app.display_info("Extensions will be compiled at runtime instead")
    
    def _check_build_requirements(self) -> bool:
        """Check if all required build tools are available."""
        try:
            # Check for nvcc
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.app.display_warning("nvcc not found, CUDA extensions will be compiled at runtime")
                return False
        except FileNotFoundError:
            self.app.display_warning("nvcc not found, CUDA extensions will be compiled at runtime")
            return False
            
        # Check for ninja (optional but recommended)
        try:
            subprocess.run(["ninja", "--version"], capture_output=True)
        except FileNotFoundError:
            self.app.display_info("ninja not found, using default build system")
            
        return True
    
    def _compile_extensions(self, build_data: Dict[str, Any]) -> None:
        """Compile all CUDA extensions and add them to the build."""
        from torch.utils.cpp_extension import load
        
        # Define extensions
        extensions = [
            ("invert_cuda", "csrc/invert.cu"),
            ("mixup_cuda", "csrc/mixup.cu"),
            ("posterize_cuda", "csrc/posterize.cu"),
            ("noise_cuda", "csrc/noise.cu"),
        ]
        
        # CUDA compilation flags
        cuda_flags = [
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86", 
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ]
        
        compiled_extensions = []
        
        # Create a build directory
        build_dir = Path.cwd() / "build" / "cuda_build_hook"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        for name, source_file in extensions:
            try:
                self.app.display_info(f"Compiling {name} during build...")
                
                # Compile the extension
                module = load(
                    name=name,
                    sources=[source_file],
                    extra_cuda_cflags=cuda_flags,
                    extra_cflags=["-O3"],
                    build_directory=str(build_dir / name),
                    verbose=False,
                )
                
                # Find the compiled .so file and copy it to package directory
                so_files = list((build_dir / name).glob(f"{name}*.so"))
                
                if so_files:
                    # Copy to package directory for inclusion in wheel
                    package_dir = Path("mjepa")
                    package_dir.mkdir(exist_ok=True)
                    
                    dest_path = package_dir / f"{name}.so"
                    shutil.copy2(so_files[0], dest_path)
                    
                    # Tell hatchling to include this file
                    if "force_include" not in build_data:
                        build_data["force_include"] = {}
                    build_data["force_include"][str(dest_path)] = f"mjepa/{name}.so"
                    
                    compiled_extensions.append(name)
                    self.app.display_info(f"Successfully compiled {name} during build!")
                else:
                    self.app.display_warning(f"Could not find compiled {name} shared library")
                    
            except Exception as e:
                self.app.display_warning(f"Failed to compile {name} during build: {e}")
        
        if compiled_extensions:
            self.app.display_info(f"Successfully compiled {len(compiled_extensions)} CUDA extensions during build: {', '.join(compiled_extensions)}")
        else:
            self.app.display_warning("No CUDA extensions were compiled during build - they will be compiled at runtime")
            
        # Also check for any pre-existing compiled extensions
        self._check_for_precompiled_extensions(build_data)
    
    def _check_for_precompiled_extensions(self, build_data: Dict[str, Any]) -> None:
        """Check for pre-compiled extensions and include them in the build."""
        extensions = ["invert_cuda", "mixup_cuda", "posterize_cuda", "noise_cuda"]
        found_extensions = []
        
        # Check in the mjepa directory for pre-compiled .so files
        mjepa_dir = Path("mjepa")
        if mjepa_dir.exists():
            for name in extensions:
                so_file = mjepa_dir / f"{name}.so"
                if so_file.exists():
                    # Include the pre-compiled extension
                    if "force_include" not in build_data:
                        build_data["force_include"] = {}
                    build_data["force_include"][str(so_file)] = f"mjepa/{name}.so"
                    found_extensions.append(name)
        
        if found_extensions:
            self.app.display_info(f"Including pre-compiled CUDA extensions: {', '.join(found_extensions)}")
        else:
            self.app.display_info("No pre-compiled CUDA extensions found - they will be compiled at runtime")
