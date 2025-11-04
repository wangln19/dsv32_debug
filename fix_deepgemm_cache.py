"""
Fix DeepGemm cache directory conflict for multi-process JIT compilation.

This script patches vLLM's deep_gemm.py to use rank-specific cache directories,
preventing file handle conflicts when multiple processes compile the same kernel.
"""
import os
import sys

# Find the vLLM deep_gemm module
try:
    import vllm.utils.deep_gemm as dg_module
except ImportError:
    print("Error: Could not import vllm.utils.deep_gemm")
    print("Make sure vLLM is installed in the current environment.")
    sys.exit(1)

# Get the file path
dg_file = dg_module.__file__
print(f"Found deep_gemm.py at: {dg_file}")

# Read the file
with open(dg_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'rank_' in content and 'deep_gemm' in content and 'os.makedirs' in content:
    # Check if our pattern exists
    if 'f"rank_{rank}"' in content or 'rank_{rank}' in content:
        print("Already patched with rank-specific cache directories.")
        print("Current patch looks good, no action needed.")
        sys.exit(0)

# Find the cache setup section
old_pattern = """    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = 'DG_JIT_CACHE_DIR'
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm")"""

new_pattern = """    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = 'DG_JIT_CACHE_DIR'
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        # Use rank-specific cache directory to avoid file handle conflicts
        # in multi-process JIT compilation scenarios
        # Try to get rank from multiple sources
        rank = None
        # Method 1: Try from torch.distributed
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = str(dist.get_rank())
        except (ImportError, AttributeError, RuntimeError):
            pass
        
        # Method 2: Try from environment variables
        if rank is None:
            rank = os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or os.environ.get('DATA_PARALLEL_RANK')
        
        # Method 3: Use process ID as fallback (better than shared directory)
        if rank is None:
            rank = str(os.getpid())
        
        # Create rank-specific cache directory
        cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "deep_gemm", f"rank_{rank}")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = cache_dir"""

if old_pattern in content:
    # Backup original file
    backup_file = dg_file + '.backup'
    with open(backup_file, 'w') as f:
        with open(dg_file, 'r') as orig:
            f.write(orig.read())
    print(f"Backup created: {backup_file}")
    
    # Apply patch
    content = content.replace(old_pattern, new_pattern)
    
    # Write patched file
    with open(dg_file, 'w') as f:
        f.write(content)
    print(f"✓ Successfully patched {dg_file}")
    print("✓ Each process will now use a rank-specific cache directory.")
    print("  Format: /root/.cache/vllm/deep_gemm/rank_{rank}/")
    print("\nYou can now restart vLLM. The file handle conflict should be resolved.")
else:
    print("⚠ Could not find the expected pattern to patch.")
    print("The file structure might be different. Showing relevant section:")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'deep_gemm cache path' in line.lower() or 'DG_JIT_CACHE_DIR' in line:
            start = max(0, i - 2)
            end = min(len(lines), i + 10)
            print("\n".join(f"{start+j+1:4d}: {lines[j]}" for j in range(start, end)))
            break
    sys.exit(1)
