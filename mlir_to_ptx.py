# main.py
import subprocess
import os
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

# 256 threads/block, each thread loads 4xi32 (8 f16)
# Inner dim = 1024 i32, step 4 → 256 threads
# Total per block = 256 * 8 = 2048 f16 elements

mlir_source = """
module {
  func.func @vecadd(
    %A: memref<256x256xf16>,
    %B: memref<256x256xf16>,
    %C: memref<256x256xf16>
  ) {
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %a = affine.load %A[%i, %j] : memref<256x256xf16>
        %b = affine.load %B[%i, %j] : memref<256x256xf16>
        %sum = arith.addf %a, %b : f16
        affine.store %sum, %C[%i, %j] : memref<256x256xf16>
      }
    }
    return
  }
}
"""

def save(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    print(f"  → Saved to {filename}")

with Context():
    module = Module.parse(mlir_source)
    print("=== Step 1: MLIR Source (256 threads/block) ===")
    save("step1_source.mlir", str(module))

    pm = PassManager.parse("builtin.module(func.func(convert-affine-for-to-gpu))")
    pm.run(module.operation)
    print("=== Step 2: After convert-affine-for-to-gpu ===")
    save("step2_gpu_launch.mlir", str(module))

    pm = PassManager.parse("builtin.module(gpu-kernel-outlining)")
    pm.run(module.operation)
    print("=== Step 3: After gpu-kernel-outlining ===")
    save("step3_outlined.mlir", str(module))

    pm = PassManager.parse(
        "builtin.module("
        "lower-affine,"
        "gpu-decompose-memrefs,"
        "expand-strided-metadata,"
        "normalize-memrefs,"
        "gpu.module("
        "  convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv},"
        "  canonicalize,"
        "  convert-vector-to-llvm,"
        "  canonicalize"
        "),"
        "convert-nvvm-to-llvm,"
        "canonicalize,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    print("=== Step 4: After full lowering ===")
    save("step4_nvvm.mlir", str(module))

    full_ir = str(module)
    gpu_module_lines = []
    inside_gpu = False
    brace_count = 0
    for line in full_ir.split("\n"):
        if "gpu.module" in line:
            inside_gpu = True
        if inside_gpu:
            gpu_module_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count <= 0 and inside_gpu and len(gpu_module_lines) > 1:
                break

    gpu_module_str = "\n".join(gpu_module_lines)
    kernel_module_str = "module {\n" + gpu_module_str + "\n}"
    kernel_module = Module.parse(kernel_module_str)

    kernel_func_lines = []
    inside_func = False
    brace_count = 0
    for line in str(kernel_module).split("\n"):
        if "llvm.func @vecadd_kernel" in line:
            inside_func = True
        if inside_func:
            kernel_func_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count <= 0 and inside_func and len(kernel_func_lines) > 1:
                break

    standalone_module = "module {\n" + "\n".join(kernel_func_lines) + "\n}"
    save("step5_kernel.mlir", standalone_module)

print("=== Step 5: mlir-translate ===")
result = subprocess.run(
    ["mlir-translate", "--mlir-to-llvmir", "step5_kernel.mlir"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    exit(1)
save("step6_llvmir.ll", result.stdout)

print("=== Step 6: llc → PTX ===")
for chip, suffix in [("sm_80", ""), ("sm_75", "_local")]:
    result = subprocess.run(
        ["llc", "-march=nvptx64", f"-mcpu={chip}", "-O3",
         "step6_llvmir.ll", "-o", f"step7_kernel{suffix}.ptx"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR ({chip}): {result.stderr}")
    else:
        print(f"  → step7_kernel{suffix}.ptx")

print()
ptx = open("step7_kernel.ptx").read()
print(ptx)

print()
for f in ["step1_source.mlir", "step2_gpu_launch.mlir", "step3_outlined.mlir",
          "step4_nvvm.mlir", "step5_kernel.mlir", "step6_llvmir.ll",
          "step7_kernel.ptx"]:
    if os.path.exists(f):
        print(f"  {f:30s} {os.path.getsize(f):6d} bytes")