# submission.py
import torch
import numpy as np
import ctypes
from cuda.bindings import driver as cu
from task import input_t, output_t

VECADD_PTX = r"""
.version 7.0
.target sm_80
.address_size 64

.visible .entry vecadd_kernel(
	.param .u64 vecadd_kernel_param_0,
	.param .u64 vecadd_kernel_param_1,
	.param .u64 .ptr .align 1 vecadd_kernel_param_2,
	.param .u64 .ptr .align 1 vecadd_kernel_param_3,
	.param .u64 .ptr .align 1 vecadd_kernel_param_4
)
{
	.reg .b16 	%rs<4>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<19>;

	ld.param.b64 	%rd1, [vecadd_kernel_param_0];
	ld.param.b64 	%rd2, [vecadd_kernel_param_2];
	cvta.to.global.u64 	%rd3, %rd2;
	ld.param.b64 	%rd4, [vecadd_kernel_param_1];
	ld.param.b64 	%rd5, [vecadd_kernel_param_3];
	cvta.to.global.u64 	%rd6, %rd5;
	ld.param.b64 	%rd7, [vecadd_kernel_param_4];
	cvta.to.global.u64 	%rd8, %rd7;
	mov.u32 	%r1, %ctaid.x;
	cvt.u64.u32 	%rd9, %r1;
	mov.u32 	%r2, %tid.x;
	cvt.u64.u32 	%rd10, %r2;
	add.s64 	%rd11, %rd1, %rd9;
	add.s64 	%rd12, %rd4, %rd10;
	shl.b64 	%rd13, %rd11, 8;
	add.s64 	%rd14, %rd13, %rd12;
	shl.b64 	%rd15, %rd14, 1;
	add.s64 	%rd16, %rd3, %rd15;
	ld.global.b16 	%rs1, [%rd16];
	add.s64 	%rd17, %rd6, %rd15;
	ld.global.b16 	%rs2, [%rd17];
	add.rn.f16 	%rs3, %rs1, %rs2;
	add.s64 	%rd18, %rd8, %rd15;
	st.global.b16 	[%rd18], %rs3;
	ret;
}
"""

_kernel_cache = {}

def _get_kernel():
    if "fn" not in _kernel_cache:
        cu.cuInit(0)
        err, module = cu.cuModuleLoadData(VECADD_PTX.encode("utf-8"))
        if err.value:
            raise RuntimeError(f"PTX load failed: {err}")
        err, kernel = cu.cuModuleGetFunction(module, b"vecadd_kernel")
        _kernel_cache["fn"] = kernel
        _kernel_cache["mod"] = module
    return _kernel_cache["fn"]


def custom_kernel(data: input_t) -> output_t:
    A, B, C = data
    N = A.numel()

    elements_per_launch = 65536  # 256 * 256

    padded_n = ((N + elements_per_launch - 1) // elements_per_launch) * elements_per_launch

    if padded_n != N:
        A_flat = torch.nn.functional.pad(A.reshape(-1), (0, padded_n - N))
        B_flat = torch.nn.functional.pad(B.reshape(-1), (0, padded_n - N))
        C_flat = torch.empty(padded_n, device=A.device, dtype=A.dtype)
    else:
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        C_flat = C.reshape(-1)

    num_tiles = padded_n // elements_per_launch
    kernel = _get_kernel()

    # once per 256x256 tile
    for tile in range(num_tiles):
        tile_offset = tile * elements_per_launch * 2  

        params = np.array([
            0,
            0,
            A_flat.data_ptr() + tile_offset,
            B_flat.data_ptr() + tile_offset,
            C_flat.data_ptr() + tile_offset,
        ], dtype=np.uint64)

        args_array = (ctypes.c_void_p * 5)(
            params[0:1].ctypes.data,
            params[1:2].ctypes.data,
            params[2:3].ctypes.data,
            params[3:4].ctypes.data,
            params[4:5].ctypes.data,
        )

        cu.cuLaunchKernel(
            kernel,
            256, 1, 1,
            256, 1, 1,
            0, 0,
            args_array, 0,
        )

    if padded_n != N:
        C.reshape(-1).copy_(C_flat[:N])

    return C