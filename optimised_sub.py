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
	.param .u64 vecadd_kernel_param_2,
	.param .u64 .ptr .align 1 vecadd_kernel_param_3,
	.param .u64 .ptr .align 1 vecadd_kernel_param_4,
	.param .u64 .ptr .align 1 vecadd_kernel_param_5
)
{
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<20>;

	ld.param.b64 	%rd1, [vecadd_kernel_param_0];
	ld.param.b64 	%rd2, [vecadd_kernel_param_3];
	cvta.to.global.u64 	%rd3, %rd2;
	ld.param.b64 	%rd4, [vecadd_kernel_param_1];
	ld.param.b64 	%rd5, [vecadd_kernel_param_4];
	cvta.to.global.u64 	%rd6, %rd5;
	ld.param.b64 	%rd7, [vecadd_kernel_param_2];
	ld.param.b64 	%rd8, [vecadd_kernel_param_5];
	cvta.to.global.u64 	%rd9, %rd8;
	mov.u32 	%r1, %ctaid.x;
	cvt.u64.u32 	%rd10, %r1;
	mov.u32 	%r2, %tid.x;
	cvt.u64.u32 	%rd11, %r2;
	add.s64 	%rd12, %rd1, %rd10;
	mad.lo.s64 	%rd13, %rd4, %rd11, %rd7;
	shl.b64 	%rd14, %rd12, 10;
	add.s64 	%rd15, %rd14, %rd13;
	shl.b64 	%rd16, %rd15, 2;
	add.s64 	%rd17, %rd3, %rd16;
	add.s64 	%rd18, %rd6, %rd16;
	ld.global.b32 	%r3, [%rd17+12];
	ld.global.b32 	%r4, [%rd17+8];
	ld.global.b32 	%r5, [%rd17+4];
	ld.global.b32 	%r6, [%rd17];
	ld.global.b32 	%r7, [%rd18+12];
	ld.global.b32 	%r8, [%rd18+8];
	ld.global.b32 	%r9, [%rd18+4];
	ld.global.b32 	%r10, [%rd18];
	add.rn.f16x2 	%r11, %r6, %r10;
	add.rn.f16x2 	%r12, %r5, %r9;
	add.rn.f16x2 	%r13, %r4, %r8;
	add.rn.f16x2 	%r14, %r3, %r7;
	add.s64 	%rd19, %rd9, %rd16;
	st.global.b32 	[%rd19+12], %r14;
	st.global.b32 	[%rd19+8], %r13;
	st.global.b32 	[%rd19+4], %r12;
	st.global.b32 	[%rd19], %r11;
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
    threads = 256
    f16_per_block = 2048

    padded_n = ((N + f16_per_block - 1) // f16_per_block) * f16_per_block

    if padded_n != N:
        A_flat = torch.nn.functional.pad(A.reshape(-1), (0, padded_n - N))
        B_flat = torch.nn.functional.pad(B.reshape(-1), (0, padded_n - N))
        C_flat = torch.empty(padded_n, device=A.device, dtype=A.dtype)
    else:
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        C_flat = C.reshape(-1)

    blocks = padded_n // f16_per_block

    kernel = _get_kernel()

    params = np.array([
        0, 4, 0,
        A_flat.data_ptr(), B_flat.data_ptr(), C_flat.data_ptr(),
    ], dtype=np.uint64)

    args_array = (ctypes.c_void_p * 6)(
        params[0:1].ctypes.data,
        params[1:2].ctypes.data,
        params[2:3].ctypes.data,
        params[3:4].ctypes.data,
        params[4:5].ctypes.data,
        params[5:6].ctypes.data,
    )

    cu.cuLaunchKernel(
        kernel, blocks, 1, 1, threads, 1, 1,
        0, 0, args_array, 0,
    )

    if padded_n != N:
        C.reshape(-1).copy_(C_flat[:N])

    return C