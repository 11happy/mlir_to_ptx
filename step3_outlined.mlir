module attributes {gpu.container_module} {
  func.func @vecadd(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf16>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %0 = arith.subi %c256, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c256_1 = arith.constant 256 : index
    %1 = arith.subi %c256_1, %c0_0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch_func  @vecadd_kernel::@vecadd_kernel blocks in (%0, %c1_3, %c1_3) threads in (%1, %c1_3, %c1_3)  args(%c0 : index, %c0_0 : index, %arg0 : memref<256x256xf16>, %arg1 : memref<256x256xf16>, %arg2 : memref<256x256xf16>)
    return
  }
  gpu.module @vecadd_kernel {
    gpu.func @vecadd_kernel(%arg0: index, %arg1: index, %arg2: memref<256x256xf16>, %arg3: memref<256x256xf16>, %arg4: memref<256x256xf16>) kernel {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.addi %arg1, %thread_id_x : index
      %2 = affine.load %arg2[%0, %1] : memref<256x256xf16>
      %3 = affine.load %arg3[%0, %1] : memref<256x256xf16>
      %4 = arith.addf %2, %3 : f16
      affine.store %4, %arg4[%0, %1] : memref<256x256xf16>
      gpu.return
    }
  }
}
