#map = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
module attributes {gpu.container_module} {
  func.func @vecadd(%arg0: memref<128x1024xi32>, %arg1: memref<128x1024xi32>, %arg2: memref<128x1024xi32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = arith.subi %c128, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %1 = arith.subi %c1024, %c0_0 : index
    %c4 = arith.constant 4 : index
    %2 = arith.ceildivsi %1, %c4 : index
    %c1_1 = arith.constant 1 : index
    gpu.launch_func  @vecadd_kernel::@vecadd_kernel blocks in (%0, %c1_1, %c1_1) threads in (%2, %c1_1, %c1_1)  args(%c0 : index, %c4 : index, %c0_0 : index, %arg0 : memref<128x1024xi32>, %arg1 : memref<128x1024xi32>, %arg2 : memref<128x1024xi32>)
    return
  }
  gpu.module @vecadd_kernel {
    gpu.func @vecadd_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<128x1024xi32>, %arg4: memref<128x1024xi32>, %arg5: memref<128x1024xi32>) kernel {
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
      %1 = arith.muli %arg1, %thread_id_x : index
      %2 = arith.addi %arg2, %1 : index
      %3 = affine.apply #map(%0, %2)
      %4 = vector.load %arg3[%0, %2] : memref<128x1024xi32>, vector<4xi32>
      %5 = vector.load %arg4[%0, %2] : memref<128x1024xi32>, vector<4xi32>
      %6 = vector.bitcast %4 : vector<4xi32> to vector<8xf16>
      %7 = vector.bitcast %5 : vector<4xi32> to vector<8xf16>
      %8 = arith.addf %6, %7 : vector<8xf16>
      %9 = vector.bitcast %8 : vector<8xf16> to vector<4xi32>
      vector.store %9, %arg5[%0, %2] : memref<128x1024xi32>, vector<4xi32>
      gpu.return
    }
  }
}
