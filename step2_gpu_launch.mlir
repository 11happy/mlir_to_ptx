#map = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
module {
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
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1_1, %arg11 = %c1_1) threads(%arg6, %arg7, %arg8) in (%arg12 = %2, %arg13 = %c1_1, %arg14 = %c1_1) {
      %3 = arith.addi %c0, %arg3 : index
      %4 = arith.muli %c4, %arg6 : index
      %5 = arith.addi %c0_0, %4 : index
      %6 = affine.apply #map(%3, %5)
      %7 = vector.load %arg0[%3, %5] : memref<128x1024xi32>, vector<4xi32>
      %8 = vector.load %arg1[%3, %5] : memref<128x1024xi32>, vector<4xi32>
      %9 = vector.bitcast %7 : vector<4xi32> to vector<8xf16>
      %10 = vector.bitcast %8 : vector<4xi32> to vector<8xf16>
      %11 = arith.addf %9, %10 : vector<8xf16>
      %12 = vector.bitcast %11 : vector<8xf16> to vector<4xi32>
      vector.store %12, %arg2[%3, %5] : memref<128x1024xi32>, vector<4xi32>
      gpu.terminator
    }
    return
  }
}
