module {
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
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1_3, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %1, %arg13 = %c1_3, %arg14 = %c1_3) {
      %2 = arith.addi %c0, %arg3 : index
      %3 = arith.addi %c0_0, %arg6 : index
      %4 = affine.load %arg0[%2, %3] : memref<256x256xf16>
      %5 = affine.load %arg1[%2, %3] : memref<256x256xf16>
      %6 = arith.addf %4, %5 : f16
      affine.store %6, %arg2[%2, %3] : memref<256x256xf16>
      gpu.terminator
    }
    return
  }
}
