module {
  func.func @vecadd(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf16>) {
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 256 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<256x256xf16>
        %1 = affine.load %arg1[%arg3, %arg4] : memref<256x256xf16>
        %2 = arith.addf %0, %1 : f16
        affine.store %2, %arg2[%arg3, %arg4] : memref<256x256xf16>
      }
    }
    return
  }
}
