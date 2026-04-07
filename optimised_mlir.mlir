#map = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
module {
  func.func @vecadd(%arg0: memref<128x1024xi32>, %arg1: memref<128x1024xi32>, %arg2: memref<128x1024xi32>) {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 1024 step 4 {
        %0 = affine.apply #map(%arg3, %arg4)
        %1 = vector.load %arg0[%arg3, %arg4] : memref<128x1024xi32>, vector<4xi32>
        %2 = vector.load %arg1[%arg3, %arg4] : memref<128x1024xi32>, vector<4xi32>
        %3 = vector.bitcast %1 : vector<4xi32> to vector<8xf16>
        %4 = vector.bitcast %2 : vector<4xi32> to vector<8xf16>
        %5 = arith.addf %3, %4 : vector<8xf16>
        %6 = vector.bitcast %5 : vector<8xf16> to vector<4xi32>
        vector.store %6, %arg2[%arg3, %arg4] : memref<128x1024xi32>, vector<4xi32>
      }
    }
    return
  }
}
