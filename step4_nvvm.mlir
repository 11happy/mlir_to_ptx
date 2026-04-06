module attributes {gpu.container_module} {
  func.func @vecadd(%arg0: memref<128x1024xi32>, %arg1: memref<128x1024xi32>, %arg2: memref<128x1024xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    gpu.launch_func  @vecadd_kernel::@vecadd_kernel blocks in (%c128, %c1, %c1) threads in (%c256, %c1, %c1)  args(%c0 : index, %c4 : index, %c0 : index, %arg0 : memref<128x1024xi32>, %arg1 : memref<128x1024xi32>, %arg2 : memref<128x1024xi32>)
    return
  }
  gpu.module @vecadd_kernel {
    llvm.func @vecadd_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(1024 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.mul %arg1, %4 : i64
      %7 = llvm.add %arg2, %6 : i64
      %8 = llvm.mul %5, %0 : i64
      %9 = llvm.add %8, %7 : i64
      %10 = llvm.getelementptr %arg3[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %11 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> vector<4xi32>
      %12 = llvm.mul %5, %0 : i64
      %13 = llvm.add %12, %7 : i64
      %14 = llvm.getelementptr %arg4[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %15 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> vector<4xi32>
      %16 = llvm.bitcast %11 : vector<4xi32> to vector<8xf16>
      %17 = llvm.bitcast %15 : vector<4xi32> to vector<8xf16>
      %18 = llvm.fadd %16, %17 : vector<8xf16>
      %19 = llvm.bitcast %18 : vector<8xf16> to vector<4xi32>
      %20 = llvm.mul %5, %0 : i64
      %21 = llvm.add %20, %7 : i64
      %22 = llvm.getelementptr %arg5[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %19, %22 {alignment = 4 : i64} : vector<4xi32>, !llvm.ptr
      llvm.return
    }
  }
}
