module {
    llvm.func @vecadd_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(256 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 overflow<nsw, nuw> : i64
      %8 = llvm.add %7, %6 overflow<nsw, nuw> : i64
      %9 = llvm.getelementptr inbounds|nuw %arg2[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %10 = llvm.load %9 : !llvm.ptr -> f16
      %11 = llvm.mul %5, %0 overflow<nsw, nuw> : i64
      %12 = llvm.add %11, %6 overflow<nsw, nuw> : i64
      %13 = llvm.getelementptr inbounds|nuw %arg3[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %14 = llvm.load %13 : !llvm.ptr -> f16
      %15 = llvm.fadd %10, %14 : f16
      %16 = llvm.mul %5, %0 overflow<nsw, nuw> : i64
      %17 = llvm.add %16, %6 overflow<nsw, nuw> : i64
      %18 = llvm.getelementptr inbounds|nuw %arg4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      llvm.store %15, %18 : f16, !llvm.ptr
      llvm.return
    }
}