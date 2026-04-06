; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @vecadd_kernel(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5) {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = sext i32 %7 to i64
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %10 = sext i32 %9 to i64
  %11 = add i64 %0, %8
  %12 = mul i64 %1, %10
  %13 = add i64 %2, %12
  %14 = mul i64 %11, 1024
  %15 = add i64 %14, %13
  %16 = getelementptr i32, ptr %3, i64 %15
  %17 = load <4 x i32>, ptr %16, align 4
  %18 = mul i64 %11, 1024
  %19 = add i64 %18, %13
  %20 = getelementptr i32, ptr %4, i64 %19
  %21 = load <4 x i32>, ptr %20, align 4
  %22 = bitcast <4 x i32> %17 to <8 x half>
  %23 = bitcast <4 x i32> %21 to <8 x half>
  %24 = fadd <8 x half> %22, %23
  %25 = bitcast <8 x half> %24 to <4 x i32>
  %26 = mul i64 %11, 1024
  %27 = add i64 %26, %13
  %28 = getelementptr i32, ptr %5, i64 %27
  store <4 x i32> %25, ptr %28, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
