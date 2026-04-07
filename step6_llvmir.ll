; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define ptx_kernel void @vecadd_kernel(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4) {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = sext i32 %6 to i64
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %9 = sext i32 %8 to i64
  %10 = add i64 %0, %7
  %11 = add i64 %1, %9
  %12 = mul nuw nsw i64 %10, 256
  %13 = add nuw nsw i64 %12, %11
  %14 = getelementptr inbounds nuw half, ptr %2, i64 %13
  %15 = load half, ptr %14, align 2
  %16 = mul nuw nsw i64 %10, 256
  %17 = add nuw nsw i64 %16, %11
  %18 = getelementptr inbounds nuw half, ptr %3, i64 %17
  %19 = load half, ptr %18, align 2
  %20 = fadd half %15, %19
  %21 = mul nuw nsw i64 %10, 256
  %22 = add nuw nsw i64 %21, %11
  %23 = getelementptr inbounds nuw half, ptr %4, i64 %22
  store half %20, ptr %23, align 2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
