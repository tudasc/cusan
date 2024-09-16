#include <AnalysisTransform.h>

namespace cusan::transform {
// DeviceSyncInstrumenter

DeviceSyncInstrumenter::DeviceSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaDeviceSynchronize", &decls->cusan_sync_device.f);
}
llvm::SmallVector<Value*> DeviceSyncInstrumenter::map_arguments(IRBuilder<>&, llvm::ArrayRef<Value*>) {
  return {};
}

// StreamSyncInstrumenter

StreamSyncInstrumenter::StreamSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamSynchronize", &decls->cusan_sync_stream.f);
}
llvm::SmallVector<Value*> StreamSyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  Value* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {cu_stream_void_ptr};
}

// EventSyncInstrumenter

EventSyncInstrumenter::EventSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventSynchronize", &decls->cusan_sync_event.f);
}
llvm::SmallVector<Value*> EventSyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {cu_event_void_ptr};
}

// EventRecordInstrumenter

EventRecordInstrumenter::EventRecordInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventRecord", &decls->cusan_event_record.f);
}
llvm::SmallVector<Value*> EventRecordInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
  return {cu_event_void_ptr, cu_stream_void_ptr};
}

// EventRecordFlagsInstrumenter

EventRecordFlagsInstrumenter::EventRecordFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventRecordWithFlags", &decls->cusan_event_record.f);
}
llvm::SmallVector<Value*> EventRecordFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                         llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
  return {cu_event_void_ptr, cu_stream_void_ptr};
}

// CudaMemcpyAsyncInstrumenter

CudaMemcpyAsyncInstrumenter::CudaMemcpyAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpyAsync", &decls->cusan_memcpy_async.f);
}
llvm::SmallVector<Value*> CudaMemcpyAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0
  assert(args.size() == 5);
  auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* src_ptr   = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
  auto* count     = args[2];
  auto* kind      = args[3];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[4], irb.getInt8PtrTy());
  return {dst_ptr, src_ptr, count, kind, cu_stream};
}

//  CudaMemcpyInstrumenter

CudaMemcpyInstrumenter::CudaMemcpyInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpy", &decls->cusan_memcpy.f);
}
llvm::SmallVector<Value*> CudaMemcpyInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, cudaMemcpyKind kind
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* src_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
  auto* count   = args[2];
  auto* kind    = args[3];
  return {dst_ptr, src_ptr, count, kind};
}

//  CudaMemcpy2DInstrumenter

CudaMemcpy2DInstrumenter::CudaMemcpy2DInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpy2D", &decls->cusan_memcpy_2d.f);
}
llvm::SmallVector<Value*> CudaMemcpy2DInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind
  assert(args.size() == 7);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* dpitch  = args[1];
  auto* src_ptr = irb.CreateBitOrPointerCast(args[2], irb.getInt8PtrTy());
  auto* spitch  = args[3];
  auto* width   = args[4];
  auto* height  = args[5];
  auto* kind    = args[6];
  return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind};
}

// CudaMemcpy2DAsyncInstrumenter

CudaMemcpy2DAsyncInstrumenter::CudaMemcpy2DAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpy2DAsync", &decls->cusan_memcpy_2d_async.f);
}
llvm::SmallVector<Value*> CudaMemcpy2DAsyncInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                          llvm::ArrayRef<Value*> args) {
  // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind,
  // stream
  assert(args.size() == 8);
  auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* dpitch    = args[1];
  auto* src_ptr   = irb.CreateBitOrPointerCast(args[2], irb.getInt8PtrTy());
  auto* spitch    = args[3];
  auto* width     = args[4];
  auto* height    = args[5];
  auto* kind      = args[6];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[7], irb.getInt8PtrTy());
  return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind, cu_stream};
}

// CudaMemsetAsyncInstrumenter

CudaMemsetAsyncInstrumenter::CudaMemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemsetAsync", &decls->cusan_memset_async.f);
}
llvm::SmallVector<Value*> CudaMemsetAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  // auto* value     = args[1];
  auto* count     = args[2];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[3], irb.getInt8PtrTy());
  return {dst_ptr, count, cu_stream};
}

// CudaMemsetInstrumenter

CudaMemsetInstrumenter::CudaMemsetInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset", &decls->cusan_memset.f);
}
llvm::SmallVector<Value*> CudaMemsetInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count,)
  assert(args.size() == 3);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  // auto* value   = args[1];
  auto* count = args[2];
  return {dst_ptr, count};
}

// CudaMemset2dAsyncInstrumenter

CudaMemset2dAsyncInstrumenter::CudaMemset2dAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset2DAsync", &decls->cusan_memset_2d_async.f);
}
llvm::SmallVector<Value*> CudaMemset2dAsyncInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                          llvm::ArrayRef<Value*> args) {
  // void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0
  assert(args.size() == 6);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* pitch   = args[1];
  // auto* value     = args[2];
  auto* height    = args[3];
  auto* width     = args[4];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[5], irb.getInt8PtrTy());
  return {dst_ptr, pitch, height, width, cu_stream};
}

// CudaMemset2dInstrumenter

CudaMemset2dInstrumenter::CudaMemset2dInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset2D", &decls->cusan_memset_2d.f);
}
llvm::SmallVector<Value*> CudaMemset2dInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* devPtr, size_t pitch, int  value, size_t width, size_t height
  assert(args.size() == 5);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* pitch   = args[1];
  // auto* value   = args[2];
  auto* height = args[3];
  auto* width  = args[4];
  ;
  return {dst_ptr, pitch, height, width};
}

// CudaHostAlloc

CudaHostAlloc::CudaHostAlloc(callback::FunctionDecl* decls) {
  setup("cudaHostAlloc", &decls->cusan_host_alloc.f);
}
llvm::SmallVector<Value*> CudaHostAlloc::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void** ptr, size_t size, unsigned int flags )
  assert(args.size() == 3);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* size    = args[1];
  auto* flags   = args[2];
  return {dst_ptr, size, flags};
}

// CudaMallocHost

CudaMallocHost::CudaMallocHost(callback::FunctionDecl* decls) {
  setup("cudaMallocHost", &decls->cusan_host_alloc.f);
}
llvm::SmallVector<Value*> CudaMallocHost::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void** ptr, size_t size)
  assert(args.size() == 2);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* size    = args[1];
  auto* flags   = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
  return {dst_ptr, size, flags};
}

// CudaEventCreateInstrumenter

CudaEventCreateInstrumenter::CudaEventCreateInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventCreate", &decls->cusan_event_create.f);
}
llvm::SmallVector<Value*> CudaEventCreateInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  // auto* cu_event_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
  auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {cu_event_void_ptr_ptr};
}

// CudaEventCreateWithFlagsInstrumenter

CudaEventCreateWithFlagsInstrumenter::CudaEventCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventCreateWithFlags", &decls->cusan_event_create.f);
}
llvm::SmallVector<Value*> CudaEventCreateWithFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                                 llvm::ArrayRef<Value*> args) {
  // cudaEvent_t* event, unsigned int  flags
  assert(args.size() == 2);
  auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {cu_event_void_ptr_ptr};
}

// StreamCreateInstrumenter

StreamCreateInstrumenter::StreamCreateInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamCreate", &decls->cusan_stream_create.f);
}
llvm::SmallVector<Value*> StreamCreateInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* flags                  = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {cu_stream_void_ptr_ptr, flags};
}

// StreamCreateWithFlagsInstrumenter

StreamCreateWithFlagsInstrumenter::StreamCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamCreateWithFlags", &decls->cusan_stream_create.f);
}

llvm::SmallVector<Value*> StreamCreateWithFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                              llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* flags                  = args[1];
  return {cu_stream_void_ptr_ptr, flags};
}

// StreamCreateWithPriorityInstrumenter

StreamCreateWithPriorityInstrumenter::StreamCreateWithPriorityInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamCreateWithPriority", &decls->cusan_stream_create.f);
}

llvm::SmallVector<Value*> StreamCreateWithPriorityInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                                 llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* flags                  = args[1];
  return {cu_stream_void_ptr_ptr, flags};
}

// StreamWaitEventInstrumenter

StreamWaitEventInstrumenter::StreamWaitEventInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamWaitEvent", &decls->cusan_stream_wait_event.f);
}
llvm::SmallVector<Value*> StreamWaitEventInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  // auto* cu_stream_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
  return {cu_stream_void_ptr, cu_event_void_ptr, args[2]};
}

// CudaHostRegister

CudaHostRegister::CudaHostRegister(callback::FunctionDecl* decls) {
  setup("cudaHostRegister", &decls->cusan_host_register.f);
}
llvm::SmallVector<Value*> CudaHostRegister::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 3);
  auto* ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* size  = args[1];
  auto* flags = args[2];
  return {ptr, size, flags};
}

// CudaHostUnregister

CudaHostUnregister::CudaHostUnregister(callback::FunctionDecl* decls) {
  setup("cudaHostUnregister", &decls->cusan_host_unregister.f);
}
llvm::SmallVector<Value*> CudaHostUnregister::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {ptr};
}

// CudaHostFree

CudaHostFree::CudaHostFree(callback::FunctionDecl* decls) {
  setup("cudaFreeHost", &decls->cusan_host_free.f);
}
llvm::SmallVector<Value*> CudaHostFree::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {ptr};
}

// CudaMallocManaged

CudaMallocManaged::CudaMallocManaged(callback::FunctionDecl* decls) {
  setup("cudaMallocManaged", &decls->cusan_managed_alloc.f);
}
llvm::SmallVector<Value*> CudaMallocManaged::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr, size_t size, u32 flags)
  assert(args.size() == 3);
  auto* ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* size  = args[1];
  auto* flags = args[2];
  return {ptr, size, flags};
}

// CudaMalloc

CudaMalloc::CudaMalloc(callback::FunctionDecl* decls) {
  setup("cudaMalloc", &decls->cusan_device_alloc.f);
}
llvm::SmallVector<Value*> CudaMalloc::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr, size_t size)
  assert(args.size() == 2);
  auto* ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  auto* size = args[1];
  return {ptr, size};
}

// CudaFree

CudaFree::CudaFree(callback::FunctionDecl* decls) {
  setup("cudaFree", &decls->cusan_device_free.f);
}
llvm::SmallVector<Value*> CudaFree::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {ptr};
}

// CudaMallocPitch

CudaMallocPitch::CudaMallocPitch(callback::FunctionDecl* decls) {
  setup("cudaMallocPitch", &decls->cusan_device_alloc.f);
}
llvm::SmallVector<Value*> CudaMallocPitch::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //(void** devPtr, size_t* pitch, size_t width, size_t height )
  assert(args.size() == 4);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());

  //"The function may pad the allocation"
  //"*pitch by cudaMallocPitch() is the width in bytes of the allocation"
  auto* pitch = irb.CreateLoad(irb.getIntPtrTy(irb.GetInsertBlock()->getModule()->getDataLayout()), args[1]);
  // auto* width = args[2];
  auto* height = args[3];

  auto* real_size = irb.CreateMul(pitch, height);
  return {ptr, real_size};
}

// CudaStreamQuery

CudaStreamQuery::CudaStreamQuery(callback::FunctionDecl* decls) {
  setup("cudaStreamQuery", &decls->cusan_stream_query.f);
}
llvm::SmallVector<Value*> CudaStreamQuery::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* stream)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {ptr};
}
llvm::SmallVector<Value*, 1> CudaStreamQuery::map_return_value(IRBuilder<>& irb, Value* result) {
  (void)irb;
  return {result};
}

// CudaEventQuery

CudaEventQuery::CudaEventQuery(callback::FunctionDecl* decls) {
  setup("cudaEventQuery", &decls->cusan_event_query.f);
}
llvm::SmallVector<Value*> CudaEventQuery::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* event)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
  return {ptr};
}
llvm::SmallVector<Value*, 1> CudaEventQuery::map_return_value(IRBuilder<>& irb, Value* result) {
  (void)irb;
  return {result};
}

}  // namespace cusan::transform
