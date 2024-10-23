#include "AnalysisTransform.h"

#include "support/Logger.h"
#include "support/Util.h"

namespace cusan {
auto get_void_ptr_type(IRBuilder<>& irb) {
#if LLVM_VERSION_MAJOR >= 15
  return irb.getPtrTy();
#else
  return irb.getInt8PtrTy();
#endif
}

namespace analysis {
namespace helper {


bool does_name_match(const std::string& model_kernel_name, llvm::CallBase& cb) {
  assert(cb.getFunction() != nullptr && "Callbase requires function.");
  const auto stub_name      = util::try_demangle_fully(*cb.getFunction());
  const auto searching_name = util::try_demangle_fully(model_kernel_name);

  StringRef searching_without_type{searching_name};
  if (StringRef{stub_name}.contains("lambda")) {
    LOG_DEBUG("Detected lambda function in stub name " << stub_name)
    // if we got a lambda it has a return type included that we want to shave off
    const auto first_space = searching_name.find(' ');
    searching_without_type = llvm::StringRef(searching_name).substr(first_space + 1);
  }

  LOG_DEBUG("Check stub \"" << stub_name << "\" ends with \"" << searching_name << "\" or \"" << searching_without_type
                            << "\"")
  return helper::ends_with_any_of(stub_name, searching_name, searching_without_type);
}
}  // namespace helper

std::optional<CudaKernelInvokeCollector::KernelInvokeData> CudaKernelInvokeCollector::match(llvm::CallBase& cb,
                                                                                            Function& callee) const {
  if (callee.getName() == "cudaLaunchKernel" && helper::does_name_match(model.kernel_name, cb)) {
    // && ends_with(stub_name, searching_name)
    // errs() << "Func:" << stub_name << " " << searching_name << "  == " << (stub_name == searching_name) << "\n";
    // errs() << cb.getFunction()->getName() << "  " << model.kernel_name << "\n" << cb << "\n";

    auto* cu_stream_handle      = std::prev(cb.arg_end())->get();
    auto* void_kernel_arg_array = std::prev(cb.arg_end(), 3)->get();
    auto kernel_args            = extract_kernel_args_for(void_kernel_arg_array);

    return KernelInvokeData{kernel_args, void_kernel_arg_array, cu_stream_handle};
  }
  return std::nullopt;
}

llvm::SmallVector<KernelArgInfo, 4> CudaKernelInvokeCollector::extract_kernel_args_for(
    llvm::Value* void_kernel_arg_array) const {
  unsigned index = 0;

  llvm::SmallVector<Value*, 4> real_args;

  for (auto* array_user : void_kernel_arg_array->users()) {
    if (auto* gep = dyn_cast<GetElementPtrInst>(array_user)) {
      for (auto* gep_user : gep->users()) {
        if (auto* store = dyn_cast<StoreInst>(gep_user)) {
          if (!(index < model.args.size())) {
            LOG_FATAL("In: " << *store->getParent()->getParent())
            LOG_FATAL("Out of bounds for model args: " << index << " vs. " << model.args.size());
            assert(false && "Encountered out of bounds access");
          }
          if (auto* cast = dyn_cast<BitCastInst>(store->getValueOperand())) {
            real_args.push_back(*cast->operand_values().begin());
          } else {
            real_args.push_back(*store->operand_values().begin());
          }
          index++;
        }
      }
    }
  }

  llvm::SmallVector<KernelArgInfo, 4> result = model.args;
  for (auto& res : result) {
    Value* val = real_args[real_args.size() - 1 - res.arg_pos];
    // because of ABI? clang might convert struct argument to a (byval)pointer
    // but the actual cuda argument is just a value. So we double check that it actually allocates a pointer
    bool real_ptr = false;
    if (auto* as_alloca = dyn_cast<AllocaInst>(val)) {
      real_ptr = res.is_pointer && as_alloca->getAllocatedType()->isPointerTy();
    }

    // not fake pointer from clang so load it before getting subargs
    for (auto& sub_arg : res.subargs) {
      if (real_ptr) {
        sub_arg.indices.insert(sub_arg.indices.begin(), FunctionSubArg::SubIndex{});
      }
      sub_arg.value = val;
    }
    res.value = val;
  }
  return result;
}
}  // namespace analysis
}  // namespace cusan

namespace cusan::transform {

bool KernelInvokeTransformer::transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
  using namespace llvm;
  return generate_compound_cb(data, irb);
}

short KernelInvokeTransformer::access_cast(AccessState access, bool is_ptr) {
  auto value = static_cast<short>(access);
  value <<= 1;
  if (is_ptr) {
    value |= 1;
  }
  return value;
}

llvm::Value* KernelInvokeTransformer::get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data,
                                                        IRBuilder<>& irb) {
  auto* cu_stream = data.cu_stream;
  assert(cu_stream != nullptr && "Require cuda stream!");
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(cu_stream, get_void_ptr_type(irb));
  return cu_stream_void_ptr;
}

bool KernelInvokeTransformer::generate_compound_cb(const analysis::CudaKernelInvokeCollector::Data& data,
                                                   IRBuilder<>& irb) const {
  const bool should_transform =
      llvm::count_if(data.args, [&](const auto& elem) {
        return llvm::count_if(elem.subargs, [&](const auto& sub_elem) { return sub_elem.is_pointer; }) > 0;
      }) > 0;

  uint32_t n_subargs = 0;
  for (const auto& arg : data.args) {
    n_subargs += arg.subargs.size();
  }

  if (!should_transform) {
    return false;
  }

  auto target_callback = decls_->cusan_register_access;

  auto* i16_ty      = Type::getInt16Ty(irb.getContext());
  auto* i32_ty      = Type::getInt32Ty(irb.getContext());
  auto* void_ptr_ty = get_void_ptr_type(irb);

  auto* cu_stream_void_ptr = get_cu_stream_ptr(data, irb);
  auto* arg_size           = irb.getInt32(n_subargs);
  auto* arg_access_array   = irb.CreateAlloca(i16_ty, arg_size);
  auto* arg_value_array    = irb.CreateAlloca(void_ptr_ty, arg_size);

  size_t arg_array_index = 0;
  for (const auto& arg : data.args) {
    LOG_DEBUG("Handling Arg: " << arg)
    for (const auto& sub_arg : arg.subargs) {
      LOG_DEBUG("   subarg: " << sub_arg)
      const auto access = access_cast(sub_arg.state, sub_arg.is_pointer);
      Value* idx        = ConstantInt::get(i32_ty, arg_array_index);
      Value* acc        = ConstantInt::get(i16_ty, access);
      auto* gep_acc     = irb.CreateGEP(i16_ty, arg_access_array, idx);
      irb.CreateStore(acc, gep_acc);
      // only if it is a pointer store the actual pointer in the value array
      if (sub_arg.is_pointer) {
        assert(arg.value.has_value());
        auto* value_ptr = arg.value.value();

        if (auto* alloca_value = dyn_cast_or_null<AllocaInst>(value_ptr)) {
          auto* subtype = alloca_value->getAllocatedType();

          if (sub_arg.indices.empty()) {
          } else if (sub_arg.indices.size() == 1 && sub_arg.indices[0].is_load) {
            value_ptr = irb.CreateLoad(subtype, value_ptr);
          } else if (sub_arg.indices.size() == 2 && !sub_arg.indices[0].is_load && sub_arg.indices[1].is_load) {
            llvm::SmallVector<Value*> values{llvm::map_range(
                sub_arg.indices[0].gep_indicies, [&irb](auto index) { return (Value*)irb.getInt32(index); })};
            value_ptr = irb.CreateGEP(subtype, value_ptr, values);
#if LLVM_VERSION_MAJOR >= 15
            value_ptr = irb.CreateLoad(void_ptr_ty, value_ptr);
#else
            value_ptr = irb.CreateLoad(value_ptr->getType()->getPointerElementType(), value_ptr);
#endif
          } else {
            LOG_ERROR("Cannot handle this kind of access " << sub_arg)
          }
        }

        auto* voided_ptr    = irb.CreatePointerCast(value_ptr, void_ptr_ty);
        auto* gep_val_array = irb.CreateGEP(void_ptr_ty, arg_value_array, idx);
        irb.CreateStore(voided_ptr, gep_val_array);
        arg_array_index += 1;
      }
    }
  }

  Value* args_cusan_register[] = {arg_value_array, arg_access_array, arg_size, cu_stream_void_ptr};
  irb.CreateCall(target_callback.f, args_cusan_register);
  return true;
}

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
  Value* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {cu_stream_void_ptr};
}

// EventSyncInstrumenter

EventSyncInstrumenter::EventSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventSynchronize", &decls->cusan_sync_event.f);
}
llvm::SmallVector<Value*> EventSyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {cu_event_void_ptr};
}

// EventRecordInstrumenter

EventRecordInstrumenter::EventRecordInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventRecord", &decls->cusan_event_record.f);
}
llvm::SmallVector<Value*> EventRecordInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  return {cu_event_void_ptr, cu_stream_void_ptr};
}

// EventRecordFlagsInstrumenter

EventRecordFlagsInstrumenter::EventRecordFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaEventRecordWithFlags", &decls->cusan_event_record.f);
}
llvm::SmallVector<Value*> EventRecordFlagsInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  return {cu_event_void_ptr, cu_stream_void_ptr};
}

// CudaMemcpyAsyncInstrumenter

CudaMemcpyAsyncInstrumenter::CudaMemcpyAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpyAsync", &decls->cusan_memcpy_async.f);
}
llvm::SmallVector<Value*> CudaMemcpyAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0
  assert(args.size() == 5);
  auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* src_ptr   = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  auto* count     = args[2];
  auto* kind      = args[3];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[4], get_void_ptr_type(irb));
  return {dst_ptr, src_ptr, count, kind, cu_stream};
}

//  CudaMemcpyInstrumenter

CudaMemcpyInstrumenter::CudaMemcpyInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemcpy", &decls->cusan_memcpy.f);
}
llvm::SmallVector<Value*> CudaMemcpyInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, cudaMemcpyKind kind
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* src_ptr = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
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
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* dpitch  = args[1];
  auto* src_ptr = irb.CreateBitOrPointerCast(args[2], get_void_ptr_type(irb));
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
llvm::SmallVector<Value*> CudaMemcpy2DAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind,
  // stream
  assert(args.size() == 8);
  auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* dpitch    = args[1];
  auto* src_ptr   = irb.CreateBitOrPointerCast(args[2], get_void_ptr_type(irb));
  auto* spitch    = args[3];
  auto* width     = args[4];
  auto* height    = args[5];
  auto* kind      = args[6];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[7], get_void_ptr_type(irb));
  return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind, cu_stream};
}

// CudaMemsetAsyncInstrumenter

CudaMemsetAsyncInstrumenter::CudaMemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemsetAsync", &decls->cusan_memset_async.f);
}
llvm::SmallVector<Value*> CudaMemsetAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  // auto* value     = args[1];
  auto* count     = args[2];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[3], get_void_ptr_type(irb));
  return {dst_ptr, count, cu_stream};
}

// CudaMemsetInstrumenter

CudaMemsetInstrumenter::CudaMemsetInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset", &decls->cusan_memset.f);
}
llvm::SmallVector<Value*> CudaMemsetInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count,)
  assert(args.size() == 3);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  // auto* value   = args[1];
  auto* count = args[2];
  return {dst_ptr, count};
}

// CudaMemset2dAsyncInstrumenter

CudaMemset2dAsyncInstrumenter::CudaMemset2dAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset2DAsync", &decls->cusan_memset_2d_async.f);
}
llvm::SmallVector<Value*> CudaMemset2dAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0
  assert(args.size() == 6);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* pitch   = args[1];
  // auto* value     = args[2];
  auto* height    = args[3];
  auto* width     = args[4];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[5], get_void_ptr_type(irb));
  return {dst_ptr, pitch, height, width, cu_stream};
}

// CudaMemset2dInstrumenter

CudaMemset2dInstrumenter::CudaMemset2dInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaMemset2D", &decls->cusan_memset_2d.f);
}
llvm::SmallVector<Value*> CudaMemset2dInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* devPtr, size_t pitch, int  value, size_t width, size_t height
  assert(args.size() == 5);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  // auto* cu_event_void_ptr = irb.CreateLoad(get_void_ptr_type(irb), args[0], "");
  auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {cu_event_void_ptr_ptr};
}

// StreamCreateInstrumenter

StreamCreateInstrumenter::StreamCreateInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamCreate", &decls->cusan_stream_create.f);
}
llvm::SmallVector<Value*> StreamCreateInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* flags                  = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {cu_stream_void_ptr_ptr, flags};
}

// StreamCreateWithFlagsInstrumenter

StreamCreateWithFlagsInstrumenter::StreamCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamCreateWithFlags", &decls->cusan_stream_create.f);
}

llvm::SmallVector<Value*> StreamCreateWithFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                           llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* flags                  = args[1];
  return {cu_stream_void_ptr_ptr, flags};
}

// StreamWaitEventInstrumenter

StreamWaitEventInstrumenter::StreamWaitEventInstrumenter(callback::FunctionDecl* decls) {
  setup("cudaStreamWaitEvent", &decls->cusan_stream_wait_event.f);
}
llvm::SmallVector<Value*> StreamWaitEventInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  // auto* cu_stream_void_ptr = irb.CreateLoad(get_void_ptr_type(irb), args[0], "");
  auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  return {cu_stream_void_ptr, cu_event_void_ptr, args[2]};
}

// CudaHostRegister

CudaHostRegister::CudaHostRegister(callback::FunctionDecl* decls) {
  setup("cudaHostRegister", &decls->cusan_host_register.f);
}
llvm::SmallVector<Value*> CudaHostRegister::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 3);
  auto* ptr   = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}

// CudaHostFree

CudaHostFree::CudaHostFree(callback::FunctionDecl* decls) {
  setup("cudaFreeHost", &decls->cusan_host_free.f);
}
llvm::SmallVector<Value*> CudaHostFree::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}

// CudaMallocManaged

CudaMallocManaged::CudaMallocManaged(callback::FunctionDecl* decls) {
  setup("cudaMallocManaged", &decls->cusan_managed_alloc.f);
}
llvm::SmallVector<Value*> CudaMallocManaged::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr, size_t size, u32 flags)
  assert(args.size() == 3);
  auto* ptr   = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* ptr  = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}

// CudaMallocPitch

CudaMallocPitch::CudaMallocPitch(callback::FunctionDecl* decls) {
  setup("cudaMallocPitch", &decls->cusan_device_alloc.f);
}
llvm::SmallVector<Value*> CudaMallocPitch::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //(void** devPtr, size_t* pitch, size_t width, size_t height )
  assert(args.size() == 4);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));

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
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
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
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}
llvm::SmallVector<Value*, 1> CudaEventQuery::map_return_value(IRBuilder<>& irb, Value* result) {
  (void)irb;
  return {result};
}

}  // namespace cusan::transform
