#include "AnalysisTransform.h"

namespace cusan {

namespace analysis {

std::optional<HipKernelInvokeCollector::KernelInvokeData> HipKernelInvokeCollector::match(llvm::CallBase& cb,
                                                                                          Function& callee) const {
  if (callee.getName() == "hipLaunchKernel" && helper::does_name_match(model.kernel_name, cb)) {
    // && ends_with(stub_name, searching_name)
    // errs() << "Func:" << stub_name << " " << searching_name << "  == " << (stub_name == searching_name) << "\n";
    // errs() << cb.getFunction()->getName() << "  " << model.kernel_name << "\n" << cb << "\n";

    auto* hip_stream_handle     = std::prev(cb.arg_end())->get();
    auto* void_kernel_arg_array = std::prev(cb.arg_end(), 3)->get();
    auto kernel_args            = extract_kernel_args_for(void_kernel_arg_array);

    return KernelInvokeData{kernel_args, void_kernel_arg_array, hip_stream_handle};
  }
  return std::nullopt;
}

llvm::SmallVector<KernelArgInfo, 4> HipKernelInvokeCollector::extract_kernel_args_for(
    llvm::Value* void_kernel_arg_array) const {
  unsigned index = 0;

  llvm::SmallVector<Value*, 4> real_args;

  LOG_DEBUG("Kernel argument array: " << *void_kernel_arg_array);
  auto process_store = [&](StoreInst* store) {
    if (!(index < model.args.size())) {
      LOG_FATAL("In: " << *store->getParent()->getParent());
      LOG_FATAL("Out of bounds for model args: " << index << " vs. " << model.args.size());
      assert(false && "Encountered out of bounds access");
    }
    if (auto* cast = dyn_cast<BitCastInst>(store->getValueOperand())) {
      LOG_DEBUG("Found bcast " << *cast)
      real_args.push_back(*cast->operand_values().begin());
    } else {
      LOG_DEBUG("Found store " << *store)
      real_args.push_back(*store->operand_values().begin());
    }
    index++;
  };
  for (auto* array_user : void_kernel_arg_array->users()) {
    if (auto* gep = dyn_cast<GetElementPtrInst>(array_user)) {
      for (auto* gep_user : gep->users()) {
        if (auto* store = dyn_cast<StoreInst>(gep_user)) {
          process_store(store);
        }
      }
    }
    if (auto* store = dyn_cast<StoreInst>(array_user)) {
      process_store(store);
    }
  }

  llvm::SmallVector<KernelArgInfo, 4> result = model.args;
  // if (!real_args.empty()) {
  for (auto& res : result) {
    // Value* val = real_args[real_args.size() - 1 - res.arg_pos];
    Value* val = real_args[real_args.size() - 1 - res.arg_pos];
    LOG_DEBUG("Real argument: " << *val)
    // because of ABI? clang might convert struct argument to a (byval)pointer
    // but the actual hip argument is just a value. So we double check that it actually allocates a pointer
    bool real_ptr = false;
    if (auto* as_alloca = dyn_cast<AllocaInst>(val)) {
      real_ptr = res.is_pointer && as_alloca->getAllocatedType()->isPointerTy();
    }

    // not fake pointer from clang so load it before getting subargs
    for (auto& sub_arg : res.subargs) {
      if (real_ptr) {
        sub_arg.does_load = true;
        sub_arg.gep_indicies.clear();
      }
      sub_arg.value = val;
    }
    res.value = val;
  }
  // }
  return result;
}
}  // namespace analysis

namespace transform {

bool HipKernelInvokeTransformer::transform(const analysis::HipKernelInvokeCollector::Data& data,
                                           IRBuilder<>& irb) const {
  using namespace llvm;
  return generate_compound_cb(data, irb);
}

short HipKernelInvokeTransformer::access_cast(AccessState access, bool is_ptr) {
  auto value = static_cast<short>(access);
  value <<= 1;
  if (is_ptr) {
    value |= 1;
  }
  return value;
}

llvm::Value* HipKernelInvokeTransformer::get_hip_stream_ptr(const analysis::HipKernelInvokeCollector::Data& data,
                                                            IRBuilder<>& irb) {
  auto* hip_stream = data.hip_stream;
  assert(hip_stream != nullptr && "Require Hip stream!");
  auto* hip_stream_void_ptr = irb.CreateBitOrPointerCast(hip_stream, get_void_ptr_type(irb));
  return hip_stream_void_ptr;
}

bool HipKernelInvokeTransformer::generate_compound_cb(const analysis::HipKernelInvokeCollector::Data& data,
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

  auto* hip_stream_void_ptr = get_hip_stream_ptr(data, irb);
  auto* arg_size            = irb.getInt32(n_subargs);
  auto* arg_access_array    = irb.CreateAlloca(i16_ty, arg_size);
  auto* arg_value_array     = irb.CreateAlloca(void_ptr_ty, arg_size);

  size_t arg_array_index = 0;
  for (const auto& arg : data.args) {
    LOG_TRACE("Handling Arg: " << arg)
    for (const auto& sub_arg : arg.subargs) {
      LOG_TRACE("   subarg: " << sub_arg)
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

          if (!sub_arg.gep_indicies.empty()) {
            llvm::SmallVector<Value*> values{
                llvm::map_range(sub_arg.gep_indicies, [&irb](auto index) { return (Value*)irb.getInt32(index); })};
            value_ptr = irb.CreateGEP(subtype, value_ptr, values);
#if LLVM_VERSION_MAJOR >= 15
            subtype = void_ptr_ty;
#else
            subtype = value_ptr->getType()->getPointerElementType();
#endif
          }

          if (sub_arg.does_load) {
            value_ptr = irb.CreateLoad(subtype, value_ptr);
          }
        }

        auto* voided_ptr    = irb.CreatePointerCast(value_ptr, void_ptr_ty);
        auto* gep_val_array = irb.CreateGEP(void_ptr_ty, arg_value_array, idx);
        irb.CreateStore(voided_ptr, gep_val_array);
      }
      arg_array_index += 1;
    }
  }

  Value* args_cusan_register[] = {arg_value_array, arg_access_array, arg_size, hip_stream_void_ptr};
  irb.CreateCall(target_callback.f, args_cusan_register);
  return true;
}

// HipMemcpyAsyncInstrumenter

HipMemcpyAsyncInstrumenter::HipMemcpyAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemcpyAsync", &decls->cusan_memcpy_async.f);
}
llvm::SmallVector<Value*> HipMemcpyAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream = 0
  assert(args.size() == 5);
  auto* dst_ptr    = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* src_ptr    = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  auto* count      = args[2];
  auto* kind       = args[3];
  auto* hip_stream = irb.CreateBitOrPointerCast(args[4], get_void_ptr_type(irb));
  return {dst_ptr, src_ptr, count, kind, hip_stream};
}

//  HIPMemcpyInstrumenter

HipMemcpyInstrumenter::HipMemcpyInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemcpy", &decls->cusan_memcpy.f);
}
llvm::SmallVector<Value*> HipMemcpyInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* dst, const void* src, size_t count, hipMemcpyKind kind
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* src_ptr = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  auto* count   = args[2];
  auto* kind    = args[3];
  return {dst_ptr, src_ptr, count, kind, irb.getInt8(0)};
}

// hipMalloc

HipMalloc::HipMalloc(callback::FunctionDecl* decls) {
  setup("hipMalloc", &decls->cusan_device_alloc.f);
}
llvm::SmallVector<Value*> HipMalloc::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr, size_t size)
  assert(args.size() == 2);
  auto* ptr  = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* size = args[1];
  return {ptr, size};
}

// hipFree

HipFree::HipFree(callback::FunctionDecl* decls) {
  setup("hipFree", &decls->cusan_device_free.f);
}
llvm::SmallVector<Value*> HipFree::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}

// hipMallocManaged

HipMallocManaged::HipMallocManaged(callback::FunctionDecl* decls) {
  setup("hipMallocManaged", &decls->cusan_managed_alloc.f);
}
llvm::SmallVector<Value*> HipMallocManaged::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* ptr, size_t size, u32 flags)
  assert(args.size() == 3);
  auto* ptr   = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* size  = args[1];
  auto* flags = args[2];
  return {ptr, size, flags};
}

// HipEventCreateInstrumenter

HipEventCreateInstrumenter::HipEventCreateInstrumenter(callback::FunctionDecl* decls) {
  setup("hipEventCreate", &decls->cusan_event_create.f);
}
llvm::SmallVector<Value*> HipEventCreateInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  // auto* hip_event_void_ptr = irb.CreateLoad(get_void_ptr_type(irb), args[0], "");
  auto* hip_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {hip_event_void_ptr_ptr};
}

// HipEventCreateWithFlagsInstrumenter

HipEventCreateWithFlagsInstrumenter::HipEventCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("hipEventCreateWithFlags", &decls->cusan_event_create.f);
}
llvm::SmallVector<Value*> HipEventCreateWithFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                             llvm::ArrayRef<Value*> args) {
  // hipEvent_t* event, unsigned int  flags
  assert(args.size() == 2);
  auto* hip_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {hip_event_void_ptr_ptr};
}

// HipEventSyncInstrumenter

HipEventSyncInstrumenter::HipEventSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipEventSynchronize", &decls->cusan_sync_event.f);
}
llvm::SmallVector<Value*> HipEventSyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* hip_event_void_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {hip_event_void_ptr};
}

// HipEventRecordInstrumenter

HipEventRecordInstrumenter::HipEventRecordInstrumenter(callback::FunctionDecl* decls) {
  setup("hipEventRecord", &decls->cusan_event_record.f);
}
llvm::SmallVector<Value*> HipEventRecordInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* hip_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* hip_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], get_void_ptr_type(irb));
  return {hip_event_void_ptr, hip_stream_void_ptr};
}

// HipStreamCreateInstrumenter

HipStreamCreateInstrumenter::HipStreamCreateInstrumenter(callback::FunctionDecl* decls) {
  setup("hipStreamCreate", &decls->cusan_stream_create.f);
}
llvm::SmallVector<Value*> HipStreamCreateInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  auto* flags                   = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
  auto* hip_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {hip_stream_void_ptr_ptr, flags};
}

// HipStreamCreateWithFlagsInstrumenter

HipStreamCreateWithFlagsInstrumenter::HipStreamCreateWithFlagsInstrumenter(callback::FunctionDecl* decls) {
  setup("hipStreamCreateWithFlags", &decls->cusan_stream_create.f);
}

llvm::SmallVector<Value*> HipStreamCreateWithFlagsInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                              llvm::ArrayRef<Value*> args) {
  assert(args.size() == 2);
  auto* hip_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* flags                   = args[1];
  return {hip_stream_void_ptr_ptr, flags};
}

// HipMemsetInstrumenter

HipMemsetInstrumenter::HipMemsetInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemset", &decls->cusan_memset.f);
}
llvm::SmallVector<Value*> HipMemsetInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count,)
  assert(args.size() == 3);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  // auto* value   = args[1];
  auto* count = args[2];
  return {dst_ptr, count};
}

// HipStreamSyncInstrumenter

HipStreamSyncInstrumenter::HipStreamSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipStreamSynchronize", &decls->cusan_sync_stream.f);
}
llvm::SmallVector<Value*> HipStreamSyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  assert(args.size() == 1);
  Value* hip_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {hip_stream_void_ptr};
}

// HipDeviceSyncInstrumenter

HipDeviceSyncInstrumenter::HipDeviceSyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipDeviceSynchronize", &decls->cusan_sync_device.f);
}
llvm::SmallVector<Value*> HipDeviceSyncInstrumenter::map_arguments(IRBuilder<>&, llvm::ArrayRef<Value*>) {
  return {};
}

// HipStreamQuery

HipStreamQuery::HipStreamQuery(callback::FunctionDecl* decls) {
  setup("hipStreamQuery", &decls->cusan_stream_query.f);
}
llvm::SmallVector<Value*> HipStreamQuery::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* stream)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}
llvm::SmallVector<Value*, 1> HipStreamQuery::map_return_value(IRBuilder<>& irb, Value* result) {
  (void)irb;
  return {result};
}

// HipEventQuery

HipEventQuery::HipEventQuery(callback::FunctionDecl* decls) {
  setup("hipEventQuery", &decls->cusan_event_query.f);
}
llvm::SmallVector<Value*> HipEventQuery::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* event)
  assert(args.size() == 1);
  auto* ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  return {ptr};
}
llvm::SmallVector<Value*, 1> HipEventQuery::map_return_value(IRBuilder<>& irb, Value* result) {
  (void)irb;
  return {result};
}


//  HipMemcpy2DInstrumenter

HipMemcpy2DInstrumenter::HipMemcpy2DInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemcpy2D", &decls->cusan_memcpy_2d.f);
}
llvm::SmallVector<Value*> HipMemcpy2DInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* target, size_t dpitch, const void* from, size_t spitch, size_t width, size_t height, cusan_MemcpyKind kind
  assert(args.size() == 7);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* dpitch  = args[1];
  auto* src_ptr = irb.CreateBitOrPointerCast(args[2], get_void_ptr_type(irb));
  auto* spitch  = args[3];
  auto* width   = args[4];
  auto* height  = args[5];
  auto* kind    = args[6];
  return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind, irb.getInt8(1)};
}

// HipMemcpy2DAsyncInstrumenter

HipMemcpy2DAsyncInstrumenter::HipMemcpy2DAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemcpy2DAsync", &decls->cusan_memcpy_2d_async.f);
}
llvm::SmallVector<Value*> HipMemcpy2DAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
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
  auto* hip_stream = irb.CreateBitOrPointerCast(args[7], get_void_ptr_type(irb));
  return {dst_ptr, dpitch, src_ptr, spitch, width, height, kind, hip_stream};
}


// HipStreamCreateWithPriorityInstrumenter

HipStreamCreateWithPriorityInstrumenter::HipStreamCreateWithPriorityInstrumenter(callback::FunctionDecl* decls) {
  setup("hipStreamCreateWithPriority", &decls->cusan_stream_create.f);
}

llvm::SmallVector<Value*> HipStreamCreateWithPriorityInstrumenter::map_arguments(IRBuilder<>& irb,
                                                                              llvm::ArrayRef<Value*> args) {
  assert(args.size() == 3);
  auto* hip_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* flags                  = args[1];
  return {hip_stream_void_ptr_ptr, flags};
}





// HipMemsetAsyncInstrumenter

HipMemsetAsyncInstrumenter::HipMemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemsetAsync", &decls->cusan_memset_async.f);
}
llvm::SmallVector<Value*> HipMemsetAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  //( void* devPtr, int  value, size_t count, hipStream_t stream = 0 )
  assert(args.size() == 4);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  // auto* value     = args[1];
  auto* count     = args[2];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[3], get_void_ptr_type(irb));
  return {dst_ptr, count, cu_stream};
}

// HipMemset2dAsyncInstrumenter

HipMemset2dAsyncInstrumenter::HipMemset2dAsyncInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemset2DAsync", &decls->cusan_memset_2d_async.f);
}
llvm::SmallVector<Value*> HipMemset2dAsyncInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
  // void* devPtr, size_t pitch, int  value, size_t width, size_t height, hipStream_t stream = 0
  assert(args.size() == 6);
  auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], get_void_ptr_type(irb));
  auto* pitch   = args[1];
  // auto* value     = args[2];
  auto* height    = args[3];
  auto* width     = args[4];
  auto* cu_stream = irb.CreateBitOrPointerCast(args[5], get_void_ptr_type(irb));
  return {dst_ptr, pitch, height, width, cu_stream};
}

// HipMemset2dInstrumenter

HipMemset2dInstrumenter::HipMemset2dInstrumenter(callback::FunctionDecl* decls) {
  setup("hipMemset2D", &decls->cusan_memset_2d.f);
}
llvm::SmallVector<Value*> HipMemset2dInstrumenter::map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
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



}  // namespace transform
}  // namespace cusan