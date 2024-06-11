//
// Created by ahueck on 05.07.23.
//

#include "KernelAnalysis.h"

#include "support/CudaUtil.h"
#include "support/Logger.h"
#include "support/Util.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/IPO/Attributor.h>
#include <utility>

namespace cucorr {

namespace device {

// stolen and modified from clang19 https://llvm.org/doxygen/FunctionAttrs_8cpp_source.html#l00611
static llvm::Attribute::AttrKind determinePointerAccessAttrs(llvm::Value* value) {
  using namespace llvm;
  SmallVector<Use*, 32> worklist;
  SmallPtrSet<Use*, 32> visited;

  bool is_read  = false;
  bool is_write = false;

  for (Use& u : value->uses()) {
    visited.insert(&u);
    worklist.push_back(&u);
  }

  while (!worklist.empty()) {
    if (is_write && is_read)
      // No point in searching further..
      return Attribute::None;

    Use* u         = worklist.pop_back_val();
    Instruction* i = cast<Instruction>(u->getUser());

    switch (i->getOpcode()) {
      case Instruction::BitCast:
      case Instruction::GetElementPtr:
      case Instruction::PHI:
      case Instruction::Select:
      case Instruction::AddrSpaceCast:
        // The original value is not read/written via this if the new value isn't.
        for (Use& uu : i->uses())
          if (visited.insert(&uu).second)
            worklist.push_back(&uu);
        break;

      case Instruction::Call:
      case Instruction::Invoke: {
        auto& cb = cast<CallBase>(*i);
        if (cb.isCallee(u)) {
          is_read = true;
          // Note that indirect calls do not capture, see comment in
          // CaptureTracking for context
          continue;
        }

        // Given we've explictily handled the callee operand above, what's left
        // must be a data operand (e.g. argument or operand bundle)
        const unsigned use_index = cb.getDataOperandNo(u);

        // Some intrinsics (for instance ptrmask) do not capture their results,
        // but return results thas alias their pointer argument, and thus should
        // be handled like GEP or addrspacecast above.
        if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(&cb, /*MustPreserveNullness=*/false)) {
          for (Use& uu : cb.uses())
            if (visited.insert(&uu).second)
              worklist.push_back(&uu);
        } else if (!cb.doesNotCapture(use_index)) {
          if (!cb.onlyReadsMemory())
            // If the callee can save a copy into other memory, then simply
            // scanning uses of the call is insufficient.  We have no way
            // of tracking copies of the pointer through memory to see
            // if a reloaded copy is written to, thus we must give up.
            return Attribute::None;
          // Push users for processing once we finish this one
          if (!i->getType()->isVoidTy())
            for (Use& UU : i->uses())
              if (visited.insert(&UU).second)
                worklist.push_back(&UU);
        }

        // The accessors used on call site here do the right thing for calls and
        // invokes with operand bundles.
        if (cb.doesNotAccessMemory(use_index)) {
          /* nop */
        } else if (cb.onlyReadsMemory() || cb.onlyReadsMemory(use_index)) {
          is_read = true;
        } else if (cb.dataOperandHasImpliedAttr(use_index, Attribute::WriteOnly)) {
          is_write = true;
        } else {
          return Attribute::None;
        }
        break;
      }

      case Instruction::Load:
        // A volatile load has side effects beyond what readonly can be relied
        // upon.
        if (cast<LoadInst>(i)->isVolatile())
          return Attribute::None;

        is_read = true;
        break;

      case Instruction::Store:
        if (cast<StoreInst>(i)->getValueOperand() == *u)
          // untrackable capture
          return Attribute::None;

        // A volatile store has side effects beyond what writeonly can be relied
        // upon.
        if (cast<StoreInst>(i)->isVolatile())
          return Attribute::None;

        is_write = true;
        break;

      case Instruction::ICmp:
      case Instruction::Ret:
        break;

      default:
        return Attribute::None;
    }
  }

  if (is_write && is_read)
    return Attribute::None;
  if (is_read)
    return Attribute::ReadOnly;
  if (is_write)
    return Attribute::WriteOnly;
  return Attribute::ReadNone;
}

inline AccessState state(const llvm::AAMemoryBehavior& mem) {
  if (mem.isAssumedReadNone()) {
    return AccessState::kNone;
  }
  if (mem.isAssumedReadOnly()) {
    return AccessState::kRead;
  }
  if (mem.isAssumedWriteOnly()) {
    return AccessState::kWritten;
  }
  return AccessState::kRW;
}

inline AccessState state(const llvm::Attribute::AttrKind mem) {
  using namespace llvm;
  if (mem == Attribute::ReadNone) {
    return AccessState::kNone;
  }
  if (mem == Attribute::ReadOnly) {
    return AccessState::kRead;
  }
  if (mem == Attribute::WriteOnly) {
    return AccessState::kWritten;
  }
  return AccessState::kRW;
}

struct ChildInfo {
  llvm::Value* val;
  llvm::SmallVector<int32_t> indicies;
};

void collect_children(FunctionArg& arg, llvm::Value* init_val) {
  using namespace llvm;
  llvm::SmallVector<ChildInfo, 32> work_list;
  work_list.push_back({init_val, {}});

  while (!work_list.empty()) {
    // not nice making copies of the stack all the time idk
    auto curr_info   = work_list.pop_back_val();
    auto* value      = curr_info.val;
    auto index_stack = curr_info.indicies;

    Type* value_type = value->getType();
    if (auto* ptr_type = dyn_cast<PointerType>(value_type)) {
      auto* elem_type = ptr_type->getPointerElementType();
      if (elem_type->isStructTy()) {
        for (User* value_user : value->users()) {
          if (auto* gep = dyn_cast<GetElementPtrInst>(value_user)) {
            auto gep_indicies = gep->indices();
            auto sub_index_stack = index_stack;
            for (unsigned i = 1; i < gep->getNumIndices(); i++) {
              auto* index = gep_indicies.begin() + i;
              if (auto* index_value = dyn_cast<ConstantInt>(index->get())) {
                sub_index_stack.push_back((int32_t)index_value->getSExtValue());
                work_list.push_back({gep, sub_index_stack});
              } else {
                LOG_WARNING("Failed to determine access pattern for argument '"
                            << arg.arg_pos << "' since it uses dynamic gep indices");
                break;
              }
            }
          }
        }
      }
      //{
      //  const auto res = determinePointerAccessAttrs(load);
      //  const FunctionArg kernel_arg{load, index_stack, arg_pos, true, state(res)};
      //  args.push_back(kernel_arg);
      //}
      for (User* value_user : value->users()) {
        if (auto* load = dyn_cast<LoadInst>(value_user)) {
          if (load->getType()->isPointerTy()) {
            auto sub_index_stack = index_stack;
            sub_index_stack.push_back(-1);
            work_list.push_back({load, sub_index_stack});
            const auto res = determinePointerAccessAttrs(load);
            //const FunctionArg kernel_arg{load, std::move(sub_index_stack), arg_pos, true, state(res)};
            const FunctionSubArg sub_arg{load, std::move(sub_index_stack), true, state(res)};
            arg.subargs.push_back(sub_arg);
          }
        }
      }

    } else {
      return;
    }
  }
}

void attribute_value(FunctionArg& arg, llvm::Attributor& attrib
                     ) {
  using namespace llvm;
  auto* value = arg.value.getValue();
  Type* value_type = value->getType();
  llvm::errs() << "Attributing Value: " << *value << " of type: " << *value_type << "\n";

  if (value_type->isPointerTy()) {
    IRPosition const val_pos = IRPosition::value(*value);
    const auto& mem_behavior = attrib.getOrCreateAAFor<AAMemoryBehavior>(val_pos);
    const auto res2          = determinePointerAccessAttrs(value);
    llvm::errs() << "   secRes: None:" << (res2 == Attribute::None) << " ReadNone" << (res2 == Attribute::ReadNone)
                 << " ReadOnly" << (res2 == Attribute::ReadOnly) << " WriteOnly" << (res2 == Attribute::WriteOnly)
                 << "\n";
    llvm::errs() << "     isValid:" << mem_behavior.isValidState() << "\n";
    llvm::errs() << "    Got ptr: KnownReadNone:" << mem_behavior.isKnownReadNone()
                 << " KnownReadOnly:" << mem_behavior.isKnownReadOnly()
                 << " KnownWriteOnly:" << mem_behavior.isKnownWriteOnly() << "\n";
    llvm::errs() << "    Got ptr: ReadNone:" << mem_behavior.isAssumedReadNone()
                 << " ReadOnly:" << mem_behavior.isAssumedReadOnly()
                 << " WriteOnly:" << mem_behavior.isAssumedWriteOnly() << "\n";
    const FunctionSubArg kernel_arg{value, {}, true, state(res2)};
    arg.is_pointer = true;
    arg.subargs.emplace_back(kernel_arg);
    collect_children(arg, value);
  } else {
    const FunctionSubArg kernel_arg{value, {}, false, AccessState::kRW};
    arg.subargs.emplace_back(kernel_arg);
  }
}

std::optional<KernelModel> info_with_attributor(llvm::Function* kernel) {
  using namespace llvm;

  auto* module = kernel->getParent();
  AnalysisGetter ag;
  SetVector<Function*> functions;
  for (auto& module_f : module->functions()) {
    functions.insert(&module_f);
  }
  CallGraphUpdater cg_updater;
  BumpPtrAllocator allocator;
  InformationCache info_cache(*module, ag, allocator, /* CGSCC */ nullptr);

  Attributor attrib(functions, info_cache, cg_updater);

  LOG_DEBUG("Attributing " << kernel->getName() << "\n" << *kernel << "\n")

  llvm::SmallVector<FunctionArg, 4> args{};
  for (const auto& arg_value : llvm::enumerate(kernel->args())) {
    llvm::SmallVector<int32_t> index_stack = {};
    FunctionArg arg{};
    arg.arg_pos = (uint32_t)arg_value.index();
    arg.value = &arg_value.value();
    attribute_value(arg, attrib);
    args.push_back(std::move(arg));
  }

  KernelModel model{kernel, std::string{kernel->getName()}, args};

  return model;
}

std::optional<KernelModel> analyze_device_kernel(llvm::Function* f) {
  if (!cuda::is_kernel(f)) {
    assert(f != nullptr && "Function should not be null here!");
    LOG_DEBUG("Function is not a kernel " << f->getName())
    return {};
  }
  using namespace llvm;
  const auto kernel_model = info_with_attributor(f);
  return kernel_model;
}

}  // namespace device

namespace host {

std::optional<KernelModel> kernel_model_for_stub(llvm::Function* func, const ModelHandler& models) {
  const auto stub_name = [&](const auto& name) {
    auto stub_name    = std::string{name};
    const auto prefix = std::string{"__device_stub__"};
    const auto pos    = stub_name.find(prefix);
    if (pos != std::string::npos) {
      stub_name.erase(pos, prefix.length());
    }
    return stub_name;
  }(util::try_demangle(*func));

  const auto result = llvm::find_if(models.models, [&stub_name](const auto& model_) {
    return llvm::StringRef(util::demangle(model_.kernel_name)).startswith(stub_name);
  });

  if (result != std::end(models.models)) {
    LOG_DEBUG("Found fitting kernel data " << *result)
    return *result;
  }

  LOG_DEBUG("Found no kernel data for stub: " << stub_name)
  return {};
}

}  // namespace host

}  // namespace cucorr
