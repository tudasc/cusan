// cusan library
// Copyright (c) 2023-2025 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "KernelAnalysis.h"

#include "support/CudaUtil.h"
#include "support/HipUtil.h"
#include "support/Logger.h"
#include "support/Util.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/IPO/Attributor.h>
#include <utility>

namespace cusan {

namespace device {

// when we store an argument into a local alloca we can then follow that alloca with this function
// returns a true if it escapes otherwise false
static bool determinePointerAccessAttrsForLocalPtrUse(llvm::Use& use_of_ptr,
                                                      llvm::SmallVector<llvm::Use*, 32>& worklist,
                                                      llvm::SmallPtrSet<llvm::Use*, 32>& visited) {
  using namespace llvm;
  auto* i = cast<Instruction>(use_of_ptr.getUser());
  switch (i->getOpcode()) {
    case Instruction::Load:
      if (cast<LoadInst>(i)->isVolatile())
        return true;
      for (Use& uu : i->uses())
        if (visited.insert(&uu).second)
          worklist.push_back(&uu);
      return false;
    default:
      return true;
  }
}

// Taken from (and extended to interprocedural analysis + looking through local stores) from clang19
// https://llvm.org/doxygen/FunctionAttrs_8cpp_source.html#l00611
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

    Use* u  = worklist.pop_back_val();
    auto* i = cast<Instruction>(u->getUser());
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

        // Given we've explicitly handled the callee operand above, what's left
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
          // auto called = cb.getCalledFunction();
          // if(visited_funcs.contains(called)){
          //   LOG_WARNING("Not handling recursive kernels right now");
          //   return Attribute::None;
          // }
          // if(called->isDeclaration()){
          //   LOG_WARNING("Could not determine pointer access since calling function outside of this cu: " << called);
          //   return Attribute::None;
          // }
          // visited_funcs.insert(called);
          // called->getArg(use_index);
          return Attribute::ReadNone;
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
        // A volatile store has side effects beyond what writeonly can be relied
        // upon.
        if (cast<StoreInst>(i)->isVolatile())
          return Attribute::None;

        if (cast<StoreInst>(i)->getValueOperand() == *u) {
          auto* addr_cast_ptr = cast<StoreInst>(i)->getPointerOperand();
          auto* alloca_ptr    = addr_cast_ptr;
          bool has_addr_cast  = false;
          if (auto cast = dyn_cast<AddrSpaceCastInst>(alloca_ptr)) {
            alloca_ptr    = cast->getPointerOperand();
            has_addr_cast = true;
          }
          if (auto alloc = dyn_cast<AllocaInst>(alloca_ptr)) {
            alloca_ptr = alloc;
          } else {
            alloca_ptr = nullptr;
          }
          if (alloca_ptr != nullptr) {
            if (has_addr_cast) {
              for (Use& UU : addr_cast_ptr->uses()) {
                if (visited.insert(&UU).second && cast<Instruction>(UU.getUser()) != i)
                  if (determinePointerAccessAttrsForLocalPtrUse(UU, worklist, visited)) {
                    return Attribute::None;
                  }
              }
            }
            for (Use& UU : alloca_ptr->uses()) {
              bool skip_addr_space_cast = (!has_addr_cast || cast<Instruction>(UU.getUser()) != addr_cast_ptr);
              if (visited.insert(&UU).second && cast<Instruction>(UU.getUser()) != i && skip_addr_space_cast)
                if (determinePointerAccessAttrsForLocalPtrUse(UU, worklist, visited)) {
                  return Attribute::None;
                }
            }

            continue;
          }
          // untrackable capture
          return Attribute::None;
        }

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

void collect_subsequent_load(FunctionArg& arg, llvm::Value* value, llvm::SmallVector<int64_t> index_stack) {
  using namespace llvm;
  for (User* value_user : value->users()) {
    if (auto* load = dyn_cast<LoadInst>(value_user)) {
      if (load->getType()->isPointerTy()) {
        const auto res = determinePointerAccessAttrs(load);
        const FunctionSubArg sub_arg{load, true, index_stack, true, state(res)};
        arg.subargs.push_back(sub_arg);
      }
    }
  }
}

void collect_children(FunctionArg& arg, llvm::Value* value, llvm::SmallSet<llvm::Function*, 8>& visited_funcs,
                      llvm::SmallSet<llvm::Instruction*, 8>& ignore_instrs) {
  using namespace llvm;

  Type* value_type = value->getType();
  if (auto* ptr_type = dyn_cast<PointerType>(value_type)) {
    // auto* elem_type = ptr_type->getPointerElementType();
    //  if (elem_type->isStructTy() || elem_type->isPointerTy()) {
    for (Use& value_use : value->uses()) {
      User* value_user = value_use.getUser();
      if (ignore_instrs.contains(dyn_cast<Instruction>(value_user))) {
        continue;
      }
      if (auto* call = dyn_cast<CallBase>(value_user)) {
        Function* called = call->getCalledFunction();
        if (visited_funcs.contains(called)) {
          LOG_WARNING("Not handling recursive kernels right now");
          continue;
        }
        // for memcpy we can skip parts but still need to update the access state
        if (util::starts_with_any_of(called->getName(), "llvm.memcpy", "llvm.memmove", "llvm.memset")) {
          if (value_use.getOperandNo() == 0) {  // destination
            auto* res = llvm::find_if(arg.subargs, [=](auto a) { return a.value.value_or(nullptr) == value; });
            assert(res != arg.subargs.end());
            res->state = mergeAccessState(res->state, AccessState::kWritten);
            continue;
          }
          if (value_use.getOperandNo() == 1) {  // input
            auto* res = llvm::find_if(arg.subargs, [=](auto a) { return a.value.value_or(nullptr) == value; });
            assert(res != arg.subargs.end());
            res->state = mergeAccessState(res->state, AccessState::kRead);
            ignore_instrs.insert(call);
            collect_children(arg, call->getArgOperand(0), visited_funcs, ignore_instrs);
            llvm::errs() << *call->getCaller() << "\n";
            llvm::errs() << "<<<" << *call->getArgOperand(0) << "\n";
            llvm::errs() << "<<<" << *value << "\n";
            llvm::errs() << "<<<" << arg << "\n";
            continue;
          }
        }
        if (called->isDeclaration()) {
          LOG_WARNING("Could not determine pointer access of the "
                      << arg.arg_pos
                      << " Argument since its calling function outside of this cu: " << called->getName());
          continue;
        }
        visited_funcs.insert(called);

        Argument* ipo_argument = called->getArg(value_use.getOperandNo());
        {
          const auto access_res = determinePointerAccessAttrs(ipo_argument);
          // const FunctionSubArg sub_arg{ipo_argument, index_stack, true, state(access_res)};
          // arg.subargs.push_back(sub_arg);
          //  this argument should have already been looked at in the current function so if we
          //  check it again we should merge the results to get the correct accessstate
          auto* res = llvm::find_if(arg.subargs, [=](auto a) { return a.value.value_or(nullptr) == ipo_argument; });
          if (res == arg.subargs.end()) {
            res->state = mergeAccessState(res->state, state(access_res));
          } else {
            assert(false);
          }
        }
        collect_children(arg, ipo_argument, visited_funcs, ignore_instrs);
      } else if (auto* gep = dyn_cast<GetElementPtrInst>(value_user)) {
        auto gep_indicies                  = gep->indices();
        llvm::SmallVector<int64_t> indices = {};
        bool all_constant                  = true;

        for (unsigned i = 0; i < gep->getNumIndices(); i++) {
          auto* index = gep_indicies.begin() + i;
          if (auto* index_value = dyn_cast<ConstantInt>(index->get())) {
            indices.push_back(index_value->getSExtValue());
          } else {
            LOG_WARNING("Failed to determine access pattern for argument '" << arg.arg_pos
                                                                            << "' since it uses dynamic gep indices");
            all_constant = false;
            break;
          }
        }
        if (all_constant) {
          // const FunctionSubArg sub_arg{value, true, indices, true, state(res)};
          // arg.subargs.push_back(sub_arg);
          collect_subsequent_load(arg, gep, std::move(indices));
          // work_list.push_back({gep, sub_index_stack});
        }
      } else if (auto* c = dyn_cast<AddrSpaceCastInst>(value_user)) {
        collect_children(arg, c, visited_funcs, ignore_instrs);
      }
      // else if (auto* c = dyn_cast<StoreInst>(value_user)) {
      //   if (auto t = 	dyn_cast<AllocaInstr>(c.getPointerOperand()) && !c.isVolatile()) {
      //     ignore_instrs.insert(c);
      //     collect_children(arg, t, visited_funcs, ignore_instrs);
      //   }
      // }
    }

    // for (User* value_user : value->users()) {
    //   if (value_user.contains(dyn_cast<Instruction>(value_user))) {
    //     continue;
    //   };
    // }
    collect_subsequent_load(arg, value, {});
  } else {
    return;
  }
}

void attribute_value(FunctionArg& arg) {
  using namespace llvm;
  assert(arg.value.has_value());
  auto* value      = arg.value.value();
  Type* value_type = value->getType();
  if (value_type->isPointerTy()) {
    const auto res2 = determinePointerAccessAttrs(value);
    const FunctionSubArg kernel_arg{value, false, {}, true, state(res2)};
    arg.is_pointer = true;
    arg.value      = value;
    arg.subargs.emplace_back(kernel_arg);
    llvm::SmallSet<llvm::Function*, 8> visited_funcs   = {};
    llvm::SmallSet<llvm::Instruction*, 8> ignore_instr = {};
    collect_children(arg, value, visited_funcs, ignore_instr);
  } else {
    const FunctionSubArg kernel_arg{value, false, {}, false, AccessState::kRW};
    arg.subargs.emplace_back(kernel_arg);
  }
}

std::optional<KernelModel> info_with_attributor(llvm::Function* kernel) {
  using namespace llvm;

  LOG_DEBUG("Attributing " << kernel->getName() << "\n" << *kernel << "\n")

  llvm::SmallVector<FunctionArg, 4> args{};
  for (const auto& arg_value : llvm::enumerate(kernel->args())) {
    FunctionArg arg{};
    arg.arg_pos = (uint32_t)arg_value.index();
    arg.value   = &arg_value.value();
    args.push_back(arg);
  }

  for (auto& arg : args) {
    attribute_value(arg);
  }

  KernelModel model{kernel, std::string{kernel->getName()}, args};

  return model;
}

std::optional<KernelModel> analyze_device_kernel(llvm::Function* f) {
  if (!cuda::is_kernel(f) && !hip::is_kernel(f)) {
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
  }(util::try_demangle_fully(*func));

  const auto result = llvm::find_if(models.models, [&stub_name](const auto& model_) {
    return util::starts_with_any_of(util::try_demangle_fully(model_.kernel_name), stub_name);
  });

  if (result != std::end(models.models)) {
    LOG_DEBUG("Found fitting kernel data " << *result)
    return *result;
  }

  LOG_DEBUG("Found no kernel data for stub: " << stub_name)
  return {};
}

}  // namespace host

}  // namespace cusan
