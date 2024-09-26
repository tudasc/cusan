// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "KernelModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <system_error>
#include <vector>

namespace compat {
auto open_flag() {
#if LLVM_VERSION_MAJOR < 13
  return llvm::sys::fs::OpenFlags::F_Text;
#else
  return llvm::sys::fs::OpenFlags::OF_Text;
#endif
}
}  // namespace compat

template <>
struct llvm::yaml::ScalarTraits<cusan::AccessState> {
  static void output(const cusan::AccessState& value, void*, llvm::raw_ostream& out) {
    out << cusan::access_state_string(value);
  }

  static llvm::StringRef input(llvm::StringRef scalar, void*, cusan::AccessState& value) {
    // FIXME keep stringliteral and enum value in sync, see cusan::access_state_string
    value = llvm::StringSwitch<cusan::AccessState>(scalar)
                .Case("Write", cusan::AccessState::kWritten)
                .Case("None", cusan::AccessState::kNone)
                .Case("Read", cusan::AccessState::kRead)
                .Default(cusan::AccessState::kRW);
    return StringRef();
  }

  // Determine if this scalar needs quotes.
  static QuotingType mustQuote(StringRef) {
    return QuotingType::None;
  }
};

template <>
struct llvm::yaml::MappingTraits<cusan::FunctionArg> {
  static void mapping(IO& io, cusan::FunctionArg& info) {
    if (!io.outputting()) {
      info.value = std::nullopt;
    }
    io.mapRequired("position", info.arg_pos);
    io.mapRequired("pointer", info.is_pointer);
    io.mapRequired("subargs", info.subargs);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cusan::FunctionArg)

template <>
struct llvm::yaml::MappingTraits<cusan::FunctionSubArg> {
  static void mapping(IO& io, cusan::FunctionSubArg& info) {
    if (!io.outputting()) {
      info.value = std::nullopt;
    }
    io.mapRequired("indices", info.indices);
    io.mapRequired("access", info.state);
    io.mapRequired("pointer", info.is_pointer);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cusan::FunctionSubArg)

template <>
struct llvm::yaml::MappingTraits<cusan::FunctionSubArg::SubIndex> {
  static void mapping(IO& io, cusan::FunctionSubArg::SubIndex& info) {
    io.mapRequired("index", info.index);
    io.mapRequired("is_load", info.is_load);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cusan::FunctionSubArg::SubIndex)

template <>
struct llvm::yaml::MappingTraits<cusan::KernelModel> {
  static void mapping(IO& io, cusan::KernelModel& info) {
    if (!io.outputting()) {
      info.kernel = std::nullopt;
    }
    io.mapRequired("name", info.kernel_name);
    io.mapRequired("args", info.args);
    //    info.kernel = {};
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cusan::KernelModel)

namespace cusan::io {
[[nodiscard]] llvm::ErrorOr<bool> store(const ModelHandler& kernel_db, std::string_view file) {
  using namespace llvm;

  std::error_code error;
  raw_fd_ostream oss(StringRef(file), error, compat::open_flag());

  if (oss.has_error()) {
    return error;
  }

  auto types = kernel_db.models;
  yaml::Output out(oss);
  if (!types.empty()) {
    out << types;
  } else {
    out.beginDocuments();
    out.endDocuments();
  }

  return true;
}

[[nodiscard]] llvm::ErrorOr<bool> load(ModelHandler& kernel_db, std::string_view file) {
  using namespace llvm;
  ErrorOr<std::unique_ptr<MemoryBuffer>> memBuffer = MemoryBuffer::getFile(file.data());

  if (std::error_code error = memBuffer.getError(); error) {
    return error;
  }

  kernel_db.models.clear();

  yaml::Input in(memBuffer.get()->getMemBufferRef());
  std::vector<KernelModel> models;
  in >> models;

  kernel_db.models = std::move(models);

  return !in.error();
}

}  // namespace cusan::io