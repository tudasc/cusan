//
// Created by ahueck on 22.12.23.
//

#ifndef LIB_CUCORR_LOGGER_H_
#define LIB_CUCORR_LOGGER_H_

#include "llvm/Support/raw_ostream.h"
#include <mutex>

#ifndef CUCORR_LOG_LEVEL
/*
 * Usually set at compile time: -DCUCORR_LOG_LEVEL=<N>, N in [0, 3] for output
 * 3 being most verbose
 */
#define CUCORR_LOG_LEVEL 2
#endif

#ifndef LOG_BASENAME_FILE
#define LOG_BASENAME_FILE __FILE__
#endif

namespace cucorr::detail {
static std::mutex print_mutex;

inline void log(std::string_view msg) {
  llvm::errs() << msg;
}
}  // namespace cucorr::detail

// clang-format off
#define OO_LOG_LEVEL_MSG(LEVEL_NUM, LEVEL, MSG)                                                                   \
  if constexpr ((LEVEL_NUM) <= CUCORR_LOG_LEVEL) {                                                                \
    std::lock_guard<std::mutex> lock{cucorr::detail::print_mutex};                                                \
    std::string logging_message;                                                                                  \
    llvm::raw_string_ostream rso(logging_message);                                                                \
    rso << (LEVEL) << LOG_BASENAME_FILE << ":" << __func__ << ":" << __LINE__ << ":" << MSG << "\n"; /* NOLINT */ \
    cucorr::detail::log(rso.str());                                                                               \
  }

#define OO_LOG_LEVEL_MSG_BARE(LEVEL_NUM, LEVEL, MSG)               \
  if constexpr ((LEVEL_NUM) <= CUCORR_LOG_LEVEL) {                 \
    std::lock_guard<std::mutex> lock{cucorr::detail::print_mutex}; \
    std::string logging_message;                                   \
    llvm::raw_string_ostream rso(logging_message);                 \
    rso << (LEVEL) << " " << MSG << "\n"; /* NOLINT */             \
    cucorr::detail::log(rso.str());                                \
  }
// clang-format on

#define LOG_TRACE(MSG) OO_LOG_LEVEL_MSG_BARE(3, "[Trace]", MSG)
#define LOG_DEBUG(MSG) OO_LOG_LEVEL_MSG(3, "[Debug]", MSG)
#define LOG_INFO(MSG) OO_LOG_LEVEL_MSG(2, "[Info]", MSG)
#define LOG_WARNING(MSG) OO_LOG_LEVEL_MSG(1, "[Warning]", MSG)
#define LOG_ERROR(MSG) OO_LOG_LEVEL_MSG(1, "[Error]", MSG)
#define LOG_FATAL(MSG) OO_LOG_LEVEL_MSG(0, "[Fatal]", MSG)
#define LOG_MSG(MSG) llvm::errs() << MSG << "\n"; /* NOLINT */

#endif
