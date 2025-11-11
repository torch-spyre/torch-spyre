/*
 * Copyright IBM Corp. 2025
 */
#pragma once

#include <iostream>
#include <utility>

namespace spyre {

extern bool g_debug_info_enabled;

namespace detail {
template <class Head>
void PrintLog(bool e, Head&& head) {
  std::cout << head;
  if (e)
    std::cout << std::endl << std::dec;
  else
    std::cout << ' ';
}

template <class Head, class... Tail>
void PrintLog(bool e, Head&& head, Tail&&... others) {
  std::cout << head << ' ';
  PrintLog(e, std::forward<Tail>(others)...);
}

}  // namespace detail

class SuppressDebugLog {
  bool original_state;

 public:
  SuppressDebugLog() {
    original_state = g_debug_info_enabled;
  }
  ~SuppressDebugLog() {
    g_debug_info_enabled = original_state;
  }
};

#define DEBUGINFO(...)                                                   \
  do {                                                                   \
    if (g_debug_info_enabled) {                                          \
      ::spyre::detail::PrintLog(true, "[", __func__, "] ", __VA_ARGS__); \
    }                                                                    \
  } while (0);

#define DEBUGINFO_NO_ENDL(...)                                            \
  do {                                                                    \
    if (g_debug_info_enabled) {                                           \
      ::spyre::detail::PrintLog(false, "[", __func__, "] ", __VA_ARGS__); \
    }                                                                     \
  } while (0);

}  // namespace spyre
