/*
 * Copyright IBM Corp. 2025
 */
#include <logging.h>

namespace spyre {

const char* const kDebugVarName = SPYRE_DEBUG_ENV;

bool g_debug_info_enabled = std::getenv(kDebugVarName) != nullptr
                                ? std::atoi(std::getenv(kDebugVarName)) != 0
                                : false;

}  // namespace spyre
