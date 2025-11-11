/*
 * Copyright IBM Corp. 2025
 */
#include "spyre_storage_impl.h"

namespace spyre {

SpyreStorageImpl::SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                                   at::Allocator* allocator, bool resizable)
    : c10::StorageImpl(use_byte_size_t(), size_bytes, allocator, resizable) {}

}  // namespace spyre
