/*
 * Copyright IBM Corp. 2025
 */
#pragma once
#include <ATen/ATen.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/SymInt.h>

namespace spyre {

/**
 * An SpyreStorageImpl is a storage type which always returns
 * the Spyre device through device_type, regardless of whether
 * the data is on CPU or on Spyre.
 * For now, this is actually a CPU storage class, but eventually
 * it will be used to hold Spyre custom storage format conversions,
 * like Spyre specific stickification
 */
class SpyreStorageImpl : public c10::StorageImpl {
 public:
  SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes,
                   at::Allocator* allocator, bool resizable);
};

}  // namespace spyre
