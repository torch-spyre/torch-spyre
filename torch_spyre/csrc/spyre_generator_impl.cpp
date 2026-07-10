/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_generator_impl.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/ScalarType.h>
#include <c10/util/CallOnce.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "spyre_device_enum.h"

namespace spyre {

namespace detail {

c10::DeviceIndex num_cards = -1;
std::deque<c10::once_flag> spyre_gens_init_flag;
std::vector<at::Generator> default_gens_spyre;

void initSpyreGenVector() {
  static bool init_flag [[maybe_unused]] = []() {
    num_cards = getVisibleDeviceCount();
    if (num_cards > 0) {
      spyre_gens_init_flag.resize(num_cards);
      default_gens_spyre.resize(num_cards);
    }
    return true;
  }();
}

c10::DeviceIndex resolve_device_index(c10::DeviceIndex device) {
  initSpyreGenVector();

  TORCH_CHECK(num_cards > 0,
              "No Spyre devices available. Cannot create generator.");

  if (device == -1) {
    c10::Device curr_device =
        c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
            ->getDevice();
    device = curr_device.index();
  }

  TORCH_CHECK(device >= 0 && device < num_cards,
              "Spyre device index is invalid");
  return device;
}

const at::Generator& getDefaultSpyreGenerator(c10::DeviceIndex device) {
  device = resolve_device_index(device);

  c10::call_once(spyre_gens_init_flag[device], [&]() {
    default_gens_spyre[device] = at::make_generator<SpyreGeneratorImpl>(
        device, c10::detail::getNonDeterministicRandom());
  });

  return default_gens_spyre[device];
}

at::Generator createSpyreGenerator(c10::DeviceIndex device, uint64_t seed_val) {
  device = resolve_device_index(device);
  return at::make_generator<SpyreGeneratorImpl>(device, seed_val);
}

/**
 * Helper function to concatenate two 32 bit unsigned int
 * and return them as a 64 bit unsigned int
 */
inline static uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

}  // namespace detail

/**
 * SpyreGeneratorImpl class implementation
 */
SpyreGeneratorImpl::SpyreGeneratorImpl(c10::DeviceIndex device_index,
                                       uint64_t seed_in)
    : c10::GeneratorImpl{c10::Device(c10::DeviceType::PrivateUse1,
                                     device_index),
                         c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)},
      engine_{seed_in} {}

/**
 * Manually seeds the engine with the seed input
 * See Note [Acquire lock when using random generators]
 */
void SpyreGeneratorImpl::set_current_seed(uint64_t seed) {
  engine_ = at::mt19937(seed);
}

/**
 * Sets the offset of RNG state.
 * See Note [Acquire lock when using random generators]
 */
void SpyreGeneratorImpl::set_offset(uint64_t offset [[maybe_unused]]) {
  TORCH_CHECK(false, "Spyre Generator does not use offset");
}

/**
 * Gets the current offset of SpyreGeneratorImpl.
 */
uint64_t SpyreGeneratorImpl::get_offset() const {
  TORCH_CHECK(false, "Spyre Generator does not use offset");
}

/**
 * Gets the current seed of SpyreGeneratorImpl.
 */
uint64_t SpyreGeneratorImpl::current_seed() const {
  return engine_.seed();
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the SpyreGeneratorImpl with it and then returns that number.
 */
uint64_t SpyreGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

/**
 * Sets the internal state of SpyreGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and of the same size as
 * SpyreGeneratorImplState.
 */
void SpyreGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  // Validate SpyreGeneratorImplState is POD-compatible at compile time
  static_assert(std::is_standard_layout_v<SpyreGeneratorImplState>,
                "SpyreGeneratorImplState must have standard layout for safe "
                "memory operations");
  constexpr size_t size = sizeof(SpyreGeneratorImplState);

  // Validate the input tensor state
  at::detail::check_rng_state(new_state);

  at::mt19937 engine;
  auto new_state_size = new_state.numel();

  TORCH_CHECK(new_state_size == size,
              "Expected a SpyreGeneratorImplState of size ", size,
              " but found the input RNG state size to be ", new_state_size);

  // Extract and validate state data pointer
  auto rng_state = new_state.data_ptr_impl<SpyreGeneratorImplState>();

  // Construct engine_
  at::mt19937_data_pod rng_data{};
  std::copy(std::begin(rng_state->state), std::end(rng_state->state),
            rng_data.state_.begin());
  rng_data.seed_ = rng_state->initial_seed_;
  rng_data.left_ = rng_state->left;
  rng_data.seeded_ = rng_state->seeded;
  rng_data.next_ = static_cast<uint32_t>(rng_state->next);
  engine.set_data(rng_data);
  TORCH_CHECK(engine.is_valid(), "Invalid mt19937 state");
  this->engine_ = engine;
}

/**
 * Gets the current internal state of SpyreGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> SpyreGeneratorImpl::get_state() const {
  // Validate SpyreGeneratorImplState is POD-compatible at compile time
  static_assert(std::is_standard_layout_v<SpyreGeneratorImplState>,
                "SpyreGeneratorImplState must have standard layout for safe "
                "memory operations");
  constexpr size_t size = sizeof(SpyreGeneratorImplState);

  auto state_tensor = at::detail::empty_cpu(
      {static_cast<int64_t>(size)}, c10::ScalarType::Byte, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt);
  auto rng_state = state_tensor.data_ptr();

  // accumulate generator data to be copied into byte tensor
  auto accum_state = std::make_unique<SpyreGeneratorImplState>();
  auto rng_data = this->engine_.data();
  accum_state->initial_seed_ = rng_data.seed_;
  accum_state->left = rng_data.left_;
  accum_state->seeded = rng_data.seeded_;
  accum_state->next = rng_data.next_;
  std::copy(rng_data.state_.begin(), rng_data.state_.end(),
            std::begin(accum_state->state));

  memcpy(rng_state, accum_state.get(), size);
  return state_tensor.getIntrusivePtr();
}

/**
 * Gets the DeviceType of SpyreGeneratorImpl.
 * Used for type checking during run time.
 */
c10::DeviceType SpyreGeneratorImpl::device_type() {
  return c10::DeviceType::PrivateUse1;
}

/**
 * Gets a random 32 bit unsigned integer from the engine
 *
 * See Note [Acquire lock when using random generators]
 */
uint32_t SpyreGeneratorImpl::random() {
  return engine_();
}

/**
 * Gets a random 64 bit unsigned integer from the engine
 *
 * See Note [Acquire lock when using random generators]
 */
uint64_t SpyreGeneratorImpl::random64() {
  uint32_t random1 = engine_();
  uint32_t random2 = engine_();
  return detail::make64BitsFrom32Bits(random1, random2);
}

/**
 * Get the engine of the SpyreGeneratorImpl
 */
at::mt19937 SpyreGeneratorImpl::engine() const {
  return engine_;
}

/**
 * Set the engine of the SpyreGeneratorImpl
 *
 * See Note [Acquire lock when using random generators]
 */
void SpyreGeneratorImpl::set_engine(const at::mt19937& engine) {
  engine_ = engine;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<SpyreGeneratorImpl> SpyreGeneratorImpl::clone() const {
  return std::shared_ptr<SpyreGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
SpyreGeneratorImpl* SpyreGeneratorImpl::clone_impl() const {
  auto gen = new SpyreGeneratorImpl(this->device().index());
  gen->set_engine(engine_);
  return gen;
}

/**
 * Gets the RNG state for the specified Spyre device
 */
at::Tensor get_rng_state(c10::DeviceIndex device_index) {
  auto gen = detail::getDefaultSpyreGenerator(device_index);
  std::scoped_lock<std::mutex> lock(gen.mutex());
  return gen.get_state();
}

/**
 * Sets the RNG state for the specified Spyre device
 */
void set_rng_state(const at::Tensor& new_state, c10::DeviceIndex device_index) {
  auto gen = detail::getDefaultSpyreGenerator(device_index);
  std::scoped_lock<std::mutex> lock(gen.mutex());
  gen.set_state(new_state);
}

/**
 * Manually seeds the RNG for the specified Spyre device
 */
void manual_seed(uint64_t seed, c10::DeviceIndex device_index) {
  auto gen = detail::getDefaultSpyreGenerator(device_index);
  std::scoped_lock<std::mutex> lock(gen.mutex());
  gen.set_current_seed(seed);
}

/**
 * Manually seeds the RNG for all Spyre devices
 */
void manual_seed_all(uint64_t seed) {
  detail::initSpyreGenVector();
  for (const auto device : c10::irange(detail::num_cards)) {
    manual_seed(seed, static_cast<c10::DeviceIndex>(device));
  }
}

/**
 * Gets the initial seed for the specified Spyre device
 */
uint64_t initial_seed(c10::DeviceIndex device_index) {
  auto gen = detail::getDefaultSpyreGenerator(device_index);
  std::scoped_lock<std::mutex> lock(gen.mutex());
  return gen.current_seed();
}

}  // namespace spyre
