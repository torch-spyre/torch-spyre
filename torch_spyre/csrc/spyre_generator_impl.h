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

#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/CallOnce.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace spyre {

struct SpyreGeneratorImplState {
  uint64_t initial_seed_;
  int left;   /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  uint64_t state[at::MERSENNE_STATE_N]; /* the array for the state vector  */
};

struct SpyreGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  SpyreGeneratorImpl(c10::DeviceIndex device_index = -1,
                     uint64_t seed_in = at::default_rng_seed_val);
  ~SpyreGeneratorImpl() override = default;

  // SpyreGeneratorImpl methods
  std::shared_ptr<SpyreGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

  static c10::DeviceType device_type();
  uint32_t random();
  uint64_t random64();

  at::mt19937 engine() const;
  void set_engine(const at::mt19937& engine);

 private:
  at::mt19937 engine_;

  SpyreGeneratorImpl* clone_impl() const override;
};

at::Tensor get_rng_state(c10::DeviceIndex device_index = -1);
void set_rng_state(const at::Tensor& new_state,
                   c10::DeviceIndex device_index = -1);
void manual_seed(uint64_t seed, c10::DeviceIndex device_index = -1);
void manual_seed_all(uint64_t seed);
uint64_t initial_seed(c10::DeviceIndex device_index = -1);

namespace detail {

inline c10::DeviceIndex num_cards = -1;
inline std::deque<c10::once_flag> spyre_gens_init_flag;
inline std::vector<at::Generator> default_gens_spyre;

const at::Generator& getDefaultSpyreGenerator(
    c10::DeviceIndex device_index = -1);
at::Generator createSpyreGenerator(
    c10::DeviceIndex device_index = -1,
    uint64_t seed_val = at::default_rng_seed_val);

}  // namespace detail
}  // namespace spyre
