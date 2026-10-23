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

// Native first-fit packer for the annealing backend (scratchpad/backend.py).
// The annealer searches residency and a placement order; every move re-packs
// all offsets, so the packing runs here.

#include "box_packer.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace torch_spyre {
namespace scratchpad {

namespace {

struct Placed {
  int64_t x0, x1, y0, y1;
};

}  // namespace

std::vector<int64_t> pack_boxes(const std::vector<int>& order,
                                const std::vector<int64_t>& x0,
                                const std::vector<int64_t>& x1,
                                const std::vector<int64_t>& size,
                                const std::vector<bool>& present,
                                const std::vector<int64_t>& pinned,
                                const std::vector<std::vector<int>>& ties) {
  const size_t n = x0.size();
  if (x1.size() != n || size.size() != n || present.size() != n ||
      pinned.size() != n || ties.size() != n) {
    throw std::invalid_argument("pack_boxes: per-box inputs differ in length");
  }
  std::vector<int64_t> offset(n, 0);
  std::vector<bool> is_placed(n, false);
  std::vector<Placed> placed;
  placed.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (present[i] && pinned[i] >= 0) {
      offset[i] = pinned[i];
      is_placed[i] = true;
      placed.push_back({x0[i], x1[i], pinned[i], pinned[i] + size[i]});
    }
  }
  std::vector<std::pair<int64_t, int64_t>> near;
  std::vector<int64_t> candidates;
  for (int i : order) {
    if (i < 0 || static_cast<size_t>(i) >= n) {
      throw std::out_of_range("pack_boxes: order names no box");
    }
    if (!present[i]) {
      offset[i] = 0;
      continue;
    }
    const auto tied = std::find_if(ties[i].begin(), ties[i].end(), [&](int t) {
      return t >= 0 && is_placed[t];
    });
    if (tied != ties[i].end()) {
      offset[i] = offset[*tied];
    } else {
      near.clear();
      candidates.assign(1, 0);
      for (const Placed& p : placed) {
        if (p.x0 < x1[i] && x0[i] < p.x1 && p.y0 < p.y1) {
          near.emplace_back(p.y0, p.y1);
          candidates.push_back(p.y1);
        }
      }
      std::sort(candidates.begin(), candidates.end());
      // The highest top always fits, so the loop always assigns.
      for (int64_t c : candidates) {
        bool fits = size[i] == 0 ||
                    std::all_of(near.begin(), near.end(), [&](const auto& y) {
                      return c + size[i] <= y.first || y.second <= c;
                    });
        if (fits) {
          offset[i] = c;
          break;
        }
      }
    }
    is_placed[i] = true;
    placed.push_back({x0[i], x1[i], offset[i], offset[i] + size[i]});
  }
  return offset;
}

void register_box_packer(py::module_& m) {
  m.def("pack_boxes", &pack_boxes, py::arg("order"), py::arg("x0"),
        py::arg("x1"), py::arg("size"), py::arg("present"), py::arg("pinned"),
        py::arg("ties"));
}

}  // namespace scratchpad
}  // namespace torch_spyre
