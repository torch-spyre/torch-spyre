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

// Native (C++) accelerator for the permutation-based scratchpad layout packer.
//
// This mirrors the *observable* behaviour of the canonical Python
// ``PermutationBasedLayoutSolver``
// (``torch_spyre/_inductor/scratchpad/permutation_layout.py``): given a
// permutation (allocation order), per-buffer sizes and LX-eligibility, it
// places every buffer on top of the earlier-placed, time-overlapping buffers
// (with in-place reuse and a capacity/eviction gate) and exposes the resulting
// ``addresses`` (None == evicted / HBM), ``quality()`` and
// ``count_allocated()``.
//
// Placement is a pure function of (permutation, sizes, eligibility, lifetimes),
// so after every mutating op the layout is recomputed from scratch in
// permutation order using precomputed time-overlap sets plus the saturation
// early-stop -- i.e. the same decision the Python reference/from-scratch placer
// makes, which the incremental Python packer is differentially proven equal to.
// The internal representation is therefore deliberately simpler than Python's
// below/above contact profiles (the task allows any internal representation
// that reproduces the observable state); the speedup comes from staying in C++
// and never touching a Python object per operation. Buffer fields are read
// exactly once, in the constructor.

#include "perm_layout_native.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace torch_spyre {
namespace scratchpad {

namespace {

// Default placement alignment: one Spyre stick (128 bytes).
constexpr int64_t kDefaultAlignment = 128;

// Immutable-during-planning data derived from the buffers, the capacity and the
// alignment. Shared by reference between a solver and its ``copy()`` clones.
struct StaticData {
  int n = 0;
  int64_t capacity = 0;
  int64_t alignment = kDefaultAlignment;
  std::vector<int64_t> start;  // start_time == uses.front()
  std::vector<int64_t> end;    // end_time == uses.back() + 1
  std::vector<double> weight;  // len(uses) + (first_use_is_read ? 0.0 : 0.5)
  // Time-overlap members of each buffer (order-independent, lifetimes only).
  std::vector<std::vector<int>> overlap;
  // Possible in-place partners: declared parents + children that declare it.
  std::vector<std::vector<int>> inplace_partners;
  // Parent indices this buffer names in ``in_place_parents`` (for direction).
  // A vector (not a set): a typical buffer declares 0-1 parents, and the sole
  // read is a boolean membership test, so a linear scan beats a hash lookup.
  std::vector<std::vector<int>> declared_parents;
  // Saturation early-stop interval data (lifetimes only).
  int num_intervals = 0;
  std::vector<int> total_at;                       // alive count per interval
  std::vector<std::pair<int, int>> buf_intervals;  // [lo, hi) interval range
};

// Both classes below stay in this file's anonymous namespace. They are handed
// to pybind11 by ``register_perm_layout_native`` and are never named from
// another translation unit, and internal linkage matches the hidden visibility
// of their pybind11 members (external linkage draws -Wattributes for
// ``buffers_``).
class NativePermutationLayoutSolver;

// Read-only, always-live view of a solver's current allocation order.
//
// The search captures ``perm = plan.permutation`` once and then mutates the
// plan through it -- ``annealing_step_swap`` re-reads ``perm[i]`` and ``perm[i
// + 1]`` after every ``plan.swap(i)`` -- so the object it holds must reflect
// later mutations. Reading straight through to the owner's ``std::vector``
// makes that true by construction: there is no mirrored Python list for a
// future mutator to forget to update. Read-only, so Python cannot corrupt the
// order behind the solver's back either. ``py::keep_alive`` on the property
// keeps the owner alive for as long as any view of it.
class PermutationView {
 public:
  explicit PermutationView(const NativePermutationLayoutSolver& owner)
      : owner_(owner) {}

  int64_t len() const;
  // Python list indexing, negative offsets included.
  int getitem(int64_t i) const;
  bool eq(const py::object& other) const;
  // A *detached* snapshot. ``copy.copy(view)`` must not alias the live order:
  // the search stores the best-so-far permutation that way and later compares
  // the live one against it, a test that would be vacuously equal if the copy
  // tracked the original.
  py::list to_list() const;
  std::string repr() const;

 private:
  const std::vector<int>& vec() const;

  const NativePermutationLayoutSolver& owner_;
};

class NativePermutationLayoutSolver {
 public:
  NativePermutationLayoutSolver(const py::list& buffers,
                                std::vector<int> permutation, int64_t capacity,
                                int64_t alignment, const py::object& eligible) {
    auto st = std::make_shared<StaticData>();
    // Retained (not just read once): ``finalize`` writes each address back onto
    // the Python buffer objects. Shared, never deep-copied by ``copy()`` --
    // the same contract as the Python packer's ``self.buffers = buffers``.
    buffers_ = buffers;
    const int n = static_cast<int>(buffers.size());
    st->n = n;
    st->capacity = capacity;
    st->alignment = alignment;
    if (alignment <= 0) {
      throw std::invalid_argument("alignment must be positive");
    }

    // Validate the permutation is a permutation of range(n).
    {
      std::vector<int> sorted_perm = permutation;
      std::sort(sorted_perm.begin(), sorted_perm.end());
      bool ok = static_cast<int>(sorted_perm.size()) == n;
      for (int i = 0; ok && i < n; ++i) ok = sorted_perm[i] == i;
      if (!ok) {
        throw std::invalid_argument(
            "permutation must be a permutation of range(len(buffers))");
      }
    }

    // Read every buffer field exactly once.
    st->start.resize(n);
    st->end.resize(n);
    st->weight.resize(n);
    sizes_.resize(n);
    std::vector<std::string> names(n);
    std::vector<std::vector<std::string>> parent_names(n);
    // Whether each buffer's last use is a read, i.e. whether it has storage to
    // hand to an in-place child (checked below). Derived here because it needs
    // ``first_use_is_read``, which nothing else retains -- the rest of the test
    // is recoverable from ``st->start``/``st->end``. The test is the one in
    // ``plan_solver.check_in_place_parent_is_read``, "a use strictly after the
    // first", kept term-for-term so a repeated index cannot pass as a read.
    std::vector<char> can_hand_over(n, 0);
    for (int i = 0; i < n; ++i) {
      // ``py::object``, not ``py::handle``: a handle is a *borrowed* reference
      // owned only by ``buffers``, so Python code running during the field
      // reads below (a property, ``__getattr__``) could drop the element from
      // the list and free the object out from under us.
      py::object b = buffers[i];
      try {
        names[i] = b.attr("name").cast<std::string>();
        sizes_[i] = b.attr("size").cast<int64_t>();
        if (sizes_[i] < 0) {
          throw std::invalid_argument("buffer " + std::to_string(i) +
                                      ": size must be non-negative");
        }
        std::vector<int64_t> uses = b.attr("uses").cast<std::vector<int64_t>>();
        if (uses.empty()) {
          throw std::invalid_argument("buffer uses must be non-empty");
        }
        // end_time is uses.back() + 1, so the last use has to leave room for
        // it. A use past INT64_MAX already fails the cast above, making this
        // the only value that reaches the addition.
        if (uses.back() == std::numeric_limits<int64_t>::max()) {
          throw std::invalid_argument("buffer " + std::to_string(i) +
                                      ": last use must be below INT64_MAX");
        }
        bool first_read = b.attr("first_use_is_read").cast<bool>();
        parent_names[i] =
            b.attr("in_place_parents").cast<std::vector<std::string>>();
        st->start[i] = b.attr("start_time").cast<int64_t>();
        st->end[i] = b.attr("end_time").cast<int64_t>();
        st->weight[i] =
            static_cast<double>(uses.size()) + (first_read ? 0.0 : 0.5);
        const bool read_after_write =
            uses.size() > 1 && uses.back() > uses.front();
        can_hand_over[i] = (first_read || read_after_write) ? 1 : 0;
      }
      catch (const py::cast_error&) {
        // pybind11 turns a failed cast into a RuntimeError whose message names
        // neither the field nor the buffer (and, in a release build, not even
        // the target type). Every other rejection in this class is a
        // ValueError, so restate it as one and name the offending buffer.
        throw std::invalid_argument(
            "buffer " + std::to_string(i) +
            ": could not read fields (name, size, uses, first_use_is_read, "
            "in_place_parents)");
      }
    }

    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(n * 2);
    for (int i = 0; i < n; ++i) name_to_idx.emplace(names[i], i);
    // Names are the identity in-place parents are resolved by, so a duplicate
    // makes ``in_place_parents=["a"]`` ambiguous: first-wins (``emplace``) and
    // last-wins (Python's dict comprehension) are both arbitrary. Reject
    // instead of silently picking one. Mirrors the assert in
    // ``PermutationBasedLayoutSolverBase.__init__``.
    if (static_cast<int>(name_to_idx.size()) != n) {
      throw std::invalid_argument("buffer names must be unique");
    }

    // Time-overlap sets (half-open [start, end) intervals).
    st->overlap.assign(n, {});
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) {
        if (st->start[a] < st->end[c] && st->start[c] < st->end[a]) {
          st->overlap[a].push_back(c);
          st->overlap[c].push_back(a);
        }
      }
    }

    // In-place partners and declared-parent direction.
    st->declared_parents.assign(n, {});
    std::vector<std::unordered_set<int>> partner_sets(n);
    for (int child = 0; child < n; ++child) {
      for (const std::string& pname : parent_names[child]) {
        auto it = name_to_idx.find(pname);
        if (it == name_to_idx.end()) continue;
        int parent = it->second;
        // The two invariants the Python packer asserts where it resolves the
        // same pairs (``_compute_inplace_partners``), restated so the two
        // packers reject the same inputs and not merely agree on the layouts
        // they do produce. A write-only computed parent has no storage to hand
        // over; and ``PlaceDecision`` co-locates a declared partner on nothing
        // more than a time overlap, so a pair overlapping by more than the
        // handoff tick would be handed one address while both are live.
        if (!can_hand_over[parent]) {
          throw std::invalid_argument(
              "in-place parent " + names[parent] +
              " is a computed buffer that is never read, so it cannot hand its "
              "storage to child " +
              names[child]);
        }
        if (st->end[parent] != st->start[child] + 1) {
          throw std::invalid_argument(
              "in-place pair (" + names[parent] + ", " + names[child] +
              ") must hand off at a single tick: parent end_time " +
              std::to_string(st->end[parent]) + " != child start_time " +
              std::to_string(st->start[child]) + " + 1");
        }
        st->declared_parents[child].push_back(parent);
        partner_sets[child].insert(parent);
        partner_sets[parent].insert(child);
      }
    }
    st->inplace_partners.assign(n, {});
    for (int i = 0; i < n; ++i) {
      st->inplace_partners[i].assign(partner_sets[i].begin(),
                                     partner_sets[i].end());
    }

    BuildIntervalData(*st);

    st_ = std::move(st);

    // Dynamic state.
    permutation_ = std::move(permutation);
    position_.assign(n, 0);
    for (int pos = 0; pos < n; ++pos) position_[permutation_[pos]] = pos;
    if (eligible.is_none()) {
      eligible_.assign(n, 1);
    } else {
      std::vector<bool> flags = eligible.cast<std::vector<bool>>();
      if (static_cast<int>(flags.size()) != n) {
        throw std::invalid_argument("eligible must have one flag per buffer");
      }
      eligible_.resize(n);
      for (int i = 0; i < n; ++i) eligible_[i] = flags[i] ? 1 : 0;
    }
    addr_.assign(n, 0);
    allocated_.assign(n, 0);
    cand_mark_.assign(n, 0);
    RecomputeAll();
  }

  double swap(int i) {
    const int n = st_->n;
    if (i < 0 || i >= n - 1) {
      throw std::invalid_argument("swap index out of range");
    }
    const int x = permutation_[i];
    const int y = permutation_[i + 1];
    permutation_[i] = y;
    permutation_[i + 1] = x;
    position_[x] = i + 1;
    position_[y] = i;
    // No-op cases (identical to the Python packer): a transparent (HBM) member
    // is outside the eligible stacking order, and independent buffers do not
    // affect any address -- only the positions moved.
    if (!(eligible_[x] && eligible_[y])) return 0.0;
    if (!Overlaps(x, y)) return 0.0;
    const double old_total = total_quality_;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double rotate(int i, int j) {
    // Bounds first, then the i == j no-op: the other way round, an out-of-range
    // ``rotate(999, 999)`` would return 0.0 instead of raising.
    const int n = st_->n;
    if (i < 0 || i >= n || j < 0 || j >= n) {
      throw std::invalid_argument("rotate index out of range");
    }
    if (i == j) return 0.0;
    if (!eligible_[permutation_[i]]) {
      // Moving a transparent (HBM) buffer changes no eligible buffer's relative
      // order, so no address moves; just relocate its slot.
      MoveInPermutation(i, j);
      return 0.0;
    }
    const double old_total = total_quality_;
    MoveInPermutation(i, j);
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double resize(int idx, int64_t new_size) {
    if (idx < 0 || idx >= st_->n) {
      throw std::invalid_argument("resize index out of range");
    }
    // A negative footprint puts a buffer's top below its own address, breaking
    // the "stack on the max top" invariant the placer rests on -- and it is the
    // one input that made AlignUp overflow and the fast path disagree with the
    // candidate scan.
    if (new_size < 0) {
      throw std::invalid_argument("resize size must be non-negative");
    }
    if (sizes_[idx] == new_size) return 0.0;  // no-op: footprint unchanged
    const double old_total = total_quality_;
    sizes_[idx] = new_size;
    // An ineligible buffer is in HBM: its size affects nothing observable, but
    // the new size is recorded so a later set_eligible(True) uses it.
    if (!eligible_[idx]) return 0.0;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double set_eligible(int idx, bool flag) {
    if (idx < 0 || idx >= st_->n) {
      throw std::invalid_argument("set_eligible index out of range");
    }
    if (static_cast<bool>(eligible_[idx]) == flag) return 0.0;
    const double old_total = total_quality_;
    eligible_[idx] = flag ? 1 : 0;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  NativePermutationLayoutSolver copy() const {
    return *this;
  }

  double quality() const {
    return total_quality_;
  }
  int count_allocated() const {
    return total_allocated_count_;
  }

  py::list addresses() const {
    py::list out;
    for (int i = 0; i < st_->n; ++i) {
      if (allocated_[i]) {
        out.append(py::cast(addr_[i]));
      } else {
        out.append(py::none());
      }
    }
    return out;
  }

  // Current allocation order, as a live read-only view (see PermutationView).
  PermutationView permutation() const {
    return PermutationView(*this);
  }

  const std::vector<int>& permutation_vec() const {
    return permutation_;
  }

  py::list buffers() const {
    return buffers_;
  }

  // Exclusive top (``address + size``) of buffer ``idx``, or +inf when it is
  // evicted. Lives on the packer rather than in the search so it reads the
  // plan-local ``sizes_`` -- which ``resize`` mutates -- instead of the shared
  // buffer object's ``size``, which goes stale the moment a co-optimizer starts
  // driving resizes. An evicted buffer sorting as arbitrarily high is what
  // keeps it from being reordered below a placed one.
  double top_or_inf(int idx) const {
    if (idx < 0 || idx >= st_->n) {
      throw std::invalid_argument("top_or_inf index out of range");
    }
    if (!allocated_[idx]) {
      return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(addr_[idx] + sizes_[idx]);
  }

  // Write each buffer's address back onto the Python buffer object, for EVERY
  // index -- an evicted buffer gets ``None`` -- exactly like
  // ``PermutationBasedLayoutSolverBase.finalize``.
  void finalize() const {
    for (int i = 0; i < st_->n; ++i) {
      py::object value =
          allocated_[i] ? py::cast(addr_[i]) : py::object(py::none());
      buffers_[i].attr("address") = value;
    }
  }

  // True if buffers ``i`` and ``j`` are alive at a common tick. Buffer-index
  // args, bounds-checked; mirrors the Python base ``overlaps(i, j)``.
  bool overlaps(int i, int j) const {
    const int n = st_->n;
    if (i < 0 || i >= n || j < 0 || j >= n) {
      throw std::invalid_argument("overlaps index out of range");
    }
    return Overlaps(i, j);
  }

  // True if buffer ``idx`` has an address (fits below capacity).
  // Bounds-checked; mirrors the Python base ``is_fully_allocated(idx)``.
  bool is_fully_allocated(int idx) const {
    if (idx < 0 || idx >= st_->n) {
      throw std::invalid_argument("is_fully_allocated index out of range");
    }
    return allocated_[idx] != 0;
  }

 private:
  // Mirrors PermutationBasedLayoutSolverBase._build_interval_data. Takes a
  // reference (not a pointer): the StaticData is a required, non-null argument.
  static void BuildIntervalData(StaticData& st) {
    const int n = st.n;
    std::vector<int64_t> pts;
    pts.reserve(2 * n);
    // Order does not matter -- pts is sorted below -- so bulk-copy both arrays.
    pts.insert(pts.end(), st.start.begin(), st.start.end());
    pts.insert(pts.end(), st.end.begin(), st.end.end());
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    const int k = std::max(0, static_cast<int>(pts.size()) - 1);
    st.num_intervals = k;
    st.total_at.assign(k, 0);
    st.buf_intervals.assign(n, {0, 0});
    if (k == 0) return;
    // Delta sweep: +1 at each start interval, -1 at each end interval.
    std::vector<int> deltas(k + 1, 0);
    auto bisect_left = [&pts](int64_t v) {
      return static_cast<int>(std::lower_bound(pts.begin(), pts.end(), v) -
                              pts.begin());
    };
    for (int i = 0; i < n; ++i) {
      int lo = bisect_left(st.start[i]);
      int hi = bisect_left(st.end[i]);
      deltas[lo] += 1;
      deltas[hi] -= 1;
      st.buf_intervals[i] = {lo, hi};
    }
    int running = 0;
    for (int i = 0; i < k; ++i) {
      running += deltas[i];
      st.total_at[i] = running;
    }
  }

  bool Overlaps(int x, int y) const {
    return st_->start[x] < st_->end[y] && st_->start[y] < st_->end[x];
  }

  int64_t AlignUp(int64_t addr) const {
    const int64_t a = st_->alignment;
    return ((addr + a - 1) / a) * a;
  }

  // Returns (parent, child) for an in-place pair. One of the two members must
  // declare the other as an in-place parent. Mirrors
  // PermutationBasedLayoutSolverBase._in_place_pair.
  std::pair<int, int> InPlacePair(int i, int j) const {
    const std::vector<int>& dp = st_->declared_parents[i];
    if (std::find(dp.begin(), dp.end(), j) != dp.end()) {
      return {j, i};  // j parents i
    }
    return {i, j};  // i parents j
  }

  // Mirrors PermutationBasedLayoutSolverBase._placement_decision. ``cand`` are
  // the already-placed, time-overlapping, eligible candidates for ``idx``.
  // Returns {placed, addr}: {true, addr} if placed, {false, 0} if evicted
  // (None).
  std::pair<bool, int64_t> PlaceDecision(int idx,
                                         const std::vector<int>& cand) {
    const int64_t cap = st_->capacity;
    if (cand.empty()) {
      // Lone buffer sits on the floor unless it alone exceeds capacity.
      if (sizes_[idx] > cap) return {false, 0};
      return {true, 0};
    }
    // A None (evicted) candidate dominates: idx would rest on it.
    for (int p : cand) {
      if (!allocated_[p]) return {false, 0};
    }
    int64_t max_top = std::numeric_limits<int64_t>::min();
    for (int p : cand) {
      const int64_t top = addr_[p] + sizes_[p];
      if (top > max_top) max_top = top;
    }
    // Try to drop into an in-place partner's slot.
    const std::vector<int>& partners = st_->inplace_partners[idx];
    if (!partners.empty()) {
      ++cand_gen_;
      for (int p : cand) cand_mark_[p] = cand_gen_;
      for (int partner : partners) {
        if (cand_mark_[partner] != cand_gen_) continue;
        std::pair<int, int> pr = InPlacePair(idx, partner);
        const int parent = pr.first;
        const int child = pr.second;
        if (sizes_[child] > sizes_[parent]) continue;  // _can_inplace
        const int64_t partner_addr = addr_[partner];
        int64_t others_top = 0;
        for (int q : cand) {
          if (q == partner) continue;
          const int64_t top = addr_[q] + sizes_[q];
          if (top > others_top) others_top = top;
        }
        if (others_top <= partner_addr) {
          if (partner_addr + sizes_[idx] > cap) return {false, 0};
          return {true, partner_addr};
        }
      }
    }
    const int64_t aligned = AlignUp(max_top);
    if (aligned + sizes_[idx] > cap) return {false, 0};
    return {true, aligned};
  }

  // Places every buffer in permutation order with the saturation early-stop,
  // rebuilding addresses, quality and the allocated count from scratch. Mirrors
  // PermutationBasedLayoutSolverBase._sequential_place.
  // --- RecomputeAll phases ---

  // Reset the saturation early-stop state and return the count of not-yet-
  // saturated intervals (those with at least one live buffer).
  int InitIntervals() {
    const int k = st_->num_intervals;
    placed_at_.assign(k, 0);
    has_none_at_.assign(k, 0);
    done_at_.assign(k, 0);
    max_top_at_.assign(k, 0);
    int not_done = 0;
    for (int t = 0; t < k; ++t) {
      if (st_->total_at[t] == 0) {
        done_at_[t] = 1;
      } else {
        ++not_done;
      }
    }
    return not_done;
  }

  // Gather idx's already-placed, time-overlapping, eligible candidates into
  // cand_ (the buffers idx would stack on top of).
  void GatherCandidates(int idx, int pos) {
    cand_.clear();
    for (int w : st_->overlap[idx]) {
      if (position_[w] < pos && eligible_[w]) cand_.push_back(w);
    }
  }

  // Advance the per-interval aggregates over idx's live intervals and update
  // not_done. ``set_none`` marks the interval as holding an evicted (None)
  // buffer (only ever set on the eligible-and-evicted path). ``top`` is a
  // placed buffer's exclusive top (addr + size), folded into the running
  // per-interval max; pass a negative value for buffers that do not stack
  // (evicted, or ineligible/HBM) so it never updates the max (max_top_at_ is
  // always >= 0). These aggregates let the common no-in-place-partner placement
  // read its floor and eviction straight off its intervals -- see RecomputeAll.
  //
  // (The former ineligible branch tested placed_at_ == total_at without the
  // has_none_at_ term; the unified test here is equivalent because has_none_at_
  // can only be true once done_at_ is already set, so it never fires alone.)
  void CommitIntervals(int idx, bool set_none, int64_t top, int& not_done) {
    const int lo = st_->buf_intervals[idx].first;
    const int hi = st_->buf_intervals[idx].second;
    for (int t = lo; t < hi; ++t) {
      ++placed_at_[t];
      if (set_none) {
        has_none_at_[t] = 1;
      } else if (top > max_top_at_[t]) {
        max_top_at_[t] = top;
      }
      if (!done_at_[t] &&
          (has_none_at_[t] || placed_at_[t] == st_->total_at[t])) {
        done_at_[t] = 1;
        --not_done;
      }
    }
  }

  void RecomputeAll() {
    const int n = st_->n;
    const int64_t cap = st_->capacity;
    // position_ is the inverse of permutation_ and is maintained incrementally
    // by every mutator (constructor, swap, rotate/MoveInPermutation) -- resize
    // and set_eligible leave the permutation untouched -- so it is always in
    // sync here and needs no rebuild.
    total_quality_ = 0.0;
    total_allocated_count_ = 0;
    int not_done = InitIntervals();

    int stop = n;
    for (int pos = 0; pos < n; ++pos) {
      if (not_done == 0) {
        stop = pos;
        break;
      }
      const int idx = permutation_[pos];
      if (!eligible_[idx]) {
        // Ineligible: routed to HBM, transparent to the stack. No address / no
        // quality, and it does not mark its intervals has_none (nothing rests
        // on it), but it is still counted into placed_at so an interval whose
        // remaining occupants are all ineligible can still saturate.
        allocated_[idx] = 0;
        CommitIntervals(idx, /*set_none=*/false, /*top=*/-1, not_done);
        continue;
      }
      const int lo = st_->buf_intervals[idx].first;
      const int hi = st_->buf_intervals[idx].second;
      bool evicted;
      int64_t a;
      if (st_->inplace_partners[idx].empty()) {
        // Fast path (no in-place partner): PlaceDecision needs only whether an
        // overlapping earlier buffer was evicted (a None dominates) and the max
        // top to stack on -- both are per-interval aggregates, so we skip
        // gathering the candidate list. Bit-exact with the scan: idx's
        // intervals are exactly the times it overlaps an earlier buffer, so the
        // max over them of max_top_at_ equals the candidate max-top, and
        // has_none_at_ on any of them means a None candidate exists.
        bool none = false;
        int64_t max_top = 0;
        for (int t = lo; t < hi; ++t) {
          if (has_none_at_[t]) {
            none = true;
            break;
          }
          if (max_top_at_[t] > max_top) max_top = max_top_at_[t];
        }
        if (none) {
          evicted = true;
          a = 0;
        } else {
          const int64_t aligned = AlignUp(max_top);
          evicted = aligned + sizes_[idx] > cap;
          a = evicted ? 0 : aligned;
        }
      } else {
        // In-place partner present: needs the real candidate list for the
        // partner-slot logic.
        GatherCandidates(idx, pos);
        const auto [placed, addr] = PlaceDecision(idx, cand_);
        evicted = !placed;
        a = addr;
      }
      allocated_[idx] = evicted ? 0 : 1;
      addr_[idx] = a;
      if (!evicted) {
        total_quality_ += st_->weight[idx] * static_cast<double>(sizes_[idx]);
        ++total_allocated_count_;
        CommitIntervals(idx, /*set_none=*/false, /*top=*/a + sizes_[idx],
                        not_done);
      } else {
        CommitIntervals(idx, /*set_none=*/true, /*top=*/-1, not_done);
      }
    }
    for (int pos = stop; pos < n; ++pos) allocated_[permutation_[pos]] = 0;
  }

  void MoveInPermutation(int i, int j) {
    const auto begin = permutation_.begin();
    // Move the element at i to j by rotating the [lo, hi] subrange by one: a
    // left rotate when moving forward (i < j), a right rotate when moving back.
    if (i < j) {
      std::rotate(begin + i, begin + i + 1, begin + j + 1);
    } else {
      std::rotate(begin + j, begin + i, begin + i + 1);
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    for (int p = lo; p <= hi; ++p) position_[permutation_[p]] = p;
  }

  std::shared_ptr<const StaticData> st_;

  // The Python buffer objects, retained for finalize(). Shared with clones (the
  // copy ctor copies the handle, not the list). Deliberately NOT in StaticData:
  // that struct is held by const shared_ptr and is Python-free, so it can be
  // destroyed without the GIL -- and finalize() writes through this handle.
  py::list buffers_;

  // Dynamic per-plan state (deep-copied by copy()).
  std::vector<int> permutation_;
  std::vector<int> position_;   // inverse of permutation_
  std::vector<int64_t> sizes_;  // mutable footprint (resize)
  std::vector<char> eligible_;  // LX-eligibility (set_eligible)
  std::vector<int64_t> addr_;   // address; valid iff allocated_[i]
  std::vector<char> allocated_;
  double total_quality_ = 0.0;
  int total_allocated_count_ = 0;

  // Scratch reused across RecomputeAll / PlaceDecision calls. cand_mark_ is a
  // generation-stamped membership set (replacing Python's
  // partners.intersection(candidates)): PlaceDecision bumps cand_gen_ and
  // stamps every candidate, so cand_mark_[p] == cand_gen_ tests "p is a
  // candidate" without clearing the array each call. uint64_t so the counter
  // never wraps.
  std::vector<int> cand_;
  std::vector<uint64_t> cand_mark_;
  uint64_t cand_gen_ = 0;
  std::vector<int> placed_at_;
  std::vector<char> has_none_at_;
  std::vector<char> done_at_;
  std::vector<int64_t> max_top_at_;  // running max exclusive-top per interval
};

const std::vector<int>& PermutationView::vec() const {
  return owner_.permutation_vec();
}

int64_t PermutationView::len() const {
  return static_cast<int64_t>(vec().size());
}

int PermutationView::getitem(int64_t i) const {
  const int64_t n = len();
  if (i < 0) i += n;  // list semantics
  if (i < 0 || i >= n) {
    // py::index_error, not invalid_argument: ``list(view)`` and ``for x in
    // view`` walk the sequence protocol until IndexError stops them.
    throw py::index_error("permutation index out of range");
  }
  return vec()[static_cast<size_t>(i)];
}

bool PermutationView::eq(const py::object& other) const {
  if (py::isinstance<PermutationView>(other)) {
    return vec() == other.cast<const PermutationView&>().vec();
  }
  // Elementwise against a plain list -- how the search compares the live order
  // with its stored best-so-far snapshot.
  try {
    return vec() == other.cast<std::vector<int>>();
  }
  catch (const py::cast_error&) {
    return false;
  }
}

py::list PermutationView::to_list() const {
  py::list out;
  for (int idx : vec()) out.append(py::cast(idx));
  return out;
}

std::string PermutationView::repr() const {
  std::string s = "PermutationView([";
  const std::vector<int>& v = vec();
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(v[i]);
  }
  return s + "])";
}

}  // namespace

void register_perm_layout_native(py::module_& m) {
  py::class_<PermutationView>(m, "PermutationView")
      .def("__len__", &PermutationView::len)
      .def("__getitem__", &PermutationView::getitem, py::arg("i"))
      .def("__eq__", &PermutationView::eq, py::arg("other"))
      .def("__copy__", &PermutationView::to_list)
      .def(
          "__deepcopy__",
          [](const PermutationView& v, const py::object&) {
            return v.to_list();
          },
          py::arg("memo"))
      .def("__repr__", &PermutationView::repr);

  py::class_<NativePermutationLayoutSolver>(m, "NativePermutationLayoutSolver")
      .def(py::init<const py::list&, std::vector<int>, int64_t, int64_t,
                    const py::object&>(),
           py::arg("buffers"), py::arg("permutation"), py::arg("capacity"),
           py::arg("alignment") = kDefaultAlignment,
           py::arg("eligible") = py::none())
      .def("swap", &NativePermutationLayoutSolver::swap, py::arg("i"))
      .def("rotate", &NativePermutationLayoutSolver::rotate, py::arg("i"),
           py::arg("j"))
      .def("resize", &NativePermutationLayoutSolver::resize, py::arg("idx"),
           py::arg("new_size"))
      .def("set_eligible", &NativePermutationLayoutSolver::set_eligible,
           py::arg("idx"), py::arg("flag"))
      .def("copy", &NativePermutationLayoutSolver::copy)
      .def("quality", &NativePermutationLayoutSolver::quality)
      .def("count_allocated", &NativePermutationLayoutSolver::count_allocated)
      .def("overlaps", &NativePermutationLayoutSolver::overlaps, py::arg("i"),
           py::arg("j"))
      .def("is_fully_allocated",
           &NativePermutationLayoutSolver::is_fully_allocated, py::arg("idx"))
      .def("top_or_inf", &NativePermutationLayoutSolver::top_or_inf,
           py::arg("idx"))
      .def("finalize", &NativePermutationLayoutSolver::finalize)
      .def_property_readonly("buffers", &NativePermutationLayoutSolver::buffers)
      // The view borrows the solver, so the solver must outlive it. The
      // keep_alive has to be attached to an explicit cpp_function: passed as a
      // trailing extra to def_property_readonly it is applied to the property
      // rather than to the getter, silently does nothing, and the view is left
      // reading freed memory once the last reference to the solver goes away.
      .def_property_readonly(
          "permutation",
          py::cpp_function(&NativePermutationLayoutSolver::permutation,
                           py::keep_alive<0, 1>()))
      .def_property_readonly("addresses",
                             &NativePermutationLayoutSolver::addresses);
}

}  // namespace scratchpad
}  // namespace torch_spyre
