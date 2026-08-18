/*
 * Copyright 2025 The Torch-Spyre Authors.
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

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>  // TORCH_WARN_ONCE
#include <module.h>
#include <util/sendefs/sendefs.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spyre {

inline std::unordered_map<c10::ScalarType, std::string> torchScalarToString = {
    /* this ensures the same representation regardless of how PyTorch changes
       its type names we will use this to map to DT and SenDnn names
    */
    {c10::kByte, "uint8"},
    {c10::kChar, "int8"},
    {c10::kFloat8_e4m3fn, "fp8_143"},    // fn=finite-form
    {c10::kFloat8_e5m2fnuz, "fp8_152"},  // fnuz=finite-form+unsigned zero
    {c10::kShort, "int16"},
    {c10::kInt, "int32"},
    {c10::kLong, "int64"},
    {c10::kHalf, "float16"},
    {c10::kFloat, "float32"},
    {c10::kDouble, "float64"},
    {c10::kBool, "bool"},
    {c10::kBFloat16, "bfloat16"},
    {c10::kComplexHalf, "complex32"},
    {c10::kComplexFloat, "complex64"},
    {c10::kComplexDouble, "complex128"},
    {c10::kQInt8, "qint8"},
    {c10::kQUInt8, "quint8"},
    {c10::kQInt32, "qint32"},
    {c10::kQUInt4x2, "quint4x2"},
    {c10::kQUInt2x4, "quint2x4"},
    {c10::ScalarType::Undefined, "undefined"}};

inline std::pair<DataFormats, DataFormats> stringToDTDataFormatPair(
    const std::string& type_name) {
  /* val-1 = type on CPU-side
   * val-2 = type on Spyre-side
   */
  static const std::unordered_map<std::string,
                                  std::pair<DataFormats, DataFormats>>
      type_map = {
          {"float16", {DataFormats::IEEE_FP16, DataFormats::SEN169_FP16}},
          {"float32", {DataFormats::IEEE_FP32, DataFormats::IEEE_FP32}},
          {"int8", {DataFormats::SENINT8, DataFormats::SENINT8}},
          {"int16", {DataFormats::SENINT16, DataFormats::SENINT16}},
          {"int32", {DataFormats::IEEE_INT32, DataFormats::IEEE_INT32}},
          {"int64", {DataFormats::IEEE_INT64, DataFormats::IEEE_INT32}},
          {"bool", {DataFormats::BOOL, DataFormats::SEN169_FP16}},
          {"bfloat16", {DataFormats::BFLOAT16, DataFormats::SEN169_FP16}},
          {"quint8", {DataFormats::SENUINT32, DataFormats::SENUINT32}},
          {"qint8", {DataFormats::SENINT8, DataFormats::SENINT8}},
          {"quint4x2", {DataFormats::SENUINT2, DataFormats::SENUINT2}},
          {"quint2x4", {DataFormats::SENUINT2, DataFormats::SENUINT2}},
          {"uint8", {DataFormats::SENUINT32, DataFormats::SENUINT32}},
          {"int4", {DataFormats::SENINT4, DataFormats::SENINT4}},
          {"int2", {DataFormats::SENINT2, DataFormats::SENINT2}},
          {"fp8_143", {DataFormats::SEN143_FP8, DataFormats::SEN143_FP8}},
          {"fp8_152", {DataFormats::SEN152_FP8, DataFormats::SEN152_FP8}},
          {"fp9_153", {DataFormats::SEN153_FP9, DataFormats::SEN153_FP9}},
          {"int24", {DataFormats::SENINT24, DataFormats::SENINT24}},
          // Add more mappings as needed
      };

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    if (spyre::get_downcast_warn_enabled()) {
      std::vector<std::string> allowed = {
          "int64",
      };
      if (std::find(allowed.begin(), allowed.end(), type_name) !=
          allowed.end()) {
        TORCH_WARN_ONCE(
            "Backend Spyre does not support int64; downcasting to int32 may "
            "change values "
            "outside the 32-bit range. "
            "You can silence this via warnings.filterwarnings(...) or "
            "spyre.set_downcast_warning(False) or " SPYRE_DOWNCAST_ENV
            " env. variable.");
      }
    }
    return it->second;
  }
  return {DataFormats::INVALID, DataFormats::INVALID};
}

// Returns true if deeptools ConvertData_general_shuffle supports the given
// (src, dst) DataFormats pair.  The set of supported pairs is derived from
// the exhaustive if/else-if chain in
// deeptools/spyrecode-host-functions/sendataconvert/sen_data_convert.cpp ::
// ConvertData_general_shuffle.
inline bool isDCIConversionSupported(DataFormats src, DataFormats dst) {
  // clang-format off
  using DF = DataFormats;
  // Pairs listed as (src, dst) matching the order in
  // ConvertData_general_shuffle.
  static const std::pair<DataFormats, DataFormats> kSupported[] = {
      {DF::IEEE_FP32,   DF::IEEE_FP32},
      {DF::SEN169_FP16, DF::IEEE_FP32},
      {DF::IEEE_FP16,   DF::IEEE_FP32},
      {DF::IEEE_FP32,   DF::SEN169_FP16},
      {DF::SEN169_FP16, DF::SEN169_FP16},
      {DF::SEN169_FP16, DF::SENINT16},
      {DF::SENINT16,    DF::SEN169_FP16},
      {DF::IEEE_FP32,   DF::BFLOAT16},
      {DF::IEEE_FP16,   DF::BFLOAT16},
      {DF::BFLOAT16,    DF::IEEE_FP32},
      {DF::SEN143_FP8,  DF::IEEE_FP32},
      {DF::IEEE_FP32,   DF::SEN143_FP8},
      {DF::SEN143_FP8,  DF::SEN143_FP8},
      {DF::SENINT4,     DF::IEEE_FP32},
      {DF::IEEE_FP32,   DF::SENINT4},
      {DF::SENINT8,     DF::IEEE_FP32},
      {DF::IEEE_FP32,   DF::SENINT8},
      {DF::SENUINT2,    DF::IEEE_FP32},
      {DF::IEEE_FP32,   DF::SENUINT2},
      {DF::BOOL,        DF::SEN169_FP16},
      {DF::SEN169_FP16, DF::BOOL},
      {DF::IEEE_INT64,  DF::SEN169_FP16},
      {DF::IEEE_FP16,   DF::SEN169_FP16},
      {DF::SEN169_FP16, DF::IEEE_FP16},
      {DF::SENINT4,     DF::SENINT4},
      {DF::SENINT8,     DF::SENINT8},
      {DF::IEEE_INT64,  DF::IEEE_INT32},
      {DF::IEEE_INT32,  DF::IEEE_INT64},
      {DF::IEEE_INT32,  DF::IEEE_INT32},
      {DF::SENUINT32,   DF::IEEE_INT64},
      {DF::IEEE_INT64,  DF::SENUINT32},
      {DF::SENUINT32,   DF::SENUINT32},
      {DF::SEN169_FP16, DF::IEEE_INT64},
      {DF::BFLOAT16,    DF::SEN169_FP16},
      {DF::SEN169_FP16, DF::BFLOAT16},
  };
  // clang-format on
  for (const auto& p : kSupported) {
    if (p.first == src && p.second == dst) {
      return true;
    }
  }
  return false;
}

inline std::pair<size_t, size_t> elementSize(const c10::ScalarType& dtype) {
  /* return size (bytes) on CPU and on Spyre*/
  static const std::unordered_map<c10::ScalarType, std::pair<size_t, size_t>>
      itemsize_map = {
          {c10::kBool, {1, 2}},
      };
  auto it = itemsize_map.find(dtype);
  if (it != itemsize_map.end()) {
    return it->second;
  }
  auto val = c10::elementSize(dtype);
  return {val, val};
}
}  // namespace spyre
