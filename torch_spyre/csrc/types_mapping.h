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
#include <util/sendefs.h>

#include <sendnn/tensor/sendatatype.hpp>
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
          {"bfloat16", {DataFormats::BFLOAT16, DataFormats::BFLOAT16}},
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

inline std::pair<sendnn::sen_datatype_enum, sendnn::sen_datatype_enum>
stringToSenDatatypePair(const std::string& type_name) {
  /* val-1 = type on CPU-side
   * val-2 = type on Spyre-side
   */
  static const std::unordered_map<
      std::string,
      std::pair<sendnn::sen_datatype_enum, sendnn::sen_datatype_enum>>
      type_map = {
          // Boolean and string
          {"bool",
           {sendnn::sen_datatype_enum::boolean,
            sendnn::sen_datatype_enum::sen_fp16}},
          {"string",
           {sendnn::sen_datatype_enum::string,
            sendnn::sen_datatype_enum::string}},

          // IEEE floats
          {"fp8_143",
           {sendnn::sen_datatype_enum::float8,
            sendnn::sen_datatype_enum::sen_fp8}},
          // TODO(tmhoangt): figure out why there is not FP8 variant specific in
          // sen_datatype_enum
          {"fp8_152",
           {sendnn::sen_datatype_enum::float8,
            sendnn::sen_datatype_enum::sen_fp8}},
          {"float16",
           {sendnn::sen_datatype_enum::float16,
            sendnn::sen_datatype_enum::sen_fp16}},
          {"float32",
           {sendnn::sen_datatype_enum::float32,
            sendnn::sen_datatype_enum::float32}},
          {"float64",
           {sendnn::sen_datatype_enum::float64,
            sendnn::sen_datatype_enum::float64}},
          {"float128",
           {sendnn::sen_datatype_enum::float128,
            sendnn::sen_datatype_enum::float128}},
          {"float256",
           {sendnn::sen_datatype_enum::float256,
            sendnn::sen_datatype_enum::float256}},

          // Decimal
          {"decimal32",
           {sendnn::sen_datatype_enum::decimal32,
            sendnn::sen_datatype_enum::decimal32}},
          {"decimal64",
           {sendnn::sen_datatype_enum::decimal64,
            sendnn::sen_datatype_enum::decimal64}},
          {"decimal128",
           {sendnn::sen_datatype_enum::decimal128,
            sendnn::sen_datatype_enum::decimal128}},

          // bfloat
          {"bfloat16",
           {sendnn::sen_datatype_enum::bfloat16,
            sendnn::sen_datatype_enum::bfloat16}},
          {"bfloat16_compute",
           {sendnn::sen_datatype_enum::bfloat16,
            sendnn::sen_datatype_enum::float32}},

          // Signed ints
          {"int1",
           {sendnn::sen_datatype_enum::int1,
            sendnn::sen_datatype_enum::sen_int1}},
          {"int2",
           {sendnn::sen_datatype_enum::int2,
            sendnn::sen_datatype_enum::sen_int2}},
          {"int4",
           {sendnn::sen_datatype_enum::int4,
            sendnn::sen_datatype_enum::sen_int4}},
          {"int8",
           {sendnn::sen_datatype_enum::int8,
            sendnn::sen_datatype_enum::sen_int8}},
          {"int16",
           {sendnn::sen_datatype_enum::int16,
            sendnn::sen_datatype_enum::sen_int16}},
          {"int32",
           {sendnn::sen_datatype_enum::int32,
            sendnn::sen_datatype_enum::sen_int32}},
          {"int64",
           {sendnn::sen_datatype_enum::int64,
            sendnn::sen_datatype_enum::sen_int32}},

          // Unsigned ints
          {"uint1",
           {sendnn::sen_datatype_enum::uint1,
            sendnn::sen_datatype_enum::sen_uint1}},
          {"uint2",
           {sendnn::sen_datatype_enum::uint2,
            sendnn::sen_datatype_enum::sen_uint2}},
          {"uint4",
           {sendnn::sen_datatype_enum::uint4,
            sendnn::sen_datatype_enum::sen_uint4}},
          {"uint8",
           {sendnn::sen_datatype_enum::uint8,
            sendnn::sen_datatype_enum::sen_uint8}},
          {"uint16",
           {sendnn::sen_datatype_enum::uint16,
            sendnn::sen_datatype_enum::sen_uint16}},
          {"uint32",
           {sendnn::sen_datatype_enum::uint32,
            sendnn::sen_datatype_enum::sen_uint32}},
          {"uint64",
           {sendnn::sen_datatype_enum::uint64,
            sendnn::sen_datatype_enum::sen_uint32}},

          // Quantized ints
          {"qint1",
           {sendnn::sen_datatype_enum::qint1,
            sendnn::sen_datatype_enum::qint1}},
          {"qint2",
           {sendnn::sen_datatype_enum::qint2,
            sendnn::sen_datatype_enum::qint2}},
          {"qint4",
           {sendnn::sen_datatype_enum::qint4,
            sendnn::sen_datatype_enum::qint4}},
          {"qint8",
           {sendnn::sen_datatype_enum::qint8,
            sendnn::sen_datatype_enum::qint8}},
          {"qint16",
           {sendnn::sen_datatype_enum::qint16,
            sendnn::sen_datatype_enum::qint16}},
          {"qint32",
           {sendnn::sen_datatype_enum::qint32,
            sendnn::sen_datatype_enum::qint32}},
          {"qint64",
           {sendnn::sen_datatype_enum::qint64,
            sendnn::sen_datatype_enum::qint64}},

          {"quint1",
           {sendnn::sen_datatype_enum::quint1,
            sendnn::sen_datatype_enum::quint1}},
          {"quint2",
           {sendnn::sen_datatype_enum::quint2,
            sendnn::sen_datatype_enum::quint2}},
          {"quint4",
           {sendnn::sen_datatype_enum::quint4,
            sendnn::sen_datatype_enum::quint4}},
          {"quint8",
           {sendnn::sen_datatype_enum::quint8,
            sendnn::sen_datatype_enum::quint8}},
          {"quint16",
           {sendnn::sen_datatype_enum::quint16,
            sendnn::sen_datatype_enum::quint16}},
          {"quint32",
           {sendnn::sen_datatype_enum::quint32,
            sendnn::sen_datatype_enum::quint32}},
          {"quint64",
           {sendnn::sen_datatype_enum::quint64,
            sendnn::sen_datatype_enum::quint64}},

          // Complex
          {"complex64",
           {sendnn::sen_datatype_enum::complex64,
            sendnn::sen_datatype_enum::complex64}},
          {"complex128",
           {sendnn::sen_datatype_enum::complex128,
            sendnn::sen_datatype_enum::complex128}},

          // Special
          {"variant",
           {sendnn::sen_datatype_enum::variant,
            sendnn::sen_datatype_enum::variant}},
          {"resource",
           {sendnn::sen_datatype_enum::resource,
            sendnn::sen_datatype_enum::resource}},

          // Sentient types
          {"sen_fp8",
           {sendnn::sen_datatype_enum::sen_fp8,
            sendnn::sen_datatype_enum::sen_fp8}},
          {"sen_fp16",
           {sendnn::sen_datatype_enum::sen_fp16,
            sendnn::sen_datatype_enum::sen_fp16}},
          {"sen_fp8_compute",
           {sendnn::sen_datatype_enum::sen_fp8,
            sendnn::sen_datatype_enum::float32}},
          {"sen_fp16_compute",
           {sendnn::sen_datatype_enum::sen_fp16,
            sendnn::sen_datatype_enum::float32}},

          {"sen_int1",
           {sendnn::sen_datatype_enum::sen_int1,
            sendnn::sen_datatype_enum::sen_int1}},
          {"sen_int2",
           {sendnn::sen_datatype_enum::sen_int2,
            sendnn::sen_datatype_enum::sen_int2}},
          {"sen_int4",
           {sendnn::sen_datatype_enum::sen_int4,
            sendnn::sen_datatype_enum::sen_int4}},
          {"sen_int8",
           {sendnn::sen_datatype_enum::sen_int8,
            sendnn::sen_datatype_enum::sen_int8}},
          {"sen_int16",
           {sendnn::sen_datatype_enum::sen_int16,
            sendnn::sen_datatype_enum::sen_int16}},
          {"sen_int24",
           {sendnn::sen_datatype_enum::sen_int24,
            sendnn::sen_datatype_enum::sen_int24}},
          {"sen_int32",
           {sendnn::sen_datatype_enum::sen_int32,
            sendnn::sen_datatype_enum::sen_int32}},
          {"sen_int4_compute",
           {sendnn::sen_datatype_enum::sen_int4,
            sendnn::sen_datatype_enum::int32}},
          {"sen_int8_compute",
           {sendnn::sen_datatype_enum::sen_int8,
            sendnn::sen_datatype_enum::int32}},

          {"sen_uint1",
           {sendnn::sen_datatype_enum::sen_uint1,
            sendnn::sen_datatype_enum::sen_uint1}},
          {"sen_uint2",
           {sendnn::sen_datatype_enum::sen_uint2,
            sendnn::sen_datatype_enum::sen_uint2}},
          {"sen_uint4",
           {sendnn::sen_datatype_enum::sen_uint4,
            sendnn::sen_datatype_enum::sen_uint4}},
          {"sen_uint8",
           {sendnn::sen_datatype_enum::sen_uint8,
            sendnn::sen_datatype_enum::sen_uint8}},
          {"sen_uint16",
           {sendnn::sen_datatype_enum::sen_uint16,
            sendnn::sen_datatype_enum::sen_uint16}},
          {"sen_uint24",
           {sendnn::sen_datatype_enum::sen_uint24,
            sendnn::sen_datatype_enum::sen_uint24}},
          {"sen_uint32",
           {sendnn::sen_datatype_enum::sen_uint32,
            sendnn::sen_datatype_enum::sen_uint32}},
      };

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    return it->second;
  }
  return {sendnn::sen_datatype_enum::dt_undef,
          sendnn::sen_datatype_enum::dt_undef};
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
