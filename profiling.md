# Test Performance Summary

## Suite 1

| Test Name                               | Real Time | Max Mem (MB) |
|-----------------------------------------|-----------|---------------|
| `test_view_ops_config`                  | 0m16.812s | 728           |
| `test_profiler_config`                  | 0m40.291s | 696           |

---

## Suite 2

| Test Name                               | Real Time | Max Mem (MB) |
|-----------------------------------------|-----------|---------------|
| `granite_3_3_8b_instruct_spyre`         | 0m17.722s | 522           |

---

## Suite 3

| Test Name                               | Real Time | Max Mem (MB) |
|-----------------------------------------|-----------|---------------|
| `test_device_enum_config`               | 0m37.550s | 903           |
| `test_fallbacks_config`                 | 0m14.716s | 527           |
| `test_modules_config`                   | 0m37.292s | 12103         |
| `test_regex_config`                     | 0m8.462s  | 469           |
| `test_spyre_config`                     | 0m19.957s | 638           |
| `test_spyre_lazy_silent_config`         | 0m11.720s | 927           |
| `test_stream_config`                    | 0m16.175s | 598           |
| `test_spyre_profiler_config`            | 0m0.002s  | 16            |
| `test_prepare_kernel_config`            | 0m0.002s  | 16            |
| `test_building_blocks_config`           | 0m56.015s | 1065          |
| `test_codegen_config`                   | 0m17.274s | 703           |
| `test_decomp_config`                    | 0m51.633s | 985           |
| `test_inductor_fx_passes_config`        | 0m54.178s | 1022          |
| `test_normalization_scalars_config`     | 1m2.548s  | 3223          |
| `test_inductor_ops_config`              | 8m22.035s | 11051         |
| `test_ops_lx_planning_config`           | 2m55.050s | 9246          |
| `test_inductor_scalar_config`           | 0m35.969s | 765           |
| `test_logging_config`                   | 0m8.519s  | 467           |
| `test_dedup_constants_config`           | 0m18.394s | 710           |
| `test_padding_config`                   | 0m22.853s | 717           |
| `test_overwrite`                        | 0m0.002s  | 16            |
| `test_restickify_config`                | 1m21.014s | 916           |
| `test_scratchpad_patterns_config`       | 0m9.372s  | 502           |
| `test_scratchpad_use_config`            | 0m16.156s | 698           |
| `test_dtype_scalars_config`             | 0m34.384s | 733           |
| `test_cache_config`                     | 0m16.160s | 695           |
| `test_coordinates_config`               | 0m9.144s  | 494           |
| `test_it_space_splits_config`           | 0m9.360s  | 498           |
| `test_tensor_layout_config`             | 0m16.405s | 667           |

---

## Highlights

- **Longest running test:** `test_inductor_ops_config` — `8m22.035s`
- **Highest memory usage:** `test_modules_config` — `12103 MB`
