# Granite LX relayout inventory

This index names all 130 records in the pinned historical ownership catalog: 55 prefill and 75 decode. It is a list of required tests, **not KTIR test results**. The rows are checked in here so workload coverage does not depend on a mutable external summary.

Source: [pinned SenDNN ownership catalog](https://github.com/AdnanHoque/torch-spyre/blob/36804f23ede70325c21ba234a7e102537a8eda95/experiments/granite_relayout/artifacts/catalogs/sendnn_sdsc_lx_replay_manifest.json)

SHA-256: `4c8aca2e1989eefb76c4ee40b99a4a87800aac4134cbae68c41810ab98c747d4`

Use `relayouts[index]` in that file for each row below. The index is zero-based. The exact fixture is the complete source/destination rectangle and owner tables in that record; the summary counts here are not a substitute. IDs are stable for this file hash.

For every record, use the shared contract in [LX relayout workload coverage](lx_relayout_workload_coverage.md). The original consumer and input number are part of the test. Preserve core IDs, offsets, layout, and the order of values. Run the real consumer after transport when its operation is available; copy-only and consumer-execution verdicts are separate.

“Pieces” counts distinct logical regions. “Copies” is the set of destination owners-per-region counts. “Inputs per region” counts source pieces intersecting a destination region. Thus `32 → 1; copies 28` means 32 pieces gathered into one complete region, delivered to 28 cores—not 32 cores sending only to one core. Where pieces are split as well as gathered, use their intersections rather than copying each whole source piece.

The saved consumer name identifies the operation, not a manually inferred model layer. The semantic P01–P14 discussion is in the main document. The family-level meaning and evidence rules are in [LX relayout workload coverage](lx_relayout_workload_coverage.md).

Preparation check only: every destination rectangle is covered exactly once by the non-overlapping source rectangles; no two distinct destination rectangles overlap; no owner is duplicated inside a piece. This does not validate physical addresses or KTIR lowering.

## Prefill

| Test ID | JSON index | Consumer / input | Source shape | Pieces → pieces; copies | Inputs per region | Fold count |
|---|---:|---|---|---|---|---:|
| GR-PF-001 | 0 | `add_3` / 1 | `mb=512, out=4096, y=1` | 16 → 32; 1 | 2 | 1 |
| GR-PF-002 | 1 | `mm-BMM_1` / 0 | `in=4096, mb=512, y=1` | 16 → 8; 4 | 2 | 1 |
| GR-PF-003 | 2 | `bmm_2-BMM_1` / 0 | `in=128, mb=512, x=8, x1=1, y=4` | 32 → 32; 1 | 4 | 38 |
| GR-PF-004 | 3 | `bmm_2-BMM_1` / 1 | `in=128, out=512, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 38 |
| GR-PF-005 | 4 | `bmm_3-BMM_1` / 1 | `in=512, out=128, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 38 |
| GR-PF-006 | 5 | `mm_10-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 38 |
| GR-PF-007 | 6 | `mm_11-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 38 |
| GR-PF-008 | 7 | `mm_12-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 38 |
| GR-PF-009 | 8 | `mm_1-BMM_1` / 0 | `in=4096, mb=512, y=1` | 16 → 8; 4 | 2 | 1 |
| GR-PF-010 | 9 | `bmm_78-BMM_1` / 0 | `in=128, mb=512, x=8, x1=1, y=4` | 32 → 32; 1 | 4 | 1 |
| GR-PF-011 | 10 | `bmm_78-BMM_1` / 1 | `in=128, out=512, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 1 |
| GR-PF-012 | 11 | `bmm_79-BMM_1` / 1 | `in=512, out=128, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 1 |
| GR-PF-013 | 12 | `mm_276-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-014 | 13 | `mm_277-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-015 | 14 | `mm_278-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-016 | 15 | `mm_280-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 28 | 32 | 1 |
| GR-PF-017 | 16 | `mm_2-BMM_1` / 0 | `in=4096, mb=512, y=1` | 16 → 8; 4 | 2 | 1 |
| GR-PF-018 | 17 | `bmm-BMM_1` / 0 | `in=128, mb=512, x=8, x1=1, y=4` | 32 → 32; 1 | 4 | 1 |
| GR-PF-019 | 18 | `bmm-BMM_1` / 1 | `in=128, out=512, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 1 |
| GR-PF-020 | 19 | `bmm_1-BMM_1` / 1 | `in=512, out=128, x=8, x1=1, y=1` | 32 → 1; 32 | 32 | 1 |
| GR-PF-021 | 20 | `mm_3-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-022 | 21 | `mm_4-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-023 | 22 | `mm_5-BMM_1` / 0 | `in=4096, mb=512, y=1` | 32 → 8; 4 | 4 | 1 |
| GR-PF-024 | 23 | `mean_1-Exx2` / 0 | `mb=512, out=4096, y=1` | 32 → 8; 1 | 4 | 1 |
| GR-PF-025 | 24 | `mean_3-Exx2` / 0 | `mb=512, out=4096, y=1` | 32 → 8; 1 | 4 | 38 |
| GR-PF-026 | 25 | `mean_79-Exx2` / 0 | `mb=512, out=4096, y=1` | 32 → 8; 1 | 4 | 1 |
| GR-PF-027 | 26 | `mean_80-Exx2` / 0 | `mb=512, out=4096, y=1` | 32 → 8; 1 | 4 | 1 |
| GR-PF-028 | 27 | `mean_1-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-029 | 28 | `mean_1-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-030 | 29 | `mean_2-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-031 | 30 | `mean_2-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-032 | 31 | `mean_3-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 38 |
| GR-PF-033 | 32 | `mean_3-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 38 |
| GR-PF-034 | 33 | `mean_4-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 38 |
| GR-PF-035 | 34 | `mean_4-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 38 |
| GR-PF-036 | 35 | `mean_79-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-037 | 36 | `mean_79-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-038 | 37 | `mean_80-LayerNormNorm` / 1 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-039 | 38 | `mean_80-LayerNormNorm` / 2 | `mb=512, out=64, y=1` | 8 → 8; 4 | 1 | 1 |
| GR-PF-040 | 39 | `mul_14-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 38 |
| GR-PF-041 | 40 | `mul_14-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 38 |
| GR-PF-042 | 41 | `mul_15-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 38 |
| GR-PF-043 | 42 | `mul_15-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 38 |
| GR-PF-044 | 43 | `mul_3-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-045 | 44 | `mul_432-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-046 | 45 | `mul_432-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-047 | 46 | `mul_433-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-048 | 47 | `mul_433-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-049 | 48 | `mul_3-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-050 | 49 | `mul_4-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-051 | 50 | `mul_4-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=512` | 16 → 8; 4 | 2 | 1 |
| GR-PF-052 | 51 | `bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=512, out=128, x=1, y=1` | 32 → 32; 1 | 1 | 1 |
| GR-PF-053 | 52 | `bmm_2-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=512, out=128, x=1, y=1` | 32 → 32; 1 | 1 | 38 |
| GR-PF-054 | 53 | `bmm_78-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=512, out=128, x=1, y=1` | 32 → 32; 1 | 1 | 1 |
| GR-PF-055 | 54 | `slice_161-Stcdp` / 0 | `mb=512, out=4096, y=1` | 32 → 32; 1 | 1 | 1 |

## Decode

| Test ID | JSON index | Consumer / input | Source shape | Pieces → pieces; copies | Inputs per region | Fold count |
|---|---:|---|---|---|---|---:|
| GR-DC-001 | 55 | `mm-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 32 | 32 | 1 |
| GR-DC-002 | 56 | `bmm_2-BMM_1` / 0 | `in=128, mb=1, x=8, x1=1, y=4` | 32 → 8; 4 | 4 | 38 |
| GR-DC-003 | 57 | `bmm_3-BMM_1` / 0 | `in=768, mb=1, x=8, x1=1, y=4` | 32 → 16; 2 | 4 | 38 |
| GR-DC-004 | 58 | `bmm_3-BMM_1` / 1 | `in=768, out=128, x=8, x1=1, y=1` | 32 → 16; 2 | 32 | 38 |
| GR-DC-005 | 59 | `mm_11-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 38 |
| GR-DC-006 | 60 | `mm_12-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 38 |
| GR-DC-007 | 61 | `mm_13-BMM_1` / 0 | `in=12800, mb=1, y=1` | 25 → 1; 32 | 25 | 38 |
| GR-DC-008 | 62 | `mm_1-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 16 | 32 | 1 |
| GR-DC-009 | 63 | `bmm_78-BMM_1` / 0 | `in=128, mb=1, x=8, x1=1, y=4` | 32 → 8; 4 | 4 | 1 |
| GR-DC-010 | 64 | `bmm_79-BMM_1` / 0 | `in=768, mb=1, x=8, x1=1, y=4` | 32 → 16; 2 | 4 | 1 |
| GR-DC-011 | 65 | `bmm_79-BMM_1` / 1 | `in=768, out=128, x=8, x1=1, y=1` | 32 → 16; 2 | 32 | 1 |
| GR-DC-012 | 66 | `mm_277-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 1 |
| GR-DC-013 | 67 | `mm_278-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 1 |
| GR-DC-014 | 68 | `mm_279-BMM_1` / 0 | `in=12800, mb=1, y=1` | 25 → 1; 32 | 25 | 1 |
| GR-DC-015 | 69 | `mm_280-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 28 | 32 | 1 |
| GR-DC-016 | 70 | `mm_2-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 16 | 32 | 1 |
| GR-DC-017 | 71 | `bmm-BMM_1` / 0 | `in=128, mb=1, x=8, x1=1, y=4` | 32 → 8; 4 | 4 | 1 |
| GR-DC-018 | 72 | `bmm-BMM_1` / 1 | `in=128, out=768, x=8, x1=1, y=1` | 32 → 32; 1 | 2 | 1 |
| GR-DC-019 | 73 | `bmm_1-BMM_1` / 0 | `in=768, mb=1, x=8, x1=1, y=4` | 32 → 16; 2 | 4 | 1 |
| GR-DC-020 | 74 | `bmm_1-BMM_1` / 1 | `in=768, out=128, x=8, x1=1, y=1` | 32 → 16; 2 | 32 | 1 |
| GR-DC-021 | 75 | `mm_4-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 1 |
| GR-DC-022 | 76 | `mm_5-BMM_1` / 0 | `in=4096, mb=1, y=1` | 32 → 1; 25 | 32 | 1 |
| GR-DC-023 | 77 | `mm_6-BMM_1` / 0 | `in=12800, mb=1, y=1` | 25 → 1; 32 | 25 | 1 |
| GR-DC-024 | 78 | `mean-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 1 |
| GR-DC-025 | 79 | `mean_1-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 1 |
| GR-DC-026 | 80 | `mean_2-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 1 |
| GR-DC-027 | 81 | `mean_3-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 38 |
| GR-DC-028 | 82 | `mean_4-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 38 |
| GR-DC-029 | 83 | `mean_79-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 1 |
| GR-DC-030 | 84 | `mean_80-Exx2` / 0 | `mb=1, out=4096, y=1` | 32 → 1; 1 | 32 | 1 |
| GR-DC-031 | 85 | `mean-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-032 | 86 | `mean-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-033 | 87 | `mean_1-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-034 | 88 | `mean_1-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-035 | 89 | `mean_2-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-036 | 90 | `mean_2-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-037 | 91 | `mean_3-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 38 |
| GR-DC-038 | 92 | `mean_3-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 38 |
| GR-DC-039 | 93 | `mean_4-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 38 |
| GR-DC-040 | 94 | `mean_4-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 38 |
| GR-DC-041 | 95 | `mean_79-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-042 | 96 | `mean_79-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-043 | 97 | `mean_80-LayerNormNorm` / 1 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-044 | 98 | `mean_80-LayerNormNorm` / 2 | `mb=1, out=64, y=1` | 1 → 1; 32 | 1 | 1 |
| GR-DC-045 | 99 | `_safe_softmax-Max` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 1 |
| GR-DC-046 | 100 | `_safe_softmax_1-Max` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 38 |
| GR-DC-047 | 101 | `_safe_softmax_39-Max` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 1 |
| GR-DC-048 | 102 | `mul_137-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 38 |
| GR-DC-049 | 103 | `mul_137-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 38 |
| GR-DC-050 | 104 | `mul_138-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 38 |
| GR-DC-051 | 105 | `mul_138-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 38 |
| GR-DC-052 | 106 | `_safe_softmax_1-Mul` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 38 |
| GR-DC-053 | 107 | `mul_3-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 1 |
| GR-DC-054 | 108 | `mul_5077-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 1 |
| GR-DC-055 | 109 | `mul_5077-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 1 |
| GR-DC-056 | 110 | `mul_5078-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 1 |
| GR-DC-057 | 111 | `mul_5078-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 1 |
| GR-DC-058 | 112 | `_safe_softmax_39-Mul` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 1 |
| GR-DC-059 | 113 | `mul_3-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 32 | 2 | 1 |
| GR-DC-060 | 114 | `mul_4-mul_1` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 1 |
| GR-DC-061 | 115 | `mul_4-mul_2` / 1 | `i=1, j=1, mb=2, out=64, x=1, y=1` | 2 → 1; 8 | 2 | 1 |
| GR-DC-062 | 116 | `_safe_softmax-Mul` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 1 |
| GR-DC-063 | 117 | `bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=1, out=128, x=1, y=768` | 32 → 32; 1 | 8 | 1 |
| GR-DC-064 | 118 | `bmm_2-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=1, out=128, x=1, y=768` | 32 → 32; 1 | 1 | 38 |
| GR-DC-065 | 119 | `bmm_78-wtAttnHeadBreak-VirtualReshape-Output-Restickify` / 0 | `j=8, mb=1, out=128, x=1, y=768` | 32 → 32; 1 | 1 | 1 |
| GR-DC-066 | 120 | `cat_1-kvCacheScatter` / 0 | `mb=8, out=128, x=1, y=1` | 16 → 1; 1 | 16 | 1 |
| GR-DC-067 | 121 | `cat_3-kvCacheScatter` / 0 | `mb=8, out=128, x=1, y=1` | 16 → 1; 1 | 16 | 38 |
| GR-DC-068 | 122 | `cat_79-kvCacheScatter` / 0 | `mb=8, out=128, x=1, y=1` | 16 → 1; 1 | 16 | 1 |
| GR-DC-069 | 123 | `embedding-Output-Stcdp` / 0 | `mb=1, out=4096, y=1` | 1 → 32; 1 | 1 | 1 |
| GR-DC-070 | 124 | `_safe_softmax-Sub` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 1 |
| GR-DC-071 | 125 | `_safe_softmax_1-Sub` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 38 |
| GR-DC-072 | 126 | `_safe_softmax_39-Sub` / 1 | `i=1, mb=32, out=64, x=1` | 8 → 8; 4 | 1 | 1 |
| GR-DC-073 | 127 | `_safe_softmax-Sum` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 1 |
| GR-DC-074 | 128 | `_safe_softmax_1-Sum` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 38 |
| GR-DC-075 | 129 | `_safe_softmax_39-Sum` / 0 | `i=1, mb=32, out=768, x=1` | 32 → 8; 1 | 4 | 1 |

## Matching the prefill family catalog

The [pinned P01–P14 catalog](https://github.com/AdnanHoque/torch-spyre/blob/36804f23ede70325c21ba234a7e102537a8eda95/experiments/granite_relayout/artifacts/catalogs/prefill_relayout_templates.json) groups the prefill records into 14 families. These are additional labels, not replacement IDs:

| Family | Individual tests |
|---|---|
| P01 | GR-PF-004, GR-PF-011, GR-PF-019 |
| P02 | GR-PF-005, GR-PF-012, GR-PF-020 |
| P03 | GR-PF-007, GR-PF-008, GR-PF-014, GR-PF-015, GR-PF-022, GR-PF-023 |
| P04 | GR-PF-006, GR-PF-013, GR-PF-021 |
| P05 | GR-PF-024, GR-PF-025, GR-PF-026, GR-PF-027 |
| P06 | GR-PF-003, GR-PF-010, GR-PF-018 |
| P07 | GR-PF-040, GR-PF-041, GR-PF-042, GR-PF-043, GR-PF-044, GR-PF-045, GR-PF-046, GR-PF-047, GR-PF-048, GR-PF-049, GR-PF-050, GR-PF-051 |
| P08 | GR-PF-052, GR-PF-053, GR-PF-054 |
| P09 | GR-PF-002, GR-PF-009, GR-PF-017 |
| P10 | GR-PF-028, GR-PF-030, GR-PF-032, GR-PF-034, GR-PF-036, GR-PF-038 |
| P11 | GR-PF-029, GR-PF-031, GR-PF-033, GR-PF-035, GR-PF-037, GR-PF-039 |
| P12 | GR-PF-001 |
| P13 | GR-PF-016 |
| P14 | GR-PF-055 |

## Acceptance recording

Keep each ID in the results even if tests share a runner. For each, record the proposed KTIR file, exact KTIR compiler commit, verifier result, numerical result, emitted LX/HBM evidence, actual-consumer result, and any failure reason. Missing cases are NOT_RUN, never implicitly passing. A compressed family example is useful for design discussion but cannot silently replace the saved core IDs and extents.

The proposal's separate 51-file inventory is not assumed to be this catalog. Establish its mapping by original artifact identifiers before claiming one subsumes the other.
