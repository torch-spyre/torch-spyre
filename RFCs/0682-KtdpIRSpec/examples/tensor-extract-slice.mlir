// RUN: ktir-opt %s | ktir-opt | FileCheck --check-prefix=CHECK-IR %s
// Round-tripping dummy test


// An example to demonstrate the usage of tensor-extract operation.
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct a memory view of A from a given address
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of A
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of B
    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    // Load data from the corresponding access tile
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    // Perform add operation on the data tiles.
    %C_data_tile = tensor.empty() : tensor<3x64xf16>
    linalg.add ins(%A_data_tile, %B_data_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                outs(%C_data_tile: tensor<3x64xf16>) -> tensor<3x64xf16>

    // extract 1x64 slice of C_data_tile
    %C_data_subtile = tensor.extract_slice %C_data_tile[0, 0] [1, 64] [1, 1]
      : tensor<3x64xf16> to tensor<1x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of C
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Store data into the access tile.
    ktdp.store %C_data_subtile, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
