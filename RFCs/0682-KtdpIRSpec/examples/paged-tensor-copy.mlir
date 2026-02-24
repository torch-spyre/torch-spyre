// This is an example program to illustrate the use of ktdp.construct_access_tile and ktdp.construct_indirect_access_tile
// The program makes a sub-tensor copy operation from an input (stored as paged tensor) 
// into an output (stored as contiguous tensor). 
// An index tensor is used to indicate the portion (pages) of the input tensor that must copied.

#X_coord_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 9999 >= 0, 
                                              d1 >= 0, -d1 + 7 >=0, 
                                              d2 >= 0, -d2 + 63 >= 0, 
                                              d3 >= 0, -d3 + 127>= 0)>
#Y_coord_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0, 
                                              d1 >= 0, -d1 + 2047 >=0, 
                                              d2 >= 0, -d2 + 7 >= 0, 
                                              d3 >= 0, -d3 + 127>= 0)>
#XY_var_space_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0, 
                                                  d1 >= 0, -d1 + 7 >= 0, 
                                                  d2 >= 0, -d2 + 2047>= 0, 
                                                  d3 >= 0, -d3 + 127 >= 0)>

module {
  func.func @paged_tensor_copy_1core() {
        // The program follows semantics of a typical "paged attention" kernel implementation: 
        // Ndkv = feature size per head, 
        // Nhkv = number of kv heads, 
        // Nb = batch size, 
        // Ntkv = context size
        // Nb and Ntkv are paged. size of tkv within a page (Ptkv) > 1, and size of batch within a page = 1
        // ==> dkv, hkv are directly accessed, b is indirectly accessed, 
        //     tkv is hybrid - directly accessed until page size, and then indirectly accessed across pages
        // Npages = number of total pages 
        // size_of_page = Ndkv * Nhkv * Ptkv
        // Copy is performed across Nb * Ntkv/Ptkv pages. Page ids to copy are provided by the index tensor

        // For this example, we will use Ndkv=128, Nhkv=8, Ntkv=2048 with Ptkv=64, Nb=4, Npages=10000
        // X input = 4D tensor => shape {Ndkv=128, Nhkv=8, Ptkv=64, Npages=10000}  
        // Idx input = 2D tensor => shape {Nb=4, Ntkv/Ptkv=32}
        // Y output = 4D tensor => shape {Ndkv=128, Nhkv=8, Nb=4, Ntkv=2048}
        // for dkv in 0..Ndkv
        //    for h in 0..Nhkv
        //        for b in 0..Nb
        //            for tkv in 0..Ntkv
        //                Y[dkv][h][b][tkv] = X[dkv][h][tkv%Ptkv][Idx[b][tkv/Ptkv]]
        

        %Nb = arith.constant 4 : index
        %Ntkv = arith.constant 2048 : index
        %Ptkv = arith.constant 64 : index
        %Ndkv = arith.constant 128 : index
        %Nhkv = arith.constant 8 : index
        %Npages = arith.constant 10000 : index
        %Ntkv_Ptkv = arith.divui %Ntkv, %Ptkv : index

        // In this example, all tensors Index, X, Y are in single memory space (e.g. DDR)
        %X_start_address = arith.constant 30000000 : index
        %Idx_start_address = arith.constant 20000000 : index
        %Y_start_address = arith.constant 10000000 : index

        // Accessing a tensor in KTIR follows a 3 step process:
        // Note1: Accesses are single-ended i.e., 
        //        load = read from memory to produce a data-tile
        //        store = write a data-tile to memory
        // Note2: To accomplish a data-transfer (or copy) from source to destination, 
        //        we need a seperate load from source and a store to destination 
        // (1) Create memory view: Informs how the tensor is present in memory
        //    Note: the tensor can be spread across multiple memory spaces
        // (2) Create access tile: Informs the logical coordinates of the tensor that must be accessed
        //    Note1: the coordinates that are accessed can be part of multiple memory spaces
        //    Note2: The access tile does not dictate the order in which its coordinates are accesssed
        //    Note3: Explicit access ordering can be enforced by creating multiple smaller-sized access tiles in a loop
        // (3) Create data tile: Extract a sub-portion of the tensor corresponding to the coordinates present in the access tile

        // (1) Construct memory view for Index tensor
        // Note: number of entries in sizes, strides, dims in coordinate_set, shape of memref must be identical
        %Idx_mem_view = ktdp.construct_memory_view %Idx_start_address, 
                        sizes: [%Nb, %Ntkv_Ptkv], strides: [%Ntkv_Ptkv, 1] {
                        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<4x32xi32>

        // (1) Construct memory view for input X 
        // Note: affine_set (e.g., #X_coord_set) can be declared outside the module 
        %X_mem_view = ktdp.construct_memory_view %X_start_address, 
                        sizes: [%Npages, %Nhkv, %Ptkv, %Ndkv], strides: [65536, 8192, %Ndkv, 1] {
                        coordinate_set = #X_coord_set,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<10000x8x64x128xf16>

        // (1) Construct memory view for output Y
        // Note: strides can be constructed with arit operations
        %Y_str_Ntkv = arith.muli  %Ndkv, %Nh : index
        %Y_str_Nb = arith.muli %Y_str_Ntkv, %Ntkv : index
        %Y_mem_view = ktdp.construct_memory_view %Y_start_address, 
                        sizes: [%Nb, %Ntkv, %Nh, %Ndkv], strides: [%Y_str_Nb, %Y_str_Ntkv, %Ndkv , 1] {
                        coordinate_set = #Y_coord_set,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<4x2048x8x128xf16>

        // (2) Construct indirect access tile X[dkv, h, tkv%Ptkv, Idx[b][tkv/Ptkv]]
        // Note: Number of entries in intermediate_variables and access_tile_set must be equal
        // Note: Number of entries in subscript of mem_view, shape of memref, shape of ktdp.tile must be equal
        %X_access_tile = ktdp.construct_indirect_access_tile 
                            intermediate_variables(%b, %h, %tkv, %dkv) 
                            %X_mem_view[Idx_mem_view[%b, %tkv / 64] , %hkv, %tkv % 64, %dkv] {
            variables_space_set = #XY_var_space_set
        } : memref<10000x8x64x128xf16> -> !ktdp.tile<4x8x2048x128xindex>

        // (2) Construct access tile for Y[dkv, h, b, tkv]
        // Note: No need for intermediate_variables in direct accessed tiles
        // Note: Number of entries in subscript of mem_view, access_tile_set, shape of memref, shape of ktdp.tile must be equal
        %Y_access_tile = ktdp.construct_access_tile %Y_mem_view[0, 0, 0, 0] {
            access_tile_set = #XY_var_space_set
        } : memref<4x2048x8x128xf16> -> !ktdp.tile<4x8x2048x128xindex>

        // (3) Create data_tile for X from its access tile
        %X_data_tile = ktdp.load %X_access_tile : !ktdp.tile<4x8x2048x128xindex> -> tensor<4x8x2048x128xf16>

        // (3) Store Y[...] = X_data_tile
        ktdp.store %X_data_tile, %Y_access_tile : tensor<4x8x2048x128xf16>, !ktdp.tile<4x8x2048x128xindex>

        return
  }
}