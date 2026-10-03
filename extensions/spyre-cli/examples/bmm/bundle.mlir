module {
	func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>, %arg_1_base_addr: !sdscbundle.input_arg<index>, %arg_2_base_addr: !sdscbundle.input_arg<index>) {
		%arg_0 = sdscbundle.input_arg_extract value from %arg_0_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_1 = sdscbundle.input_arg_extract value from %arg_1_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_2 = sdscbundle.input_arg_extract value from %arg_2_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_0_core_offset_128 = arith.constant 128 : index
		%arg_0_core_128 = arith.addi %arg_0, %arg_0_core_offset_128 : index
		%arg_0_core_offset_256 = arith.constant 256 : index
		%arg_0_core_256 = arith.addi %arg_0, %arg_0_core_offset_256 : index
		%arg_0_core_offset_384 = arith.constant 384 : index
		%arg_0_core_384 = arith.addi %arg_0, %arg_0_core_offset_384 : index
		%arg_0_core_offset_512 = arith.constant 512 : index
		%arg_0_core_512 = arith.addi %arg_0, %arg_0_core_offset_512 : index
		%arg_0_core_offset_640 = arith.constant 640 : index
		%arg_0_core_640 = arith.addi %arg_0, %arg_0_core_offset_640 : index
		%arg_0_core_offset_768 = arith.constant 768 : index
		%arg_0_core_768 = arith.addi %arg_0, %arg_0_core_offset_768 : index
		%arg_0_core_offset_896 = arith.constant 896 : index
		%arg_0_core_896 = arith.addi %arg_0, %arg_0_core_offset_896 : index
		%arg_1_core_offset_128 = arith.constant 128 : index
		%arg_1_core_128 = arith.addi %arg_1, %arg_1_core_offset_128 : index
		%arg_1_core_offset_256 = arith.constant 256 : index
		%arg_1_core_256 = arith.addi %arg_1, %arg_1_core_offset_256 : index
		%arg_1_core_offset_384 = arith.constant 384 : index
		%arg_1_core_384 = arith.addi %arg_1, %arg_1_core_offset_384 : index
		%arg_1_core_offset_512 = arith.constant 512 : index
		%arg_1_core_512 = arith.addi %arg_1, %arg_1_core_offset_512 : index
		%arg_1_core_offset_640 = arith.constant 640 : index
		%arg_1_core_640 = arith.addi %arg_1, %arg_1_core_offset_640 : index
		%arg_1_core_offset_768 = arith.constant 768 : index
		%arg_1_core_768 = arith.addi %arg_1, %arg_1_core_offset_768 : index
		%arg_1_core_offset_896 = arith.constant 896 : index
		%arg_1_core_896 = arith.addi %arg_1, %arg_1_core_offset_896 : index
		%arg_2_core_offset_128 = arith.constant 128 : index
		%arg_2_core_128 = arith.addi %arg_2, %arg_2_core_offset_128 : index
		%arg_2_core_offset_256 = arith.constant 256 : index
		%arg_2_core_256 = arith.addi %arg_2, %arg_2_core_offset_256 : index
		%arg_2_core_offset_384 = arith.constant 384 : index
		%arg_2_core_384 = arith.addi %arg_2, %arg_2_core_offset_384 : index
		%arg_2_core_offset_512 = arith.constant 512 : index
		%arg_2_core_512 = arith.addi %arg_2, %arg_2_core_offset_512 : index
		%arg_2_core_offset_640 = arith.constant 640 : index
		%arg_2_core_640 = arith.addi %arg_2, %arg_2_core_offset_640 : index
		%arg_2_core_offset_768 = arith.constant 768 : index
		%arg_2_core_768 = arith.addi %arg_2, %arg_2_core_offset_768 : index
		%arg_2_core_offset_896 = arith.constant 896 : index
		%arg_2_core_896 = arith.addi %arg_2, %arg_2_core_offset_896 : index
		sdscbundle.sdsc_execute (%arg_0, %arg_0_core_128, %arg_0_core_256, %arg_0_core_384, %arg_0_core_512, %arg_0_core_640, %arg_0_core_768, %arg_0_core_896, %arg_1, %arg_1_core_128, %arg_1_core_256, %arg_1_core_384, %arg_1_core_512, %arg_1_core_640, %arg_1_core_768, %arg_1_core_896, %arg_2, %arg_2_core_128, %arg_2_core_256, %arg_2_core_384, %arg_2_core_512, %arg_2_core_640, %arg_2_core_768, %arg_2_core_896) {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24]}
		return
	}
}
