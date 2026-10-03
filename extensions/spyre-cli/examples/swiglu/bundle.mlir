module {
	func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>, %arg_1_base_addr: !sdscbundle.input_arg<index>, %arg_2_base_addr: !sdscbundle.input_arg<index>) {
		%arg_0 = sdscbundle.input_arg_extract value from %arg_0_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_1 = sdscbundle.input_arg_extract value from %arg_1_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_2 = sdscbundle.input_arg_extract value from %arg_2_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_0_core_offset_131072 = arith.constant 131072 : index
		%arg_0_core_131072 = arith.addi %arg_0, %arg_0_core_offset_131072 : index
		%arg_0_core_offset_262144 = arith.constant 262144 : index
		%arg_0_core_262144 = arith.addi %arg_0, %arg_0_core_offset_262144 : index
		%arg_0_core_offset_393216 = arith.constant 393216 : index
		%arg_0_core_393216 = arith.addi %arg_0, %arg_0_core_offset_393216 : index
		%arg_0_core_offset_524288 = arith.constant 524288 : index
		%arg_0_core_524288 = arith.addi %arg_0, %arg_0_core_offset_524288 : index
		%arg_0_core_offset_655360 = arith.constant 655360 : index
		%arg_0_core_655360 = arith.addi %arg_0, %arg_0_core_offset_655360 : index
		%arg_0_core_offset_786432 = arith.constant 786432 : index
		%arg_0_core_786432 = arith.addi %arg_0, %arg_0_core_offset_786432 : index
		%arg_0_core_offset_917504 = arith.constant 917504 : index
		%arg_0_core_917504 = arith.addi %arg_0, %arg_0_core_offset_917504 : index
		%arg_1_core_offset_131072 = arith.constant 131072 : index
		%arg_1_core_131072 = arith.addi %arg_1, %arg_1_core_offset_131072 : index
		%arg_1_core_offset_262144 = arith.constant 262144 : index
		%arg_1_core_262144 = arith.addi %arg_1, %arg_1_core_offset_262144 : index
		%arg_1_core_offset_393216 = arith.constant 393216 : index
		%arg_1_core_393216 = arith.addi %arg_1, %arg_1_core_offset_393216 : index
		%arg_1_core_offset_524288 = arith.constant 524288 : index
		%arg_1_core_524288 = arith.addi %arg_1, %arg_1_core_offset_524288 : index
		%arg_1_core_offset_655360 = arith.constant 655360 : index
		%arg_1_core_655360 = arith.addi %arg_1, %arg_1_core_offset_655360 : index
		%arg_1_core_offset_786432 = arith.constant 786432 : index
		%arg_1_core_786432 = arith.addi %arg_1, %arg_1_core_offset_786432 : index
		%arg_1_core_offset_917504 = arith.constant 917504 : index
		%arg_1_core_917504 = arith.addi %arg_1, %arg_1_core_offset_917504 : index
		%arg_2_core_offset_131072 = arith.constant 131072 : index
		%arg_2_core_131072 = arith.addi %arg_2, %arg_2_core_offset_131072 : index
		%arg_2_core_offset_262144 = arith.constant 262144 : index
		%arg_2_core_262144 = arith.addi %arg_2, %arg_2_core_offset_262144 : index
		%arg_2_core_offset_393216 = arith.constant 393216 : index
		%arg_2_core_393216 = arith.addi %arg_2, %arg_2_core_offset_393216 : index
		%arg_2_core_offset_524288 = arith.constant 524288 : index
		%arg_2_core_524288 = arith.addi %arg_2, %arg_2_core_offset_524288 : index
		%arg_2_core_offset_655360 = arith.constant 655360 : index
		%arg_2_core_655360 = arith.addi %arg_2, %arg_2_core_offset_655360 : index
		%arg_2_core_offset_786432 = arith.constant 786432 : index
		%arg_2_core_786432 = arith.addi %arg_2, %arg_2_core_offset_786432 : index
		%arg_2_core_offset_917504 = arith.constant 917504 : index
		%arg_2_core_917504 = arith.addi %arg_2, %arg_2_core_offset_917504 : index
		sdscbundle.sdsc_execute (%arg_0, %arg_0_core_131072, %arg_0_core_262144, %arg_0_core_393216, %arg_0_core_524288, %arg_0_core_655360, %arg_0_core_786432, %arg_0_core_917504) {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1, -2, -3, -4, -5, -6, -7, -8]}
		sdscbundle.sdsc_execute (%arg_1, %arg_1_core_131072, %arg_1_core_262144, %arg_1_core_393216, %arg_1_core_524288, %arg_1_core_655360, %arg_1_core_786432, %arg_1_core_917504, %arg_2, %arg_2_core_131072, %arg_2_core_262144, %arg_2_core_393216, %arg_2_core_524288, %arg_2_core_655360, %arg_2_core_786432, %arg_2_core_917504) {sdsc_filename="sdsc_1.json", "symbol_ids"=[-9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24]}
		return
	}
}
