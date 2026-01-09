# Spyre Device Construct in Pytorch

**Authors:**
* @JRosenkranz

## **Summary**

This RFC describes additions necessary to have a proper "Spyre" device within the Pytorch ecosystem. This entails implementing the necessary Pytorch device interfaces as well as registering Spyre backend through PrivateUse1.

## **Motivation**

The goals of this proposal are as follows:

1. User friendly interface to the Spyre Device
2. Support proper tensor residency
3. Improved developer productivity
4. Support existing out-of-the-box Pytorch implementations
5. Improve stability of the software stack
6. Explicit over implicit whereever possible

## **Proposed Implementation**

### Registering Spyre as an out-of-tree accelerator through privateuse1 construct

```python
torch.utils.rename_privateuse1_backend(“spyre”)
torch._register_device_module(“spyre”, make_spyre_module())
```

### Implementing a custom c10::impl::DeviceGuardImplInterface

We will implement a specific SpyreGuardImpl, that will be used for device management, allowing to track current device and synchronization between devices. There are open questions here about whether Spyre can support streams, as well as how to properly support device counts/indexes.

### Implement a custom at::Allocator

In order to properly allocate space on the device and manage tensors, we will implement a custom at::Allocator (SpyreAllocator). Due to Z security requirements, we will ensure that the tensor handle returned to the user must not have a physical pointer to the memory.

With the switch from PF to VF mode (no physical addresses will be present), we will now have a more strict constraint on the number of handles we get from the backend runtime. As such, this requires us to re-think the Pytorch allocator as the current design will only allow a small number of handles to resident tensors when in VF mode.

Instead of allocating in the backend runtime on a tensor-by-tensor basis (as is currently the case), allocate large chunks (lazily as necessary). We will continue to use the TryAllocate method from flex to allocate these large chunks. Within the Pytorch allocator, we can do our own virtual memory management given the large allocated chunk (similar to the cuda cache allocator). This way, we can create any number of handles in the Pytorch allocator, while only holding the large chunk handles from the backend runtime.

For more information, please visit the following [epic](https://github.com/torch-spyre/torch-spyre/issues/200)

### Implement a custom at::TensorImpl

In Spyre, tensor representations differ from that of cpu/gpu. All memory and compute operations operate on chunks of 128 bytes. We call this chunk of 128 bytes a stick. The in-memory format of tensors on the Spyre device is designed to support efficient SIMD computations on sticks of data. In particular, one or more of every tensor’s dimensions are designated as stick dimensions. Stick dimensions are padded to be multiples of 128 bytes. To maximize reuse and to enable efficient device memory/scratchpad transfers, stick dimensions are laid out in a tiled fashion in the device’s memory (sticks that are consecutive in a stick dimension from the perspective of PyTorch-level indexing may not actually be assigned consecutive memory addresses on the device). The importance of memory layout for efficient computation is familiar from GPUs, but it is even more important on Spyre. Furthermore, the compute operations of Spyre’s SIMD dataflow engine impose a number of legality constraints on the memory layout of their inputs and the layout of the resulting output.

Because size and strides does not fully provide all of the information required about layout of a Spyre tensor, we must keep extra stick metadata as part of the TensorImpl to properly capture the layout of the tensor. As part of this, we must include a method to set and get the stick dimensions from a tensor, as well as allocate a tensor with a specific stick format. Additionally, we will require to keep translation information (DCI) between CPU and Spyre tensors to properly handle conversions.

More info can be found in the following [RFC](https://github.com/torch-spyre/torch-spyre/pull/59)

### Provide a mechanism for creating Spyre Tensors

When creating tensors, either through eager mode with torch factory methods, or automatically generated through inductor codegen, the `empty_strided`, `empty.memory_format` method will be typically called to allocate and get a handle to the tensor. As part of this work, we will implement this method, and return the proper SpyreTensor backed by a SpyreTensorImpl.

### Torch.compile support

In order to support torch.compile using a Spyre device backend, we will implement a method to launch a Spyre kernel. When running torch.compile, the Spyre inductor pathway will generate an artifact for each kernel that will be launched through this method.

```c++
void launchKernel(std::string g2_path, std::vector<at::Tensor> args);
```

Once we have a runnable binary, we will then augment this artifact with the proper tensor memory handles and execute the program.

#### Caching

We will re-use the existing torch.compile cache to save the artifact for later use, such that each subsequent launch does not require full compilation (similar to the current triton cache).

### Supporting Eager Operations

Our plan here is to use the torch.compile (aot path) to generate the artifact we will use to run eager operations. We also plan to ship these generated binaries as part of the distribution of torch-spyre. (re-using the torch.compile caching mechanism) Each op need not be hand-written, instead we can codegen the desired graph and launch kernel calls. By going through torch.compile, this should support a larger set of operations and workflows, unifying the two pathways.

```python
@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)
```

For more information, please visit [epic](https://github.com/torch-spyre/torch-spyre/issues/183)

#### Handling Symbolic Shapes during eager execution

Our plan to handle symbolic shapes will be to compile for the largest possible practical unit of an op and encapsulate this unit within a Control Block. We plan to build a mechanism (Control Block Stream) which allows the ability to compose a sequence of these control blocks during runtime. In the case where the largest possible unit cannot fit the given shape, we plan to append to this control block stream until the target shape is reached. If the control block / composed control block stream is too large for a given shape, we will build in a mechanism to mask out portions from the result.

<!-- TODO: Larger Design document regarding CBs and CB Streams to be added -->

### Multi-Device Support

Our plan is to implement the standard torch collective communications APIs similar to NCCL, either through a custom ProcessGroup or using the new TorchComms APIs (all2all, all_reduce, reduce_scatter, all_gather)

## **Metrics **

To properly measure the success of this feature, requires the following criteria be met:

* [ ] Pytorch upstream tests running to completion with at least 80% passing
* [ ] Support for transformers model implementations
* [ ] TBD

## **Drawbacks**

The main drawback to this new addition is the implementation cost, as interfacing with the Spyre device through the above mechanism is quite different from the current workflow. This requires a lot of re-implementation, and replacement of existing code. Some of the problems here are known and have been implemented, but a large portion of this effort still contains open questions. Please visit [Unresolved Questions](#Unresolved-questions) for more details.

## **Alternatives**

There are no other alternatives to this method as this method will bring long-term stability. If we do not do this, we would require continuing with the existing stack, which would be much more costly to bring in new features.

## **Prior Art**

Prior to this method, we did an eager support POC using the existing software stack which did not include a device construct. This feature taught us a lot about the handling eager tensor allocation, eager copies, and eager compute which are all widely used in the above RFC.

## **How we teach this**

TBD

## **Unresolved questions**

The following are some unresolved questions:

1. Tensor Allocation - When moving from PF to VF mode on Spyre, large changes may be required in the allocator. There may be some performance implications as well that will need to be addressed as part of this new allocator
2. CUDA Stream Support - Can we support a CUDA Stream like concept using spyre?
3. Control Block Streams - What level of exposure should we be giving to torch-spyre for Control Block Streams? What should the overall structure be of the Control Block Stream? Will we be caching Control Blocks and composing Control Block Streams on the fly, Or also caching the Control Block Streams as well? Will we need to finalize CB Streams, or can we keep them open?
4. It is still not fully known the full scope of collective communications APIs we can support with spyre natively.
5. We will need to see how to expose device indexes and counts from the backend runtime.

## Resolution

TBD

### Level of Support

<!--
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.
-->

#### Additional Context

TODO

### Next Steps

TODO

#### Tracking issue

Issues and priorities will be tracked in github through our [github Project](https://github.com/orgs/torch-spyre/projects/2)

#### Exceptions

TODO
