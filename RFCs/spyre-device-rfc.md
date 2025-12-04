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
4. Support existing out-of-the-box pytorch implementations
5. Improve stability of the software stack
6. Explicit over implicit whereever possible

## **Proposed Implementation**

### Registering Spyre as an out-of-tree accelerator through privateuse1 construct

```python
torch.utils.rename_privateuse1_backend(“spyre”)
torch._register_device_module(“spyre”, make_spyre_module())
```

### Implementing a custom c10::impl::DeviceGuardImplInterface

We will implement a specific SpyreGuardImpl, that will be used for device management, allowing to track current device and synchronization between devices.

### Implement a custom at::Allocator

In order to properly allocate space on the device and manage tensors, we will implement a custom at::Allocator (SpyreAllocator). As part of this implementation, we will ensure that what is returned to the user (tensor handle) is only an indirect index to the allocated memory, and not the actual pointer (This is a Z security requirement). We plan to allocate memory through the Spyre Flex runtime interface (TryAllocate). The allocator will also be responsible for cleaning up tensor handles that go out of scope.

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

### Multi-Device Support

Our plan is to implement the standard torch collective communications APIs similar to NCCL, either through a custom ProcessGroup or using the new TorchComms APIs (all2all, all_reduce, reduce_scatter, all_gather)

## **Metrics **

To properly measure the success of this feature, requires the following criteria be met:

- [ ] Pytorch upstream tests running to completion with at least 80% passing
- [ ] Support for transformers model implementations
- [ ] TBD

## **Drawbacks**

The main drawback to this new addition is the implementation cost, as interfacing with the Spyre device through the above mechanism is quite different from the current workflow. This requires a lot of re-implementation, and replacement of existing code.

## **Alternatives**

There are no other alternatives to this method as this method will bring long-term stability. If we do not do this, we would require continuing with the existing stack, which would be much more costly to bring in new features.

## **Prior Art**

Prior to this method, we did an eager support POC using the existing software stack which did not include a device construct. This feature taught us a lot about the handling eager tensor allocation, eager copies, and eager compute which are all widely used in the above RFC.

## **How we teach this**

TBD

## **Unresolved questions**

The following are some unresolved questions:

1. Tensor Allocation - When moving from PF to VF mode on Spyre, large changes may be required in the allocator

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
