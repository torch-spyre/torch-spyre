# IBM Spyre device

This document provides an overview of the Spyre device.

## What is Spyre

The Spyre AI Card, also known as the IBM Spyre Accelerator, is a high-performance, energy-efficient
AI accelerator. Currently, it is generally available for IBM Z, LinuxONE, and Power systems.

Spyre Accelerators are engineered to support the development of higher-accuracy AI techniques, enabling real-time generative asset creation, customer data ingestion and interpretation for outreach, cross-selling, and risk assessments.

## Key features

Some of the key features of the Spyre device are listed below:

* It is equipped with 32 AI accelerator cores, capable of handling matrix operations and low‑precision workloads for high throughput.
* It is manufactured using advanced 5nm node technology.
* Each card supports up to 128 GB of LPDDR5 memory, with ensembles of up to eight cards delivering 1 TB memory and massive AI performance.
* It delivers exceptional AI compute, exceeding 300 TOPS per card, while consuming just 75W.

## Use cases

The Spyre device is designed for enterprise AI workloads including:
* Real-time fraud detection
* Code generation and assistance
* Large language model inference
* Multi-model ensemble inferencing

## Integration with PyTorch

The Spyre device is integrated with PyTorch as a custom backend device, enabling standard PyTorch models to leverage Spyre's AI acceleration capabilities. See the [README](../README.md) for setup and usage instructions. The [doc](../docs/) and [examples](../examples/) provide in-depth documentation on various topics and examples, respectively.

## Learn more

Refer to the online docs and blogs to learn more about the Spyre device.

* [Introduction to IBM Spyre Accelerator](https://www.ibm.com/docs/es/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator)
* [IBM Spyre Accelerator and Telum II Processor](https://www.ibm.com/new/announcements/ibm-spyre-accelerator-and-telum-ii-processor-capturing-ai-value-at-a-trusted-enterprise-level)
