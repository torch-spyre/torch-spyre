# IBM Spyre device

This document provides an overview of the Spyre device.

## What is Spyre

The Spyre AI Card, also known as the IBM Spyre Accelerator, is a high-performance, energy-efficient
AI accelerator. Currently, it is generally available for IBM Z, LinuxONE, and Power systems.

Spyre Accelerators are engineered to support the development of higher-accuracy AI techniques, enabling real-time generative asset creation, customer data ingestion and interpretation for outreach, cross-selling, and risk assessments.

:::{figure} https://research-website-prod-cms-uploads.s3.us.cloud-object-storage.appdomain.cloud/IBM_Telum_Spyre_Chip_4k_02_e4a8dfccec.jpg
:alt: The IBM Spyre AI Card
:width: 680px
:align: center

The IBM Spyre AI Card. *Image credit: [IBM Research](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator).*
:::

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

The Spyre device is integrated with PyTorch as a custom backend device, enabling standard PyTorch models to leverage Spyre's AI acceleration capabilities. See the [Getting Started](../getting_started/installation.md) guide for setup and usage instructions. The [examples](../user_guide/examples.md) section provides annotated code examples.

IBM and the PyTorch community are collaborating to broaden Spyre's integration into the open-source AI stack, including vLLM and torchtitan. See [Expanding AI model training and inference for the open-source community](https://research.ibm.com/blog/pytorch-expanding-ai-model-training-and-inference-for-the-open-source-community) for details.

## Learn more

Refer to the official product pages and IBM Research blogs to learn more about the Spyre device.

**Product pages**
* [IBM Spyre Accelerator for Z and LinuxONE](https://www.ibm.com/support/z-content-solutions/spyre-accelerator-z-and-linuxone/)

**IBM Research blogs**
* [Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator) — architecture deep-dive and full-stack approach
* [Enhancing enterprise AI with the IBM Spyre Accelerator](https://research.ibm.com/blog/spyre-for-z) — Spyre for IBM Z mainframe AI inference
* [Expanding AI model training and inference for the open-source community](https://research.ibm.com/blog/pytorch-expanding-ai-model-training-and-inference-for-the-open-source-community) — Spyre in the PyTorch ecosystem

**Internal reference**
* [Dataflow Architecture Reference](dataflow_architecture.md)
