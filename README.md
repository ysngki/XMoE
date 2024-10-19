# XMoE: Sparse Models with Fine-grained and Adaptive Expert Selection

# Abstract

Sparse models, including sparse Mixture-of-Experts (MoE) models, have emerged as an effective approach for scaling Transformer models. However, they often suffer from **computational inefficiency since a significant number of parameters are unnecessarily involved in computations by multiplying values by zero or low activation values**. To address this issue, we present XMoE, a novel MoE designed to enhance both the efficacy and efficiency of sparse MoE models. **XMoE leverages small experts and a threshold-based router to enable tokens to selectively engage only essential parameters**. Our extensive experiments on language modeling and machine translation tasks demonstrate that XMoE enhances model performance and can decrease the computation load at MoE layers by over 50\% without sacrificing performance. Furthermore, we present the versatility of XMoE by applying it to dense models, enabling sparse computation during inference.

# Codebase

Pretraining leverages the [Deepspeed-Megatron](https://github.com/microsoft/Megatron-DeepSpeed.git) framework, while NMT is built upon the [fairseq](https://github.com/facebookresearch/fairseq/tree/da8fb630880d529ab47e53381c30ddc8ad235216) library.

For details on the implementation and experimental setup, please refer to the respective instructions:

- [Pretraining Instructions](MoE-Megatron-DeepSpeed/README.md)
- [Translation Instructions](fairseq/README.md)



## Citations

If you find XMoE useful or relevant to your research, please kindly cite our paper:

```latex
@inproceedings{yang-etal-2024-xmoe,
    title = "{XM}o{E}: Sparse Models with Fine-grained and Adaptive Expert Selection",
    author = "Yang, Yuanhang  and
      Qi, Shiyi  and
      Gu, Wenchao  and
      Wang, Chaozheng  and
      Gao, Cuiyun  and
      Xu, Zenglin",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.694",
    doi = "10.18653/v1/2024.findings-acl.694",
}

```