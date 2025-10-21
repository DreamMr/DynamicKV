# 🚀 DynamicKV: Task-Aware Adaptive KV Cache Compression for Long-Context LLMs  

---


[![Paper](https://img.shields.io/badge/arXiv-2412.14838-b31b1b.svg)](https://arxiv.org/abs/2412.14838)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)]()

**DynamicKV** is a task-aware, layer-adaptive KV cache compression method for long-context LLM inference. It dynamically allocates KV cache budgets per layer based on task-specific attention patterns, achieving **~90% of FullKV performance with only 1.7% cache retention**.

> 💡 **Key Insight**: Different tasks (e.g., QA, summarization, code completion) exhibit distinct token importance distributions across transformer layers. Fixed-pattern compression (e.g., pyramid, sliding window) fails to capture this variability.

---

## 🔍 Method Overview

### Why DynamicKV?
Existing KV compression methods (e.g., StreamingLLM, PyramidKV) use **fixed retention patterns** across layers and tasks, ignoring task-specific attention dynamics.

### How It Works
1. **Dynamic Budget Allocation**: For each layer, retain top-K tokens based on attention scores with the most recent window.
2. **Progressive Cache Update**: Every `m` layers, globally re-normalize and adjust historical KV cache sizes to respect total memory budget.

### Advantages
- ✅ **Task-aware**: Adapts to QA, summarization, code, etc.
- ✅ **High compression**: 1.7% cache → 90% performance.
- ✅ **No training required**.
- ⌛️ **Plug-and-play**: Only modifies prefill phase; compatible with vLLM, FlashAttention.

---

## 📊 Model Comparison（LongBench, KV Cache = 512）

| Model | FullKV | StreamingLLM | H2O | SnapKV | PyramidKV | **DynamicKV (Ours)** |
|-------|--------|--------------|-----|--------|-----------|----------------------|
| Llama-3-8B-Instruct | 41.95 | 34.70 | 37.20 | 40.30 | 40.18 | **40.73** |
| Mistral-7B-Instruct-v0.2 | 42.71 | 30.06 | 37.37 | 40.71 | 40.47 | **40.90** |
| Qwen2-7B-Instruct | 40.71 | 29.65 | 35.63 | 38.47 | 38.19 | **39.16** |
| InternLM-2.5-7B-Chat-1M | 43.21 | 32.25 | 34.65 | 37.84 | 37.86 | **38.39** |

> 💡 **Conclusion**: DynamicKV **consistently outperforms SOTA** under extreme compression (6.9% context ratio).

### Needle-in-a-Haystack (32K context, 64 cache)
| Method | Accuracy |
|--------|----------|
| FullKV | 92% |
| StreamingLLM | 26% |
| PyramidKV | 72% |
| **DynamicKV** | **83%** |

---

## ⚡ Quick Start

### Install
```bash
git clone https://github.com/DreamMr/DynamicKV.git
cd DynamicKV
pip install transformers>=0.44.1
```

### Run Example
```bash
bash run/longbench/scripts/run_qwen2/run_qwen2_7b_instruct_dynamic_v11_maxpool.sh 
```

### Supported Models
- `meta-llama/Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2-7B-Instruct`
- `internlm/internlm2_5-7b-chat-1m`

---

## 📚 Citation

If you find DynamicKV useful, please cite our paper:

```bibtex
@inproceedings{
    zhou2025dynamickv,
    title={Dynamic{KV}: Task-Aware Adaptive {KV} Cache Compression for Long Context {LLM}s},
    author={Xiabin Zhou and Wenbin Wang and Minyan Zeng and Jiaxian Guo and Xuebo Liu and Li Shen and Min Zhang and Liang Ding},
    booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
    year={2025},
    url={https://openreview.net/forum?id=eDc56RuoC6}
}
```

> 🔗 **Code**: [https://github.com/DreamMr/DynamicKV](https://github.com/DreamMr/DynamicKV)  
> 📄 **Paper**: [arXiv:2412.14838](https://arxiv.org/abs/2412.14838)