# 🚀 DynamicKV: Task-Aware Adaptive KV Cache Compression for Long-Context LLMs  
# 🚀 DynamicKV：面向任务自适应的长上下文大模型 KV 缓存压缩方法

<div align="center">

<!-- Language Toggle -->
<button onclick="toggleLang('en')" style="margin:5px;">English</button>
<button onclick="toggleLang('zh')" style="margin:5px;">中文</button>

<script>
function toggleLang(lang) {
  const enElems = document.querySelectorAll('.lang-en');
  const zhElems = document.querySelectorAll('.lang-zh');
  if (lang === 'en') {
    enElems.forEach(el => el.style.display = 'block');
    zhElems.forEach(el => el.style.display = 'none');
  } else {
    enElems.forEach(el => el.style.display = 'none');
    zhElems.forEach(el => el.style.display = 'block');
  }
}
// Default: show English
document.addEventListener('DOMContentLoaded', () => {
  toggleLang('en');
});
</script>

</div>

---

<div class="lang-en">

[![Paper](https://img.shields.io/badge/arXiv-2412.14838-b31b1b.svg)](https://arxiv.org/abs/2412.14838)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)]()

**DynamicKV** is a task-aware, layer-adaptive KV cache compression method for long-context LLM inference. It dynamically allocates KV cache budgets per layer based on task-specific attention patterns, achieving **~90% of FullKV performance with only 1.7% cache retention**.

> 💡 **Key Insight**: Different tasks (e.g., QA, summarization, code completion) exhibit distinct token importance distributions across transformer layers. Fixed-pattern compression (e.g., pyramid, sliding window) fails to capture this variability.

</div>

<div class="lang-zh" style="display:none">

[![论文](https://img.shields.io/badge/arXiv-2412.14838-b31b1b.svg)](https://arxiv.org/abs/2412.14838)  
[![许可证](https://img.shields.io/badge/许可证-MIT-green)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)]()

**DynamicKV** 是一种面向任务自适应的长上下文大语言模型（LLM）KV 缓存压缩方法。它根据任务特有的注意力分布，动态为每一层分配缓存预算，在仅保留 **1.7% KV 缓存** 的情况下仍能达到 **约 90% 的原始性能**。

> 💡 **核心洞察**：不同任务（如问答、摘要、代码补全）在 Transformer 各层对 token 的重要性分布显著不同。固定模式压缩方法（如金字塔结构、滑动窗口）无法适配这种差异。

</div>

---

## 🔍 Method Overview / 方法简介

<div class="lang-en">

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

</div>

<div class="lang-zh" style="display:none">

### 为什么需要 DynamicKV？
现有 KV 压缩方法（如 StreamingLLM、PyramidKV）在所有任务和层上使用**固定保留模式**，忽略了任务特有的注意力动态。

### 工作原理
1. **动态预算分配**：每层根据与最近窗口 token 的注意力得分，保留 top-K 重要 token。
2. **渐进式缓存更新**：每 `m` 层，全局重新归一化并调整历史层的 KV 缓存大小，确保总内存预算不超限。

### 优势
- ✅ **任务感知**：自动适配问答、摘要、代码等任务。
- ✅ **高压缩率**：仅 1.7% 缓存即可保留 90% 性能。
- ✅ **无需训练**。
- ⌛️ **即插即用**：仅修改 prefill 阶段，兼容 vLLM、FlashAttention。

</div>

---

## 📊 Model Comparison / 模型对比（LongBench, KV Cache = 512）

<div class="lang-en">

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

</div>

<div class="lang-zh" style="display:none">

| 模型 | FullKV | StreamingLLM | H2O | SnapKV | PyramidKV | **DynamicKV（ours）** |
|------|--------|--------------|-----|--------|-----------|----------------------|
| Llama-3-8B-Instruct | 41.95 | 34.70 | 37.20 | 40.30 | 40.18 | **40.73** |
| Mistral-7B-Instruct-v0.2 | 42.71 | 30.06 | 37.37 | 40.71 | 40.47 | **40.90** |
| Qwen2-7B-Instruct | 40.71 | 29.65 | 35.63 | 38.47 | 38.19 | **39.16** |
| InternLM-2.5-7B-Chat-1M | 43.21 | 32.25 | 34.65 | 37.84 | 37.86 | **38.39** |

> 💡 **结论**：在极端压缩（6.9% 上下文比例）下，DynamicKV **全面超越现有 SOTA 方法**。

### Needle-in-a-Haystack（32K 上下文，64 缓存）
| 方法 | 准确率 |
|------|--------|
| FullKV | 92% |
| StreamingLLM | 26% |
| PyramidKV | 72% |
| **DynamicKV** | **83%** |

</div>

---

## ⚡ Quick Start / 一键启动

<div class="lang-en">

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

</div>

<div class="lang-zh" style="display:none">

### 安装
```bash
git clone https://github.com/DreamMr/DynamicKV.git
cd DynamicKV
pip install -r requirements.txt
```

### 运行示例
```bash
bash run/longbench/scripts/run_qwen2/run_qwen2_7b_instruct_dynamic_v11_maxpool.sh 
```

### 支持模型
- `meta-llama/Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2-7B-Instruct`
- `internlm/internlm2_5-7b-chat-1m`

</div>

---

## 📚 Citation / 引用

<div class="lang-en">

If you find DynamicKV useful, please cite our paper:

```bibtex
@misc{zhou2025dynamickvtaskawareadaptivekv,
      title={DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs}, 
      author={Xiabin Zhou and Wenbin Wang and Minyan Zeng and Jiaxian Guo and Xuebo Liu and Li Shen and Min Zhang and Liang Ding},
      year={2025},
      eprint={2412.14838},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14838}, 
}
```

</div>

<div class="lang-zh" style="display:none">

如果您觉得 DynamicKV 对您的研究有帮助，请引用我们的论文：

```bibtex
@misc{zhou2025dynamickvtaskawareadaptivekv,
      title={DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs}, 
      author={Xiabin Zhou and Wenbin Wang and Minyan Zeng and Jiaxian Guo and Xuebo Liu and Li Shen and Min Zhang and Liang Ding},
      year={2025},
      eprint={2412.14838},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14838}, 
}
```

</div>


> 🔗 **Code**: [https://github.com/DreamMr/DynamicKV](https://github.com/DreamMr/DynamicKV)  
> 📄 **Paper**: [arXiv:2412.14838](https://arxiv.org/abs/2412.14838)


✅ **使用说明**：
- 此 README 在 GitHub 上默认显示英文，点击“中文”按钮可切换为中文（依赖浏览器 JavaScript）。
- 若需纯静态版本（无 JS），也可提供双语并列版，欢迎告知。