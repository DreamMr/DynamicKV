import torch
from typing import Optional, Tuple, Dict, Any
from importlib.metadata import version
import transformers

from .llama_model_impl.dynamickv_v11 import llama_flash_attn2_forward_DynamicKV_V11
from .mistral_model_impl.dynamickv_v11 import mistral_flash_attn2_forward_DynamicKV_V11
from .qwen2_model_impl.dynamickv_v11 import qwen2_flash_attn2_forward_DynamicKV_V11
from .internlm_model_impl.dynamickv_v11 import internlm_flash_attn2_forward_DynamicKV_V11

from .llama_model_impl.utils import prepare_inputs_for_generation_llama
from .mistral_model_impl.utils import prepare_inputs_for_generation_mistral
from .qwen2_model_impl.utils import prepare_inputs_for_generation_qwen2
from .internlm_model_impl.utils import prepare_inputs_for_generation_internlm

import sys
sys.path.append('/root/.cache/huggingface/modules')
import transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2

llama_forward_function_map = {
    "dynamickv_v11": llama_flash_attn2_forward_DynamicKV_V11
}

mistral_forward_function_map = {
    "dynamickv_v11": mistral_flash_attn2_forward_DynamicKV_V11,

}



qwen2_forward_function_map = {
    "dynamickv_v11": qwen2_flash_attn2_forward_DynamicKV_V11
}

internlm_forward_function_map = {
    "dynamickv_v11": internlm_flash_attn2_forward_DynamicKV_V11
}

def replace_attention(model_type: str, method: str):
    if method.lower() == "fullkv":
        return
    # if method not in llama_forward_function_map:
    #     raise ValueError(f"Unknown method: {method}")
    print(f"Using method: {method}!!!")
    print(f"Replacing attention forward: {model_type}!!!")
    if model_type == "llama" or model_type == "lwm":
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_forward_function_map[method]
    elif model_type == "mistral":
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_forward_function_map[method]
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_forward_function_map[method]
    elif model_type == "internlm":
        transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2.InternLM2FlashAttention2.forward = internlm_forward_function_map[method]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen2
        transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2.InternLM2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_internlm

    
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version