"""Ming audio integration fixes for the pinned vLLM runtime."""


def register():
    from .fp4_runtime import install as install_fp4_runtime

    install_fp4_runtime()
    from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
    from vllm_omni.model_executor.models.ming3_audio.ming3_audio import (
        BailingMoeV3AudioForConditionalGeneration,
    )

    audio = BailingMoeV3AudioForConditionalGeneration
    if getattr(audio, "_ming_nvfp4_mapping_installed", False):
        return
    # Quantization exclusions are mapped at the outer model. The upstream audio
    # wrapper maps model.* -> language_model.model.* but omits attention ->
    # self_attn, causing BF16 attention tensors to be loaded into packed linears.
    audio.hf_to_vllm_mapper = (
        BailingMoeV3ForCausalLM.hf_to_vllm_mapper | audio.hf_to_vllm_mapper
    )
    audio.packed_modules_mapping = dict(BailingMoeV3ForCausalLM.packed_modules_mapping)
    audio._ming_nvfp4_mapping_installed = True
