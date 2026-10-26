from pathlib import Path
from types import SimpleNamespace

import torch

from ming3_nvfp4.adapter import quant_config
from ming3_nvfp4.common import read_recipe


def fixture_model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        expert = torch.nn.Module()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            setattr(expert, projection, torch.nn.Linear(16, 16, bias=False))
        layer.mlp.experts = torch.nn.ModuleList([expert])
        layer.mlp.gate = torch.nn.Linear(16, 1, bias=False)
        layer.mlp.shared_experts = torch.nn.Linear(16, 16, bias=False)
        layer.attention = torch.nn.Linear(16, 16, bias=False)
        model.model.layers.append(layer)
    model.model.audio = torch.nn.Linear(16, 16, bias=False)
    model.config = SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=2, first_k_dense_replace=1))
    return model


def test_only_routed_base_experts_are_enabled():
    import modelopt.torch.quantization as mtq

    model = fixture_model()
    recipe = read_recipe(Path("recipes/tiny.toml"))
    recipe["mode"] = "w4a16"
    cfg = quant_config(model, recipe)
    cfg["algorithm"] = None
    mtq.quantize(model, cfg)
    enabled = {name for name, module in model.named_modules()
               if name.endswith("weight_quantizer") and module.is_enabled}
    assert enabled == {f"model.layers.1.mlp.experts.0.{p}.weight_quantizer"
                       for p in ("gate_proj", "up_proj", "down_proj")}
    assert not any(module.is_enabled for name, module in model.named_modules()
                   if name.endswith("input_quantizer"))


def test_mse_recipe_keeps_static_weight_scale_algorithm():
    recipe = read_recipe(Path("recipes/tiny.toml"))
    recipe.update(algorithm="mse", mode="w4a4")
    cfg = quant_config(fixture_model(), recipe)
    assert cfg["algorithm"]["method"] == "mse"
    assert cfg["algorithm"]["fp8_scale_sweep"]
    assert any(rule["quantizer_name"].endswith("input_quantizer") for rule in cfg["quant_cfg"])


def test_layerwise_capture_replays_through_accelerate_and_restores_mtp():
    from accelerate import cpu_offload
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    from ming3_nvfp4.adapter import BaseLayerView, prepare_layerwise_offload

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                torch.nn.Linear(4, 4, bias=False) for _ in range(4)])
            self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=3))

        def forward(self, value):
            for layer in self.model.layers[:3]:
                value = layer(value)
            return value

    model = SmallModel()
    value = torch.ones(1, 4)
    expected = model(value).detach().clone()
    originals = list(model.model.layers)
    cpu_offload(model, execution_device=torch.device("cpu"))
    prepare_layerwise_offload(model)
    layers = BaseLayerView(model)
    collector = LayerActivationCollector(model)
    collector._patch_all_layers(layers)
    try:
        for layer in layers:
            inputs = collector.get_input_activations(layer, lambda m: m(value))
        args, kwargs = inputs[0]
        assert torch.allclose(layers[-1](*args, **kwargs), expected)
        assert model.model.layers[0] is not originals[0]
        assert model.model.layers[3] is originals[3]
    finally:
        collector._unpatch_all_layers()
    assert list(model.model.layers) == originals
    assert torch.allclose(model(value), expected)


def test_restore_materializes_meta_scales_without_replacing_weights():
    from ming3_nvfp4.adapter import restore_quantizer_buffers

    model = torch.nn.Linear(4, 4)
    model.weight_quantizer = torch.nn.Module()
    model.weight_quantizer.register_buffer("_amax", torch.empty((), device="meta"))
    weight = model.weight
    restore_quantizer_buffers(model, {"weight_quantizer._amax": torch.tensor(0.25)}, "cpu")
    assert model.weight is weight
    assert model.weight_quantizer._amax.item() == 0.25
