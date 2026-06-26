import torchvision
from torch import nn
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
)

__all__ = ["get_model"]

def get_torch_model_baseline(arch_name: str, model_config) -> nn.Module:
    """Return a standard torchvision model (3-channel input) for baseline runs."""
    num_classes = int(model_config.get("num_classes", 2))

    if arch_name == "resnet18":
        weights = None
        if model_config.get("weights"):
            weights = ResNet18_Weights.verify(model_config["weights"])
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch_name == "resnet50":
        weights = None
        if model_config.get("weights"):
            weights = ResNet50_Weights.verify(model_config["weights"])
        m = torchvision.models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch_name == "densenet121":
        weights = None
        if model_config.get("weights"):
            weights = DenseNet121_Weights.verify(model_config["weights"])
        m = torchvision.models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise ValueError(f"Unknown baseline arch_name={arch_name!r}")

def get_model(model_config) -> nn.Module:
    # extract args
    arch_name = model_config["name"]

    # Baseline (non-B-cos) path: standard torchvision models (3-channel)
    if not model_config.get("is_bcos", False):
        print(f"Using baseline (non-B-cos) model for arch_name={arch_name!r}")
        return get_torch_model_baseline(arch_name, model_config)
    else:
        error_msg = f"get_model() currently only supports baseline (non-B-cos) models. Received model_config with name={arch_name!r} and is_bcos={model_config.get('is_bcos', False)}"
        raise NotImplementedError(error_msg)