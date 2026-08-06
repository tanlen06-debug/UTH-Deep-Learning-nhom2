import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes=10,
    strategy="fc_only",
    use_pretrained=True
):
    """
    Xây dựng mô hình ResNet-18 cho bài toán phân loại CIFAR-10.

    Tham số:
        num_classes:
            Số lớp đầu ra. CIFAR-10 có 10 lớp.

        strategy:
            - "fc_only": chỉ huấn luyện lớp fc.
            - "layer4_fc": fine-tuning layer4 và fc.

        use_pretrained:
            True: sử dụng trọng số pre-trained trên ImageNet.
            False: không tải trọng số pre-trained.

    Trả về:
        Mô hình ResNet-18 đã được điều chỉnh.
    """

    # Kiểm tra chiến lược có hợp lệ không
    if strategy not in ["fc_only", "layer4_fc"]:
        raise ValueError(
            "strategy phải là 'fc_only' hoặc 'layer4_fc'"
        )

    # Tải ResNet-18
    if use_pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    # Đóng băng toàn bộ tham số của mô hình
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Lấy số đặc trưng đầu vào của lớp fc gốc
    in_features = model.fc.in_features

    # Thay lớp fc để phù hợp với CIFAR-10
    model.fc = nn.Linear(
        in_features=in_features,
        out_features=num_classes
    )

    # Nếu fine-tuning thì mở lại layer4
    if strategy == "layer4_fc":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    return model


def build_resnet18_fc_only(
    num_classes=10,
    use_pretrained=True
):
    """
    ResNet-18 A:
    Đóng băng backbone và chỉ huấn luyện lớp fc.
    """

    model = build_resnet18(
        num_classes=num_classes,
        strategy="fc_only",
        use_pretrained=use_pretrained
    )

    return model


def build_resnet18_finetune_layer4(
    num_classes=10,
    use_pretrained=True
):
    """
    ResNet-18 B:
    Fine-tuning layer4 và lớp fc.
    """

    model = build_resnet18(
        num_classes=num_classes,
        strategy="layer4_fc",
        use_pretrained=use_pretrained
    )

    return model


def count_parameters(model):
    """
    Đếm tổng số tham số, số tham số trainable
    và số tham số đang bị đóng băng.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    return {
        "total": total_parameters,
        "trainable": trainable_parameters,
        "frozen": frozen_parameters
    }


def get_trainable_parameter_names(model):
    """
    Lấy tên các tham số đang được phép huấn luyện.
    """

    trainable_names = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_names.append(name)

    return trainable_names