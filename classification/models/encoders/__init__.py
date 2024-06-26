# import timm
import functools
import torch.utils.model_zoo as model_zoo

import torch
from .resnet import resnet_encoders
from .mobilenet import mobilenet_encoders
from .mcunet import mcunet_encoders

from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(mobilenet_encoders)
encoders.update(mcunet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs): # pretrained là thuộc tính chỉ của mcunet (nó có thể thuộc kwargs)

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(
            name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(log_grad='log_grad' in kwargs and kwargs['log_grad'])
    encoder = Encoder(**params)
    if weights is not None and weights != "full_imagenet":
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))    
        
    if weights == "full_imagenet":
        if name == "resnet18":
            model_state_dict = torch.load("pretrained_ckpts/pre_trained_resnet18_raw.ckpt")['state_dict']
        elif name == "mobilenet_v2":
            model_state_dict = torch.load("pretrained_ckpts/pre_trained_mbv2_raw.ckpt")['state_dict']
        # elif name == "mcunet":  (Triển khai sau) (Không cần triển khai nữa vì họ đã tự load)
        encoder.load_state_dict(model_state_dict)
        
    if "mcunet" in name:
        assert "pretrained" in kwargs, "[Warning] pretrained condition is not defined for mcunet"
        encoder.set_in_channels(in_channels, pretrained=kwargs["pretrained"])
    else:
        encoder.set_in_channels(in_channels, pretrained=weights is not None) # Có vẻ không đúng, vì mcunet là bản được pretrained nhưng weight trong file config họ không set, tức là = None => không pretrained (Đã sửa)
    
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):

    all_settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in all_settings.keys():
        raise ValueError(
            "Available pretrained options {}".format(all_settings.keys()))
    settings = all_settings[pretrained]

    formatted_settings = {}
    formatted_settings["input_space"] = settings.get("input_space", "RGB")
    formatted_settings["input_range"] = list(
        settings.get("input_range", [0, 1]))
    formatted_settings["mean"] = list(settings.get("mean"))
    formatted_settings["std"] = list(settings.get("std"))

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
