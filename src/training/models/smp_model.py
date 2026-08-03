"""SMP model wrappers for configurable synthesis architectures."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from training.models.aux_attention import (
    AuxDecoderAttention,
    normalize_aux_segmentation_config,
)


ARCHITECTURE_ALIASES = {
    "smp_unet": "unet",
    "unet": "unet",
    "unet++": "unetplusplus",
    "unetpp": "unetplusplus",
    "unet_plus_plus": "unetplusplus",
    "unetplusplus": "unetplusplus",
    "manet": "manet",
    "ma_net": "manet",
    "linknet": "linknet",
    "fpn": "fpn",
    "pspnet": "pspnet",
    "pan": "pan",
    "deeplabv3": "deeplabv3",
    "deeplab_v3": "deeplabv3",
    "deeplabv3+": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "deeplab_v3_plus": "deeplabv3plus",
    "upernet": "upernet",
    "uper_net": "upernet",
}
SUPPORTED_ARCHITECTURES = tuple(dict.fromkeys(ARCHITECTURE_ALIASES.values()))

SMP_CLASS_NAMES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "manet": "MAnet",
    "linknet": "Linknet",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "deeplabv3": "DeepLabV3",
    "deeplabv3plus": "DeepLabV3Plus",
    "upernet": "UPerNet",
}

ALLOWED_ARCHITECTURE_KWARGS = {
    "unet": {
        "encoder_depth",
        "decoder_use_norm",
        "decoder_channels",
        "decoder_attention_type",
        "decoder_interpolation",
    },
    "unetplusplus": {
        "encoder_depth",
        "decoder_use_norm",
        "decoder_channels",
        "decoder_attention_type",
        "decoder_interpolation",
    },
    "manet": {
        "encoder_depth",
        "decoder_use_norm",
        "decoder_channels",
        "decoder_pab_channels",
        "decoder_interpolation",
    },
    "linknet": {
        "encoder_depth",
        "decoder_use_norm",
    },
    "fpn": {
        "encoder_depth",
        "decoder_pyramid_channels",
        "decoder_segmentation_channels",
        "decoder_merge_policy",
        "decoder_dropout",
        "decoder_interpolation",
        "upsampling",
    },
    "pspnet": {
        "encoder_depth",
        "psp_out_channels",
        "decoder_use_norm",
        "psp_dropout",
        "upsampling",
    },
    "pan": {
        "encoder_depth",
        "encoder_output_stride",
        "decoder_channels",
        "decoder_interpolation",
        "upsampling",
    },
    "deeplabv3": {
        "encoder_depth",
        "encoder_output_stride",
        "decoder_channels",
        "decoder_atrous_rates",
        "decoder_aspp_separable",
        "decoder_aspp_dropout",
        "upsampling",
    },
    "deeplabv3plus": {
        "encoder_depth",
        "encoder_output_stride",
        "decoder_channels",
        "decoder_atrous_rates",
        "decoder_aspp_separable",
        "decoder_aspp_dropout",
        "upsampling",
    },
    "upernet": {
        "encoder_depth",
        "decoder_channels",
        "decoder_use_norm",
        "upsampling",
    },
}


def normalize_backbone_name(backbone: str) -> str:
    normalized = backbone.strip().lower()
    if not normalized:
        raise ValueError("model backbone/encoder_name must not be empty")
    return normalized


def normalize_architecture_name(architecture: str) -> str:
    normalized = architecture.strip().lower()
    normalized = ARCHITECTURE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_ARCHITECTURES:
        available = ", ".join(sorted(ARCHITECTURE_ALIASES))
        raise ValueError(
            f"Unsupported SMP architecture={architecture!r}. "
            f"Available model.type values: {available}."
        )
    return normalized


def is_smp_model_type(model_type: str) -> bool:
    normalized = model_type.strip().lower()
    return normalized in ARCHITECTURE_ALIASES or normalized in SUPPORTED_ARCHITECTURES


def normalize_encoder_weights(
    encoder_name: str,
    encoder_weights: str | bool | None,
) -> str | bool | None:
    if encoder_weights is None:
        return None
    if isinstance(encoder_weights, str):
        normalized = encoder_weights.strip().lower()
        if normalized in {"", "none", "null", "false", "0"}:
            return None
        if normalized in {"true", "pretrained"}:
            return True if encoder_name.startswith("tu-") else "imagenet"
    if encoder_weights is False:
        return None
    if encoder_weights is True:
        return True if encoder_name.startswith("tu-") else "imagenet"
    if encoder_name.startswith("tu-"):
        return True
    return encoder_weights


def _normalize_sequence(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _prepare_smp_kwargs(
    architecture: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    allowed = ALLOWED_ARCHITECTURE_KWARGS[architecture]
    provided = {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }
    unknown = sorted(set(provided) - allowed)
    if unknown:
        allowed_text = ", ".join(sorted(allowed))
        unknown_text = ", ".join(unknown)
        raise ValueError(
            f"Unsupported config field(s) for model.type={architecture!r}: "
            f"{unknown_text}. Allowed fields: {allowed_text}."
        )

    return {
        key: _normalize_sequence(value)
        for key, value in provided.items()
    }


class SMPModel(nn.Module):
    """SMP architecture that predicts a post-contrast enhancement residual."""

    def __init__(
        self,
        architecture: str = "unet",
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_name: str | None = None,
        backbone: str | None = None,
        encoder_weights: str | bool | None = "imagenet",
        aux_segmentation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        normalized_architecture = normalize_architecture_name(architecture)
        aux_config = normalize_aux_segmentation_config(aux_segmentation)
        self.aux_segmentation_enabled = bool(aux_config.get("enabled", False))
        aux_out_channels = int(aux_config.get("out_channels", 1))
        attention_config = aux_config["attention"]
        self.aux_attention_enabled = bool(attention_config["enabled"])
        if self.aux_segmentation_enabled and normalized_architecture not in {
            "unet",
            "unetplusplus",
        }:
            raise ValueError(
                "model.aux_segmentation is currently supported only for SMP "
                "model.type='unet' or model.type='unetpp'."
            )
        selected_backbone = backbone or encoder_name or "resnet34"
        normalized_backbone = normalize_backbone_name(selected_backbone)
        weights = normalize_encoder_weights(normalized_backbone, encoder_weights)
        smp_kwargs = _prepare_smp_kwargs(normalized_architecture, kwargs)

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation-models-pytorch is required for SMP model types. "
                "Install it with `pip install segmentation-models-pytorch timm`."
            ) from exc

        model_cls = getattr(smp, SMP_CLASS_NAMES[normalized_architecture])
        self.architecture = normalized_architecture
        self.encoder_name = normalized_backbone
        self.encoder_weights = weights
        self.model = model_cls(
            encoder_name=normalized_backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
            **smp_kwargs,
        )
        if self.aux_segmentation_enabled:
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

            final_decoder_channels = self._infer_segmentation_head_channels()
            aux_prefixes: list[str] = []
            if self.architecture == "unet":
                decoder_channels = self._infer_decoder_channels()
                self.aux_decoder = UnetDecoder(
                    encoder_channels=self.model.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=len(decoder_channels),
                    use_norm=smp_kwargs.get("decoder_use_norm", "batchnorm"),
                    attention_type=smp_kwargs.get("decoder_attention_type", None),
                    add_center_block=not isinstance(self.model.decoder.center, nn.Identity),
                    interpolation_mode=smp_kwargs.get("decoder_interpolation", "nearest"),
                )
                aux_prefixes.append("aux_decoder.")
            self.aux_segmentation_head = nn.Conv2d(
                final_decoder_channels,
                aux_out_channels,
                kernel_size=3,
                padding=1,
            )
            aux_prefixes.append("aux_segmentation_head.")
            if self.aux_attention_enabled:
                self.aux_attention = AuxDecoderAttention(
                    final_decoder_channels,
                    strength=float(attention_config["strength"]),
                    warmup_epochs=int(attention_config["warmup_epochs"]),
                    norm="batch",
                )
                aux_prefixes.append("aux_attention.")
            self.aux_state_key_prefixes = tuple(aux_prefixes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_attention_enabled:
            prediction, _ = self.forward_with_aux(x, current_epoch=None)
            return prediction
        prediction = self.model(x)
        if prediction.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"SMP model.type={self.architecture!r} with "
                f"backbone={self.encoder_name!r} returned spatial shape "
                f"{tuple(prediction.shape[-2:])}, expected {tuple(x.shape[-2:])}. "
                "Adjust encoder_depth or upsampling in the model config."
            )
        return prediction

    def _infer_decoder_channels(self) -> tuple[int, ...]:
        channels: list[int] = []
        for block in self.model.decoder.blocks:
            conv2 = getattr(block, "conv2", None)
            if conv2 is None or not isinstance(conv2, nn.Sequential):
                raise RuntimeError("Could not infer SMP U-Net decoder channels")
            conv = conv2[0]
            if not isinstance(conv, nn.Conv2d):
                raise RuntimeError("Could not infer SMP U-Net decoder channels")
            channels.append(int(conv.out_channels))
        if not channels:
            raise RuntimeError("SMP U-Net decoder has no decoder blocks")
        return tuple(channels)

    def _infer_segmentation_head_channels(self) -> int:
        head_conv = self.model.segmentation_head[0]
        if not isinstance(head_conv, nn.Conv2d):
            raise RuntimeError("Could not infer SMP segmentation head channels")
        return int(head_conv.in_channels)

    @staticmethod
    def _decoder_spatial_shapes(features: list[torch.Tensor]) -> list[torch.Size]:
        return [feature.shape[2:] for feature in features][::-1]

    @staticmethod
    def _decoder_inputs(
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        decoder_features = features[1:][::-1]
        head = decoder_features[0]
        skip_connections = decoder_features[1:]
        return head, skip_connections

    def _decode_with_features(
        self,
        decoder: nn.Module,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.architecture == "unetplusplus":
            return decoder(features), []
        spatial_shapes = self._decoder_spatial_shapes(features)
        head, skip_connections = self._decoder_inputs(features)
        decoded_features: list[torch.Tensor] = []
        x = decoder.center(head)
        for idx, decoder_block in enumerate(decoder.blocks):
            height, width = spatial_shapes[idx + 1]
            skip = skip_connections[idx] if idx < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip)
            decoded_features.append(x)
        return x, decoded_features

    def _decode_auxiliary(
        self,
        features: list[torch.Tensor],
        generator_features: list[torch.Tensor],
    ) -> torch.Tensor:
        spatial_shapes = self._decoder_spatial_shapes(features)
        head, skip_connections = self._decoder_inputs(features)
        x = self.aux_decoder.center(head)
        for idx, decoder_block in enumerate(self.aux_decoder.blocks):
            height, width = spatial_shapes[idx + 1]
            skip = skip_connections[idx] if idx < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip)
            x = x + generator_features[idx]
        return x

    def aux_attention_factor(self, current_epoch: int | None = None) -> float | None:
        if not self.aux_attention_enabled:
            return None
        return self.aux_attention.factor(current_epoch)

    def forward_with_aux(
        self,
        x: torch.Tensor,
        current_epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.aux_segmentation_enabled:
            return self.forward(x), None
        features = self.model.encoder(x)
        decoded, generator_features = self._decode_with_features(
            self.model.decoder,
            features,
        )
        aux_decoded = (
            self._decode_auxiliary(features, generator_features)
            if self.architecture == "unet"
            else decoded
        )
        seg_logits = self.aux_segmentation_head(aux_decoded)
        if self.aux_attention_enabled:
            decoded = self.aux_attention(decoded, seg_logits, current_epoch)
        residual = self.model.segmentation_head(decoded)
        if residual.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"SMP model.type={self.architecture!r} with "
                f"backbone={self.encoder_name!r} returned spatial shape "
                f"{tuple(residual.shape[-2:])}, expected {tuple(x.shape[-2:])}."
            )
        if seg_logits.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                "Auxiliary segmentation branch returned spatial shape "
                f"{tuple(seg_logits.shape[-2:])}, expected {tuple(x.shape[-2:])}."
            )
        return residual, seg_logits


class SMPUNet(SMPModel):
    """Backward-compatible wrapper for the default SMP U-Net architecture."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(architecture="unet", **kwargs)


__all__ = [
    "ALLOWED_ARCHITECTURE_KWARGS",
    "ARCHITECTURE_ALIASES",
    "SMPModel",
    "SMPUNet",
    "SUPPORTED_ARCHITECTURES",
    "is_smp_model_type",
    "normalize_architecture_name",
    "normalize_backbone_name",
    "normalize_encoder_weights",
]
