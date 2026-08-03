"""Lightning modules for synthesis model families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
import torch.nn as nn

from training.losses import LossWeights, SynthesisLoss
from training.official_eval import (
    ValidationPrediction,
    run_official_evaluation,
)
from training.visualization import (
    close_validation_grid,
    log_validation_grid_to_tensorboard,
    make_validation_grid,
    save_validation_grid,
    select_visualization_cases,
)


class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN discriminator for paired precontrast/postcontrast images."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("discriminator_layers must be >= 1")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        previous_channels = base_channels
        for layer_idx in range(1, num_layers):
            channels = min(base_channels * (2**layer_idx), 512)
            layers.extend(
                [
                    nn.Conv2d(
                        previous_channels,
                        channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            previous_channels = channels

        channels = min(previous_channels * 2, 512)
        layers.extend(
            [
                nn.Conv2d(
                    previous_channels,
                    channels,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([condition, image], dim=1))


class ResidualTranslationModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        max_epochs: int = 100,
        loss_weights: LossWeights | None = None,
        official_models_dir: str | None = None,
        official_eval_predictions_dir: str | Path = "experiments/resunet_baseline/official_eval_predictions",
        val_visualizations_dir: str | Path = "experiments/resunet_baseline/val_visualizations",
        official_eval_every_n_epochs: int = 1,
        visualize_val_samples: bool = True,
        num_visualization_cases: int = 4,
        visualization_seed: int = 42,
        seg_loss: nn.Module | None = None,
        aux_seg_loss: nn.Module | None = None,
        kl_loss: nn.Module | None = None,
        wavelet_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = SynthesisLoss(
            loss_weights,
            seg_loss=seg_loss,
            aux_seg_loss=aux_seg_loss,
            kl_loss=kl_loss,
            wavelet_loss=wavelet_loss,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.max_epochs = max_epochs
        self.official_models_dir = official_models_dir
        self.official_eval_predictions_dir = Path(official_eval_predictions_dir)
        self.val_visualizations_dir = Path(val_visualizations_dir)
        self.official_eval_every_n_epochs = official_eval_every_n_epochs
        self.visualize_val_samples = visualize_val_samples
        self.num_visualization_cases = num_visualization_cases
        self.visualization_seed = visualization_seed
        self._val_predictions: list[ValidationPrediction] = []
        self._visualization_case_ids: list[str] | None = None
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss_weights",
                "seg_loss",
                "aux_seg_loss",
                "kl_loss",
                "wavelet_loss",
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # Sobel 算子  Loss: LPIPS  GAN
        return image + self.model(image)   # model = post - pre

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get("state_dict", {})
        self._fill_missing_aux_model_state(state_dict)
        for key in list(state_dict):
            if (
                getattr(self.loss_fn, "seg_loss", None) is None
                and key.startswith("loss_fn.seg_loss.")
            ):
                del state_dict[key]
            if (
                getattr(self.loss_fn, "aux_seg_loss", None) is None
                and key.startswith("loss_fn.aux_seg_loss.")
            ):
                del state_dict[key]
            if not hasattr(self, "discriminator") and key.startswith("discriminator."):
                del state_dict[key]
            if not hasattr(self, "adv_loss") and key.startswith("adv_loss."):
                del state_dict[key]

    def _fill_missing_aux_model_state(self, state_dict: dict[str, Any]) -> None:
        prefixes = getattr(self.model, "aux_state_key_prefixes", ())
        if not prefixes:
            return
        model_prefixes = tuple(f"model.{prefix}" for prefix in prefixes)
        for key, value in self.state_dict().items():
            if key.startswith(model_prefixes) and key not in state_dict:
                state_dict[key] = value

    def _forward_prediction_and_aux(
        self,
        image: torch.Tensor,
        current_epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if hasattr(self.model, "forward_with_aux"):
            residual, segmentation_logits = self.model.forward_with_aux(
                image,
                current_epoch=current_epoch,
            )
        else:
            residual, segmentation_logits = self.model(image), None
        aux_parts: dict[str, torch.Tensor] = {}
        factor_fn = getattr(self.model, "aux_attention_factor", None)
        if callable(factor_fn):
            factor = factor_fn(current_epoch)
            if factor is not None:
                aux_parts["aux_attention_factor"] = image.new_tensor(float(factor))
        return image + residual, segmentation_logits, aux_parts

    def _log_loss_parts(
        self,
        stage: str,
        loss: torch.Tensor,
        parts: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        on_step = stage == "train"
        self.log(
            f"{stage}/total_loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=True,
        )
        # Backward-compatible alias used by existing logs/checks.
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=False,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=True,
        )
        for name, value in parts.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=True,
                batch_size=batch_size,
                on_step=on_step,
                on_epoch=True,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        current_epoch = int(self.current_epoch)
        prediction, segmentation_logits, aux_parts = self._forward_prediction_and_aux(
            batch["input"],
            current_epoch=current_epoch,
        )
        loss, parts = self.loss_fn(
            prediction,
            batch["target"],
            batch["mask"],
            batch["valid_mask"],
            segmentation_logits=segmentation_logits,
            current_epoch=current_epoch,
        )
        parts.update(aux_parts)
        self._log_loss_parts("train", loss, parts, int(batch["input"].shape[0]))
        return loss

    def should_collect_validation_predictions(self) -> bool:
        return (
            self.official_eval_every_n_epochs > 0
            or (self.visualize_val_samples and self.num_visualization_cases > 0)
        )

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        current_epoch = int(self.current_epoch)
        prediction, segmentation_logits, aux_parts = self._forward_prediction_and_aux(
            batch["input"],
            current_epoch=current_epoch,
        )
        loss, parts = self.loss_fn(
            prediction,
            batch["target"],
            batch["mask"],
            batch["valid_mask"],
            segmentation_logits=segmentation_logits,
            current_epoch=current_epoch,
        )
        parts.update(aux_parts)
        batch_size = int(batch["input"].shape[0])
        self._log_loss_parts("val", loss, parts, batch_size)
        self.log("val_loss", loss, prog_bar=False, batch_size=batch_size)

        if not self.should_collect_validation_predictions():
            return loss

        for idx, case_id in enumerate(batch["case_id"]):
            h, w = [int(v) for v in batch["original_size"][idx].tolist()]
            aux_segmentation = None
            if segmentation_logits is not None:
                aux_segmentation = (
                    torch.sigmoid(segmentation_logits[idx, 0, :h, :w])
                    .detach()
                    .cpu()
                    .numpy()
                )
            self._val_predictions.append(
                ValidationPrediction(
                    case_id=str(case_id),
                    prediction=prediction[idx, 0, :h, :w].detach().cpu().numpy(),
                    target=batch["target"][idx, 0, :h, :w].detach().cpu().numpy(),
                    mask=batch["mask"][idx, 0, :h, :w].detach().cpu().numpy(),
                    precontrast=batch["input"][idx, 0, :h, :w].detach().cpu().numpy(),
                    input_path=batch["input_path"][idx],
                    target_path=batch["target_path"][idx],
                    mask_path=batch["mask_path"][idx],
                    aux_segmentation=aux_segmentation,
                )
            )
        return loss

    def _select_visualization_predictions(
        self,
        predictions: list[ValidationPrediction],
    ) -> list[ValidationPrediction]:
        if not self.visualize_val_samples or self.num_visualization_cases <= 0:
            return []
        if self._visualization_case_ids is None:
            selected = select_visualization_cases(
                predictions,
                self.num_visualization_cases,
                self.visualization_seed,
            )
            self._visualization_case_ids = [item.case_id for item in selected]
            return selected

        by_case_id = {item.case_id: item for item in predictions}
        selected = [
            by_case_id[case_id]
            for case_id in self._visualization_case_ids
            if case_id in by_case_id
        ]
        if selected:
            return selected
        selected = select_visualization_cases(
            predictions,
            self.num_visualization_cases,
            self.visualization_seed,
        )
        self._visualization_case_ids = [item.case_id for item in selected]
        return selected

    def _save_and_log_validation_visualization(self, epoch_number: int) -> None:
        selected = self._select_visualization_predictions(self._val_predictions)
        if not selected:
            return
        fig = make_validation_grid(selected)
        try:
            output_path = self.val_visualizations_dir / f"epoch_{epoch_number:04d}.png"
            save_validation_grid(fig, output_path)
            log_validation_grid_to_tensorboard(
                self.trainer,
                fig,
                tag="val/visualization_grid",
                epoch=epoch_number,
            )
        finally:
            close_validation_grid(fig)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._val_predictions.clear()
            return
        if not self._val_predictions:
            return
        epoch_number = int(self.current_epoch)
        self._save_and_log_validation_visualization(epoch_number)

        if self.official_eval_every_n_epochs <= 0:
            self._val_predictions.clear()
            return

        if (epoch_number + 1) % self.official_eval_every_n_epochs != 0:
            self._val_predictions.clear()
            return

        pred_dir = self.official_eval_predictions_dir / f"epoch_{epoch_number:04d}"
        metrics = run_official_evaluation(
            self._val_predictions,
            output_dir=pred_dir,
            models_dir=self.official_models_dir,
        )
        for metric_name, stats in metrics.get("aggregates", {}).items():
            for stat_name, value in stats.items():
                if value is None:
                    continue
                self.log(
                    f"val/official_{metric_name}_{stat_name}",
                    float(value),
                    prog_bar=metric_name == "mse" and stat_name == "mean",
                    sync_dist=False,
                )
                if metric_name == "mse" and stat_name == "mean":
                    self.log(
                        "val_official_mse_mean",
                        float(value),
                        prog_bar=False,
                        sync_dist=False,
                    )
        self._val_predictions.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        if self.optimizer_name.lower() != "adamw":
            raise ValueError(
                f"Unsupported optimizer={self.optimizer_name!r}. "
                "Only AdamW is currently implemented."
            )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class AdversarialTranslationModule(ResidualTranslationModule):
    """Residual generator trained with supervised loss plus conditional GAN loss."""

    def __init__(
        self,
        model: nn.Module,
        adversarial_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        config = adversarial_config or {}
        self.automatic_optimization = False
        self.adversarial_enabled = bool(config.get("enabled", True))
        self.adversarial_weight = float(config.get("weight", 0.01))
        self.adversarial_weight_ramp_epochs = int(
            config.get("weight_ramp_epochs", 0)
        )
        if self.adversarial_weight_ramp_epochs < 0:
            raise ValueError(
                "adversarial.weight_ramp_epochs must be >= 0, "
                f"got {self.adversarial_weight_ramp_epochs}."
            )
        self.discriminator_update_interval = int(
            config.get("discriminator_update_interval", 1)
        )
        if self.discriminator_update_interval < 1:
            raise ValueError(
                "adversarial.discriminator_update_interval must be >= 1, "
                f"got {self.discriminator_update_interval}."
            )
        discriminator_lr = config.get("discriminator_learning_rate", None)
        discriminator_weight_decay = config.get("discriminator_weight_decay", None)
        self.discriminator_learning_rate = (
            self.learning_rate if discriminator_lr is None else float(discriminator_lr)
        )
        self.discriminator_weight_decay = (
            self.weight_decay
            if discriminator_weight_decay is None
            else float(discriminator_weight_decay)
        )
        manual_clip_val = config.get("gradient_clip_val", None)
        self.manual_gradient_clip_val = (
            None if manual_clip_val is None else float(manual_clip_val)
        )
        self.manual_gradient_clip_algorithm = str(
            config.get("gradient_clip_algorithm", "norm")
        )
        self.discriminator = ConditionalPatchDiscriminator(
            in_channels=int(config.get("discriminator_in_channels", 2)),
            base_channels=int(config.get("discriminator_base_channels", 64)),
            num_layers=int(config.get("discriminator_layers", 3)),
        )
        self.adv_loss = nn.MSELoss()

    def _clip_manual_optimizer(self, optimizer: Any) -> None:
        clip_val = self.manual_gradient_clip_val
        if clip_val is None or float(clip_val) <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(clip_val),
            gradient_clip_algorithm=self.manual_gradient_clip_algorithm,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        state_dict = checkpoint.get("state_dict", {})
        current_state = self.state_dict()
        for key, value in current_state.items():
            if key.startswith("discriminator.") and key not in state_dict:
                state_dict[key] = value

    def _adversarial_weight_for_epoch(self, current_epoch: int) -> float:
        if (
            not self.adversarial_enabled
            or self.adversarial_weight <= 0
            or not self.loss_fn.advanced_losses_active(current_epoch)
        ):
            return 0.0
        if self.adversarial_weight_ramp_epochs == 0:
            return float(self.adversarial_weight)

        warmup_epochs = self.loss_fn.advanced_warmup_epochs()
        ramp_step = max(0, current_epoch - warmup_epochs + 1)
        ramp_factor = min(1.0, ramp_step / float(self.adversarial_weight_ramp_epochs))
        return float(self.adversarial_weight) * ramp_factor

    def _adversarial_active(self, current_epoch: int) -> bool:
        return self._adversarial_weight_for_epoch(current_epoch) > 0

    def _should_update_discriminator(self, batch_idx: int) -> bool:
        return batch_idx % self.discriminator_update_interval == 0

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        optimizer_g, optimizer_d = self.optimizers()
        batch_size = int(batch["input"].shape[0])
        adversarial_weight = self._adversarial_weight_for_epoch(
            int(self.current_epoch)
        )
        adversarial_active = adversarial_weight > 0
        update_discriminator = adversarial_active and self._should_update_discriminator(
            batch_idx
        )

        self.toggle_optimizer(optimizer_g)
        current_epoch = int(self.current_epoch)
        prediction, segmentation_logits, aux_parts = self._forward_prediction_and_aux(
            batch["input"],
            current_epoch=current_epoch,
        )
        supervised_loss, parts = self.loss_fn(
            prediction,
            batch["target"],
            batch["mask"],
            batch["valid_mask"],
            segmentation_logits=segmentation_logits,
            current_epoch=current_epoch,
        )
        parts.update(aux_parts)
        generator_adv_loss = supervised_loss.new_tensor(0.0)
        generator_adv_weight = supervised_loss.new_tensor(0.0)
        if adversarial_active:
            fake_scores_for_g = self.discriminator(batch["input"], prediction)
            generator_adv_loss = self.adv_loss(
                fake_scores_for_g,
                torch.ones_like(fake_scores_for_g),
            )
            generator_adv_weight = supervised_loss.new_tensor(adversarial_weight)
        generator_total = supervised_loss + generator_adv_weight * generator_adv_loss
        optimizer_g.zero_grad()
        self.manual_backward(generator_total)
        self._clip_manual_optimizer(optimizer_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        log_parts = {
            **parts,
            "loss_g_supervised": supervised_loss.detach(),
            "loss_g_adv": generator_adv_loss.detach(),
            "loss_g_total": generator_total.detach(),
            "loss_g_adv_weight": generator_adv_weight.detach(),
            "d_update": supervised_loss.new_tensor(float(update_discriminator)),
        }
        if update_discriminator:
            self.toggle_optimizer(optimizer_d)
            real_scores = self.discriminator(batch["input"], batch["target"])
            fake_scores = self.discriminator(batch["input"], prediction.detach())
            discriminator_real_loss = self.adv_loss(
                real_scores,
                torch.ones_like(real_scores),
            )
            discriminator_fake_loss = self.adv_loss(
                fake_scores,
                torch.zeros_like(fake_scores),
            )
            discriminator_total = 0.5 * (
                discriminator_real_loss + discriminator_fake_loss
            )
            optimizer_d.zero_grad()
            self.manual_backward(discriminator_total)
            self._clip_manual_optimizer(optimizer_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            log_parts.update(
                {
                    "loss_d_real": discriminator_real_loss.detach(),
                    "loss_d_fake": discriminator_fake_loss.detach(),
                    "loss_d_total": discriminator_total.detach(),
                    "d_real_score": real_scores.detach().mean(),
                    "d_fake_score": fake_scores.detach().mean(),
                }
            )
        self._log_loss_parts("train", generator_total.detach(), log_parts, batch_size)
        return generator_total.detach()

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        adversarial_active = self._adversarial_active(int(self.current_epoch))
        for idx, scheduler in enumerate(schedulers):
            if idx == 1 and not adversarial_active:
                continue
            scheduler.step()

    def configure_optimizers(self) -> tuple[list[Any], list[dict[str, Any]]]:
        if self.optimizer_name.lower() != "adamw":
            raise ValueError(
                f"Unsupported optimizer={self.optimizer_name!r}. "
                "Only AdamW is currently implemented."
            )
        optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            weight_decay=self.discriminator_weight_decay,
        )
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_g,
            T_max=max(1, self.max_epochs),
        )
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_d,
            T_max=max(1, self.max_epochs - self.loss_fn.advanced_warmup_epochs()),
        )
        return [optimizer_g, optimizer_d], [
            {"scheduler": scheduler_g, "interval": "epoch"},
            {"scheduler": scheduler_d, "interval": "epoch"},
        ]


def create_lightning_module(
    module_type: str,
    model: nn.Module,
    **kwargs: Any,
) -> L.LightningModule:
    normalized = module_type.lower()
    adversarial_config = kwargs.pop("adversarial_config", None)
    if normalized == "translation":
        return ResidualTranslationModule(model=model, **kwargs)
    if normalized == "adversarial_translation":
        return AdversarialTranslationModule(
            model=model,
            adversarial_config=adversarial_config,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported module_type={module_type!r}. "
        "Available: translation, adversarial_translation."
    )
