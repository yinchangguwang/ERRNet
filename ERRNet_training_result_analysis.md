# ERRNet Training Result Analysis

Analysis time: 2026-06-09

## Artifacts

- Result archive: `errnet_final_results_20260609.tar.gz`
- Extracted analysis directory: `analysis_errnet_final/`
- Final checkpoints:
  - `checkpoints/errnet/errnet_latest.pt`
  - `checkpoints/errnet/errnet_060_00514920.pt`
- Training log:
  - `logs/train_gpu_20260608_060114.log`

## Training Status

The training completed successfully.

- Epochs completed: 60
- Total iterations: 514920
- Steps per epoch: 8582
- Final checkpoint time: 2026-06-09 00:42 UTC
- No `Traceback`, `Aborted`, or `Error` was found in the training log.

## Training Loss Trend

| Stage | IPixel | VGG | D | G |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 | 0.0150 | 3.0265 | - | - |
| Epoch 20 | 0.0090 | 1.8398 | 0.6915 | 0.6956 |
| Epoch 40 | 0.0077 | 1.4814 | 0.5506 | 1.1285 |
| Epoch 59/60 before final eval | 0.0083 | 1.3327 | 0.5499 | 1.2200 |

The loss trend is healthy. Pixel loss and VGG perceptual loss both decreased clearly. GAN loss was enabled after epoch 20, and the discriminator/generator losses stayed stable rather than diverging.

## Final Validation Metrics

The training script automatically evaluated `CEILNet Table2` and `real20`.

| Dataset | LMSE | NCC | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| CEILNet Table2 | 0.0048 | 0.9701 | 26.5898 | 0.9336 |
| real20 | 0.0211 | 0.8557 | 22.7185 | 0.8141 |

## Baseline Comparison

README baseline is the provided pretrained checkpoint `checkpoints/errnet/errnet_060_00463920.pt`.

| Dataset | Metric | This Run | README Baseline | Difference |
| --- | --- | ---: | ---: | ---: |
| CEILNet Table2 | PSNR | 26.5898 | 27.88 | -1.2902 |
| CEILNet Table2 | SSIM | 0.9336 | 0.9407 | -0.0071 |
| CEILNet Table2 | NCC | 0.9701 | 0.9808 | -0.0107 |
| CEILNet Table2 | LMSE | 0.0048 | 0.0048 | 0.0000 |
| real20 | PSNR | 22.7185 | 23.55 | -0.8315 |
| real20 | SSIM | 0.8141 | 0.8285 | -0.0144 |
| real20 | NCC | 0.8557 | 0.8877 | -0.0320 |
| real20 | LMSE | 0.0211 | 0.0201 | +0.0010 |

## Conclusion

The run is valid and completed normally. The model converged and produced usable final checkpoints.

However, the final metrics are slightly below the README pretrained baseline:

- CEILNet Table2 PSNR is lower by about 1.29 dB.
- real20 PSNR is lower by about 0.83 dB.
- SSIM and NCC are also slightly lower on both evaluated datasets.

This is not a failed training run. It is a normal reproduction gap. Likely contributors include data filtering differences, random augmentation, PPU backend numerical behavior, batch size 1 training variance, and incomplete knowledge of the exact official pretrained checkpoint training procedure.

The final checkpoint is usable, but it should not be described as outperforming the official baseline. If the goal is best evaluation score, compare against the provided pretrained checkpoint and consider keeping the best validation checkpoint rather than only the final epoch checkpoint.
