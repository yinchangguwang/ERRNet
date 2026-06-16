# ERRNet Latest Training vs Baseline

Analysis time: 2026-06-15

## Compared Artifacts

- Latest remote run: `errnet_joint_balanced_late_gan`
- Remote log: `/mnt/workspace/dip26/zhengjunyin/logs/train_joint_balanced_late_gan_20260614_111206.log`
- Latest checkpoint: `/mnt/workspace/dip26/zhengjunyin/checkpoints/errnet_joint_balanced_late_gan/errnet_latest.pt`
- Joint-best checkpoint: `/mnt/workspace/dip26/zhengjunyin/checkpoints/errnet_joint_balanced_late_gan/errnet_best_joint_PSNR.pt`
- Baseline: `README_DIP26.md` pretrained checkpoint `checkpoints/errnet/errnet_060_00463920.pt`

## Latest Run Status

The latest run stopped after writing the epoch-70 checkpoint. No active training process was running during the status check.

The training log's final automatic validation block corresponds to epoch 69, just before the epoch-70 checkpoint save.

## Metric Comparison

Higher is better for PSNR, SSIM, and NCC. Lower is better for LMSE.

| Dataset | Metric | Latest Run | README Baseline | Difference |
| --- | --- | ---: | ---: | ---: |
| CEILNet Table2 | PSNR | 29.0374 | 27.8800 | +1.1574 |
| CEILNet Table2 | SSIM | 0.9483 | 0.9407 | +0.0076 |
| CEILNet Table2 | NCC | 0.9841 | 0.9808 | +0.0033 |
| CEILNet Table2 | LMSE | 0.0040 | 0.0048 | -0.0008 |
| real20 | PSNR | 23.1027 | 23.5500 | -0.4473 |
| real20 | SSIM | 0.8144 | 0.8285 | -0.0141 |
| real20 | NCC | 0.8840 | 0.8877 | -0.0037 |
| real20 | LMSE | 0.0213 | 0.0201 | +0.0012 |

## Validation Trend

| Epoch | CEILNet PSNR | CEILNet SSIM | real20 PSNR | real20 SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 29 | 26.7892 | 0.9301 | 22.8574 | 0.8150 |
| 34 | 27.9276 | 0.9384 | 23.4833 | 0.8204 |
| 39 | 28.1800 | 0.9411 | 23.2836 | 0.8148 |
| 44 | 28.8057 | 0.9465 | 23.3703 | 0.8172 |
| 49 | 28.8466 | 0.9468 | 23.2434 | 0.8159 |
| 54 | 28.8820 | 0.9464 | 23.1747 | 0.8143 |
| 59 | 28.9241 | 0.9473 | 23.1957 | 0.8149 |
| 64 | 28.8676 | 0.9469 | 23.1104 | 0.8136 |
| 69 | 29.0374 | 0.9483 | 23.1027 | 0.8144 |

## Conclusion

The latest joint-balanced run clearly beats the README baseline on CEILNet Table2 across all four tracked metrics. The gain is large on PSNR, about +1.16 dB.

On real20, the latest run is still below the README baseline: PSNR is lower by about 0.45 dB, SSIM and NCC are slightly lower, and LMSE is slightly worse.

Compared with the previous completed local report run, the new training strategy succeeded at fixing the CEILNet gap and nearly matching the real20 baseline, but it did not surpass the baseline on real20. The best practical claim is:

> The latest joint-balanced training improves synthetic CEILNet Table2 performance beyond the provided pretrained baseline while keeping real20 close to, but still slightly below, the baseline.

The current comparison only covers CEILNet Table2 and real20 because those are the datasets automatically evaluated in the training log. To compare against the full README baseline table, run `test_errnet.py` on `objects`, `postcard`, and `wild` using `errnet_latest.pt` or `errnet_best_joint_PSNR.pt`.
