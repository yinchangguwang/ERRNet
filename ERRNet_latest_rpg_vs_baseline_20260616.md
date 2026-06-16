# ERRNet Latest RPG Training vs Baseline

Analysis time: 2026-06-16 09:41:43 +08:00

## Checked Remote Artifacts

- Remote workspace: `/mnt/workspace/dip26/zhengjunyin`
- Latest run: `errnet_rpg_real20_ft80_nogan`
- Latest log: `/mnt/workspace/dip26/zhengjunyin/logs/train_rpg_real20_ft80_nogan_20260615_190037.log`
- PID file: `/mnt/workspace/dip26/zhengjunyin/logs/train_rpg_real20_ft80_nogan.pid`
- Latest checkpoint: `/mnt/workspace/dip26/zhengjunyin/checkpoints/errnet_rpg_real20_ft80_nogan/errnet_latest.pt`
- Best real20 checkpoint: `/mnt/workspace/dip26/zhengjunyin/checkpoints/errnet_rpg_real20_ft80_nogan/errnet_best_PSNR_testdata_real20.pt`
- Baseline: README pretrained checkpoint `checkpoints/errnet/errnet_060_00463920.pt`

## Training Status

The latest training task has ended.

- PID file value: `1657457`
- `ps -p 1657457` found no running process.
- No active `python ... train_errnet.py` process was found for the user.
- Latest log modification time: `2026-06-16 00:27:48 +0000`
- Latest checkpoint modification time: `2026-06-16 00:27:37 +0000`
- Training option `nEpochs: 35`; the final validation block is epoch 34, which matches the loop condition `epoch < nEpochs`.
- Periodic checkpoint saves were logged at epochs 10, 20, and 30. The final state is represented by `errnet_latest.pt`.
- No `Traceback`, `Error`, `Aborted`, `Killed`, exception, or CUDA OOM marker was found in the parsed log.

## Latest Run Metrics

The training script automatically evaluated CEILNet Table2, inferred from 100 validation samples, and real20, inferred from 20 validation samples.

| Epoch | Dataset | LMSE | NCC | PSNR | SSIM |
| ---: | --- | ---: | ---: | ---: | ---: |
| 34 | CEILNet Table2 | 0.0045 | 0.9801 | 28.0048 | 0.9413 |
| 34 | real20 | 0.0210 | 0.8810 | 22.9969 | 0.8116 |

The best real20 checkpoint was written earlier in the run:

- `errnet_best_PSNR_testdata_real20.pt`
- Modification time: `2026-06-15 19:36:32 +0000`

Its corresponding logged validation point is epoch 4:

| Epoch | Dataset | LMSE | NCC | PSNR | SSIM |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | CEILNet Table2 | 0.0039 | 0.9843 | 29.1605 | 0.9489 |
| 4 | real20 | 0.0208 | 0.8882 | 23.2546 | 0.8157 |

## Baseline Comparison

Higher is better for PSNR, SSIM, and NCC. Lower is better for LMSE.

### Latest Checkpoint vs Baseline

| Dataset | Metric | Latest Run | README Baseline | Difference |
| --- | --- | ---: | ---: | ---: |
| CEILNet Table2 | PSNR | 28.0048 | 27.8800 | +0.1248 |
| CEILNet Table2 | SSIM | 0.9413 | 0.9407 | +0.0006 |
| CEILNet Table2 | NCC | 0.9801 | 0.9808 | -0.0007 |
| CEILNet Table2 | LMSE | 0.0045 | 0.0048 | -0.0003 |
| real20 | PSNR | 22.9969 | 23.5500 | -0.5531 |
| real20 | SSIM | 0.8116 | 0.8285 | -0.0169 |
| real20 | NCC | 0.8810 | 0.8877 | -0.0067 |
| real20 | LMSE | 0.0210 | 0.0201 | +0.0009 |

### Best real20 Checkpoint vs Baseline

| Dataset | Metric | Best real20 Checkpoint | README Baseline | Difference |
| --- | --- | ---: | ---: | ---: |
| real20 | PSNR | 23.2546 | 23.5500 | -0.2954 |
| real20 | SSIM | 0.8157 | 0.8285 | -0.0128 |
| real20 | NCC | 0.8882 | 0.8877 | +0.0005 |
| real20 | LMSE | 0.0208 | 0.0201 | +0.0007 |

## Validation Trend

| Epoch | CEILNet PSNR | CEILNet SSIM | real20 PSNR | real20 SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 29.1605 | 0.9489 | 23.2546 | 0.8157 |
| 9 | 27.2147 | 0.9343 | 22.7026 | 0.8162 |
| 14 | 28.6305 | 0.9450 | 23.1231 | 0.8121 |
| 19 | 28.7474 | 0.9460 | 22.9694 | 0.8097 |
| 24 | 28.8696 | 0.9467 | 23.0264 | 0.8098 |
| 29 | 28.9399 | 0.9471 | 22.9911 | 0.8098 |
| 34 | 28.0048 | 0.9413 | 22.9969 | 0.8116 |

## Conclusion

The latest RPG finetune completed without a visible crash and produced a fresh `errnet_latest.pt`.

Compared with the README baseline, the final checkpoint is slightly better on CEILNet Table2 PSNR, SSIM, and LMSE, but slightly worse on CEILNet NCC. On real20 it remains below the baseline on PSNR, SSIM, and LMSE; NCC only exceeds the baseline at the best-real20 checkpoint.

The strongest result from this run is the early best-real20 checkpoint at epoch 4, but even that does not surpass the README baseline on real20 PSNR or SSIM. It should be reported as a completed finetune that improves or preserves parts of the synthetic CEILNet behavior, while still not beating the provided pretrained baseline on the main real20 PSNR/SSIM metrics.
