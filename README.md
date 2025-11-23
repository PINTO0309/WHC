# WHC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17670546.svg)](https://doi.org/10.5281/zenodo.17670546) ![GitHub License](https://img.shields.io/github/license/pinto0309/WHC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/whc)

Waving Hand Classification.

https://github.com/user-attachments/assets/c6b38d56-48b7-4609-bae1-f607c21ba423

https://github.com/user-attachments/assets/7e9f8763-839f-46d2-98b1-320170f8ed10

|Variant|Size|Seq|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|:-:|
|S|1.1 MB|4|0.9821|0.31 ms|[Download]()|
|M|1.1 MB|6|0.| ms|[Download]()|
|L|1.1 MB|8|0.| ms|[Download]()|

## Data sample

|1|2|3|4|
|:-:|:-:|:-:|:-:|
|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/a5ac4472-9ab9-42ce-85f3-baa93cfb2884" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/dc9bb1c7-8757-4fe7-823f-a1be1ac3b5b7" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/1399a80d-b249-4c0b-8636-4e58a0ba4188" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/c3edcc98-a17b-4c4f-93ab-596f521bb27c" />|

## Setup

```bash
git clone https://github.com/PINTO0309/WHC.git && cd WHC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference

```bash
uv run python demo_whc.py \
-wm whc_seq_3dcnn_4x32x32.onnx \
-v 0 \
-ep cuda \
-dlr -dnm -dgm -dhm -dhd

uv run python demo_whc.py \
-wm whc_seq_3dcnn_4x32x32.onnx \
-v 0 \
-ep tensorrt \
-dlr -dnm -dgm -dhm -dhd
```

## Dataset Preparation

```bash
uv run python 01_data_prep_realdata.py
```

<img width="400" alt="class_distribution" src="https://github.com/user-attachments/assets/7eade7cc-3d2c-416d-9079-cb18c0a5d10e" />

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `whc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `whc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/whc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=32x32
uv run python -m whc train \
--data_root data/dataset.parquet \
--output_dir runs/whc_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
SIZE=32x32
VAR=s
uv run python -m whc train \
--data_root data/dataset.parquet \
--output_dir runs/whc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp

3DCNN:

```bash
SEQ=4
SIZE=32x32
uv run python -m whc train \
--data_root data/dataset.parquet \
--output_dir runs/whc_seq_3dcnn_${SEQ}x${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--seed 42 \
--device auto \
--use_amp \
--use_sequence 3dcnn \
--sequence_len ${SEQ}
```

LSTM:

```bash
SEQ=4
SIZE=32x32
uv run python -m whc train \
--data_root data/dataset.parquet \
--output_dir runs/whc_seq_lstm_${SEQ}x${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size 32x32 \
--base_channels 32 \
--seed 42 \
--device auto \
--use_amp \
--use_sequence lstm \
--sequence_len ${SEQ}
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
SIZE=32x32
uv run python -m whc train \
--data_root data/dataset.parquet \
--output_dir runs/whc_convnext_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 3x3 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `whc_epoch_*.pt`, the latest 10 `whc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/whc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 32`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/whc
  ```

### ONNX Export

```bash
uv run python -m whc exportonnx \
--checkpoint runs/whc_is_s_32x32/whc_best_epoch0049_f1_0.9939.pt \
--output whc_s_32x32.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_smiling` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="350" alt="whc_p_32x32" src="https://github.com/user-attachments/assets/b3c79843-004d-4b12-a51a-34d707242f6c" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [WHC: Happy smile classifier](https://github.com/PINTO0309/WHC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025whc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/WHC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17670546},
  url       = {https://github.com/PINTO0309/whc},
  abstract  = {Happy smile classifier.},
}
```

## Acknowledgments

- https://github.com/microsoft/FERPlus: MIT License
  ```bibtex
  @inproceedings{BarsoumICMI2016,
      title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
      author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
      booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
      year={2016}
  }
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.17625710}
  }
  ```
