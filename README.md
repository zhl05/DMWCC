
# DMWCC

PyTorch implementation of *Unsupervised Domain Adaptation for Medical Image Segmentation via Dynamic Matrix and Wavelet Consistency Constraints*. DMWCC uses a DeepLabv2 backbone, dynamic matrix feature enhancement (DMFE), wavelet consistency, adversarial alignment, and MK-MMD.

## Environment and installation

The experiments were run on one NVIDIA RTX 4060 Ti GPU. Use a CUDA-enabled PyTorch installation compatible with the local NVIDIA driver.

```bash
conda create -n dmwcc python=3.8 -y
conda activate dmwcc
pip install -r requirements.txt
```

`DeepLab_resnet_pretrained_imagenet.pth` is required by every provided YAML through `TRAIN.RESTORE_FROM`. Configuration, checkpoint, data-list, and output paths may be absolute or relative to the repository root. Environment variables and `~` are expanded at runtime, so no machine-specific path needs to be hard-coded in the Python files.

## Dataset preparation

All loaders consume one 2D sample per `.npy` file. Images are normalized to `[-1, 1]`; labels are integer arrays. The loader expands each image to three channels, converts it to BGR, and subtracts the ImageNet mean. Each image-list text file and label-list text file must have the same order and number of lines.

```text
data/MMWHS17/train/image_000.npy
data/MMWHS17/train/image_001.npy
```

The released lists are selected from `datalist/MMWHS17/`, `datalist/BraTS2018/`, and `datalist/Pro12/` through the `--datalist-root` option. Relative entries inside a list are interpreted from the repository root during training and from `--data-root` during evaluation; absolute entries are also accepted.

The list-generation utilities no longer contain local path constants. Use `dataset/create_datalist.py --data-dir ... --label-dir ... --image-list ... --label-list ...` for paired `.npy` lists and `dataset/create_test_datalist.py --data-dir ... --data-list ...` for `.npz` test lists. Add `--relative-to <base-directory>` to write portable relative paths.

The source and target datasets share a common loader interface, so the target-domain label is returned with each target sample but ignored by the training routine. Only source-domain labels contribute to the supervised segmentation loss. Target-domain labels were not used for hyperparameter tuning, early stopping, or checkpoint selection, and were used only for final evaluation.

## Dataset splits and preprocessing

| Dataset | Adaptation tasks | Split used in the manuscript | Preprocessing |
| --- | --- | --- | --- |
| MMWHS17 | CT to MRI; MRI to CT | 20 unpaired CT and 20 unpaired MRI volumes; 16 volumes per modality for training and 4 for testing | SIFA v2 preprocessed data; 2D slices |
| BraTS2018 | T2 to FLAIR; FLAIR to T2 | 20 patient volumes in total; 16 patient volumes for training and 4 for the shared validation/test set; the same patient-level split is used for both modalities | central brain crop to 128 x 128 x 128, followed by axial 2D slice extraction; empty-label training slices removed; 1,746 FLAIR and 1,569 T2 training slices; 512 validation/test slices per modality |
| Pro12 | HK to BIDMC; BIDMC to HK | 12 cases per site; seeded random selection of 10 training and 2 testing cases | axial slices from NIfTI volumes, resized to 256 x 256 |

**BraTS2018 split clarification.** Dataset partitioning was performed at the patient level before 2D slice extraction. Specifically, **16 patients were used for training, while the remaining 4 patients constituted the shared validation/test set**. The same patient-level partition was applied to both T2 and FLAIR, and all slices from the same patient remained within the same subset. During preprocessing, slices with entirely empty segmentation labels were removed from both training lists. After filtering, the training lists contained **1,746 FLAIR slices and 1,569 T2 slices**, while the 4 validation/test patient volumes provided **512 slices per modality**. The different training slice counts resulted from empty-label removal and did not affect the patient-level partition.

### Fixed data lists

The fixed data lists used in the experiments are provided in `datalist/Pro12/`, `datalist/BraTS2018/`, and `datalist/MMWHS17/`. All entries use paths relative to the repository root. Training image and label lists are kept in separate files and retain their original line order.

For Pro12, the released files record the fixed 10/2 patient-level split. The HK training list contains `hk_0` through `hk_9`, with `Case48` and `Case49` in the test list. The BIDMC training list contains `bidmc_0` through `bidmc_9`, with `Case11` and `Case12` in the test list.

The MMWHS lists currently in `datalist/MMWHS17/` contain 2,304 training slices, 576 validation slices, and 4 test-volume entries per modality. Keep splits at the volume level to prevent slices from one subject appearing in both training and testing sets.

## Training

Run commands from the repository root. On Windows, select the GPU with `set CUDA_VISIBLE_DEVICES=0`; on Linux/macOS, prepend `CUDA_VISIBLE_DEVICES=0`.

```bash
python train.py --dataset mmwhs --cfg configs/ours_CT2MR.yml --datalist-root datalist
python train.py --dataset mmwhs --cfg configs/ours_MR2CT.yml --datalist-root datalist
python train.py --dataset brats --cfg configs/brats/ours_t22flair.yml --datalist-root datalist
python train.py --dataset brats --cfg configs/brats/ours_flair2t2.yml --datalist-root datalist
python train.py --dataset pro --cfg configs/pro/HK2BIDMC.yml --datalist-root datalist
python train.py --dataset pro --cfg configs/pro/BIDMC2HK.yml --datalist-root datalist
```

`--cfg` and `--datalist-root` accept either repository-relative or absolute paths. The pretrained model and output directories specified in the YAML are resolved in the same way.

The default configuration specifies batch size 4, 50,000 iterations, Adam optimization with learning rate `3e-4`, MK-MMD `kernel_num=5` and `kernel_mul=2.0`, Haar wavelets with 3 levels and enhancement factor 1.5, and adversarial/MK-MMD/consistency weights of 0.003/0.05/0.01, respectively. Run artifacts are saved under `experiments/snapshots/<source>2<target>/<experiment_name>/`.

## Evaluation

Pass the same YAML used for training so the model architecture and class count match the checkpoint.

```bash
python test.py --cfg configs/ours_CT2MR.yml --dataset mmwhs --target_modality MR --num_class 5 --pretrained_model_pth experiments/snapshots/CT2MR/dmwcc_feature-dmfe_cons-wavelet/model_50000.pth --data-root .
python test.py --cfg configs/brats/ours_t22flair.yml --dataset brats --target_modality flair --num_class 2 --pretrained_model_pth checkpoints/brats.pth --data-root .
python test.py --cfg configs/pro/HK2BIDMC.yml --dataset pro --target_modality bidmc --num_class 2 --pretrained_model_pth checkpoints/pro.pth --data-root .
```

The evaluator selects the released test list from the dataset and modality. Use `--test-list <path>` to override it. `--data-root` sets the base directory for relative entries inside that file.

The evaluator reports mean Dice, Dice standard deviation, mean ASSD, and ASSD standard deviation. It also saves prediction images. Checkpoint filenames depend on `TRAIN.SAVE_PRED_EVERY`; use the checkpoint that exists in the snapshot directory rather than assuming `model_50000.pth`.

## Configuration and random seed

All experiment parameters are YAML files under `configs/`. `domain_adaptation/config.py` provides defaults; the YAML overrides these defaults. The key experiment fields are `SOURCE`, `TARGET`, `NUM_WORKERS`, `TRAIN.RESTORE_FROM`, `TRAIN.BATCH_SIZE`, `TRAIN.MAX_ITERS`, `TRAIN.feature_enhance_type`, `TRAIN.consistency_aug_type`, and loss weights.

The default random seed is `1234` (`TRAIN.RANDOM_SEED`). Training seeds Python `random`, NumPy, PyTorch CPU, and all CUDA devices; dataloader worker seeds are derived as `1234 + worker_id`. CUDA deterministic mode is not enabled in the current implementation, so minor run-to-run variation may remain. To require deterministic CUDA kernels, set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` before constructing the model, noting the potential speed reduction.
