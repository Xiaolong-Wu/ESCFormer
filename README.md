# ESCFormer



ESCFormer: a spatiotemporal Transformer developed for EEG-based classification of disorders of consciousness (DoC)


ESCFormer combines multi-segment EEG sequences with virtual source-space representations (V-cubes) generated using Hilbert-based stable-fastICA. The framework was developed for the classification of MCS- and VS and was additionally evaluated on the BCI Competition IV 2a motor-imagery dataset.

## Article

**ESCFormer: A Spatiotemporal Transformer Integrating EEG Sequences and Virtual Source-Space Representations for EEG-Based Classification of Disorders of Consciousness**

Publication information will be added after the article is published.

## Code Execution

Code execution is divided into two stages:

1. MATLAB processing for EEG preprocessing, virtual source-space construction, data partitioning, and result analysis.
2. Python/PyTorch processing for model training and testing.

The scripts currently use absolute Windows paths. For direct execution without modifying the paths, place `test_matlab` and `test_pycharm` directly under the `D:\` drive:

```text
D:\
├── test_matlab\
│   ├── project_EEG\
│   └── toolbox\
│       ├── eeglab\
│       ├── eegGA\
│       └── pca_ica\
└── test_pycharm\
    ├── project_LORETA\
    └── project_eegICAcloud\
```

Do not rename these directories unless the corresponding paths in the MATLAB and Python scripts are also changed.

# Stage One: MATLAB Phase

The MATLAB code is located in:

```text
D:\test_matlab\project_EEG
```

It contains separate processing directories for the DoC and BCI Competition IV 2a datasets:

```text
data_2024_DoC
data_2024_Competition
```

## 1. Data Preparation

The BCI Competition IV 2a dataset can be downloaded from:

https://www.bbci.de/competition/iv/

The scripts for importing, filtering, segmenting, and organizing the BCI data are located in:

```text
D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\raw
```

Run the `run_rawTranslateE*` scripts in numerical order according to the comments in the scripts.

The clinical DoC EEG data are not publicly distributed because they contain sensitive patient information. Access may be considered upon reasonable request and subject to institutional and ethical approval.

## 2. V-Cube Construction

After the EEG trials have been preprocessed and organized, run the source-space calculation scripts in the corresponding dataset directory:

```text
data_003_ICA(-100,700)\run_ICAspace.m
```

The scripts perform seven-segment EEG construction, Hilbert transformation, stable-fastICA decomposition, and virtual source-space mapping. The resulting EEG sequences and V-cube features are saved as MATLAB files.

The scripts call functions from the following MATLAB toolbox directories:

```text
D:\test_matlab\toolbox\eeglab
D:\test_matlab\toolbox\eegGA
D:\test_matlab\toolbox\pca_ica
```

They also use MATLAB functions associated with signal processing, constrained optimization, and GPU computation.

## 3. Export Data to Python

Set the dataset and data-split parameters at the beginning of:

```text
D:\test_matlab\project_EEG\runICA_dataResult2dataPython.m
```

Then run the script from the `project_EEG` directory. It generates the Python training and test datasets under:

```text
D:\test_pycharm\project_eegICAcloud\dataset_train
D:\test_pycharm\project_eegICAcloud\dataset_test
```

# Stage Two: Python Phase

Open the following directory as a project in PyCharm:

```text
D:\test_pycharm\project_LORETA
D:\test_pycharm\project_eegICAcloud
```

Make sure that this directory is also set as the working directory.

## 1. Python Environment

The tested environment is recorded in `environment.txt`. It can be created using:

```bash
conda create -n eegICA-gpu python=3.9
conda activate eegICA-gpu

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scipy==1.10.1
conda install numpy==1.23.5
conda install matplotlib==3.7.0
```

GPU availability can be checked using:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
```

## 2. Model Configuration

The main program is:

```text
D:\test_pycharm\project_LORETA\run_LORETA.py
D:\test_pycharm\project_LORETA\run.py
D:\test_pycharm\project_eegICAcloud\run.py
```

Before running it, configure the following variables near the beginning of the file:

- `dataNo`: selects the BCI or DoC dataset and the DoC auditory paradigm.
- `modelEpoch`: sets the number of training epochs.
- `modelBatchSize`: sets the batch size.
- `project_LORETA`:
- `modelNo[0]`: selects EEG combined with sLORETA-cube input.
- `project_eegICAcloud`:
- `modelNo[0]`: selects EEG-sequence-only input or EEG combined with V-cube input.
- `modelNo[1]`: selects the convolutional, LSTM, Transformer, or TimesBlock-based architecture.

The detailed network parameters, including convolutional dimensions, dropout, Transformer layers, and attention heads, are defined in:

```text
config.json
```

## 3. Model Training and Testing

Run:

```bash
python run.py
```

If the corresponding model checkpoint does not exist, the program trains the selected model and saves the checkpoint and loss values.

Run the same command again after training. The saved checkpoint will then be loaded automatically for test-set inference.

The outputs are stored in:

```text
dataset_model
```

The principal output files include:

```text
*.pth          trained model checkpoint
*_loss.txt     training loss
*_target.txt   ground-truth labels
*_test.txt     model outputs
```

To retrain an existing configuration, remove or rename its corresponding checkpoint and output files before running the program again.

## Result Analysis

After Python training and testing, the MATLAB script

```text
D:\test_matlab\project_EEG\runICA_dataResult4dataModel.m
```

can be used to calculate evaluation metrics and generate the loss curves, confusion matrices, and individual-level results reported in the study.

## Data Availability

The BCI Competition IV 2a dataset is publicly available from the competition website.

The clinical DoC dataset is not publicly available because of patient privacy and ethical restrictions. Requests for access are subject to approval by the corresponding institutions and ethics committee.
