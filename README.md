# External validation of APCONet: a 1d CNN model for estimating cardiac output based on arterial pressure waveform
APCONet is an open-source machine learning model developed by Yang et al., trained on data from surgical patients at Seoul National University Hospital (SNUH). It estimates stroke volume (SV) from 100 Hz, 20-second arterial pressure waveform segments combined with demographic data (age, height, weight, and sex). SV is used to calculate cardiac output (CO) by multiplying it with heartrate. This model was further improved by Van Mierlo et al. However, as the final improved model performed insufficiently on our data, an earlier version closer to the original model by Yang et al. was used here for external validation on patients from Medisch Spectrum Twente (MST). After external validation, the model was fine-tuned on the MST dataset to assess whether fine-tuning improved performance.

> Author: [Hyun-Lim Yang](https://sites.google.com/view/hyunlim-yang) 
([VitalLab](https://vitallab.ai/), SNUH, South Korea) <br/>

> Credits: <br/>
> * Hyung-Chul Lee ([VitalLab](https://vitaldb.net/), SNUH, South Korea)
> * Chul-Woo Jung ([VitalLab](https://vitaldb.net/), SNUH, South Korea)
> * [Min-Soo Kim](http://infolab.kaist.ac.kr/members/Min-Soo%20Kim/) 
([InfoLab](http://infolab.kaist.ac.kr/), KAIST, South Korea)

> Code adjustments made by:
> * [Roy van Mierlo](https://research.tue.nl/en/persons/roy-van-mierlo/) (TU/e, the Netherlands)

> External validation done by:
> * Lucas Halman (UT, [l.m.halman@student.utwente.nl](mailto:l.m.halman@student.utwente.nl))
> * Nienke Rietdijk (UT, [n.e.rietdijk@student.utwente.nl](mailto:n.e.rietdijk@student.utwente.nl))
> * Ruben Tielen (UT, [r.g.tielen@student.utwente.nl](mailto:r.g.tielen@student.utwente.nl))
> * Irma van der Werf (UT, [i.vanderwerf@student.utwente.nl](mailto:i.vanderwerf@student.utwente.nl))

---

## Citation
If you use this code in your research, cite both the paper by Yang et al. and by Van Mierlo et al.

### Original APCONet paper

Yang, H. L., Jung, C. W., Yang, S. M., Kim, M. S., Shim, S., Lee, K. H., & Lee, H. C. (2021).
<a href="https://medinform.jmir.org/2021/8/e24762/">
Development and validation of an arterial pressure-based cardiac output algorithm using a convolutional neural network: Retrospective study based on prospective registry data.
</a>
<i>JMIR Medical Informatics</i>, 9(8), e24762.

```
@article{yang2021apconet,
  title={Development and validation of an arterial pressure-based cardiac output algorithm using a convolutional neural network},
  author={Yang, Hyun-Lim and Jung, Chul-Woo and Yang, Seung-Mok and Kim, Min-Soo and Shim, Sung and Lee, Kwang-Ho and Lee, Hyo-Chang},
  journal={JMIR Medical Informatics},
  volume={9},
  number={8},
  pages={e24762},
  year={2021}
}
```
### Improved model
Van Mierlo, R. R., Bouwman, R. A., & Van Riel, N. A. (2024).  
<a href="https://ieeexplore.ieee.org/abstract/document/10596819">
Reproducing and improving one-dimensional convolutional neural networks for arterial blood pressure-based cardiac output estimation.
</a>  In <i>Proceedings of the IEEE International Symposium on Medical Measurements and Applications (MeMeA)</i> (pp. 1–6).

```
@inproceedings{vanmierlo2024cnn,
  title={Reproducing and improving one-dimensional convolutional neural networks for arterial blood pressure-based cardiac output estimation},
  author={Van Mierlo, R. R. and Bouwman, R. A. and Van Riel, N. A.},
  booktitle={IEEE International Symposium on Medical Measurements and Applications (MeMeA)},
  year={2024},
  pages={1--6}
}
```

---

## Dataset
A pseudonymized dataset from MST containing 94 included surgical patients under general anesthesia was used. Two types of monitor data sources were used throughout the code: vital monitor data and Hemosphere monitor data. These datasets differed in structure and duration. The vital data contains arterial blood pressure (ABP) data and the Hemosphere data contained the SV reference measurements. The dataset also contained demographic information: age, sex, height and weight. Demographic characteristics, SV and ABP data were stored in separate CSV files for further processing.

The preprocessing steps and data loading were specifically adapted to the monitoring systems and data formats used at MST (MARTINI and Philips IntelliVue), including specific file paths and time handling. Because of this, adjustments may be required before using the code with data from other hospitals or monitoring systems. 

---

## Requirements
Python 3.11.8. was used. 

```
numpy
pandas
matplotlib
scipy
statsmodels
pyvital
tqdm
dotenv
os
```

Install all dependencies with:
 
```bash
pip install -r requirements.txt
```
---

## Project structure

### Folder structure

The code requires a demographics, vital data and Hemosphere data folder. The script automatically pairs them based on the patient ID. Data is loaded automatically based on the following folder structure:

```
project/
├── demographics.csv
│
├── vital data/
│   ├── patient01_vital/
│   │   └── patient01_vital.csv
│   ├── patient02_vital/
│   │   └── patient02_vital.csv
│   ├── patient03_vital/
│   │   └── patient03_vital.csv
│   ├── ...
│   └── patient94_vital/
│       └── patient94_vital.csv
│
├── hemosphere data/
│   ├── patient01_hs/
│   │   └── patient01_hs.csv
│   ├── patient02_hs/
│   │   └── patient02_hs.csv
│   ├── patient03_hs/
│   │   └── patient03_hs.csv
│   ├── ...
│   └── patient94_hs/
│       └── patient94_hs.csv
```

### Data arrays

Processed `.npy` files are stored in a separate directory for each patient, where the folder name corresponds to the patient number. The dataset must be named and shaped as follows:
 
| Variable Name | Data Type | Shape | Description |
|---|---|---|---|
| `np_w_$VERSION$` | Numpy array | `(batch, 2000)` | ABP waveform data, 20-second segments |
| `np_sv_$VERSION$` | Numpy array | `(batch,)` | Target stroke volume |
| `np_a_$VERSION$` | Numpy array | `(batch, 4)` | Demographic data: age, sex, weight, height |
| `np_c_$VERSION$` | Numpy array | `(batch,)` | Patient ID |
 
`$VERSION$` indicates the dataset version in `yymmdd` format. Default: `200101`.

---
## Configuration
 
Settings are loaded from a `vars.env` file in the project directory. Create this file based on the structure below: 
 
```env
# Paths
DATA_PATH=path/to/data
DEMOGRAPHICS=path/to/demographic/data
STAT_PATH=?
 
# Signal
SAMPLE_LENGTH=2000
SAMPLING_RATE=0.01
 
# Filtering
FILTER_HR=True
FILTER_PP=True
 
# Visualisation
PLOT_LINKING=False
PLOT_HR=False
PLOT_PP=False
PLOT_FINAL=False

# Save as npy files
NPY_SAVE=True

```
 
| Variable | Description |
|---|---|
| `DATA_PATH` | Path to the directory containing vital and Hemosphere data |
| `DEMOGRAPHICS` | Path to the directory containing folders with demographic data |
| `STAT_PATH` | ? |
| `SAMPLE_LENGTH` | ABP segment length in samples (2000 = 20 s at 100 Hz) |
| `SAMPLING_RATE` | Sampling interval in seconds (0.01 s = 100 Hz) |
| `FILTER_*` | Enable or disable individual filter steps |
| `PLOT_*` | Enable or disable visualisations per step |
| `NPY_SAVE` | Enable or disable saving .npy files  |

---

## Preprocessing

The preprocessing steps are based on the filtering strategy introduced by Van Mierlo et al. and can be found in `preprocessing.ipynb`. The variables are explained below.

* ```load_vital```: Reads data from the vital monitor.
* ```load_hemosphere```: Reads data from the hemosphere monitor.
* ```resample_abp```: Extracts the ABP signal and resamples it to 100 Hz using pyvital.
* ```link_abp_sv```: Matches each SV measurement to the 20-second ABP segment that preceded it.
* ```lowess_smoothing```: LOWESS smoothing filter implementation adapted from Van Mierlo et al.
* ```lowess_sv```: Applies the LOWESS smoothing to the SV signal.
* ```filter_begin```: Deletes first 10 minutes of the recording due to signal stabilization of the SV measurements.
* ```filter_abp_dip```: Removes segments with NIBP measurements that disturb the ABP waveform. 
* ```filter_physiological```: Removes segments where ABP values are <25 or >250 mmHg or where SV values are <20 or >200 mL.
* ```filter_heartrate```: Removes segments where the HR is <30 or >180 bpm.
* ```filter_pulse_pressure```: Removes segments where the mean pulse pressure is <20 mmHg.
* ```delete_segments```: Applies the collected removal mask to all arrays.
* ```save_data```: Saves the data to .npy files. 
* ```process_patient```: Runs the full pipeline for one patient.

---

## Statistical analysis

The statistical analysis of patient demographics is performed in `statistical_analysis.ipynb`. The notebook includes the following steps:

* **Normality testing**: Visual assessment through Q-Q plots, histograms, and box plots to assess the distribution of demographic variables (age, height, weight).
* **Descriptive statistics**: Summary table of demographic characteristics for the full dataset and per subset (train, validation, test).
* **Mann-Whitney U test**: Used to compare continuous variables (age, height, weight) between subsets.
* **Pearson chi-squared test**: Used to compare categorical variables (sex) between subsets.

Descriptive statistics for age, height, and weight were determined based on normality assessment. Normally distributed variables are presented as mean ± standard deviation (SD), while non-normally distributed variables are presented as median and interquartile range (IQR). Note that when applying this pipeline to a different dataset, the appropriate descriptive statistic should be reassessed based on the normality of that dataset.

---

## APCONet model
The model created by Yang et al. was used for external validation. This model was created with the code from Van Mierlo and can be downloaded via the [APCONet repository](https://github.com/Computational-Biology-TUe/APCONet). After external validation, the model was further finetuned on the MST dataset. The finetuned model can be found in this repository under `AF-07`.

The model performance is assessed using the following measures:

* **ME and MAE**: Mean error and mean absolute error (mL) to assess bias and accuracy of SV estimates.
* **MPE and MAPE**: Mean percentage error and mean absolute percentage error (%) to assess relative bias and accuracy.
* **Pearson and Spearman correlation**: Correlation between APCONet SV estimates and FloTrac reference measurements.
* **Bland-Altman analysis**: Agreement between APCONet and FloTrac over the full range of measurements, reported as bias and limits of agreement (±SD).
* **Four-quadrant plot**: Trend analysis based on concordance rate of percentage changes in SV between APCONet and FloTrac. A concordance rate of >90% indicates reliable trending ability.

---

## Usage
1. Set up the `vars.env` configuration file (see [Configuration](#configuration)).
2. Place the patient data in the correct folder structure (see [Project structure](#project-structure)).
4. Run the preprocessing pipeline in `preprocessing.ipynb` (see [Preprocessing](#preprocessing)).
5. Check the normality via `statistical_analysis.ipynb` (see [Statistical analysis](#statistical-analysis)).
6. Run the APCONet model (see [APCONet model](#apconet-model)). 
 
> **Note:** Adjust file paths, descriptive statistics and time handling when using data from a different hospital or monitoring system.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
