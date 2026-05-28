# External validation of APCONet: a 1d CNN model for estimating cardiac output based on arterial pressure waveform
APCONet is an open-source machine learning model developed by Yang et al., trained on data from surgical patients at Seoul National University Hospital (SNUH). It estimates stroke volume (SV) from 100 Hz, 20-second arterial pressure waveform segments combined with demographic data (age, height, weight, and sex). SV is used to calculate cardiac output by multiplying it with heartrate. This model was further improved by van Mierlo et al. and is used here for external validation on patients from Medisch Spectrum Twente (MST).

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
If you use this code in your research, cite both the paper by Yang et al. and by van Mierlo et al..

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

A pseudonymized dataset from MST containing 140 surgical patients under general anesthesia was used. It included demographic information and vital signs, including arterial pressure waveforms. Reference SV measurements were obtained using FloTrac (version 2.3 and higher).

Two different monitor data sources are used throughout the code: vital monitor data and Hemosphere monitor data. These datasets differ in structure and duration. The vital data contains ABP and the Hemosphere data contains stroke volume The Hemosphere data generally spans a longer time period than the vital data. 

The preprocessing steps and data loading are specifically adapted to the monitoring systems and data formats used at MST, including MARTINI and Philips IntelliVue. As a result, parts of the code, such as file paths and time handling, are specific to the MST setup. Because of this, adjustments may be required before using the code with data from other hospitals or monitoring systems. 

## Requirements

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

---

## Project structure

### Folder structure

Each patient requires a `_vital` and a `_hs` folder. The script automatically pairs them based on the patient ID. Data is loaded automatically based on the following folder structure:

```
DATA_PATH/
├── demographics.csv
│
├── patient01_vital/
│   └── vital.csv
├── patient01_hs/
│   └── hemosphere.csv
├── patient02_vital/
│   └── vital.csv
├── patient02_hs/
│   └── hemosphere.csv
└── ...
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

## Preprocessing

* ```load_vital```: Reads data from the vital monitor.
* ```load_hemosphere```: Reads data from the hemosphere monitor.
* ```resample_abp```: Extracts the ABP signal and resamples it from 125 to 100 Hz using pyvital.
* ```link_abp_sv```: Matches each SV measurement to the 20-second ABP segment that preceeded it.
* ```lowess_smoothing```: LOWESS smoothing filter implementation adapted from van Mierlo et al.
* ```lowess_sv```: Applies the LOWESS smoothing to the SV signal.
* ```filter_physiological```: Removes segments where ABP values are <25 or >250 mmHg or where SV values are <20 or >200 mL.
* ```detect_unrealistic_segment```: Detects unrealistic segments with unrealistically large jumps from >25 mmHg.
* ```filter_noise```: Removes segments with unrealistically large jumps (>25 mmHg).
* ```filter_extra_peaks```: Removes segments that contain more peaks than expected for the detected number of heartbeats. Uses a peak prominence threshold of 2 mmHg and a heartbeat factor threshold of 1.2.
* ```filter_heartrate```: Removes segments where the HR is <30 or >180 bpm.
* ```filter_pulse_pressure```: Removes segments where the mean pulse pressure is <20 mmHg
* ```delete_segments```: Applies the collected removal mask to all arrays.
* ```save_data```: Saves the data to .npy files. 
* ```process_patient```: Runs the full pipeline for one patient.

In preprocessing, the filtering strategy introduced by van Mierlo et al. is used and extended with additional filtering steps developed for the MST external validation. 

---
 
## Configuration
 
Settings are loaded from a `vars.env` file in the project directory. Create this file based on the example below:
 
```env
# Paths
DATA_PATH=path/to/data
 
# Signal
SAMPLE_LENGTH=2000
SAMPLING_RATE=0.01
 
# Filtering
FILTER_NOISE=True
FILTER_EXTRA_PEAKS=True
FILTER_HR=True
FILTER_PP=True
 
# Visualisation
PLOT_LINKING=False
PLOT_HR=False
PLOT_PP=False
PLOT_EXTRA_PEAKS=False
PLOT_FINAL=False

# Save as npy files
NPY_SAVE=True

```
 
| Variable | Description |
|---|---|
| `DATA_PATH` | Path to the directory containing patient folders |
| `SAMPLE_LENGTH` | ABP segment length in samples (2000 = 20 s at 100 Hz) |
| `SAMPLING_RATE` | Sampling rate in Hz |
| `FILTER_*` | Enable or disable individual filter steps |
| `PLOT_*` | Enable or disable visualisations per step |
| `NPY_SAVE` | Enable or disable saving .npy files  |

---
