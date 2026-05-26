# Extern validation of APCONet: a 1d CNN model for estimating cardiac output based on arterial pressure waveform
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

## File preparations


---

## Project structure

Dataset must be named and shaped as follows: <br/>

Variable Name     | Data Type   | Shape of data | Description
------------------|-------------|---------------|------------
`np_w_$VERSION$`  | Numpy array | (batch, 2000) | abp wave data with 20 second segment
`np_sv_$VERSION$` | Numpy array | (batch, )     | target stroke volume data
`np_a_$VERSION$`  | Numpy array | (batch, 4)    | demographic data with order of age, sex, weight, height
`np_c_$VERSION$`  | Numpy array | (batch, )     | chart names of each data point (it is used to split validation set)

`$VERSION$` indicates the version of dataset, which for convenience of dataset management <br/>
Default value of `$VERSION$` is `200101` of `yymmdd` format. <br/>

* ```preprocessing.ipynb```
  * ```laad_vital```: Loads data from the vital monitor.
  * ```laad_hemosphere```: Loads data from the hemosphere monitor.
  * ```resample_abp```: Resamples arterial pressure wave data from 125 to 100 Hz.
  * ```koppel_abp_sv```: Links ABP segments to SV.
  * ```lowess_smoothing```: Definition of the LOWESS filter, made by van Mierlo et al..
  * ```lowess_sv```: Performs LOWESS smoothing.
  * ```filter_fysiologisch```: Detects and removes ABP values <25 or >250 mmHg and SV values <20 or >200 mL.
  * ```detect_unrealistic_segment```: Detects unrealistic segments where the ABP rises or drops >25 mmHg.
  * ```filter_ruis```: Removes the unrealistic segments.
  * ```filter_hartslag```: Detects HR values <30 or >180 beats/min.
  * ```filter_pulse_pressure```: Detects pulse pressure values
