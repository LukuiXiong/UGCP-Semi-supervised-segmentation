# UGCP-Semi-supervised-segmentation

## Requirements
Please see requirements.txt
```
pip install -r requirements.txt
```
## Usage
### 1. Data Preparation
Please download the preprocessed dataset provided by https://github.com/HiLab-git/SSL4MIS/ first and put them into the diretory "/data/ACDC/".

   Then, please download the raw data of the ACDC dataset and put related testing cases into the diretory "/data/ACDC_raw". The testing cases and corresponding naming conventions should follow the "test.list" in the diretory "/data/ACDC/" downloaded from https://github.com/HiLab-git/SSL4MIS/. The raw data are used to calculate the 95HD and ASD metrics in mm.
