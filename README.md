# STSR-INR: Spatiotemporal Super-Resolution for Multivariate Time-Varying Volumetric Data via Implicit Neural Representation
![alt text](https://github.com/TouKaienn/STSR-INR/blob/main/assets/model.png)
## Description
This is the Pytorch implementation for STSR-INR: Spatiotemporal Super-Resolution for Multivariate Time-Varying Volumetric
Data via Implicit Neural Representation

## Installation
```
git clone https://github.com/TouKaienn/STSR-INR.git
conda create --name STSRINR python=3.9
conda activate STSRINR
pip install -r requirements.txt
```

## Data Format
The volume at each time step is saved as a .raw file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis. You could download our joint training and ionization data with the link over here: https://drive.google.com/file/d/1Tw1DYlQNLYzpHR51TYesrP40BWUsQezL/view?usp=drive_link

Unzip the downloaded file and put the data into the root dir, you could get a similar file structure like this:
```
.
├── configs
├── Data
│   ├── five_jets_norm
│   ├── GT
│   ├── H+
│   ├── H2
│   ├── PD
│   ├── tornado_norm
│   └── vorts_norm
├── dataInfo
├── dataio.py
├── Exp
├── latentInterpolation.py
├── LICENSE
├── logger
├── main.py
├── model.py
├── pretrainedIonization
├── README.md
├── requirements.txt
├── train.py
└── utils.py
```


## Training and Inference
After Saving all your data in ./Data dir and then ensure ./dataInfo/localDataInfo.json includes all the necessary information for each volume data. Use the yaml file which contains all the hyper-parameters settings within ./configs dir to train or inference.

We provide pre-trained model weight of ionization experiment, you could load and infer with:
```
python3 main.py --config_path './configs/ionization_inf.yml'
```
To train from scratch:
```
python3 main.py --config_path './configs/ionization_train.yml'
```

After training or inference finished, you should be able to find the results in ./Exp dir.

## Citation
```
@article{tang2024stsr,
  title={STSR-INR: Spatiotemporal Super-Resolution for Multivariate Time-Varying Volumetric Data via Implicit Neural Representation},
  author={Tang, Kaiyuan and Wang, Chaoli},
  journal={Computers \& Graphics},
  year={2024},
  note={Accepted}
}
```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1955395, IIS-2101696, OAC-2104158, and the U.S. Department of Energy through grant DE-SC0023145.
