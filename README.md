# STSR-INR: Spatiotemporal Super-Resolution for Multivariate Time-Varying Volumetric Data via Implicit Neural Representation

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
The volume at each time step is saved as a .raw file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis. You could download fivejet,vortex and tornado data with the link over here: https://www.dropbox.com/scl/fi/61r42t9aur5qzu6mn659l/Data.zip?rlkey=xpbo2h5m1nzfj6ai6szx8iexm&dl=0

## Training and Inference
Save all your data in ./Data dir and then modify ./dataInfo/localDataInfo.json to include the necessary information for each volume data. All the hyper-parameters settings are saved as yaml file within ./configs dir.

Then run the training and inference with:
```
python3 main.py --config_path './configs/jointTrain.yml'
```
After training and inference finished, you should be able to find the results in ../Exp dir.

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
