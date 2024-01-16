## Prestained Ionization weights
Latent and Log folder contains the pre-trained latent table and model weight to replicate the ionization experiment of STSR-INR result in Figure 6.

(1) If you did not inference, go to root dir, inference with pre-trained weights:
```
python3 main.py --config_path './configs/ionization_inf.yml'
```
After inference, you should be able to find the STSR-INR reconstruction result in /Exp/ionization/vionization/Results folder. 

(2) Download Paraview from ```https://www.paraview.org/download/```, we used Paraview 5.11.1 Linux version to render the result, but Windows or other versions should also work.

(3) Open Paraview, at top tab bar ```File -> Load State...```, select our State File ```state.pvsm```. In the ```Load State Options```, select ```Choose File Names``` from ```Load State Data File Options```. Choose the file names to ./Exp/ionization/vionization/Results/ionization_GT/STSR-INR-ionization_GT-00085.raw. Click OK to render, and you should be able to get STSR-INR iso-surface rendering result in Figure 6.