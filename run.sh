# evn install
git clone https://github.com/TouKaienn/STSR-INR.git
conda create --name STSRINR python=3.9
conda activate STSRINR
pip install -r requirements.txt

# test the code
python3 main.py --config_path './configs/ionization_inf.yml'