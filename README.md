# DeepFl-LogP

DeepFl-LogP a deep neural lipophilitcity descriptor.

Installation instruction for windows:

Prior installation: make sure to update conda & update conda-build separately.

It's recommended to create a new conda environment, as follow: 

  >> conda create --name KarLogP python==3.7

To activate the new environment: >> conda activate KarLogP

Pre-requisites packages (install in the following order): 

  >> pip install pandas

  >> pip install tensorflow==1.15

  >> pip install mordred

  >> conda install -c conda-forge rdkit

Once installation is done, use cd to navigate to the script directory.

Run the python script on the Terminal:

  >> python DeepFl-LogP.py

  >> Enter molecule: Please, enter the SMILE code of the compound of interest

  >> DeepFl-LogP =  [[value]]

Finished! 

---------------------------------------------------------------------------------------

When using the DeepFl-LogP tool, Please cite this paper: https://rdcu.be/chxaY 
