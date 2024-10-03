Packages: scipy, h5py, faiss, pandas

Please follow the following steps in order:

1. Download all the ".mat" file of ICVL hyperspectral images as described at https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html

2. Put the downloaded hyperspectral images to the folder: /train/data/

3. Run the script "train_A+_oracle.py" to train the oracle solution detailed in: Lin, Yi-Tun, and Graham D. Finlayson. "Investigating the upper-bound performance of sparse-coding-based spectral reconstruction from RGB images." Color and Imaging Conference. Vol. 2021. No. 29. 2021.

4. Run the script "train_A++.py" to train the model described in: Lin, Yi-Tun, and Graham D. Finlayson. "A rehabilitation of pixel-based spectral reconstruction from RGB images." Sensors 23.8 (2023): 4155.

5. The models will be saved in /train/trained_models/