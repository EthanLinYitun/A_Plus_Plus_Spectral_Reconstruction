Packages: scipy, h5py, faiss, pandas

Please follow the following steps in order:

1. Download the ".mat" file of ICVL hyperspectral images as described at https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html

2. Put the downloaded hyperspectral images to the folder: /test/data/hyperspectral_gt/orig/

3. Run the script "rgb_simulate.py" to simulate the RGB images and the rotated or blurred hyperspectral ground-truths

4. Run the script "recover.py" for recovering the hyperspectral images (the recovery outcomes will be stored in: /test/data/hyperspectral_rec/)

5. Run the script "evaluate.py" to calculate MRAE errors and show the error maps

----

Note: We include the trained A++ (Ours), A+ and PR-RELS models in the /test/trained_models/ folder since their file sizes are small. For testing the DNN-based AWAN, AWAN-aug3 and HSCNN-D, please retrain the network and put them in: /test/trained_models/ (rename the models as appropriate).