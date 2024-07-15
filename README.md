<h1>Chromatin NDD</h1>
This repository contains the code for the bachelor’s thesis of Dario.
It encompasses a pipeline the process, segment and extract features describing the chromatin organisation of stained nuclei in confocal images. A file describing all the extracted features is included.
Furthermore, a 3D adaptation for calculation Gray-Level-Co-occurence matrix in 3D using python is also part of this repository.

<h2>Project structure</h2>
<h4>exp1_rep1, exp1_rep2, exp3, exp3_dapi </h4>
These folders contain the notebooks used for the analysis of the experments described in the thesis after extracting the features using the code in this repository.
<h4>features</h4>
Contains the python scripts to extract the features from nuclear crops (get called by feature_extraction).
<h4>segmentation_comparison</h4>
Code and notebooks used to compare the different segmentation methods.
<h4>feature_description.csv</h4>
Description of the extracted features
<h4>glcm_testcases.ipynb</h4>
Notebook providing testcases for the 3D adaptation of Gray-Level-Co-occurence matrix to test that it works correctly
<h4>global_morphology_test.ipynb</h4>
Notebook that was used in an attempt to adapt global morphology features to a 3D version, including a suggested approach that should work (at the bottom)
<h4>simplified_uml_diagram.png</h4>
A simplified UML diagram of the pipeline (only including the most important functions and not including input parameter)

<h2>Usage of pipeline</h2>
The repository contains several scripts and notebooks to preprocess data, perform segmentation and extract features.

<h4>run_all_pipelines.py</h4> - calls the other scripts to run the whole pipeline
<h4>preprocessing.py</h4> - extracts files from microscope project file and performs the preprocessing as described in the methods section of the thesis
<h4>segmentation.py</h4> - segments the nuclei from the images using MultiOtsu thresholding
<h4>feature_extraction.py</h4> - extracts nuclear crops from images and calculates features thereof

<h4>How to run the complete pipeline:</h4>

```
python run_all_pipelines.py --input_dir <input directory> --output_dir <output directory> --run_all
```


<h2>3D adaptation of Gray-Level-Co-occurence matrix (GLCM)</h2>
The code to calculate GLCM of 3D images can be found in features/utils/graycomatrix.py </br>
Features of the calculated GLCMs were extracted in features/img_texture.py </br>
Testcases to see that the 3D adapation worked are shown in glcm_testcases.ipynb 
