# CW-SOBA
Contextual Word embeddings in Semi-automatic Ontology Building for Aspect based sentiment analysis

## Recommended software
- Anaconda 3: https://www.anaconda.com/products/individual
- Python 3: https://www.python.org/downloads/
- PyCharm: https://www.jetbrains.com/pycharm/
- Protege: https://protege.stanford.edu/
- Java Eclipse IDE: https://www.eclipse.org/downloads/packages/release/kepler/sr1/eclipse-ide-java-developers
- Notepad++: https://notepad-plus-plus.org/downloads/

### GPU acceleration
Follow these steps to set up CUDA: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
- NVIDIA CUDA 10.0: https://developer.nvidia.com/cuda-10.0-download-archive
- CuDNN 7.6.5.32 (for CUDA 10.0): https://developer.nvidia.com/cudnn (membership required)
- NVIDA Nsight Visual Studio Edition: (https://developer.nvidia.com/nsight-visual-studio-edition)
- Visual Studio 2017: https://visualstudio.microsoft.com/vs/older-downloads/

### Google Colab
Google Colab is needed for creating word embeddings and fine-tuning language models (LMs). A local runtime is advised for creating word-embeddings, as Google's TPUs are slow. When you don't have a GPU on your device, use a hosted runtime with Google's GPU's (recommended for fine-tuning LMs).

Now follow the virtual environment set-up instructions from https://github.com/stefanvanberkum/CD-ABSC#set-up-instructions. Make sure to use this repository to include in the virtual environment. It is advised to leave the `ontology` directory out of the virtual environment, as no `.ipynb` files are present. _Note: Python 3.5 is highly recommended for the virtual environment, as higher versions often don't support TensorFlow 1 anymore. Using more recent versions of Python will throw a lot of errors during the installation of the packages in requirements.txt_

Set-up local runtime with Google Colab using the command in `jupyter_colab.txt`. Copy the localhost link into the runtime of Google Colab.

## Ontology building
The ontology building code is based on https://github.com/MarkRademaker/DCWEB-SOBA. We added new functionality and fixed bugs. The ontology building code can be found in the `ontology` directory.

Set-up steps:
- Import `ontology` directory as a Maven project into Eclipse IDE. Make sure the pom.xml has no errors before running any code.
- Download JavaFX SDK 16 from: https://gluonhq.com/products/javafx/
- Get the Yelp data set: https://www.yelp.com/dataset, and extract the files into `externalData`
- Get the corpus data using the required template. In `MyCorpus.java`, change `boolean t5` and `boolean training` in `getDomainTrainingData()` to the desired values. Change `String output_filename` to the desired output name. Set `int no_reviews` to the desired number of reviews (for word embeddings, 2000 is advised; for training LMs, 50.000-200.000 is advised).
- Fine-tune and post-train LMs using 

## Evaluation
The evaluation code is based on https://github.com/stefanvanberkum/CD-ABSC.

