# SADCWEB-SOBA
Sentiment-Aware Deep Contextual Word Embeddings-Based Semi-Automatic Ontology Building for Aspect-Based Sentiment Classification

## Recommended software
- Anaconda 3: https://www.anaconda.com/products/individual
- Python 3: https://www.python.org/downloads/
- PyCharm: https://www.jetbrains.com/pycharm/
- Protege: https://protege.stanford.edu/
- Java: https://www.oracle.com/nl/java/technologies/javase-downloads.html
- Java Eclipse IDE: https://www.eclipse.org/downloads/packages/release/kepler/sr1/eclipse-ide-java-developers
- Notepad++: https://notepad-plus-plus.org/downloads/
- 7zip: https://www.7-zip.org/

### GPU acceleration
Follow these steps to set up CUDA: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
- NVIDIA CUDA 10.0: https://developer.nvidia.com/cuda-10.0-download-archive
- CuDNN 7.6.5.32 (for CUDA 10.0): https://developer.nvidia.com/cudnn (membership required)
- NVIDA Nsight Visual Studio Edition: https://developer.nvidia.com/nsight-visual-studio-edition
- Visual Studio 2017: https://visualstudio.microsoft.com/vs/older-downloads/

### Google Colab
Google Colab is needed for creating word embeddings and fine-tuning language models (LMs). A local runtime is advised for creating word-embeddings, as Google's TPUs are slow. When you don't have a GPU on your device, use a hosted runtime with Google's GPU's (recommended for fine-tuning LMs).

Now follow the virtual environment set-up instructions from https://github.com/stefanvanberkum/CD-ABSC#set-up-instructions. Make sure to use this repository to include in the virtual environment. It is advised to leave the `ontology` directory out of the virtual environment, as no `.ipynb` files are present. _Note: Python 3.5 is highly recommended for the virtual environment, as higher versions often don't support TensorFlow 1 anymore. Using more recent versions of Python will throw a lot of errors during the installation of the packages in requirements.txt_

Set-up local runtime with Google Colab using the command in `jupyter_colab.txt`. Copy the localhost link into the runtime of Google Colab.

## Ontology building
The ontology building code is based on https://github.com/MarkRademaker/DCWEB-SOBA. We added new functionality and fixed bugs. The ontology building code can be found in the `ontology` directory.

Ontology building steps:
- Import `ontology` directory as a Maven project into Eclipse IDE. Make sure the pom.xml has no errors before running any code.
- Download JavaFX SDK 16 from: https://gluonhq.com/products/javafx/
- Get the Yelp data set: https://www.yelp.com/dataset, and extract the files into `externalData`
- Get the corpus data using the required template. In `MyCorpus.java`, change `boolean t5` and `boolean training` in `getDomainTrainingData()` to the desired values. Change `String output_filename` to the desired output name. Set `int no_reviews` to the desired number of reviews (for word embeddings, 2000 is advised; for training LMs, 50.000-200.000 is advised).
- Fine-tune and post-train LMs using `FineTune.ipynb` (for BERT and RoBERTa) and `t5FineTune.ipynb` (for T5) in `wordembed`. Add these checkpoints to `models` directory in `wordembed` and put the files in `largeData`. Our models are available here: https://drive.google.com/drive/folders/1s2NIQOAYe-vAfw-lcV0IGU_w-CUtfYpu?usp=sharing
- Create word embeddings of 2000 reviews using `getWordEmbeddingsBERT` (for BERT and RoBERTa) and `getWordEmbeddingsT5` (for T5) in `wordembed`.
- Import normal and fine-tuned word embeddings in `TermSelectionAlgo.java` by uncommenting code in constructor. Change `String wordEmbeddings` and `String wordEmbeddingsFT` to the desired files. Make sure these files do not exceed ~1GB as maximum heap space can not be exceeded. 
- Change MCS thresholds to desired values in `TermSelectionAlgo.java` and `OntologyBuilder.java`.
- Now comment out the uncommented code in `TermSelectionAlgo.java` and run `MainOntologyBuilder.java`. Follow the steps of the program to create the ontology. 
- Move the created ontologies in `output` to the `evaluation/data/externalData` directory and start evaluation.

## Evaluation
The evaluation code is based on https://github.com/stefanvanberkum/CD-ABSC. We added new funtionality and fixed bugs. The evaluation code can be found in the `evaluation` directory. The 8 ontologies we evaluate in our research are included in the externalData directory. The four ontologies we have created have prefix 'final-', the four ontologies from the literature have prefix 'restaurant-'.

Set-up steps:
- Change the path of `java_path` in `ontology.py` to the java installation.
- Download required files:
  -  Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
  -  Stanford CoreNLP Language Models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar
-  Extract the zip and update `path_to_jar` and `path_to_models_jar` in `ontology.py` to these files.
-  When using k-fold cross validation, set `rest_rest_cross=True`.
-  Set the Python intepreter in PyCharm to the Python 3.5 .exe from the virtual environment. 
-  Get the raw data for the embeddings using `raw_data.py`. For the restaurant domain, SemEval 2014, SemEval 2015, and SemEval 2016 are already included in `externalData/BERT/restaurant`.
-  Get BERT embeddings, using the files in `getBERT` with a local runtime in Google Colab. This will generate multiple embedding files, which can be concetenated using `merge_textfiles.py`.
-  Run `prepare_bert.py` to prepare BERT word embeddings. Make sure to change `train_lines` to the number of desired lines for the training file (3x number of training sentences).
-  Tune hyperparameters using `main_hyper.py`.
-  Run `main_test.py` using your desired settings. Set `write_result=True` to make sure results are written to a text file.
-  Add accuracies to `evaluation.py` and run the code to obtain evaluation statistics, Welch's t-test statistics and a Box-Whisker plot after k-fold cross validation.

