This is the companion code for the paper *Segmenting Microarrays with Deep Neural Networks*.

#### Installation 
To use this code, you'll need to
 - Install [Caffe](http://caffe.berkeleyvision.org/), along with its Python interface. Be warned that you should have a compute-capable GPU, else training the network and segmenting images will be unfeasibly slow.
 - Install a variety of other Python modules. This is most simply done with `pip install requirements.txt`.
 - Download the experimental (real) and simulated datasets from [Lehmussola et al.'s website](http://www.cs.tut.fi/sgn/csb/spotseg/). The experimental images should be placed in `sources/experimental` and the simulated `.mat` files in the `sources/simulated` folder. 

#### Usage
This project contains two 'pipelines', one for processing the experimental data and one for processing the simulated data. We recommend you interact with them from an interactive console, such as the one provided by [Spyder](https://github.com/spyder-ide/spyder).

**Experimental Pipeline**. The experimental pipeline consists of the `experimental_training` and `experimental_evaluation` modules. Running the `experimental_training.make_training_files()` function will read the hand-labellings from `sources/labels` along with the corresponding images from `sources/experimental` and construct two LMDB files that contain all the data needed to train the classifier. This will take about an hour on a modern (2015) machine. The classifier can then be trained by calling 

``caffe train -solver sources/definition/experimental_solver.prototxt``

from the project directory. Once the classifier is trained (a few hours), it can be used to segment the experimental images by calling `experimental_evaluation.score_experimental_images()`, which will process each image in turn and store the results in `temporary/scores/experimental_scores.hdf5`. This can take a day or more to finish. Once finished, calling `measure_all()` will use the scores from the `.hdf5` file to measure the expression ratio of each spot in each image. Finally, the functions `correlations` and `mean_absolute_errors` in `experimental_evaluation` can be called on the expression ratios in order to calculate the results reported in the paper.

**Simulated Pipeline**. The simulated pipeline consists of the `simulated_training` and `simulated_evaluation` modules. Running the `experimental_training.make_training_files()` function will read the datafiles from `sources/simulated` and construct two LMDB files that contain all the data needed to train the classifier. This will take about an hour on a modern (2015) machine. The classifier can then be trained by calling 

``caffe train -solver sources/definition/simulated_solver.prototxt``

from the project directory. Once the classifier is trained (a few hours), it can be used to segment the simulated images by calling `simulated_evaluation.score_simulated_images()`, which will process each image in turn and store the results in `temporary/scores/simulated_scores.hdf5`. This can take a day or more to finish. Once finished, calling `measure_all()` will use the scores from the `.hdf5` file to measure the expression ratio of each spot in each image. Finally, the functions `calculate_error_rate` and `calculate_discrepency_distance` in `simulated_evaluation` can be called on the expression ratios in order to calculate the results reported in the paper.

**Interrupting training early**. In both cases, you can interrupt training the network early (after 40,000 iterations or so) if you don't mind losing a small amount of performance. In that case, set the `MODEL_PATH` in `simulated_evaluation`/`experimental_evaluation` to point towards one of the snapshots that can be found in `temporary/models`.