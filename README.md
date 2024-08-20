# Long-Term Upper-Limb Prosthesis Myocontrol via High-Density sEMG and Incremental Learning

Repository for:

_Dario Di Domenico, Nicolò Boccardo, Andrea Marinelli, Michele Canepa, Emanuele Gruppioni, Matteo Laffranchi and Raffaello Camoriano,
"Long-Term Upper-Limb Prosthesis Myocontrol via High-Density sEMG and Incremental Learning", IEEE Robotics and Automation Letters (RA-L)_;

>**Abstract**
>
> Noninvasive human-machine interfaces such as surface electromyography (sEMG) have long been employed for controlling robotic prostheses. However, classical controllers are limited to few degrees of freedom (DoF). More recently, machine learning methods have been proposed to learn personalized controllers from user data. While promising, they often suffer from distribution shift during long-term usage, requiring costly model re-training. Moreover, most prosthetic sEMG sensors have low spatial density, which limits accuracy and the number of controllable motions. In this work, we address both challenges by introducing a novel myoelectric prosthetic system integrating a high density-sEMG (HD-sEMG) setup and incremental learning methods to accurately control 7 motions of the Hannes prosthesis. First, we present a newly designed, compact HD-sEMG interface equipped with 64 dry electrodes positioned over the forearm. Then, we introduce an efficient incremental learning system enabling model adaptation on a stream of data. We thoroughly analyze multiple learning algorithms across 7 subjects, including one with limb absence, and 6 sessions held in different days covering an extended period of several months. The size and time span of the collected data represent a relevant contribution for studying long-term myocontrol performance. Therefore, we release the DELTA dataset together with our experimental code.

## DELTA Dataset

The _DELTA_ dataset is available on Zenodo: [10.5281/zenodo.10801000](https://doi.org/10.5281/zenodo.10801000).
Before running the code in this repository, please download the DELTA dataset and unzip the folder.
Place it in the cloned repository and add the path to the configuration files described below.

## Environment set-up

Tested on Python 3.10.12

We recommend using virtual environment to manage the required python dependencies.

`pip install virtualenv
`

To create the virtuale environment for this project, run in the shell:

`python3 -m virtualenv myo_ctrl`

activate it with:

`source myo_ctrl/bin/activate`

and install the dependencies:

`pip install -r requirements.txt`

Regarding the Cholesky update computation, we refer the reader to https://github.com/modusdatascience/choldate.git to install the packages, you can run:

`pip install git+git://github.com/jcrudy/choldate.git`

## Configuration files to run experiments

The relevant parameters used for the experiments are saved into the "parameters" folder.
Specifically,:
- params_MODEL.yaml: configuration file to run cross-validation experiments on the single day to optimize the hyperparameters.
- params_MODEL_batch_multiday.yaml: configuration file to run multiday experiments based on the previous optimization performed on the single day. It contains configuration parameters for the *_batch-setting_* experiments.
- params_MODEL_incr_multiday.yaml: configuration file to run multiday experiments based on the previous optimization performed on the single day. It contains configuration parameters for the *_incremental-setting_* experiments.

MODEL can be: kNN, LDA, RLSC, RFRLSC
## How to run sample experiments
Included in this repo there are two sample scripts for the experiments and are based on configuration files contained in "parameters" folder:
- Run_single_exp.py:  optimizes each model on the single-day acquisition for each subject
- Run_multiday_exp.py: runs the experiments on all multiday permutations

Overall:
- Exp_Handler.py: it is a more general purpose as it runs both the previous scripts.

`python Exp_handler.py`

#### Outputs
The script will automatically create a folder (name defined in the configuration file) where to save the logs of the experiments.

## Plot results

- plot_single_exp.py: shows the results of the hyperparameters optimization and saves figures in the results folder. Run the following code after the Run_exp.py:

`python plot_single_exp.py --params_file parameters/params_MODEL.yaml`

- plot_nComponents.py: shows the results of the optimization of the number of Random Features for the RF-RLSC algorithm.

`python plot_nComponents.py --params_file parameters/params_RFRLSC.yaml`

- plot_multiday.py: shows the results of the multiday analysis by grouping the subjects in the list (e.g., to include all healthy subjects: --subj P01,P02,P03,P04,P05,P06). The script can plot box-plots (boxPlots==True) or the trend of the accuracy for each repetition of the gestures. Give as input the folder in which the results are saved (RESULTS_FOLDERNAME):

`python plot_multiday.py --subj P01 --boxPlots True --results_fold RESULTS_FOLDERNAME`

- plot_RFRLSCvsRLSC_multiday.py: shows the results of the comparison between RFRLSC and RLSC incremental classifiers. The script can plot box-plots (boxPlots==True) or the trend of the difference in accuracy (RFRLSC-RLSC) for each repetition of the gestures. Give as input the folder in which the results are saved (RESULTS_FOLDERNAME):

`python plot_RFRLSCvsRLSC_multiday.py --subj P01 --boxPlots True --results_fold RESULTS_FOLDERNAME`

## Incremental model

The proposed incremental model is in RLS_classifier.py, defined as a class named rls_classifier.
