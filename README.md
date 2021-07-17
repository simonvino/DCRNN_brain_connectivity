# A graph neural network framework for causal inference in brain networks

<img src="https://github.com/simonvino/DCRNN_for_brain_connectivity/blob/main/figures/DCRNN.png" width="800">


This is the implementation of the graph neural network model used in our paper:

S. Wein, W. M. Malloni, A. M. Tomé, S. M. Frank, G. -I. Henze, S. Wüst, M. W. Greenlee & E. W. Lang,
[A graph neural network framework for causal inference in brain networks](https://www.nature.com/articles/s41598-021-87411-8), Scientific Reports 11, 8061 (2021).

The implementation is based on the [DCRNN](https://github.com/liyaguang/DCRNN) proposed by:

Y. Li, R. Yu, C. Shahabi & Y. Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.

## Requirements

- scipy>=0.19.0
- numpy>=1.12.1
- pyaml
- tensorflow=1.xx

## Run demo version

A short demo version is included in this repository, which can serve as a template to process your own MRI data. Artificial fMRI data is provided in the directory ``` MRI_data/fMRI_sessions/ ``` and the artificial timecourses have the shape ``` (nodes,time) ```. 
The adjacency matrix in form of the structural connectivity (SC) between brain regions can be stored in ``` MRI_data/SC_matrix/ ```. An artificial SC matrix with shape ``` (nodes,nodes) ``` is also provided in this demo version.

The training samples can be generated from the subject session data by running: 

```
python generate_samples.py --input_dir=./MRI_data/fMRI_sessions/ --output_dir=./MRI_data/training_samples
```

The model can then be trained by running:

```
python dcrnn_for_brain_connectivity_train.py --config_filename="./configs/dcrnn_demo_config.yaml" --save_predictions=True
```

## Data availability

Preprocessed functional and structural MRI data from Human Connectome Project data is publicly available under: https://db.humanconnectome.org.

A nice tutorial on white matter tracktograph for creating a SC matrix is available under: https://osf.io/fkyht/. 

## Citations

If you apply this graph neural network model for MRI analysis, please cite the following paper: 

```
@article{Wein2021,
  title = {A graph neural network framework for causal inference in brain networks},
  author = {Wein, Simon and Malloni, Wilhelm and Tomé, Ana and Frank, S. and Henze, Gina-Isabelle and Wüst, S. and Greenlee, Mark and Lang, Elmar},
  year = {2021},
  month = {04},
  volume = {11},
  journal = {Scientific Reports},
  doi = {10.1038/s41598-021-87411-8}
}
```

And the model architecture was originally proposed by Li et al.:

```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
