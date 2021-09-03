# Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations 
## Accepted at ICCV2021 [[Paper]](https://arxiv.org/abs/2103.10226)

<p align="center" width="100%">
<img width="100%" src="docs/main.png">
</p>

### 0. Download the Dataset
* [Images](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)
* [Labels](https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q)
* Uncompress them into a path of your choice. Assuming that `$DATA` corresponds to the path where celebA has been placed, move `./data/celeba_meta` into `$DATA`. 

### 1. Install requirements

`pip install -r requirements.txt` 

### 2. Train and Validate

```python
python trainval.py -e tcvae -sb ../results -d $DATA -r 1
```

Argument Descriptions:
```
-e  [Experiment group to run like 'vae' (the rest of the experiment groups are in exp_configs/main_exps.py)] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-d  [Directory where the datasets are aved]
```

### 3. Visualize the Results

Follow these steps to visualize plots. Open `results.ipynb`, run the first cell to get a dashboard like in the gif below, click on the "plots" tab, then click on "Display plots". Parameters of the plots can be adjusted in the dashboard for custom visualizations.

<p align="center" width="100%">
<img width="100%" src="https://raw.githubusercontent.com/haven-ai/haven-ai/master/docs/vis.gif">
</p>


## Cite
```
@article{rodriguez2021beyond,
  title={Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations},
  author={Rodriguez, Pau and Caccia, Massimo and Lacoste, Alexandre and Zamparo, Lee and Laradji, Issam and Charlin, Laurent and Vazquez, David},
  journal={arXiv preprint arXiv:2103.10226},
  year={2021}
}
```
