# Multi-Task Learning as Multi-Objective Optimization

#This code repository includes the source code for the [Paper](https://arxiv.org/abs/1810.04650):

```
Multi-Task Reinforcement Learning with Multi-Objective Optimization
Bowei He
```

The experimentation framework is based on PyTorch; 
PyTorch version Frank_Wolfe_Solver and gradient descent method are implemented in `min_norm_solvers.py`, generic version using only Numpy is implemented in file `min_norm_solvers_numpy.py`.

 It also has smart initialization and gradient normalization tricks which are described with inline comments.

The source code and MetaWorld environment are released under the MIT License. See the License file for details.


# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

We adapt and use some code snippets from:
* [CSAILVision Semanti Segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch)
* [PyTorch-SemSeg](https://github.com/meetshah1995/pytorch-semseg/)



# Usage
The code base uses `configs.json` for the global configurations like dataset directories, etc.. Experiment specific parameters are provided seperately as a json file. See the `sample.json` for an example.

To train MT-SAC, use the command: 
```bash
python  train_multi_task.py --config meta_config/mt10/mtsac.json --id MT10_MTSAC --method 'multitask_SAC'  --seed 1 --worker_nums 10 --eval_worker_nums 10

```

# Contact
To Train PO-MT-SAC, use the command:


For any question, you can contact bokwaiho200010@gmail.com

# Citation
