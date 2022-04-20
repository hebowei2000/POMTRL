**Multi-Task Reinforcment Learning with Multi-Objective Optimization**
#

```
Multi-Task Reinforcement Learning with Multi-Objective Optimization
Bowei He
```

The experimentation framework is based on PyTorch; 
PyTorch version Frank_Wolfe_Solver and gradient descent method are implemented in `min_norm_solvers.py`, generic version using only Numpy is implemented in file `min_norm_solvers_numpy.py`.

 It also has smart initialization and gradient normalization tricks which are described with inline comments.

The experiment configurations are defined in the filefolder `meta_config`.
The source code and MetaWorld environment are released under the MIT License. See the License file for details.


# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.




# Usage


To train MT-SAC and PO-MT-SAC, use the command: 
```bash
python  train_multi_task.py --config meta_config/mt10/mtsac.json --id MT10_MTSAC --method 'multitask_SAC'  --seed 1 --worker_nums 10 --eval_worker_nums 10

python  train_multi_task_MGDA.py --config meta_config/mt10/mtsac_MGDA.json --id MT10_MTSAC_MGDA --method 'multitask_SAC' --seed 1 --worker_nums 10 --eval_worker_nums 10

```

# Contact



For any question, you can contact bokwaiho200010@gmail.com

# Citation
