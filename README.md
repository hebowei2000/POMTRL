# Multi-Task Reinforcement Learning with Multi-Objective Optimization

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

Nohup is strongly recommended due to the multi-process experiments. You can use the nohup to run the above commands.

```
nohup python  train_multi_task.py --config meta_config/mt10/mtsac.json --id MT10_MTSAC --method 'multitask_SAC'  --seed 1 --worker_nums 10 --eval_worker_nums 10  >> MT10_multitask_SAC_New_2022_04_20.log 2>&1 &

nohup python  train_multi_task_MGDA.py --config meta_config/mt10/mtsac_MGDA.json --id MT10_MTSAC_MGDA --method 'multitask_SAC' --seed 1 --worker_nums 10 --eval_worker_nums 10  >> MT10_multitask_SAC_MGDA_New_2022_04_20.log 2>&1 &
```

All the experiments results wiil be recorded in the folder `log` named with the experiment id like `MT10_MTSAC_MGDA`

# Contact

For any question, you can contact bokwaiho200010@gmail.com
