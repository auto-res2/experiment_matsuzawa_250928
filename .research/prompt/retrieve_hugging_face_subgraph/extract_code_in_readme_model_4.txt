
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
tags:
- Bowling-v5
- deep-reinforcement-learning
- reinforcement-learning
- custom-implementation
library_name: cleanrl
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: Bowling-v5
      type: Bowling-v5
    metrics:
    - type: mean_reward
      value: 30.20 +/- 1.17
      name: mean_reward
      verified: false
---

# (CleanRL) **PPO** Agent Playing **Bowling-v5**

This is a trained model of a PPO agent playing Bowling-v5.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the most up-to-date training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sebulba_ppo_envpool_impala_atari_wrapper.py).

## Get Started

To use this model, please install the `cleanrl` package with the following command:

```
pip install "cleanrl[jax,envpool,atari]"
python -m cleanrl_utils.enjoy --exp-name sebulba_ppo_envpool_impala_atari_wrapper --env-id Bowling-v5
```

Please refer to the [documentation](https://docs.cleanrl.dev/get-started/zoo/) for more detail.


## Command to reproduce the training

```bash
curl -OL https://huggingface.co/cleanrl/Bowling-v5-sebulba_ppo_envpool_impala_atari_wrapper-seed1/raw/main/sebulba_ppo_envpool_impala_atari_wrapper.py
curl -OL https://huggingface.co/cleanrl/Bowling-v5-sebulba_ppo_envpool_impala_atari_wrapper-seed1/raw/main/pyproject.toml
curl -OL https://huggingface.co/cleanrl/Bowling-v5-sebulba_ppo_envpool_impala_atari_wrapper-seed1/raw/main/poetry.lock
poetry install --all-extras
python sebulba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 2 3 4 5 6 --track --save-model --upload-model --hf-entity cleanrl --env-id Bowling-v5 --seed 1
```

# Hyperparameters
```python
{'actor_device_ids': [0],
 'anneal_lr': True,
 'async_batch_size': 20,
 'async_update': 3,
 'batch_size': 7680,
 'capture_video': False,
 'clip_coef': 0.1,
 'cuda': True,
 'ent_coef': 0.01,
 'env_id': 'Bowling-v5',
 'exp_name': 'sebulba_ppo_envpool_impala_atari_wrapper',
 'gae_lambda': 0.95,
 'gamma': 0.99,
 'hf_entity': 'cleanrl',
 'learner_device_ids': [1, 2, 3, 4, 5, 6],
 'learning_rate': 0.00025,
 'max_grad_norm': 0.5,
 'minibatch_size': 1920,
 'norm_adv': True,
 'num_actor_threads': 1,
 'num_envs': 60,
 'num_minibatches': 4,
 'num_steps': 128,
 'num_updates': 6510,
 'profile': False,
 'save_model': True,
 'seed': 1,
 'target_kl': None,
 'test_actor_learner_throughput': False,
 'torch_deterministic': True,
 'total_timesteps': 50000000,
 'track': True,
 'update_epochs': 4,
 'upload_model': True,
 'vf_coef': 0.5,
 'wandb_entity': None,
 'wandb_project_name': 'cleanRL'}
```
    
Output:
{
    "extracted_code": ""
}
