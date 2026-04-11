## Ant-v5
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`1_000_000` | 
| `memory_size`         |`1_000_000` | 
| `batch_size`          |`256`       | 
| `learning_start`      |`10_000`    | 
| `hidden_size_actor`   |`[256, 256]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |
## HalfCheetah-v5
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`500_000`   | 
| `memory_size`         |`200_000`   | 
| `batch_size`          |`128`       | 
| `learning_start`      |`5_000`     | 
| `hidden_size_actor`   |`[128, 128]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |

## Hopper-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`500_000`   | 
| `memory_size`         |`200_000`   | 
| `batch_size`          |`128`       | 
| `learning_start`      |`5_000`     | 
| `hidden_size_actor`   |`[128, 128]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |

## Humanoid-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `0.1`      | 
| `total_timesteps`     |`3_000_000` | 
| `memory_size`         |`1_500_000` | 
| `batch_size`          |`256`       | 
| `learning_start`      |`10_000`    | 
| `hidden_size_actor`   |`[512, 512]`| 
| `hidden_size_critic`  |`[512, 512]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `alpha_lr`            |`0.0003`    | 
| `auto_alpha`          |`true`      |

## HumanoidStandup-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `0.1`      | 
| `total_timesteps`     |`1_000_000` | 
| `memory_size`         |`1_500_000` | 
| `batch_size`          |`512`       | 
| `learning_start`      |`50_000`    | 
| `hidden_size_actor`   |`[512, 512]`| 
| `hidden_size_critic`  |`[512, 512]`| 
| `actor_lr`            |`0.0005`    | 
| `critic_lr`           |`0.0003`    | 
| `alpha_lr`            |`0.0001`    | 
| `auto_alpha`          |`true`      |

| env_kwargs            |   Value    |
| ----------------------| ---------: | 
| `ctrl_cost_weight`    | `0.01`     | 
| `uph_cost_weight`     | `2.0`      | 
| `reset_noise_scale`   | `0.03`     | 


## InvertedDoublePendulum-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`60_000`    | 
| `memory_size`         |`200_000`   | 
| `batch_size`          |`128`       | 
| `learning_start`      |`5_000`     | 
| `hidden_size_actor`   |`[128, 128]`| 
| `hidden_size_critic`  |`[128, 128]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |

## InvertedPendulum-v5
| Parameter             |   Value  |
| ----------------------| -------: | 
| `reward_scaler`       | `1.0`    | 
| `total_timesteps`     |`60_000`  | 
| `memory_size`         |`100_000` | 
| `batch_size`          |`128`     | 
| `learning_start`      |`5_000`   | 
| `hidden_size_actor`   |`[64, 64]`| 
| `hidden_size_critic`  |`[64, 64]`| 
| `actor_lr`            |`0.0003`  | 
| `critic_lr`           |`0.0003`  | 
| `auto_alpha`          |`false`   |

## Pusher-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`200_000`   | 
| `memory_size`         |`1_000_000` | 
| `batch_size`          |`256`       | 
| `learning_start`      |`10_000`    | 
| `hidden_size_actor`   |`[256, 256]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `alpha_lr`            |`0.0003`    | 
| `auto_alpha`          |`true`      |

## Reacher-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`100_000`   | 
| `memory_size`         |`50_000`    | 
| `batch_size`          |`128`       | 
| `learning_start`      |`1_000`     | 
| `hidden_size_actor`   |`[64, 64]`  | 
| `hidden_size_critic`  |`[128, 128]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |

## Swimmer-v5 
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`500_000`   | 
| `memory_size`         |`100_000`   | 
| `batch_size`          |`256`       | 
| `learning_start`      |`1_0000`    | 
| `hidden_size_actor`   |`[256, 256]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `alpha_lr`            |`0.0003`    | 
| `auto_alpha`          |`true`      |

## Walker2d-v5
| Parameter             |   Value    |
| ----------------------| ---------: | 
| `reward_scaler`       | `1.0`      | 
| `total_timesteps`     |`1_000_000` | 
| `memory_size`         |`1_000_000` | 
| `batch_size`          |`256`       | 
| `learning_start`      |`5_000`     | 
| `hidden_size_actor`   |`[256, 256]`| 
| `hidden_size_critic`  |`[256, 256]`| 
| `actor_lr`            |`0.0003`    | 
| `critic_lr`           |`0.0003`    | 
| `auto_alpha`          |`false`     |

