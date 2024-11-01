import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--n_workers", default=2, type=int, help="Number of parallel environments.")
    parser.add_argument("--interval", default=50, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--do_test", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                        help="The flag determines whether to render each agent or not.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--quantization", default='no', type=str,
                        help="The flag determines type of quantization: no, static, dynamic")
    parser.add_argument("--pruning", action="store_true",
                        help="The flag determines whether to prune policy or not.")
    parser.add_argument("--is_structured", action="store_true",
                        help="The flag determines which type of pruning to use: structured or not.")
    parser.add_argument("--network_part", default='RL_only', type=str,
                        help="Which part of the network to prune.")
    parser.add_argument("--binary_qat", action="store_true",
                        help="The flag determines whether to do binary quantization-aware training or not.")
    parser.add_argument("--num_episodes", default=100, type=int,
                        help="The number of episodes to evaluate the resulting model.")
    parser.add_argument("--test_bs", default=100, type=int,
                        help="Batch size when measuring the time of the model forward pass.")
    parser_params = parser.parse_args()

    """ 
     Parameters based on the "Exploration By Random Network Distillation" paper.
     https://arxiv.org/abs/1810.12894    
    """
    # region default parameters
    default_params = {"env_name": "MontezumaRevengeNoFrameskip-v4",
                      "state_shape": (4, 84, 84),
                      "obs_shape": (1, 84, 84),
                      "total_rollouts_per_env": int(30e3),
                      "max_frames_per_episode": 4500,  # 4500 * 4 = 18K :D
                      "rollout_length": 128,
                      "n_epochs": 4,
                      "n_mini_batch": 4,
                      "lr": 1e-4,
                      "ext_gamma": 0.999,
                      "int_gamma": 0.99,
                      "lambda": 0.95,
                      "ext_adv_coeff": 2,
                      "int_adv_coeff": 1,
                      "ent_coeff": 0.001,
                      "clip_range": 0.1,
                      "pre_normalization_steps": 50,
                      }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
