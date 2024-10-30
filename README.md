# tiny_rl

## Project Description:
This project explores the applicability of neural network optimization techniques for reinforcement learning tasks. 
Specifically, it is based on the [implementation](https://github.com/alirezakazemipour/PPO-RND) 
of "Exploration by Random Network Distillation" applied to the Atari game Montezuma's Revenge. 
The methods used include post-training quantization, binary quantization-aware training, and pruning.

## Technologies:
* **Post-training quantization** – torch.quantization (Eager Mode).
* **Binary quantization-aware training** – https://github.com/1adrianb/binary-networks-pytorch.
* **Pruning** – torch.nn.utils.prune, https://github.com/VainF/Torch-Pruning.

## Results:
The neural network architecture includes both convolutional and fully connected layers:
* **CNN Part**: Contains the convolutional layers.
* **RL Part**: Contains the fully connected layers.

The average reward per episode is determined by averaging the rewards across 100 episodes. 
The computation time provided represents the average duration of 500 forward passes, each with a batch size of 100, using observations stored in `data.npy`. 
This file contains a pre-collected set of observations that serve as input data for consistent benchmarking during the forward pass measurements.

### Post-training quantization
Parameters  | Data type | Quantization type | Mean episode reward | Time (s)  | Size (MB) |
----------  | --------- | ----------------- | ------------------- | --------  | --------- |
–           | float32   | –                 | 6600                | 0.0148    | 5.6354    |
RL(weights) | float16   | dynamic(eager)    | 6600                | 0.0287    | 5.6402    |
RL(weights) | qint8     | dynamic(eager)    | 6600                | 0.0155    | 1.6568    |
CNN + RL    | float16   | static(eager)     | 6600                | 0.0197    | 5.6355    |
**CNN + RL**| **qint8** | **static(eager)** | **6600**            | **0.0188**|**1.4264** |

The best result is shown in the last row – a 75% model size reduction while maintaining the same reward level. 
The results with float16 are somewhat unusual, and the exact reason for this behavior hasn't been identified. 
Some discussions have reported similar issues 
(see [discussion1](https://discuss.pytorch.org/t/float16-dynamic-quantization-has-no-model-size-benefit/99675), 
[discussion2](https://discuss.pytorch.org/t/quantization-not-decreasing-model-size-static-and-qat/87319)).

### Unstructured pruning (torch.nn.utils.prune)

The L1Unstructured method from torch.nn.utils.prune zeros out parts of the network's weights based on the L1 criterion. 
The problem is that memory is still allocated for the zeroed weights, and they continue to be used in computations. 
An attempt was made to use the torch.sparse format for the model's parameters. 
The last two columns refer to the sparse format, and the two preceding columns refer to dense.

Part | Method, amount       | Global sparsity | Mean episode reward | Time (s) | Size (MB) | Time (s) | Size (MB) |
---- | -------------------- | --------------- | ------------------- | -------- | --------- | -------- | --------- |
RL   | L1Unstructured, 0.5  | 47.17%          | 6600                | 0.0160   | 5.6353    | 0.1049   | 13.6048   |
ALL  | L1Unstructured, 0.5  | 49.94%          | 6600                | 0.0148   | 5.6353    | 1.2456   | 14.0726   |
RL   | L1Unstructured, 0.6  | 56.61%          | 6488                | 0.0146   | 5.6353    | 0.0920   | 10.9489   |
ALL  | L1Unstructured, 0.6  | 59.92%          | 5162                | 0.0162   | 5.6353    | 1.3546   | 11.2611   |
RL   | L1Unstructured, 0.75 | 70.76%          | 5781                | 0.0151   | 5.6353    | 0.0733   | 6.9652    |
ALL  | L1Unstructured, 0.75 | 74.91%          | 69                  | 0.0139   | 5.6353    | 1.1668   | 7.0439    |

A comparison of the last two columns shows that the attempt was not successful: 
the computation time increased significantly, especially when pruning the CNN part 
(as seen in the rows labeled "ALL" in the Part column), and the model size also grew. 
However, as sparsity increases, the performance gap narrows. 
Therefore, with more aggressive pruning (90% or higher), we might see reductions in both model size and computation time. 
In this case, pruning was done without fine-tuning, 
so iterative pruning (pruning → training → pruning → etc.) with over 90% pruning of the RL part, 
followed by converting the weights to a sparse format, could potentially lead to better outcomes.

### Structured pruning (torch.nn.utils.prune)
Layer(s)               | Method                  | Global sparsity | Mean episode reward | Time (s) | Size (MB) |
---------------------- | ----------------------- | --------------- | ------------------- | -------- | --------- | 
fc1                    | amount=0.15, n=2, dim=1 | 8.55 %          | 6360                | 0.0172   | 5.635433  |
fc1                    | amount=0.15, n=1, dim=1 | 8.55 %          | 5636                | 0.0142   | 5.635433  |
fc1, fc2               | amount=0.1,  n=1, dim=1 | 6.54 %          | 6518                | 0.0146   | 5.635433  |
fc1, fc2, extra_policy | amount=0.1,  n=1, dim=1 | 7.97 %          | 6447                | 0.0142   | 5.635433  |

The same issue arises as in the previous section: 
the weights are zeroed out but not removed, so the model size and computation speed remain unchanged. 
This leads to the next section.


### Structured pruning (torch-pruning)

Layer(s)               | Method                  | Global sparsity, n neurons          | Mean episode reward | Time (s)  | Size (MB) |
---------------------- | ----------------------- | -------------------------------     | ------------------- | --------  | --------- | 
**fc1**                | **L1, 0.3**             | **19.36 %, 256 -> 180**             | **6600**            | **0.0143**| **4.5456**| 
fc1                    | L1, 0.75                | 48.90 %, 256 -> 64                  | 499                 | 0.0153    | 2.8821    | 
**fc1, fc2**           | **L1, (0.3, 0.1)**      | **25.32 %, 256 -> 180, 448 -> 404** | **6535**            | **0.0187**| **4.2098**| 
**fc1, fc2**           | **L1, (0.3, 0.2)**      | **30.85 %, 256 -> 180, 448 -> 359** | **6445**            | **0.0152**| **3.8986**| 
**fc1, fc2**           | **L1, (0.3, 0.3)**      | **35.80 %, 256 -> 180, 448 -> 314** | **6251**            | **0.0148**| **3.6197**| 
conv3                  | L1, 0.05                | 2.80 %, channels: 64 -> 61          | 6600                | 0.0145    | 5.4779    | 

Unlike in the previous section, here the weights are actually removed. 
The best results can be seen in rows 1, 3, 4, and 5. 
However, it's clear that the speed did not improve: 
comparing rows 1 and 2 shows that the model with fewer parameters takes longer to compute.

## Requirements:
The easiest way to set up all required dependencies is by creating an Anaconda environment:
```
conda create -n test_env python=3.7
conda activate test_env
pip install -r requirements.txt
```
To use Atari environments, you'll need to import the ROM files. 
Follow the instructions provided [here](https://github.com/openai/atari-py?tab=readme-ov-file#roms) 
to download and install the Atari ROMs.

## How to Run:
### Quantization
To test different quantization types, use the following command:

`python main.py --do_test --quantization <quantization_type> --num_episodes 100 --test_bs 100`

Options for <quantization_type>:
* dynamic_int8
* dynamic_float16
* static_int8
* static_float16

### Pruning
To test pruning, you can run either of the following commands:

**Structured Pruning**

`python main.py --do_test --pruning --is_structured --num_episodes 100 --test_bs 100`

**Unstructured Pruning**

`python main.py --do_test --pruning --network_part <network_part> --num_episodes 100 --test_bs 100`

Options for <network_part>:
* RL_only: apply pruning to the reinforcement learning (RL) module only
* all_net: apply pruning to the entire network
