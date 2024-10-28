from abc import ABC
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.utils.prune as prune
import torch_pruning as tp
# from torchsummary import summary


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class PolicyModel(nn.Module, ABC):

    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                               stride=1)  # Nature paper -> kernel_size = 3, OpenAI repo -> kernel_size = 4

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=448)

        self.extra_value_fc = nn.Linear(in_features=448, out_features=448)
        self.extra_policy_fc = nn.Linear(in_features=448, out_features=448)

        self.policy = nn.Linear(in_features=448, out_features=self.n_actions)
        self.int_value = nn.Linear(in_features=448, out_features=1)
        self.ext_value = nn.Linear(in_features=448, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.fc2.bias.data.zero_()

        nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        self.extra_policy_fc.bias.data.zero_()
        nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        self.extra_value_fc.bias.data.zero_()

        nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        self.policy.bias.data.zero_()
        nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        self.int_value.bias.data.zero_()
        nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        self.ext_value.bias.data.zero_()

        self.dg_mode = False    # dependency graph mode: used for describing dependency between layers (pruning)

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        policy = self.policy(x_pi)
        probs = F.softmax(policy, dim=1)
        if self.dg_mode:    # pytorch-pruning can't establish dg with Categorical
            return int_value, ext_value, probs
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs

    def pruning(self, is_structured, part, sparse_layers):
        """Prune policy model.

        :param is_structured: if set to ``True``, structured pruning will be used
        :param part: a set of layers that need to be pruned ('RL_only' or 'all_net')
        :param sparse_layers: if set to ``True``, the weights will be in sparse format
        """
        n_all = sum(p.numel() for p in self.parameters())
        if is_structured:
            # pruning from repo https://github.com/VainF/Torch-Pruning
            self.dg_mode = True
            dg = tp.DependencyGraph().build_dependency(self, torch.randn(2, *self.state_shape))

            print(f"Before pruning: out_channels={self.conv3.out_channels}")
            strategy = tp.strategy.L1Strategy()
            pruning_index = strategy(self.conv3.weight, amount=0.05)
            plan = dg.get_pruning_plan(self.conv3, tp.prune_conv, pruning_index)
            plan.exec()
            print(f"After pruning: out_channels={self.conv3.out_channels}")

            n_all_after_prune = sum(p.numel() for p in self.parameters())
        else:
            # 1. Define the modules to prune
            parameters_to_prune = (
                (self.fc1, 'weight'),
                (self.fc2, 'weight'),
                (self.extra_value_fc, 'weight'),
                (self.extra_policy_fc, 'weight'),
                (self.policy, 'weight'),
                (self.int_value, 'weight'),
                (self.ext_value, 'weight')
            )
            if part != 'RL_only':
                parameters_to_prune += (
                    (self.conv1, 'weight'),
                    (self.conv2, 'weight'),
                    (self.conv3, 'weight')
                )

            # 2. Prune the modules
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.75
            )

            # 3. Remove pruning re-parametrization
            for layer, name in parameters_to_prune:
                prune.remove(layer, name)

            # 4. Convert matrices to sparse format
            if sparse_layers:
                for name, layer in self.named_modules():
                    if isinstance(layer, nn.Linear):
                        setattr(self, name, LinearSparseWeight(layer.weight, layer.bias))
                    elif isinstance(layer, nn.Conv2d) and part != 'RL_only':
                        setattr(self, name, Conv2dSparseWeight(layer))
        if self.dg_mode:
            self.dg_mode = False
            n_pruned = n_all - n_all_after_prune
        else:
            n_pruned = sum(int(layer.weight.numel() - layer.weight.count_nonzero()) for layer, _ in parameters_to_prune)
        print("Global sparsity: {:.2f}%".format(100 * n_pruned / n_all))


class TargetModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.encoded_features = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.encoded_features(x)


class PredictorModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.encoded_features = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.encoded_features(x)


class QuantPolicyModel(PolicyModel):
    """Quantized policy model. Child class of PolicyModel. It differs only in the presence of quantization blocks."""

    def __init__(self, state_shape, n_actions):
        super(QuantPolicyModel, self).__init__(state_shape, n_actions)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, inputs):
        x = inputs / 255.
        x = self.quant(x)   # change data type
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = F.relu(self.extra_value_fc(x))
        x_pi = F.relu(self.extra_policy_fc(x))
        x_v = self.skip_add.add(x, x_v)         # x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = self.skip_add.add(x, x_pi)       # x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        policy = self.policy(x_pi)

        # return to original data type
        int_value = self.dequant(int_value)
        ext_value = self.dequant(ext_value)
        policy = self.dequant(policy)

        probs = F.softmax(policy, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs


class HalfPolicyModel(PolicyModel):
    """Policy model with float16 weights. Child class of PolicyModel."""

    def __init__(self, state_shape, n_actions):
        super(HalfPolicyModel, self).__init__(state_shape, n_actions)

    def forward(self, inputs):
        return super(HalfPolicyModel, self).forward(inputs.half())


class NonTraceable(nn.Module):
    """Class for non-traceable part of PolicyModel forward. Used for fx-mode quantization."""

    def forward(self, policy):
        probs = F.softmax(policy, dim=1)
        dist = Categorical(probs)
        return dist, probs


class FxQuantPolicyModel(PolicyModel):
    """Fx-mode quantized policy model. Child class of PolicyModel."""

    def __init__(self, state_shape, n_actions):
        super(FxQuantPolicyModel, self).__init__(state_shape, n_actions)
        self.non_traceable_submodule = NonTraceable()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        policy = self.policy(x_pi)
        dist, probs = self.non_traceable_submodule(policy)
        return dist, int_value, ext_value, probs


class LinearSparseWeight(nn.Linear):
    """Linear layer with sparse weights and dense bias. Used in unstructured pruning. Child class of nn.Linear."""

    def __init__(self, weight, bias):
        in_features, out_features = weight.shape
        super(LinearSparseWeight, self).__init__(in_features, out_features)
        self.weight = torch.nn.Parameter(weight.data.to_sparse(), requires_grad=False)
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight @ x.T).T + self.bias


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    """Calculate conv2d output shape."""
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class Conv2dSparseWeight(nn.Conv2d):
    """nn.Conv2d layer with sparse weights and dense bias. Used in unstructured pruning. Child class of nn.Conv2d.

    Convolution is implemented as matrix multiplication.
    Note: works only for square kernel and stride.

    Methods:
        _convert_kernel: convert weights to 2d matrix for effective computation.
        _convert_input: convert input to 2d matrix for effective computation.
        forward: similar to forward of nn.Conv2d, used _convert_kernel and _convert_input.
    """

    def __init__(self, dense_layer) -> None:
        in_channels = dense_layer.in_channels
        out_channels = dense_layer.out_channels
        kernel_size = dense_layer.kernel_size
        stride = dense_layer.stride
        super(Conv2dSparseWeight, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=stride)
        self._convert_kernel(dense_layer.weight.data)
        self.bias = dense_layer.bias

    def _convert_kernel(self, weight):
        weight = torch.reshape(weight, (self.out_channels, -1)).to_sparse()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def _convert_input(self, torch_input, output_height, output_width):
        batch_size = torch_input.shape[0]
        in_channels = torch_input.shape[1]
        input_height = torch_input.shape[2]
        input_width = torch_input.shape[3]
        kernel_size = self.kernel_size[0]

        converted_input = torch.zeros((kernel_size**2 * in_channels, output_width * output_height * batch_size))
        col = 0
        for k_img in range(batch_size):
            for i in range(0, input_height - kernel_size + 1, self.stride[0]):
                for j in range(0, input_width - kernel_size + 1, self.stride[0]):
                    window = torch_input[k_img, :, i:i+kernel_size, j:j+kernel_size]
                    converted_input[:, col] = torch.flatten(window)
                    col += 1

        return converted_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)

        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=x.shape,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=0)
        converted_input = self._convert_input(x, output_height, output_width)

        output = self.weight @ converted_input + self.bias.unsqueeze(1)
        return output.transpose(1, 0).reshape(batch_size, -1, out_channels).permute(0, 2, 1).\
            reshape(batch_size, out_channels, output_height, output_width)
