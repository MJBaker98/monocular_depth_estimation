import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig


class Embedding(nn.Module):
    """
    Class which contains the convolutional layer to convert a full image into downsampled patches for encoding as a 2D vector
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # use a convolutional layer to extract pixel patches from the provided image without overlap
        # embed the patched kernels to the model dimension out_channels
        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.downsample(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Encoding(nn.Module):
    """
    Class containing the encoding representation
    """

    pos_encs: torch.Tensor

    def __init__(self, out_channels, sequence_length):
        super().__init__()

        # create class labels for each out channel
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, out_channels))

        # create position encodings which map each input to a unique position identifier
        pos_encs = torch.zeros(sequence_length, out_channels)
        for i in range(sequence_length):
            for j in range(out_channels):
                if j % 2 == 0:
                    pos_encs[i][j] = torch.sin(
                        torch.tensor(i, dtype=torch.float32)
                        / 10000 ** (2 * j / out_channels)
                    )
                else:
                    pos_encs[i][j] = torch.cos(
                        torch.tensor(i, dtype=torch.float32)
                        / 10000 ** (2 * (j - 1) / out_channels)
                    )

        self.register_buffer("pos_encs", pos_encs.unsqueeze(0))

    def forward(self, x):
        # generate class tokens for each image in the batch
        tokens = self.class_token.expand(x.shape[0], -1, -1)

        # add the encoding labels at index 0 of each batch
        x = torch.cat([tokens, x], dim=1)

        # add positional embeddings to the tokens
        x = x + self.pos_encs

        return x


class SelfAttention(nn.Module):
    """
    Representation of the Scaled Dot-Product Attention module from the paper
    "Attention Is All You Need"

    """

    def __init__(self, model_size: int, n_head_nodes: int):
        super().__init__()
        self.n_head_nodes = n_head_nodes

        self.Q = nn.Linear(model_size, n_head_nodes)
        self.K = nn.Linear(model_size, n_head_nodes)
        self.V = nn.Linear(model_size, n_head_nodes)

    def forward(self, x):
        """
        Implement torch functional dot product attention
        """
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        attn_res = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v
        )

        return attn_res


class MultiheadAttention(nn.Module):
    """
    Representation of multi-head attention from Attention is All You Need

    """

    def __init__(self, n_total_nodes: int, n_heads: int):
        super().__init__()
        assert n_total_nodes % n_heads == 0, (
            f"Number of multi-head attention heads must be divisible by the number of heads: received {n_total_nodes} total nodes and {n_heads}"
        )

        self.nodes_per_head = int(
            n_total_nodes / n_heads
        )  # wrap to an int for making layers later

        self.final_linear = nn.Linear(n_total_nodes, n_total_nodes)

        self.heads = nn.ModuleList(
            [
                SelfAttention(
                    model_size=n_total_nodes, n_head_nodes=self.nodes_per_head
                )
                for _ in range(n_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        cat_output = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.final_linear(cat_output)


class MLP(nn.Module):
    """
    Simple fully connected MLP with dropout and configurable activation
    """

    def __init__(
        self,
        n_hidden_layers: int,
        n_inputs: int,
        n_outputs: int,
        n_hidden_nodes: int,
        activation: str,
    ):
        super().__init__()

        assert n_hidden_layers >= 1, "MLP does not support less than one hidden layer"
        self.first = nn.Linear(n_inputs, n_hidden_nodes)
        self.middle = nn.Linear(n_hidden_nodes, n_hidden_nodes)
        self.last = nn.Linear(n_hidden_nodes, n_outputs)

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.2)

        ## build model from parts
        modules = []
        modules.append(self.first)

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        for _ in range(n_hidden_layers):
            modules.append(self.middle)
            modules.append(self.activation)
            modules.append(self.dropout)
        modules.append(self.last)

        self.model = nn.Sequential(*[m for m in modules])

    def forward(self, x):
        """
        use ModuleList directly to pass input through all layers of the MLP
        """
        return self.model(x)


class TransformerEncoder(nn.Module):
    """
    Implementation of the full vision transformer inspired by the encoder transformer from Attention is all you need
    """

    def __init__(self, config: DictConfig | ListConfig):
        super().__init__()

        self.n1 = nn.LayerNorm(config.vit.model_dim)
        self.n2 = nn.LayerNorm(config.vit.model_dim)
        self.mlp = MLP(
            n_hidden_layers=config.vit.mlp.n_hidden_layers,
            n_inputs=config.vit.model_dim,
            n_outputs=config.vit.model_dim,
            n_hidden_nodes=config.vit.mlp.n_hidden_nodes,
            activation=config.vit.mlp.activation,
        )
        self.mha = MultiheadAttention(
            n_total_nodes=config.vit.model_dim,
            n_heads=config.vit.transformer.n_heads,
        )

    def forward(self, x: torch.Tensor):
        """
        Process encoded images keeping skip-level connections from the paper
            - here x represents the encoded image from the vision encoder forward method

        """
        n1 = self.n1(x)
        attn = self.mha(n1)

        # intermediate results also include skip connection from the encoder
        middle = x + attn

        n2 = self.n2(middle)
        mlp_out = self.mlp(n2)

        final = middle + mlp_out

        return final


def regression_cutmix(
    images: torch.Tensor, targets: torch.Tensor, beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Method that implements the CutMix regularizer for regression tasks
        Much of this follows the implementation from the CutMix-PyTorch repo

    Args:
        images: torch.Tensor, Images to process
        targets: torch.Tensor, real value points to transfer (e.g., depth)

    Returns:
        transformed_images: torch.Tensor, Images with replaced regions
        transformed_targets: torch.Tensor, value maps with replaced regions
    """
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(images.size()[0]).to("mps")
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    return lam, images, target_a, target_b


def rand_bbox(size, lam):
    """
    random bounding box code from CutMix Pytorch library
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
