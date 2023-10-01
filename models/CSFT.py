from einops import rearrange
from os.path import join as pjoin
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import math
import ml_collections
import torch
import torch.nn as nn

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = LearnedPositionalEncoding(n_patches, config.hidden_size, n_patches)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = self.position_embeddings(x)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention_Before(nn.Module):
    def __init__(self, config):
        super(Attention_Before, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        return query_layer, key_layer, value_layer


class Attention_After(nn.Module):
    def __init__(self, config):
        super(Attention_After, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def forward(self, query_layer, key_layer, value_layer):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.hidden_size*4)
        self.fc2 = Linear(config.hidden_size*4, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CrossScaleFusionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=True):
        super(CrossScaleFusionTransformer, self).__init__()
        config = get_b16_config()
        self.img_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.patch_size = _pair(config.patches["size"])
        self.hidden_size = config.hidden_size

        self.ps2 = nn.PixelShuffle(2)
        self.ps4 = nn.PixelShuffle(4)
        out_c = config.hidden_size
        in_1 = config.hidden_size + config.hidden_size//4
        in_2 = config.hidden_size + config.hidden_size//4 + config.hidden_size//16
        self.down1 = Conv2d(in_channels=in_1, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.down2 = Conv2d(in_channels=in_2, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.down3 = Conv2d(in_channels=in_1, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.down4 = Conv2d(in_channels=in_2, out_channels=out_c, kernel_size=3, stride=1, padding=1)

        self.embeddings1 = Embeddings(config, img_size)
        self.embeddings2 = Embeddings(config, img_size//2)
        self.embeddings3 = Embeddings(config, img_size//4)

        self.attention_norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.SA_B1 = Attention_Before(config)
        self.SA_A1 = Attention_After(config)
        self.ffn_norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.Mlp1 = Mlp(config)

        self.attention_norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.SA_B2 = Attention_Before(config)
        self.SA_A2 = Attention_After(config)
        self.ffn_norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.Mlp2 = Mlp(config)

        self.attention_norm3 = LayerNorm(config.hidden_size, eps=1e-6)
        self.SA_B3 = Attention_Before(config)
        self.SA_A3 = Attention_After(config)
        self.ffn_norm3 = LayerNorm(config.hidden_size, eps=1e-6)
        self.Mlp3 = Mlp(config)

        self.attention_norm4 = LayerNorm(config.hidden_size, eps=1e-6)
        self.SA_B4 = Attention_Before(config)
        self.SA_A4 = Attention_After(config)
        self.ffn_norm4 = LayerNorm(config.hidden_size, eps=1e-6)
        self.Mlp4 = Mlp(config)

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, x2, x4):
        n = (self.img_size//self.patch_size[0])*(self.img_size//self.patch_size[1])  # number of patches
        embedding_output1 = self.embeddings1(x)
        embedding_output2 = self.embeddings2(x2)
        embedding_output3 = self.embeddings3(x4)
        # level 1
        attention_norm1 = self.attention_norm1(embedding_output1)
        q1, k1, v1 = self.SA_B1(attention_norm1)
        sa_a1 = self.SA_A1(q1, k1, v1)
        sa_a1 = sa_a1 + embedding_output1
        mlp1 = self.ffn_norm1(sa_a1)
        mlp1 = self.Mlp1(mlp1)
        mlp1 = mlp1 + sa_a1
        # level 2
        attention_norm2 = self.attention_norm2(mlp1)
        q2, k2, v2 = self.SA_B2(attention_norm2)
        sa_a2 = self.SA_A2(q2, k2, v2)
        sa_a2 = sa_a2 + mlp1
        mlp2 = self.ffn_norm2(sa_a2)
        mlp2 = self.Mlp2(mlp2)
        mlp2 = mlp2 + sa_a2

        attention_norm2 = self.attention_norm2(embedding_output2)
        q2, k2, v2 = self.SA_B2(attention_norm2)
        sa_a2 = self.SA_A2(q2, k2, v2)
        sa_a2 = sa_a2 + embedding_output2
        mlp5 = self.ffn_norm2(sa_a2)
        mlp5 = self.Mlp2(mlp5)
        mlp5 = mlp5 + sa_a2
        cat_mlp2 = rearrange(mlp2, "b (h w) d -> b d h w", h=int(n**0.5))
        cat_mlp5 = rearrange(mlp5, "b (h w) d -> b d h w", h=int((n//4)**0.5))
        cat_mlp5 = self.ps2(cat_mlp5)
        x2_1 = torch.cat((cat_mlp2, cat_mlp5), dim=1)
        x2_1 = self.down1(x2_1)
        x2_1 = rearrange(x2_1, "b d h w->b (h w) d")
        # level 3
        attention_norm3 = self.attention_norm3(x2_1)
        q3, k3, v3 = self.SA_B3(attention_norm3)
        sa_a3 = self.SA_A3(q3, k3, v3)
        sa_a3 = sa_a3 + x2_1
        mlp3 = self.ffn_norm3(sa_a3)
        mlp3 = self.Mlp3(mlp3)
        mlp3 = mlp3 + sa_a3

        attention_norm3 = self.attention_norm3(mlp5)
        q3, k3, v3 = self.SA_B3(attention_norm3)
        sa_a3 = self.SA_A3(q3, k3, v3)
        sa_a3 = sa_a3 + mlp5
        mlp6 = self.ffn_norm3(sa_a3)
        mlp6 = self.Mlp3(mlp6)
        mlp6 = mlp6 + sa_a3

        attention_norm3 = self.attention_norm3(embedding_output3)
        q3, k3, v3 = self.SA_B3(attention_norm3)
        sa_a3 = self.SA_A3(q3, k3, v3)
        sa_a3 = sa_a3 + embedding_output3
        mlp8 = self.ffn_norm3(sa_a3)
        mlp8 = self.Mlp3(mlp8)
        mlp8 = mlp8 + sa_a3
        cat_mlp3 = rearrange(mlp3, "b (h w) d -> b d h w", h=int(n**0.5))
        cat_mlp6 = rearrange(mlp6, "b (h w) d -> b d h w", h=int((n//4)**0.5))
        cat_mlp6 = self.ps2(cat_mlp6)
        cat_mlp8 = rearrange(mlp8, "b (h w) d -> b d h w", h=int((n//16)**0.5))
        cat_mlp8 = self.ps4(cat_mlp8)
        x3_1 = torch.cat((cat_mlp3, cat_mlp6, cat_mlp8), dim=1)
        x3_1 = self.down2(x3_1)
        x3_1 = rearrange(x3_1, "b d h w->b (h w) d")
        cat_mlp6 = rearrange(mlp6, "b (h w) d -> b d h w", h=int((n//4)**0.5))
        cat_mlp8 = rearrange(mlp8, "b (h w) d -> b d h w", h=int((n//16) ** 0.5))
        cat_mlp8 = self.ps2(cat_mlp8)
        x3_2 = torch.cat((cat_mlp6, cat_mlp8), dim=1)
        x3_2 = self.down3(x3_2)
        x3_2 = rearrange(x3_2, "b d h w->b (h w) d")
        # level 4
        attention_norm4 = self.attention_norm4(x3_1)
        q4, k4, v4 = self.SA_B4(attention_norm4)
        sa_a4 = self.SA_A4(q4, k4, v4)
        sa_a4 = sa_a4 + x3_1
        mlp4 = self.ffn_norm4(sa_a4)
        mlp4 = self.Mlp4(mlp4)
        mlp4 = mlp4 + sa_a4

        attention_norm4 = self.attention_norm4(x3_2)
        q4, k4, v4 = self.SA_B4(attention_norm4)
        sa_a4 = self.SA_A4(q4, k4, v4)
        sa_a4 = sa_a4 + x3_2
        mlp7 = self.ffn_norm4(sa_a4)
        mlp7 = self.Mlp4(mlp7)
        mlp7 = mlp7 + sa_a4

        attention_norm4 = self.attention_norm4(mlp8)
        q4, k4, v4 = self.SA_B4(attention_norm4)
        sa_a4 = self.SA_A4(q4, k4, v4)
        sa_a4 = sa_a4 + mlp8
        mlp9 = self.ffn_norm4(sa_a4)
        mlp9 = self.Mlp4(mlp9)
        mlp9 = mlp9 + sa_a4
        cat_mlp4 = rearrange(mlp4, "b (h w) d -> b d h w", h=int(n**0.5))
        cat_mlp7 = rearrange(mlp7, "b (h w) d -> b d h w", h=int((n//4) ** 0.5))
        cat_mlp7 = self.ps2(cat_mlp7)
        cat_mlp9 = rearrange(mlp9, "b (h w) d -> b d h w", h=int((n//16)**0.5))
        cat_mlp9 = self.ps4(cat_mlp9)
        x = torch.cat((cat_mlp4, cat_mlp7, cat_mlp9), dim=1)
        x = self.down4(x)
        x = rearrange(x, "b d h w->b (h w) d")

        x = x.mean(dim=1)
        encoded = self.encoder_norm(x)
        logits = self.head(encoded)

        return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.ones_(self.head.weight)
                nn.init.ones_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.embeddings1.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embeddings1.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings2.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embeddings2.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings3.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embeddings3.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            # self.embeddings.cls_token.copy_(np2th(weights["cls"]))

            self.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            # posemb_new = self.embeddings.position_embeddings
            # if posemb.size() == posemb_new.size():
            #     self.embeddings.position_embeddings.copy_(posemb)

            #----------------------------------layer 1------------------------------------------
            ROOT = f"Transformer/encoderblock_0"
            query_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "kernel")]).view(self.hidden_size,
                                                                                               self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "bias")]).view(-1)

            self.SA_B1.query.weight.copy_(query_weight)
            self.SA_B1.key.weight.copy_(key_weight)
            self.SA_B1.value.weight.copy_(value_weight)
            self.SA_A1.out.weight.copy_(out_weight)

            self.SA_B1.query.bias.copy_(query_bias)
            self.SA_B1.key.bias.copy_(key_bias)
            self.SA_B1.value.bias.copy_(value_bias)
            self.SA_A1.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "bias")]).t()

            self.Mlp1.fc1.weight.copy_(mlp_weight_0)
            self.Mlp1.fc2.weight.copy_(mlp_weight_1)
            self.Mlp1.fc1.bias.copy_(mlp_bias_0)
            self.Mlp1.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm1.weight.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "scale")]))
            self.attention_norm1.bias.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "bias")]))
            self.ffn_norm1.weight.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "scale")]))
            self.ffn_norm1.bias.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "bias")]))

            # ----------------------------------layer 2------------------------------------------
            ROOT = f"Transformer/encoderblock_1"
            query_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "kernel")]).view(self.hidden_size,
                                                                                               self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "bias")]).view(-1)

            self.SA_B2.query.weight.copy_(query_weight)
            self.SA_B2.key.weight.copy_(key_weight)
            self.SA_B2.value.weight.copy_(value_weight)
            self.SA_A2.out.weight.copy_(out_weight)

            self.SA_B2.query.bias.copy_(query_bias)
            self.SA_B2.key.bias.copy_(key_bias)
            self.SA_B2.value.bias.copy_(value_bias)
            self.SA_A2.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "bias")]).t()

            self.Mlp2.fc1.weight.copy_(mlp_weight_0)
            self.Mlp2.fc2.weight.copy_(mlp_weight_1)
            self.Mlp2.fc1.bias.copy_(mlp_bias_0)
            self.Mlp2.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm2.weight.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "scale")]))
            self.attention_norm2.bias.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "bias")]))
            self.ffn_norm2.weight.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "scale")]))
            self.ffn_norm2.bias.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "bias")]))

            # ----------------------------------layer 3------------------------------------------
            ROOT = f"Transformer/encoderblock_2"
            query_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "kernel")]).view(self.hidden_size,
                                                                                               self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "bias")]).view(-1)

            self.SA_B3.query.weight.copy_(query_weight)
            self.SA_B3.key.weight.copy_(key_weight)
            self.SA_B3.value.weight.copy_(value_weight)
            self.SA_A3.out.weight.copy_(out_weight)

            self.SA_B3.query.bias.copy_(query_bias)
            self.SA_B3.key.bias.copy_(key_bias)
            self.SA_B3.value.bias.copy_(value_bias)
            self.SA_A3.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "bias")]).t()

            self.Mlp3.fc1.weight.copy_(mlp_weight_0)
            self.Mlp3.fc2.weight.copy_(mlp_weight_1)
            self.Mlp3.fc1.bias.copy_(mlp_bias_0)
            self.Mlp3.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm3.weight.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "scale")]))
            self.attention_norm3.bias.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "bias")]))
            self.ffn_norm3.weight.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "scale")]))
            self.ffn_norm3.bias.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "bias")]))

            # # ----------------------------------layer 4------------------------------------------
            ROOT = f"Transformer/encoderblock_3"
            query_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "kernel")]).view(self.hidden_size,
                                                                                               self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "kernel")]).view(self.hidden_size,
                                                                                                 self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_Q + "/" + "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_K + "/" + "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_V + "/" + "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT + "/" + ATTENTION_OUT + "/" + "bias")]).view(-1)

            self.SA_B4.query.weight.copy_(query_weight)
            self.SA_B4.key.weight.copy_(key_weight)
            self.SA_B4.value.weight.copy_(value_weight)
            self.SA_A4.out.weight.copy_(out_weight)

            self.SA_B4.query.bias.copy_(query_bias)
            self.SA_B4.key.bias.copy_(key_bias)
            self.SA_B4.value.bias.copy_(value_bias)
            self.SA_A4.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT + "/" + FC_0 + "/" + "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT + "/" + FC_1 + "/" + "bias")]).t()

            self.Mlp4.fc1.weight.copy_(mlp_weight_0)
            self.Mlp4.fc2.weight.copy_(mlp_weight_1)
            self.Mlp4.fc1.bias.copy_(mlp_bias_0)
            self.Mlp4.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm4.weight.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "scale")]))
            self.attention_norm4.bias.copy_(np2th(weights[pjoin(ROOT + "/" + ATTENTION_NORM + "/" + "bias")]))
            self.ffn_norm4.weight.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "scale")]))
            self.ffn_norm4.bias.copy_(np2th(weights[pjoin(ROOT + "/" + MLP_NORM + "/" + "bias")]))
