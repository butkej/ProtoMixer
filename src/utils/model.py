## MODELS
#########
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision


# PROTOMIXER

# gradient reversal layer
from torch.autograd import Function

"""
lambda: learning rate
Propagate with flat feature map in forward propagation
Propagate with negating gradients of weight in backpropagation
"""


class AdaptiveGradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam, attention):
        ctx.lam = lam
        ctx.attention = attention
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.attention != 0:
            attention = ctx.attention.squeeze(0)  # delete 1st dim [1,100] -> [100
            max_attention = torch.max(attention)
            adaptive_attention = max_attention - attention
            adaptive_attention = adaptive_attention.unsqueeze(2)
            # adaptive_attention = torch.transpose(adaptive_attention, 0, 2)
            output = grad_output.neg() * ctx.lam
            adaptive_output = adaptive_attention * output
        else:
            adaptive_output = grad_output.neg() * ctx.lam
        return adaptive_output, None, None


class TokenMixer(nn.Module):
    # MLP-Mixer like
    def __init__(
        self, token_dim: int, num_tokens: int, hidden_dim: int, dropout: float
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(token_dim)
        self.linear = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # print(x.shape)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.linear(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        x = x + residual
        return x


class ChannelMixer(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.layer_norm = nn.LayerNorm(token_dim)
        self.linear = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if len(x) == 2:
            x, attention = x
            x = x + self.linear(self.layer_norm(x))
            return x, attention
        else:
            x = x + self.linear(self.layer_norm(x))
            return x


class ProtoMixer(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pool: str,
        domain_num: int,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.domain_num = domain_num

        self.mixer = nn.Sequential(
            *(
                [
                    TokenMixer(token_dim, num_tokens, hidden_dim, dropout),
                    ChannelMixer(token_dim, hidden_dim, dropout),
                ]
                * num_layers
            )
        )

        if self.domain_num > 0:
            self.domain_predictor = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 4096),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4096, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, domain_num),
                nn.Dropout(dropout),
            )

        self.to_latent = nn.Identity()
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, num_classes)
        )

    def forward(self, x, mode: str, DA_rate: float):
        attention = 0
        x = self.mixer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        result = self.classifier_head(x)

        if mode == "train":
            if self.domain_num > 0:
                # input to gradient reversal layer
                adapGR_features = AdaptiveGradReverse.apply(x, DA_rate, attention)
                # domain predictor
                domain_prob = self.domain_predictor(adapGR_features)
                return result, domain_prob, attention
            else:
                return result, _, attention

        elif mode == "test":
            return result, attention


# ABMIL - classic Ilse gated attention MIL


class ABMIL(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ABMIL, self).__init__()
        self.L = input_size
        self.D = 256
        self.K = 1

        self.attention_a = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L, num_classes)

    def forward(self, x):
        H = x
        H = H.squeeze(0)
        a = self.attention_a(H)  # NxK
        b = self.attention_b(H)
        A = self.attention_weights(a * b)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob, A


# CLAM


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes"""

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem"""

    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
    ):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device=device)
        n_targets = self.create_negative_targets(self.k_sample, device=device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device=device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        device = h.device
        h = h.squeeze(0)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        # return logits, Y_prob, Y_hat, A_raw # original
        return logits, A_raw
