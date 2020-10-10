#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement causally masked linear attention."""

import torch
from fast_transformers.attention_registry import AttentionRegistry, Optional, Callable
from fast_transformers.causal_product import causal_dot_product
from torch.nn import Module
import numpy as np


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def causal_linear(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0, 2, 1, 3).contiguous()


#
# class CausalLinearRelativeAttention(Module):
#     """Implement causally masked attention using dot product of feature maps in
#     O(N D^2) complexity.
#
#     See fast_transformers.attention.linear_attention.LinearAttention for the
#     general concept of replacing the softmax with feature maps. In addition to
#     that, we also make use of the fact that causal masking is a triangular mask
#     which allows us to apply the masking and still compute the attention in O(N
#     D^2) complexity.
#
#     Arguments
#     ---------
#         feature_map: callable, a callable that applies the feature map to the
#                      last dimension of a tensor (default: elu(x)+1)
#         eps: float, a small number to ensure the numerical stability of the
#              denominator (default: 1e-6)
#     """
#
#     def __init__(self, feature_map=None, eps=1e-6):
#         super(CausalLinearRelativeAttention, self).__init__()
#         self.feature_map = feature_map or elu_feature_map
#         self.eps = eps
#         self.lambdas = torch.nn.Parameter(np.pi * torch.linspace(start=-64, end=64,
#                                                             steps=64)
#                                      )
#
#     def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         # Apply the feature map to the queries and keys
#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)
#
#         # ====== Original Term =======
#         # Apply the key padding mask and make sure the attn_mask is a
#         # lower triangular causal mask
#         if not attn_mask.lower_triangular:
#             raise RuntimeError(("CausalLinearAttention only supports full "
#                                 "lower triangular masks"))
#         K = K * key_lengths.float_matrix[:, :, None, None]
#
#         # Compute the normalizers
#         Z = (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
#
#         # Compute the unnormalized result
#         V = causal_linear(
#             Q,
#             K,
#             values
#         )
#
#         # Relative attention
#         # ====== Common terms =======
#         batch_size, length, num_heads, feature_dim = queries.size()
#         Delta = torch.arange(start=0, end=length, step=1).cuda()
#
#         # fixed
#         # lambdas = np.pi * torch.linspace(start=-64, end=64, steps=feature_dim).cuda()
#         # or leart
#         lambdas = self.lambdas
#
#
#         Delta = Delta.unsqueeze(1) * lambdas.unsqueeze(0)
#         Delta = Delta.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, num_heads, 1)
#
#         sin2 = lambda x: torch.sin(x) ** 2
#         cos2 = lambda x: torch.cos(x) ** 2
#
#         sin_Q = sin2(q2)
#         sin_Delta = torch.sin(Delta)
#         cos_Delta = torch.cos(Delta)
#
#         # ====== First Term =======
#         Q1 = sin_Q * sin_Delta ** 2
#         K1 = cos_Delta ** 2
#
#         Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)) + self.eps)
#
#         # Compute the unnormalized result
#         V1 = causal_linear(
#             Q1,
#             K1,
#             values
#         )
#
#         # ====== Second Term =======
#         Q2 = sin_Q * cos_Delta ** 2
#         K2 = sin_Delta ** 2
#
#         Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)) + self.eps)
#
#         # Compute the unnormalized result
#         V2 = causal_linear(
#             Q2,
#             K2,
#             values
#         )
#
#         # ====== Third Term =======
#         # TODO save compute
#         Q3 = sin_Q * cos_Delta * sin_Delta
#         K3 = sin_Delta * cos_Delta
#
#         Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)) + self.eps)
#
#         # Compute the unnormalized result
#         V3 = causal_linear(
#             Q3,
#             K3,
#             values
#         )
#
#         numerator = V + V1 + V2 - 2 * V3
#         # TODO only one epsilon
#         denominator = Z + Z1 + Z2 - 2 * Z3
#
#         res = numerator / denominator[:, :, :, None]
#         return res

# Product form sin**2
# class CausalLinearRelativeAttention(Module):
#     """Implement causally masked attention using dot product of feature maps in
#     O(N D^2) complexity.
#
#     See fast_transformers.attention.linear_attention.LinearAttention for the
#     general concept of replacing the softmax with feature maps. In addition to
#     that, we also make use of the fact that causal masking is a triangular mask
#     which allows us to apply the masking and still compute the attention in O(N
#     D^2) complexity.
#
#     Arguments
#     ---------
#         feature_map: callable, a callable that applies the feature map to the
#                      last dimension of a tensor (default: elu(x)+1)
#         eps: float, a small number to ensure the numerical stability of the
#              denominator (default: 1e-6)
#     """
#
#     def __init__(self, feature_map=None, eps=1e-6):
#         super(CausalLinearRelativeAttention, self).__init__()
#         self.feature_map = feature_map or elu_feature_map
#         self.eps = eps
#         # self.lambdas = torch.nn.Parameter(np.pi * torch.linspace(start=0, end=1024,
#         #                                                     steps=64) / 1024
#         #                              )
#         self.lambdas = torch.nn.Parameter(
#             np.pi * torch.rand(8, 64) / (1024 )
#         )
#
#         # self.mix = torch.nn.Linear(64, 64)
#
#
#
#
#     def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         # self.feature_map = lambda x: self.mix(x)**2
#         # Apply the feature map to the queries and keys
#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)
#
#         # ====== Original Term =======
#         # Apply the key padding mask and make sure the attn_mask is a
#         # lower triangular causal mask
#         if not attn_mask.lower_triangular:
#             raise RuntimeError(("CausalLinearAttention only supports full "
#                                 "lower triangular masks"))
#         K = K * key_lengths.float_matrix[:, :, None, None]
#
#         # Relative attention
#         # ====== Common terms =======
#         batch_size, length, num_heads, feature_dim = queries.size()
#         Delta = torch.arange(start=0, end=length, step=1).cuda()
#
#         # fixed
#         # lambdas = np.pi * torch.linspace(start=-64, end=64, steps=feature_dim).cuda()
#         # or leart
#         lambdas = self.lambdas
#
#         # if Delta has different values for each head
#         Delta = Delta.unsqueeze(1).unsqueeze(1) * lambdas.unsqueeze(0)
#         Delta = Delta.unsqueeze(0).repeat(batch_size, 1, 1, 1)
#
#         # otherwise
#         # Delta = Delta.unsqueeze(1) * lambdas.unsqueeze(0)
#         # Delta = Delta.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, num_heads, 1)
#
#         sin2 = lambda x: torch.sin(x) ** 2
#         cos2 = lambda x: torch.cos(x) ** 2
#
#         sin_Q = sin2(q2)
#         sin_Delta = torch.sin(Delta)
#         cos_Delta = torch.cos(Delta)
#
#         # ====== First Term =======
#         Q1 = sin_Q * sin_Delta ** 2 * Q
#         K1 = cos_Delta ** 2 * K
#
#         Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)))
#
#         # Compute the unnormalized result
#         V1 = causal_linear(
#             Q1,
#             K1,
#             values
#         )
#
#         # ====== Second Term =======
#         Q2 = sin_Q * cos_Delta ** 2 * Q
#         K2 = sin_Delta ** 2 * K
#
#         Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)))
#
#         # Compute the unnormalized result
#         V2 = causal_linear(
#             Q2,
#             K2,
#             values
#         )
#
#         # ====== Third Term =======
#         # TODO save compute
#         Q3 = sin_Q * cos_Delta * sin_Delta * Q
#         K3 = sin_Delta * cos_Delta * K
#
#         Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)))
#
#         # Compute the unnormalized result
#         V3 = causal_linear(
#             Q3,
#             K3,
#             values
#         )
#
#         numerator = V1 + V2 - 2 * V3
#         # TODO only one epsilon
#         denominator = Z1 + Z2 - 2 * Z3 + self.eps
#
#         res = numerator / denominator[:, :, :, None]
#         return res

# with offsets, additive, 1 + sin
# class CausalLinearRelativeAttention(Module):
#     """Implement causally masked attention using dot product of feature maps in
#     O(N D^2) complexity.
#
#     See fast_transformers.attention.linear_attention.LinearAttention for the
#     general concept of replacing the softmax with feature maps. In addition to
#     that, we also make use of the fact that causal masking is a triangular mask
#     which allows us to apply the masking and still compute the attention in O(N
#     D^2) complexity.
#
#     Arguments
#     ---------
#         feature_map: callable, a callable that applies the feature map to the
#                      last dimension of a tensor (default: elu(x)+1)
#         eps: float, a small number to ensure the numerical stability of the
#              denominator (default: 1e-6)
#     """
#
#     def __init__(self, feature_map=None, eps=1e-6):
#         super(CausalLinearRelativeAttention, self).__init__()
#         self.feature_map = feature_map or elu_feature_map
#         self.eps = eps
#         # self.lambdas = torch.nn.Parameter(np.pi * torch.linspace(start=0, end=1024,
#         #                                                     steps=64) / 1024
#         #                              )
#         self.lambdas = torch.nn.Parameter(
#             2 * np.pi * torch.rand(8, 64)
#         )
#
#         self.offsets = torch.nn.Parameter(
#             2 * np.pi * torch.rand(8, 64)
#         )
#
#
#     def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         # Apply the feature map to the queries and keys
#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)
#
#         # ====== Original Term =======
#         # Apply the key padding mask and make sure the attn_mask is a
#         # lower triangular causal mask
#         if not attn_mask.lower_triangular:
#             raise RuntimeError(("CausalLinearAttention only supports full "
#                                 "lower triangular masks"))
#         K = K * key_lengths.float_matrix[:, :, None, None]
#
#         V = causal_linear(
#             Q,
#             K,
#             values
#         )
#         Z = (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)))
#         # Relative attention
#         # ====== Common terms =======
#         batch_size, length, num_heads, feature_dim = queries.size()
#         Delta = torch.arange(start=0, end=length, step=1).cuda() / length
#
#         lambdas = self.lambdas
#         offsets = self.offsets.unsqueeze(0).unsqueeze(0).repeat(batch_size,
#                                                                 length,
#                                                                 1,
#                                                                 1
#                                                                 )
#
#         # if lambdas has different values for each head
#         Delta = Delta.unsqueeze(1).unsqueeze(1) * lambdas.unsqueeze(0)
#         Delta = Delta.unsqueeze(0).repeat(batch_size, 1, 1, 1)
#
#
#         Q_term = (1 + torch.sin(q2 + offsets))
#
#         # ====== First Term =======
#         Q1 = Q_term
#         K1 = torch.ones_like(Q1)
#         Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)))
#
#         # Compute the unnormalized result
#         V1 = causal_linear(
#             Q1,
#             K1,
#             values
#         )
#
#         # ====== Second Term =======
#         Q2 = Q_term * torch.sin(Delta)
#         K2 = torch.cos(Delta + offsets)
#
#         Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)))
#
#         # Compute the unnormalized result
#         V2 = causal_linear(
#             Q2,
#             K2,
#             values
#         )
#
#         # ====== Third Term =======
#         Q3 = Q_term * torch.cos(Delta)
#         K3 = torch.sin(Delta - offsets)
#
#         Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)))
#
#         # Compute the unnormalized result
#         V3 = causal_linear(
#             Q3,
#             K3,
#             values
#         )
#
#         numerator = V + V1 + V2 - V3
#         denominator = Z + Z1 + Z2 - Z3 + self.eps
#
#         res = numerator / denominator[:, :, :, None]
#         return res

# random sin^2 additive
# class CausalLinearRelativeAttention(Module):
#     """Implement causally masked attention using dot product of feature maps in
#     O(N D^2) complexity.
#
#     See fast_transformers.attention.linear_attention.LinearAttention for the
#     general concept of replacing the softmax with feature maps. In addition to
#     that, we also make use of the fact that causal masking is a triangular mask
#     which allows us to apply the masking and still compute the attention in O(N
#     D^2) complexity.
#
#     Arguments
#     ---------
#         feature_map: callable, a callable that applies the feature map to the
#                      last dimension of a tensor (default: elu(x)+1)
#         eps: float, a small number to ensure the numerical stability of the
#              denominator (default: 1e-6)
#     """
#
#     def __init__(self, feature_map=None, eps=1e-6):
#         super(CausalLinearRelativeAttention, self).__init__()
#         self.feature_map = feature_map or elu_feature_map
#         self.eps = eps
#
#     def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         # Apply the feature map to the queries and keys
#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)
#
#         # ====== Original Term =======
#         # Apply the key padding mask and make sure the attn_mask is a
#         # lower triangular causal mask
#         if not attn_mask.lower_triangular:
#             raise RuntimeError(("CausalLinearAttention only supports full "
#                                 "lower triangular masks"))
#         K = K * key_lengths.float_matrix[:, :, None, None]
#
#         V = causal_linear(
#             Q,
#             K,
#             values
#         )
#         Z = (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)))
#         # Relative attention
#         # ====== Common terms =======
#         batch_size, length, num_heads, feature_dim = queries.size()
#         Delta = torch.arange(start=0, end=length, step=1).float().cuda() / length
#
#         Delta = Delta.unsqueeze(1).unsqueeze(1)
#         Delta = Delta.unsqueeze(0).repeat(batch_size, 1, num_heads, feature_dim)
#
#         omega = torch.randn(num_heads, feature_dim, feature_dim).to('cuda')
#
#         def phi(f, q):
#             return f(torch.einsum('nlhi,hij->nlhj', q, omega))
#
#         Q_term = phi(lambda t: torch.sin(t) ** 2,
#                      q2)
#
#         # ====== First Term =======
#         Q1 = Q_term * phi(lambda t: torch.sin(t) ** 2,
#                           Delta)
#         K1 = phi(lambda t: torch.cos(t) ** 2,
#                  Delta
#                  )
#         Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)))
#
#         # Compute the unnormalized result
#         V1 = causal_linear(
#             Q1,
#             K1,
#             values
#         )
#
#         # ====== Second Term =======
#         Q2 = Q_term * phi(lambda t: torch.cos(t) ** 2,
#                           Delta)
#         K2 = phi(lambda t: torch.sin(t) ** 2,
#                  Delta
#                  )
#
#         Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)))
#
#         # Compute the unnormalized result
#         V2 = causal_linear(
#             Q2,
#             K2,
#             values
#         )
#
#         # ====== Third Term =======
#         Q3 = Q_term * phi(lambda t: torch.sin(t) * torch.cos(t),
#                           Delta)
#         K3 = phi(lambda t: torch.sin(t) * torch.cos(t),
#                  Delta
#                  )
#
#         Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)))
#
#         # Compute the unnormalized result
#         V3 = causal_linear(
#             Q3,
#             K3,
#             values
#         )
#
#         numerator = V + V1 + V2 - 2 * V3
#         denominator = Z + Z1 + Z2 - 2 * Z3 + self.eps
#
#         res = numerator / denominator[:, :, :, None]
#         return res

class CausalLinearRelativeAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """

    def __init__(self, feature_map=None, eps=1e-6):
        super(CausalLinearRelativeAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # ====== Original Term =======
        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(("CausalLinearAttention only supports full "
                                "lower triangular masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Relative attention
        # ====== Common terms =======
        batch_size, length, num_heads, feature_dim = queries.size()
        Delta = torch.arange(start=0, end=length, step=1).float().cuda() / length

        Delta = Delta.unsqueeze(1).unsqueeze(1)
        Delta = Delta.unsqueeze(0).repeat(batch_size, 1, num_heads, feature_dim)

        omega = torch.randn(num_heads, feature_dim, feature_dim).to('cuda')

        def phi(f, q):
            return f(torch.einsum('nlhi,hij->nlhj', q, omega))

        Q_term = phi(lambda t: torch.sin(t) ** 2,
                     q2) * Q

        # ====== First Term =======
        Q1 = Q_term * phi(lambda t: torch.sin(t) ** 2,
                          Delta)
        K1 = phi(lambda t: torch.cos(t) ** 2,
                 Delta
                 ) * K
        Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)))

        # Compute the unnormalized result
        V1 = causal_linear(
            Q1,
            K1,
            values
        )

        # ====== Second Term =======
        Q2 = Q_term * phi(lambda t: torch.cos(t) ** 2,
                          Delta)
        K2 = phi(lambda t: torch.sin(t) ** 2,
                 Delta
                 ) * K

        Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)))

        # Compute the unnormalized result
        V2 = causal_linear(
            Q2,
            K2,
            values
        )

        # ====== Third Term =======
        Q3 = Q_term * phi(lambda t: torch.sin(t) * torch.cos(t),
                          Delta)
        K3 = phi(lambda t: torch.sin(t) * torch.cos(t),
                 Delta
                 ) * K

        Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)))

        # Compute the unnormalized result
        V3 = causal_linear(
            Q3,
            K3,
            values
        )

        numerator = V1 + V2 - 2 * V3
        denominator = Z1 + Z2 - 2 * Z3 + self.eps

        res = numerator / denominator[:, :, :, None]
        return res


# with offsets, additive, 1 + sin
# class CausalLinearRelativeAttention(Module):
#     """Implement causally masked attention using dot product of feature maps in
#     O(N D^2) complexity.
#
#     See fast_transformers.attention.linear_attention.LinearAttention for the
#     general concept of replacing the softmax with feature maps. In addition to
#     that, we also make use of the fact that causal masking is a triangular mask
#     which allows us to apply the masking and still compute the attention in O(N
#     D^2) complexity.
#
#     Arguments
#     ---------
#         feature_map: callable, a callable that applies the feature map to the
#                      last dimension of a tensor (default: elu(x)+1)
#         eps: float, a small number to ensure the numerical stability of the
#              denominator (default: 1e-6)
#     """
#
#     def __init__(self, feature_map=None, eps=1e-6):
#         super(CausalLinearRelativeAttention, self).__init__()
#         self.feature_map = feature_map or elu_feature_map
#         self.eps = eps
#         # self.lambdas = torch.nn.Parameter(np.pi * torch.linspace(start=0, end=1024,
#         #                                                     steps=64) / 1024
#         #                              )
#         self.lambdas = torch.nn.Parameter(
#             2 * np.pi * torch.rand(8, 64)
#         )
#
#         self.offsets = torch.nn.Parameter(
#             2 * np.pi * torch.rand(8, 64)
#         )
#
#
#     def forward(self, queries, q2, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         # Apply the feature map to the queries and keys
#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)
#
#         # ====== Original Term =======
#         # Apply the key padding mask and make sure the attn_mask is a
#         # lower triangular causal mask
#         if not attn_mask.lower_triangular:
#             raise RuntimeError(("CausalLinearAttention only supports full "
#                                 "lower triangular masks"))
#         K = K * key_lengths.float_matrix[:, :, None, None]
#
#         # Compute the unnormalized result
#         V = causal_linear(
#             Q,
#             K,
#             values
#         )
#
#         # Relative attention
#         # ====== Common terms =======
#         batch_size, length, num_heads, feature_dim = queries.size()
#         Delta = torch.arange(start=0, end=length, step=1).cuda() / length
#
#         lambdas = self.lambdas
#         offsets = self.offsets.unsqueeze(0).unsqueeze(0).repeat(batch_size,
#                                                                 length,
#                                                                 1,
#                                                                 1
#                                                                 )
#
#         # if lambdas has different values for each head
#         Delta = Delta.unsqueeze(1).unsqueeze(1) * lambdas.unsqueeze(0)
#         Delta = Delta.unsqueeze(0).repeat(batch_size, 1, 1, 1)
#
#
#         Q_term = 1 + torch.sin(q2 + offsets)
#
#         # ====== First Term =======
#         Q1 = Q_term
#         K1 = torch.ones_like(Q1)
#
#         Z1 = (torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)))
#
#         # Compute the unnormalized result
#         V1 = causal_linear(
#             Q1,
#             K1,
#             values
#         )
#
#         # ====== Second Term =======
#         Q2 = Q_term * torch.sin(Delta)
#         K2 = torch.cos(Delta + offsets) * K
#
#         Z2 = (torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)))
#
#         # Compute the unnormalized result
#         V2 = causal_linear(
#             Q2,
#             K2,
#             values
#         )
#
#         # ====== Third Term =======
#         # TODO save compute
#         Q3 = Q_term * torch.cos(Delta)
#         K3 = torch.sin(Delta - offsets) * K
#
#         Z3 = (torch.einsum("nlhi,nlhi->nlh", Q3, K3.cumsum(1)))
#
#         # Compute the unnormalized result
#         V3 = causal_linear(
#             Q3,
#             K3,
#             values
#         )
#
#         numerator = V1 + V2 - V3
#         denominator = Z1 + Z2 - Z3 + self.eps
#
#         res = numerator / denominator[:, :, :, None]
#         return res


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "causal-relative-linear", CausalLinearRelativeAttention,
    [("feature_map", Optional(Callable))]
)
