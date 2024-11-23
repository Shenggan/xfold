from typing import Optional

import einops
import torch
import torch.nn as nn


def dot_product_attention(q, k, v, mask=None, bias=None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))

    if bias is not None:
        logits += bias

    if mask is not None:
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, v)


class GridSelfAttention(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        bias = torch.squeeze(bias, 0)

        weighted_avg = dot_product_attention(q, k, v,
                                             mask=mask,
                                             bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair, mask):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        mask = mask[:, None, None, :].to(dtype=torch.bool)

        pair = self._attention(pair, mask, nonbatched_bias)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class AttentionPairBias(nn.Module):
    def __init__(self, c_single: int = 384, num_head: int = 16, use_single_cond: bool = False) -> None:
        super(AttentionPairBias, self).__init__()

        self.c_single = c_single
        self.num_head = num_head

        self.qkv_dim = self.c_single // self.num_head
        self.use_single_cond = use_single_cond

        if self.use_single_cond is False:
            self.layer_norm = nn.LayerNorm(self.c_single)

        self.q_projection = nn.Linear(self.c_single, self.c_single, bias=True)
        self.k_projection = nn.Linear(self.c_single, self.c_single, bias=False)
        self.v_projection = nn.Linear(self.c_single, self.c_single, bias=False)

        self.gating_query = nn.Linear(self.c_single, self.c_single, bias=False)
        self.transition2 = nn.Linear(self.c_single, self.c_single, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pair_logits: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """
        bias = (1e9 * (mask.to(dtype=x.dtype) - 1.0))[..., None, None, :]

        x = self.layer_norm(x)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'n (h c) -> n h c', h=self.num_head), [q, k, v])

        logits = torch.einsum('...qhc,...khc->...hqk',
                              q * self.qkv_dim ** (-0.5), k) + bias
        if pair_logits is not None:
            logits += pair_logits  # (num_heads, seq_len, seq_len)

        weights = torch.softmax(logits, dim=-1)
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)
        return self.transition2(weighted_avg)


class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = nn.LayerNorm(self.c_msa)
        self.pair_norm = nn.LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)