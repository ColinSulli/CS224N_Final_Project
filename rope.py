import torch


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###

    # dimension dim/2
    theta_i = torch.tensor(
        [10000 ** (-2 * (i - 1) / dim) for i in range(1, dim // 2 + 1)]
    )

    # create the index for positions
    positions = torch.tensor([i for i in range(0, max_positions)])

    # take outer product
    position_theta = torch.outer(positions, theta_i)

    rope_cache = torch.stack(
        [torch.cos(position_theta), torch.sin(position_theta)], axis=2
    )

    assert rope_cache.shape == (max_positions, dim // 2, 2)

    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### YOUR CODE HERE ###
    # x is (B, nh, T, hs)
    # rope cache is of shape (T, hs//2, 2)

    B, nh, T, hs = x.shape

    truncated_rope_cache = rope_cache[None, None, :T, :, :]

    x = x.view(B, nh, T, hs // 2, 2)

    # apply the rope to the query and key
    rope = torch.view_as_complex(truncated_rope_cache) * torch.view_as_complex(x)
    rope = torch.view_as_real(rope)

    rotated_x = rope.view(B, nh, T, hs)

    ### END YOUR CODE ###
    return rotated_x
