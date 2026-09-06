import torch

from diffbir.model.attention import SDPCrossAttention
from diffbir.model.identity import IdentityTokenProjector


def main() -> None:
    projector = IdentityTokenProjector(
        embedding_dim=8,
        hidden_dim=16,
        context_dim=12,
        num_tokens=4,
    )
    tokens = projector(torch.randn(2, 8))
    assert tokens.shape == (2, 4, 12)

    attention = SDPCrossAttention(
        query_dim=16,
        context_dim=12,
        heads=2,
        dim_head=8,
        identity_context_dim=12,
    ).eval()
    x = torch.randn(2, 5, 16)
    text = torch.randn(2, 7, 12)
    identity = torch.randn(2, 4, 12)

    text_only = attention(x, text)
    identity_disabled = attention(
        x,
        {
            "text": text,
            "identity": identity,
            "identity_scale": torch.zeros(2),
        },
    )
    torch.testing.assert_close(identity_disabled, text_only, atol=0, rtol=0)

    for parameter in attention.parameters():
        parameter.requires_grad = False
    attention.to_k_id.weight.requires_grad = True
    attention.to_v_id.weight.requires_grad = True
    torch.nn.init.normal_(attention.to_v_id.weight, std=0.01)
    output = attention(
        x,
        {
            "text": text,
            "identity": identity,
            "identity_scale": torch.ones(2),
        },
    )
    output.square().mean().backward()

    assert attention.to_k_id.weight.grad is not None
    assert attention.to_v_id.weight.grad is not None
    assert attention.to_q.weight.grad is None
    print("identity attention smoke test passed")


if __name__ == "__main__":
    main()
