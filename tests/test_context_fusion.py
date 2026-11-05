
import torch

from models.context_fusion import ContextFusion


def test_context_fusion():
    model = ContextFusion(
        in_channels=128,
        embedding_dim=256,
        num_contexts=9,
    )
    model.eval()

    roi_features = torch.randn(18, 9, 128, 7, 7)

    with torch.no_grad():
        embeddings, weights = model(roi_features)

    print("Embeddings:", embeddings.shape)
    print("Attention weights:", weights.shape)

    assert embeddings.shape == (18, 256)
    assert weights.shape == (18, 9)

    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(weights).all()

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(18),
        atol=1e-5,
    )

    print("Context Fusion test PASSED")


if __name__ == "__main__":
    test_context_fusion()
