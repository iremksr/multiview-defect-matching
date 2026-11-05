
import torch

from models.multiview_transformer import (
    MultiViewTransformer,
)


def create_test_data():
    torch.manual_seed(42)

    counts = [18, 7, 0]
    total = sum(counts)

    embeddings = torch.randn(
        total, 256, requires_grad=True
    )

    metadata_list = []

    for batch_id, count in enumerate(counts):
        for i in range(count):
            view_index = i % 9
            box_index = i // 9

            metadata_list.append([
                batch_id,
                view_index,
                box_index,
            ])

    metadata = torch.tensor(
        metadata_list,
        dtype=torch.long,
    )

    normalized_boxes = torch.rand(total, 4)

    # Keep widths and heights relatively small.
    normalized_boxes[:, 2:] *= 0.2

    return (
        embeddings,
        metadata,
        normalized_boxes,
    )


def test_forward_shapes():
    model = MultiViewTransformer(
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
    )

    model.eval()

    embeddings, metadata, boxes = (
        create_test_data()
    )

    with torch.no_grad():
        contextual, padded, mask, indices = model(
            embeddings=embeddings,
            metadata=metadata,
            normalized_boxes=boxes,
            batch_size=3,
        )

    assert contextual.shape == (25, 256)
    assert padded.shape == (3, 18, 256)
    assert mask.shape == (3, 18)
    assert indices.shape == (3, 18)

    assert (~mask[0]).sum().item() == 18
    assert (~mask[1]).sum().item() == 7
    assert (~mask[2]).sum().item() == 0

    assert torch.isfinite(contextual).all()

    # Padding must remain zero.
    assert torch.all(padded[mask] == 0)

    # Original ROI ordering must be preserved.
    valid = ~mask

    assert torch.allclose(
        contextual[indices[valid]],
        padded[valid],
        atol=1e-5,
    )

    print("Forward shapes test PASSED")


def test_backward():
    model = MultiViewTransformer(
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
    )

    embeddings, metadata, boxes = (
        create_test_data()
    )

    contextual, _, _, _ = model(
        embeddings=embeddings,
        metadata=metadata,
        normalized_boxes=boxes,
        batch_size=3,
    )

    loss = contextual.square().mean()

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()

    assert (
        model.geometry_encoder[0].weight.grad
        is not None
    )

    print("Backward gradient test PASSED")


def test_permutation_equivariance():
    model = MultiViewTransformer(
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    model.eval()

    embeddings, metadata, boxes = (
        create_test_data()
    )

    permutation = torch.randperm(
        embeddings.shape[0]
    )

    with torch.no_grad():
        original, _, _, _ = model(
            embeddings,
            metadata,
            boxes,
            batch_size=3,
        )

        shuffled, _, _, _ = model(
            embeddings[permutation],
            metadata[permutation],
            boxes[permutation],
            batch_size=3,
        )

    # Restore original ordering.
    restored = torch.empty_like(shuffled)
    restored[permutation] = shuffled

    assert torch.allclose(
        original,
        restored,
        atol=1e-5,
        rtol=1e-5,
    )

    print("Permutation equivariance test PASSED")


def test_without_geometry():
    model = MultiViewTransformer(
        embedding_dim=256,
        use_geometry=False,
    )

    model.eval()

    embeddings, metadata, _ = (
        create_test_data()
    )

    with torch.no_grad():
        contextual, _, _, _ = model(
            embeddings=embeddings,
            metadata=metadata,
            normalized_boxes=None,
            batch_size=3,
        )

    assert contextual.shape == (25, 256)

    print("Without geometry test PASSED")


def test_all_empty():
    model = MultiViewTransformer()

    embeddings = torch.empty(0, 256)
    metadata = torch.empty(0, 3, dtype=torch.long)
    boxes = torch.empty(0, 4)

    contextual, padded, mask, indices = model(
        embeddings=embeddings,
        metadata=metadata,
        normalized_boxes=boxes,
        batch_size=2,
    )

    assert contextual.shape == (0, 256)
    assert padded.shape == (2, 0, 256)
    assert mask.shape == (2, 0)
    assert indices.shape == (2, 0)

    print("All-empty test PASSED")


if __name__ == "__main__":
    test_forward_shapes()
    test_backward()
    test_permutation_equivariance()
    test_without_geometry()
    test_all_empty()

    print(
        "\nAll MultiViewTransformer tests PASSED!"
    )
