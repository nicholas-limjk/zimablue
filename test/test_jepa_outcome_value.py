from pathlib import Path

import torch

from zima_py.jepa_outcome_value import JepaOutcomeValueHead


def test_outcome_value_shapes_and_round_trip(tmp_path: Path):
    model = JepaOutcomeValueHead(latent_size=8, hidden_size=12)
    belief = torch.randn(3, 2, 2, 2)
    candidate = torch.randn(3, 2, 2, 2)
    outputs = model(belief, candidate, torch.tensor([0.0, 0.5, 1.0]))
    assert outputs["success_logit"].shape == (3,)
    assert outputs["remaining"].shape == (3,)
    assert torch.all((outputs["remaining"] >= 0.0) & (outputs["remaining"] <= 1.0))

    checkpoint = tmp_path / "value.pt"
    model.save(checkpoint, metadata={"objective": "outcome-only"})
    restored = JepaOutcomeValueHead.load(checkpoint)
    restored_outputs = restored(belief, candidate, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(outputs["success_logit"], restored_outputs["success_logit"])
    assert torch.allclose(outputs["remaining"], restored_outputs["remaining"])
