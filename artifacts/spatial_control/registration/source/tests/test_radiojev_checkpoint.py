"""Exercise the E3-to-E4 checkpoint handoff without downloading the backbone."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wirelessjev.radiojev import e3, e45
from wirelessjev.radiojev.trainable import LoRALinear, lora_state


@pytest.fixture
def model(tmp_path, monkeypatch):
    def backbone(engine, *, lora_layers, readout_rows):
        return SimpleNamespace(
            device=torch.device("cpu"), embed_rms=0.1,
            readout=nn.Parameter(torch.zeros(readout_rows, 1024)),
            lora_layers=[nn.Sequential(LoRALinear(nn.Linear(2, 2), r=1))])

    monkeypatch.setattr(e3, "TrainableD3", backbone)
    monkeypatch.setattr(e45, "OUT", tmp_path)
    trained = e3.MultiTask(None, None)
    with torch.no_grad():
        for parameter in trained.heads.parameters():
            parameter.fill_(0.25)
        trained.m.readout.fill_(0.75)
        for layer in trained.m.lora_layers:
            layer[0].A.fill_(0.5)
            layer[0].B.fill_(0.125)
    return trained, tmp_path


@pytest.mark.parametrize("has_e2_checkpoint", [False, True])
def test_bank_restores_all_e3_heads_readout_and_lora(model, has_e2_checkpoint):
    trained, directory = model
    expected = {"heads": trained.heads.state_dict(),
                "readout": trained.m.readout.detach(),
                "lora": lora_state(trained.m.lora_layers)}
    torch.save(expected, directory / "ckpt_e3.pt")
    if has_e2_checkpoint:
        # E3 must take precedence over an older association-only checkpoint.
        torch.save({"encoder": {}}, directory / "ckpt_radio_q.pt")

    bank = e45.Bank(None)
    for key, value in expected["heads"].items():
        torch.testing.assert_close(bank.mt.heads.state_dict()[key], value)
    torch.testing.assert_close(bank.mt.m.readout, expected["readout"])
    for key, value in expected["lora"].items():
        torch.testing.assert_close(bank.init["lora"][key], value)
    torch.testing.assert_close(bank.init["readout"], expected["readout"])


def test_bank_can_still_warm_start_from_e2_without_e3(model):
    trained, directory = model
    expected = {"encoder": trained.heads["assoc"].state_dict(),
                "readout": trained.m.readout[:2].detach(),
                "lora": lora_state(trained.m.lora_layers)}
    torch.save(expected, directory / "ckpt_radio_q.pt")

    bank = e45.Bank(None)
    for key, value in expected["encoder"].items():
        torch.testing.assert_close(bank.init["enc"][key], value)
    torch.testing.assert_close(bank.init["readout"][:2], expected["readout"])
