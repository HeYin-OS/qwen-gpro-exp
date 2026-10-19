import pytest

from qwen_grpo.train import TrainConfig, load_config


class TestLoadConfig:
    def test_loads_default_yaml(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.num_generations == 8
        assert cfg.lora_r == 32
        assert cfg.loss_type == "dapo"

    def test_defaults_when_no_file(self):
        cfg = TrainConfig()
        assert cfg.train_size == 7200
        assert cfg.beta == 0.04

    def test_unknown_key_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("lora_r: 16\nno_such_key: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no_such_key"):
            load_config(bad)

    def test_partial_override_keeps_defaults(self, tmp_path):
        f = tmp_path / "partial.yaml"
        f.write_text("lora_r: 8\n", encoding="utf-8")
        cfg = load_config(f)
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 64  # untouched default
