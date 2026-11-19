"""Milestone unit & integration tests for Stylistic Text Generator Comparator."""
from app.core import (
    LoRALinear,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)


def test_milestone_1():
    stats = milestone_1()
    assert stats["lines"] > 0
    assert stats["tokens"] > 0


def test_milestone_2():
    text = milestone_2(prompt="Short of a length", max_words=12, temperature=0.8)
    assert isinstance(text, str)
    assert len(text.split()) >= 4


def test_milestone_3():
    model, tokenizer, stats = milestone_3()
    assert tokenizer is not None
    assert 0 < stats["trainable"] < stats["total"]
    assert any(isinstance(m, LoRALinear) for m in model.modules())


def test_milestone_4(tmp_path):
    adapter_out = tmp_path / "test_lora_adapter.pt"
    result = milestone_4(epochs=1, batch_size=2, block_size=32, max_steps=1, adapter_path=adapter_out)
    assert adapter_out.exists()
    assert result["steps"] == 1
    assert result["final_loss"] > 0


def test_milestone_5():
    text = milestone_5(prompt="Full toss on middle stump,", max_new_tokens=12, temperature=0.8)
    assert isinstance(text, str)
    assert "Full toss" in text


def test_milestone_6(capsys):
    res = milestone_6(prompt="Short and wide outside off,", max_tokens=12, temperature=0.8, render=True)
    assert "ngram" in res and "base_llm" in res and "lora_llm" in res
    out = capsys.readouterr().out
    assert "Trigram N-Gram" in out
    assert "LoRA distilgpt2" in out
