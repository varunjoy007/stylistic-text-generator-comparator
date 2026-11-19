"""All 6 milestones for the Stylistic Text Generator Comparator."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import time

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("data")
RAW_CORPUS_PATH = DATA_DIR / "raw_commentary.txt"
CORPUS_PATH = DATA_DIR / "commentary.txt"
NGRAM_PATH = DATA_DIR / "ngram_model.json"
ADAPTER_PATH = DATA_DIR / "lora_adapter.pt"

DOMAIN_TERMS = {
    "off", "leg", "stump", "stumps", "cover", "covers", "midwicket", "mid-off",
    "mid-on", "point", "gully", "slip", "slips", "keeper", "bowler", "batter",
    "batsman", "length", "yorker", "bouncer", "full", "short", "wide", "driven",
    "defended", "pulled", "cut", "edged", "four", "six", "boundary", "over",
    "crease", "pad", "pads", "spin", "seam", "swing", "foot", "front", "back",
}


# --- Milestone 1: Prepare a small corpus of sports commentary text.
def clean_commentary_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", "", text)
    return text.strip()


def tokenize_for_ngram(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9'-]+|[.,!?]", text.lower())


def milestone_1(corpus_path: Path = CORPUS_PATH) -> dict:
    """Clean and tokenize the sports commentary corpus."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    if not corpus_path.exists():
        sample_lines = [
            "Short of a length and wide outside off, defended to cover.",
            "Full toss on middle stump, whipped away through midwicket for four.",
            "He steps down the track and lofts it cleanly over long-off for six.",
            "Good length delivery angling across, edged and taken at second slip.",
        ] * 10
        corpus_path.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    cleaned = [clean_commentary_line(line) for line in raw_lines if line.strip()]
    corpus_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    tokens = tokenize_for_ngram(" ".join(cleaned))
    return {"lines": len(cleaned), "tokens": len(tokens), "path": str(corpus_path)}


# --- Milestone 2: Implement the n-gram model and generate text from it.
class NGramModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.trigrams: dict[str, Counter] = defaultdict(Counter)
        self.bigrams: dict[str, Counter] = defaultdict(Counter)
        self.unigrams: Counter = Counter()

    def train(self, lines: list[str]) -> None:
        for line in lines:
            tokens = tokenize_for_ngram(line)
            if len(tokens) < self.n:
                continue
            for tok in tokens:
                self.unigrams[tok] += 1
            for i in range(len(tokens) - 1):
                self.bigrams[tokens[i]][tokens[i + 1]] += 1
            for i in range(len(tokens) - 2):
                ctx = f"{tokens[i]} {tokens[i + 1]}"
                self.trigrams[ctx][tokens[i + 2]] += 1

    def _sample(self, counter: Counter, temperature: float = 0.8) -> str:
        words = list(counter.keys())
        counts = list(counter.values())
        temp = max(temperature, 1e-3)
        logits = [math.log(c + 1e-9) / temp for c in counts]
        max_l = max(logits)
        weights = [math.exp(l - max_l) for l in logits]
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, prompt: str, max_words: int = 25, temperature: float = 0.8) -> str:
        tokens = tokenize_for_ngram(prompt)
        if not tokens:
            tokens = [random.choice(list(self.unigrams.keys()))]
        generated = list(tokens)
        for _ in range(max_words):
            next_tok = None
            if len(generated) >= 2:
                ctx2 = f"{generated[-2]} {generated[-1]}"
                if ctx2 in self.trigrams:
                    next_tok = self._sample(self.trigrams[ctx2], temperature)
            if next_tok is None and generated[-1] in self.bigrams:
                next_tok = self._sample(self.bigrams[generated[-1]], temperature)
            if next_tok is None and self.unigrams:
                next_tok = self._sample(self.unigrams, temperature)
            if next_tok is None:
                break
            generated.append(next_tok)

        out = ""
        for tok in generated:
            if tok in {".", ",", "!", "?"}:
                out += tok
            else:
                out = f"{out} {tok}".strip()
        return out.capitalize()

    def save(self, path: Path = NGRAM_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": self.n,
            "trigrams": {k: dict(v) for k, v in self.trigrams.items()},
            "bigrams": {k: dict(v) for k, v in self.bigrams.items()},
            "unigrams": dict(self.unigrams),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = NGRAM_PATH) -> "NGramModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(n=payload.get("n", 3))
        model.trigrams = defaultdict(Counter, {k: Counter(v) for k, v in payload["trigrams"].items()})
        model.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in payload["bigrams"].items()})
        model.unigrams = Counter(payload["unigrams"])
        return model


def milestone_2(
    prompt: str = "Short of a length",
    max_words: int = 24,
    temperature: float = 0.8,
    corpus_path: Path = CORPUS_PATH,
    model_path: Path = NGRAM_PATH,
) -> str:
    """Train or load the N-gram model and generate sports commentary."""
    if not model_path.exists():
        milestone_1(corpus_path)
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
        model = NGramModel(n=3)
        model.train(lines)
        model.save(model_path)
    else:
        model = NGramModel.load(model_path)
    return model.generate(prompt, max_words=max_words, temperature=temperature)


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
class LoRALinear(nn.Module):
    """From-scratch Low-Rank Adaptation wrapper for GPT-2 Conv1D / Linear layers."""

    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if hasattr(base_layer, "nf"):  # HuggingFace GPT2 Conv1D
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
        else:
            in_features = base_layer.in_features
            out_features = base_layer.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)

        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16, target_modules: tuple[str, ...] = ("c_attn", "c_proj")) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    for name, module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_modules):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
    return model


def save_lora_weights(model: nn.Module, path: Path = ADAPTER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: Path = ADAPTER_PATH, device: str = "cpu") -> nn.Module:
    lora_state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(lora_state, strict=False)
    return model


def milestone_3(model_name: str = "distilgpt2", r: int = 8, alpha: int = 16) -> tuple[nn.Module, AutoTokenizer, dict]:
    """Load distilgpt2 and inject custom from-scratch LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = inject_lora(model, r=r, alpha=alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stats = {"trainable": trainable, "total": total, "pct": round(100.0 * trainable / total, 2)}
    return model, tokenizer, stats


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
def milestone_4(
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 3e-4,
    block_size: int = 64,
    max_steps: int | None = None,
    corpus_path: Path = CORPUS_PATH,
    adapter_path: Path = ADAPTER_PATH,
) -> dict:
    """Fine-tune the injected LoRA adapter on the sports commentary corpus."""
    milestone_1(corpus_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = milestone_3()
    model.to(device)

    lines = [l.strip() for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    joined = f" {tokenizer.eos_token} ".join(lines[:400]) + f" {tokenizer.eos_token}"
    token_ids = tokenizer.encode(joined)
    examples = [
        torch.tensor(token_ids[i : i + block_size], dtype=torch.long)
        for i in range(0, max(len(token_ids) - block_size, block_size), block_size)
        if i + block_size <= len(token_ids)
    ]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    model.train()
    losses = []
    step = 0
    for _ in range(epochs):
        for i in range(0, len(examples), batch_size):
            batch = torch.stack(examples[i : i + batch_size]).to(device)
            optimizer.zero_grad()
            loss = model(input_ids=batch, labels=batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    save_lora_weights(model, adapter_path)
    return {"steps": step, "final_loss": round(losses[-1], 4), "adapter_path": str(adapter_path)}


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
def generate_with_hf(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 28, temperature: float = 0.8) -> tuple[str, float]:
    device = next(model.parameters()).device
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.1),
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = " ".join(tokenizer.decode(out[0], skip_special_tokens=True).split())
    return text, elapsed_ms


def milestone_5(
    prompt: str = "Short of a length",
    max_new_tokens: int = 28,
    temperature: float = 0.8,
    adapter_path: Path = ADAPTER_PATH,
) -> str:
    """Generate sports commentary using the fine-tuned LoRA-adapted distilgpt2 model."""
    if not adapter_path.exists():
        milestone_4(epochs=1, max_steps=2, adapter_path=adapter_path)
    model, tokenizer, _ = milestone_3()
    model = load_lora_weights(model, adapter_path)
    text, _ = generate_with_hf(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return text


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
def style_density(text: str) -> str:
    words = re.findall(r"[a-z\-]+", text.lower())
    if not words:
        return "0.0%"
    hits = sum(1 for w in words if w in DOMAIN_TERMS)
    return f"{(100.0 * hits / len(words)):.1f}% ({hits}/{len(words)})"


def milestone_6(
    prompt: str = "Full toss on middle stump,",
    max_tokens: int = 28,
    temperature: float = 0.8,
    render: bool = True,
) -> dict:
    """Compare Trigram N-Gram, Base distilgpt2, and LoRA-Adapted distilgpt2 side-by-side."""
    t0 = time.perf_counter()
    ngram_text = milestone_2(prompt=prompt, max_words=max_tokens, temperature=temperature)
    ngram_ms = (time.perf_counter() - t0) * 1000.0

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_text, base_ms = generate_with_hf(base_model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=temperature)

    if not ADAPTER_PATH.exists():
        milestone_4(epochs=1, max_steps=2, adapter_path=ADAPTER_PATH)
    lora_model = inject_lora(AutoModelForCausalLM.from_pretrained("distilgpt2"), r=8, alpha=16)
    lora_model = load_lora_weights(lora_model, ADAPTER_PATH)
    lora_text, lora_ms = generate_with_hf(lora_model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=temperature)

    results = {
        "prompt": prompt,
        "ngram": {"text": ngram_text, "density": style_density(ngram_text), "latency_ms": round(ngram_ms, 1)},
        "base_llm": {"text": base_text, "density": style_density(base_text), "latency_ms": round(base_ms, 1)},
        "lora_llm": {"text": lora_text, "density": style_density(lora_text), "latency_ms": round(lora_ms, 1)},
    }

    if render:
        console = Console()
        table = Table(title=f'Prompt: "{prompt}" | Temp: {temperature}', show_lines=True)
        table.add_column("Model Architecture", style="bold cyan", width=22)
        table.add_column("Generated Commentary", ratio=3)
        table.add_column("Domain Jargon", justify="right", style="magenta", width=16)
        table.add_column("Latency", justify="right", style="green", width=11)

        table.add_row("Trigram N-Gram", ngram_text, results["ngram"]["density"], f"{ngram_ms:.1f} ms")
        table.add_row("Base distilgpt2", base_text, results["base_llm"]["density"], f"{base_ms:.1f} ms")
        table.add_row("LoRA distilgpt2 (0.49%)", f"[bold]{lora_text}[/bold]", results["lora_llm"]["density"], f"{lora_ms:.1f} ms")
        console.print(table)

    return results
