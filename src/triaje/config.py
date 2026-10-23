"""
Configuración central del proyecto.
====================================

`Config` es una ``@dataclass`` que concentra TODOS los hiperparámetros, rutas y
precios. Es la fuente única de parámetros (requisito de la rúbrica: POO, nada de
números mágicos sueltos). Está basada en la ``Config`` del profe
(`referencia_profe/main.ipynb`) pero adaptada a nuestra tarea y a la T4.

Diferencias deliberadas respecto a la demo del profe (ver `CLAUDE.md` §2):

- Modelo ``Qwen/Qwen2.5-0.5B-Instruct`` en vez de ``google/gemma-2b`` (el
  enunciado prohíbe usar el mismo modelo/familia que la clase). Es multilingüe
  (buen español) y lo bastante pequeño para correr todo el diseño experimental
  en ~1-2 h de una T4. Antes usábamos Phi-3.5-mini (3.8B), pero estimamos
  ~1.5-2.5 h por corrida y el diseño completo no entraba en el presupuesto de Colab.
- Con 0.5B parámetros el modelo cabe holgado en fp16 en la T4 (16 GB): no hace
  falta cuantizar a 4-bit (QLoRA) ni ``gradient_checkpointing``, que solo
  harían más lento el entrenamiento.
- La T4 **no** soporta bf16 → ``fp16=True`` (el profe usaba bfloat16, que en T4
  fallaría).
- ``attn_implementation="sdpa"``: atención nativa de PyTorch (no hay
  flash-attention 2 en T4, pero SDPA sí funciona y es más rápida que "eager").
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Config:
    """Todos los parámetros del proyecto en un solo lugar."""

    # ── Modelo base ─────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Modelo diminuto para el smoke-test en CPU (ver `smoke_test`).
    model_name_smoke: str = "sshleifer/tiny-gpt2"

    # ── Cuantización 4-bit (QLoRA) ──────────────────────────────────────────
    # Desactivada: el modelo de 0.5B cabe en fp16. Si se activa, requiere
    # instalar bitsandbytes (no viene preinstalado en Colab).
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    # float16 (NO bfloat16): la T4 no tiene soporte de bf16.
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = True  # doble cuantización, ahorra VRAM

    # ── Atención / precisión en T4 ──────────────────────────────────────────
    attn_implementation: str = "sdpa"  # no hay flash-attention 2 en T4
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False  # innecesario con 0.5B; solo ralentiza

    # ── LoRA / DoRA ─────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Capas objetivo de Qwen2.5 (atención + MLP). Fijas entre todas las corridas.
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # Técnica a comparar contra LoRA: DoRA (use_dora=True en PEFT).
    use_dora: bool = False

    # ── Entrenamiento (fijo entre corridas, salvo los factores del diseño) ──
    num_train_epochs: int = 3
    # 100 pasos × batch efectivo 16 = 1600 ejemplos ≈ 1.5 épocas de train (1050).
    # Suficiente para que un modelo pequeño aprenda el formato y las clases, y
    # mantiene cada corrida en pocos minutos (presupuesto total ~1-2 h).
    max_steps: int = 100  # si > 0, manda sobre las épocas
    per_device_train_batch_size: int = 2
    # Lote de la loss de validación del Trainer. Pequeño a propósito: con el
    # vocabulario de Qwen (~152k tokens) los logits de 16 × 512 tokens ocupan
    # ~4.6 GB y no caben en la T4 junto al entrenamiento.
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # batch efectivo = 2 * 8 = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"  # paged_adamw_8bit requeriría bitsandbytes
    max_seq_length: int = 512  # mensajes + JSON de salida
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2

    # ── Generación / inferencia ─────────────────────────────────────────────
    max_new_tokens: int = 160
    # Lote de generación en la evaluación del test (solo inferencia, sin logits
    # de toda la secuencia): 16 prompts a la vez caben de sobra en la T4.
    batch_generacion: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False  # determinístico: para medir reproducible

    # ── Dataset ─────────────────────────────────────────────────────────────
    # Proporciones de los splits (estratificados por `prioridad`, semilla fija).
    frac_train: float = 0.70
    frac_val: float = 0.15
    frac_test: float = 0.15
    seed_split: int = 42  # semilla de los splits (independiente de las réplicas)

    # ── Reproducibilidad ────────────────────────────────────────────────────
    seed: int = 42

    # ── Rutas ───────────────────────────────────────────────────────────────
    # En Colab se sobreescriben para apuntar a Google Drive. Por defecto, local.
    repo_dir: Path = Path(".")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    # Adaptadores y checkpoints: a Drive, NUNCA a git.
    output_dir: Path = Path("outputs")
    drive_dir: Optional[Path] = None  # p. ej. /content/drive/MyDrive/triaje

    # ── Costo simulado (análisis económico) ─────────────────────────────────
    # Precios de GPU por hora (USD) de un proveedor cloud, configurables.
    precio_gpu_usd_hora: float = 0.35  # T4 spot aproximado
    precio_gpu_a100_usd_hora: float = 1.50  # para comparar escenarios
    # Precio de una API de pago por 1M tokens (entrada/salida), para comparar el
    # costo de resolver el test set con una API vs. con nuestro LoRA.
    precio_api_usd_por_1m_tokens_in: float = 0.15
    precio_api_usd_por_1m_tokens_out: float = 0.60

    # ── Modo smoke-test (pipeline completo en CPU, modelo diminuto) ─────────
    # Permite verificar que el código corre antes de gastar GPU en Colab.
    smoke_test: bool = False
    smoke_n_ejemplos: int = 10
    smoke_max_steps: int = 2

    def aplicar_smoke_test(self) -> "Config":
        """
        Devuelve una copia lógica (in-place) ajustada para correr en CPU y rápido:
        modelo diminuto, sin 4-bit, sin fp16, pocos pasos y pocos ejemplos.
        """
        self.smoke_test = True
        self.model_name = self.model_name_smoke
        # Carpetas propias: si compartiera `results/` con la suite real, sus filas
        # (mismos run_id: base_s42, lora_r4_a4_s42, ...) harían que el runner
        # se saltara esas corridas reales creyendo que ya están hechas.
        self.results_dir = self.results_dir.parent / f"{self.results_dir.name}_smoke"
        self.output_dir = self.output_dir.parent / f"{self.output_dir.name}_smoke"
        # El modelo diminuto (GPT-2) nombra sus capas distinto que Qwen2.5.
        self.lora_target_modules = ["c_attn"]
        self.use_4bit = False
        self.fp16 = False
        self.bf16 = False
        self.gradient_checkpointing = False
        self.attn_implementation = "eager"
        self.optim = "adamw_torch"
        self.max_steps = self.smoke_max_steps
        self.num_train_epochs = 1
        self.max_seq_length = 128
        self.max_new_tokens = 40
        return self

    @property
    def ratio_alpha_r(self) -> float:
        """Ratio alpha/r, el factor que realmente controla la influencia del LoRA."""
        return self.lora_alpha / self.lora_r

    def run_id(self) -> str:
        """
        ID determinístico de la corrida, p. ej. ``lora_r16_a32_s42`` o
        ``dora_r16_a32_s42``. Lo usa el runner para saltarse corridas ya hechas.
        """
        tecnica = "dora" if self.use_dora else "lora"
        return f"{tecnica}_r{self.lora_r}_a{self.lora_alpha}_s{self.seed}"


def set_seed(seed: int) -> None:
    """
    Fija la semilla en random, numpy y torch (si está disponible) y en
    transformers. Se llama al inicio de cada corrida para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # En un entorno sin torch (p. ej. solo validando datos) no pasa nada.
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass
