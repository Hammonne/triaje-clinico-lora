"""
Carga del modelo base (4-bit) y aplicación de LoRA / DoRA.
==========================================================

Basado en `referencia_profe/main.ipynb` (`load_model_and_tokenizer`, `apply_lora`)
pero adaptado a Qwen2.5-0.5B en T4 y a nuestra `Config`:

- En smoke-test (CPU) NO se usa cuantización ni dtypes de GPU: se carga un modelo
  diminuto en fp32 para verificar el pipeline.
- En GPU se carga en fp16 (la T4 no tiene bf16). La ruta 4-bit NF4 (QLoRA) se
  conserva por si se cambia a un modelo más grande (`config.use_4bit`).
- La técnica a comparar (DoRA) se activa con `config.use_dora`.
"""

from typing import Tuple

import torch

from .config import Config


def cargar_modelo_y_tokenizer(config: Config, para_entrenar: bool = True):
    """
    Carga el modelo base y su tokenizer según `config`.

    Devuelve ``(model, tokenizer)``. En modo smoke-test carga en CPU sin 4-bit;
    en GPU usa QLoRA (4-bit NF4).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Qwen2.5 está soportado nativamente por transformers: no hace falta
    # `trust_remote_code` (ejecutar código remoto del Hub).
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Qwen2.5 trae pad token, pero por robustez lo fijamos si faltara.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # correcto para causal LM en entrenamiento

    if config.use_4bit and not config.smoke_test:
        # ── Ruta GPU: QLoRA 4-bit ──
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=config.attn_implementation,
        )
    elif torch.cuda.is_available() and not config.smoke_test:
        # ── Ruta GPU: fp16 sin cuantizar (el modelo cabe entero en la T4) ──
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float16 if config.fp16 else torch.float32,
            attn_implementation=config.attn_implementation,
        ).to("cuda")
    else:
        # ── Ruta smoke-test / CPU: sin cuantización, fp32 ──
        model = AutoModelForCausalLM.from_pretrained(config.model_name)

    if para_entrenar and config.use_4bit and not config.smoke_test:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    return model, tokenizer


def aplicar_lora(model, config: Config):
    """
    Envuelve el modelo con LoRA o DoRA (según `config.use_dora`) usando PEFT.

    Imprime y devuelve, junto con el modelo, el número de parámetros entrenables
    (métrica de costo del enunciado).
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config.use_dora,  # DoRA: la técnica PEFT a comparar
    )
    model = get_peft_model(model, lora_cfg)

    entrenables, total = contar_parametros(model)
    tecnica = "DoRA" if config.use_dora else "LoRA"
    print(
        f"{tecnica} aplicado - entrenables: {entrenables:,} / {total:,} "
        f"({100 * entrenables / total:.3f}%)"
    )
    return model


def contar_parametros(model) -> Tuple[int, int]:
    """Devuelve (parámetros entrenables, parámetros totales)."""
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return entrenables, total
