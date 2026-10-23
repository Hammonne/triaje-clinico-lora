"""
Entrenamiento del LoRA / DoRA.
==============================

Arma el dataset tokenizado (prompt + JSON esperado), entrena con `Trainer` y mide
el costo (tiempo de entrenamiento y VRAM pico). Sigue la estructura del `train`
del profe, adaptado a `Config` y a Qwen2.5.
"""

import json
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import Config
from .data import Ejemplo
from .prompts import SYSTEM_PROMPT, formatear_manual


@dataclass
class ResultadoEntrenamiento:
    """Métricas de costo del entrenamiento (se guardan en resultados.csv)."""

    segundos_entrenamiento: float
    vram_pico_gb: float
    parametros_entrenables: int
    parametros_totales: int
    loss_final: Optional[float] = None


def _texto_entrenamiento(ejemplo: Ejemplo, tokenizer, config: Config) -> str:
    """
    Convierte un ejemplo en el texto completo de entrenamiento: system + user +
    assistant(JSON). Usa el chat template del tokenizer si existe; si no (modelo
    diminuto del smoke-test), usa el formateo manual de respaldo.
    """
    respuesta = json.dumps(ejemplo.salida, ensure_ascii=False)
    plantilla = getattr(tokenizer, "chat_template", None)
    if plantilla:
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ejemplo.mensaje},
            {"role": "assistant", "content": respuesta},
        ]
        return tokenizer.apply_chat_template(mensajes, tokenize=False)
    return formatear_manual(ejemplo.mensaje, respuesta)


def construir_dataset(ejemplos: List[Ejemplo], tokenizer, config: Config):
    """Crea un ``datasets.Dataset`` tokenizado a partir de los ejemplos."""
    from datasets import Dataset

    textos = [_texto_entrenamiento(e, tokenizer, config) for e in ejemplos]

    def tokenizar(lote):
        return tokenizer(
            lote["texto"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,  # el data collator hace el padding dinámico
        )

    ds = Dataset.from_dict({"texto": textos})
    ds = ds.map(tokenizar, batched=True, remove_columns=["texto"])
    return ds


def entrenar(
    model,
    tokenizer,
    train_ejs: List[Ejemplo],
    val_ejs: List[Ejemplo],
    config: Config,
) -> ResultadoEntrenamiento:
    """
    Entrena el modelo (ya envuelto con LoRA/DoRA) y devuelve las métricas de costo.

    Mide tiempo de entrenamiento y VRAM pico (`torch.cuda.max_memory_allocated`).
    """
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from .modelo import contar_parametros

    train_ds = construir_dataset(train_ejs, tokenizer, config)
    eval_ds = construir_dataset(val_ejs, tokenizer, config) if val_ejs else None

    usar_max_steps = config.max_steps > 0
    args = TrainingArguments(
        output_dir=str(config.output_dir / config.run_id()),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if usar_max_steps else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        # transformers 5 eliminó `warmup_ratio`: un float en [0, 1) en `warmup_steps`
        # se interpreta como proporción del total de pasos (mismo efecto).
        warmup_steps=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if (eval_ds and usar_max_steps) else "no",
        eval_steps=config.eval_steps if (eval_ds and usar_max_steps) else None,
        save_strategy="no",  # el adaptador se guarda al final a Drive
        # Precisión acorde a la T4 (sin bf16). En CPU/smoke, ambos en False.
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing and not config.smoke_test,
        optim=config.optim,
        seed=config.seed,
        report_to="none",
        disable_tqdm=config.smoke_test,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Medición de VRAM pico: reseteamos el contador antes de entrenar.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    salida = trainer.train()
    segundos = time.perf_counter() - t0

    vram_pico_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    entrenables, total = contar_parametros(model)

    return ResultadoEntrenamiento(
        segundos_entrenamiento=segundos,
        vram_pico_gb=vram_pico_gb,
        parametros_entrenables=entrenables,
        parametros_totales=total,
        loss_final=float(salida.training_loss) if salida else None,
    )


def guardar_adaptador(model, tokenizer, config: Config) -> str:
    """
    Guarda el adaptador LoRA/DoRA y el tokenizer en ``output_dir/run_id`` (en
    Colab apuntará a Drive). Devuelve la ruta.
    """
    destino = config.output_dir / config.run_id()
    destino.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destino))
    tokenizer.save_pretrained(str(destino))
    return str(destino)
