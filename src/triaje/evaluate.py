"""
Inferencia + métricas + medición de costo.
===========================================

Evalúa un modelo (base o con LoRA/DoRA) sobre el test set y calcula TODAS las
métricas del enunciado (CLAUDE.md §4):

Desempeño en la tarea
- % de JSON válido (parseable) y % que cumple el esquema (raw y post-extraído)
- accuracy y macro-F1 por campo categórico
- F1 de conjuntos para `sintomas`
- exact match de los campos categóricos juntos
- **tasa de sub-triaje** (métrica de seguridad clínica)

Costo computacional
- tiempo de inferencia por ejemplo, tokens/s
- VRAM pico de inferencia
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .config import Config
from .data import Ejemplo
from .prompts import SYSTEM_PROMPT, formatear_manual
from .schema import (
    CAMPOS_CATEGORICOS,
    ORDEN_PRIORIDAD,
    extraer_objeto_json,
    validar_salida,
)

# Etiqueta sentinel para predicciones inválidas (cuenta como fallo en los campos).
_INVALIDO = "__invalido__"


@dataclass
class Metricas:
    """Todas las métricas de una evaluación. Serializable a JSON/CSV."""

    n: int = 0
    # Validez del JSON
    json_valido_raw: float = 0.0
    esquema_ok_raw: float = 0.0
    json_valido_post: float = 0.0
    esquema_ok_post: float = 0.0
    # Desempeño por campo (sobre salidas post-extraídas)
    accuracy: Dict[str, float] = field(default_factory=dict)
    macro_f1: Dict[str, float] = field(default_factory=dict)
    sintomas_f1: float = 0.0
    exact_match_categoricos: float = 0.0
    # Seguridad clínica
    tasa_subtriaje: float = 0.0
    tasa_subtriaje_cobertura: float = 0.0  # sobre cuántos se pudo medir
    # Costo
    segundos_inferencia_total: float = 0.0
    segundos_por_ejemplo: float = 0.0
    tokens_por_segundo: float = 0.0
    vram_pico_inferencia_gb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def _texto_entrada(mensaje: str, tokenizer) -> str:
    """Prompt de inferencia: chat template si existe; si no, el formateo manual."""
    if getattr(tokenizer, "chat_template", None):
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mensaje},
        ]
        return tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
    return formatear_manual(mensaje, respuesta=None)


def generar_respuestas(
    mensajes: List[str], model, tokenizer, config: Config
) -> List[Tuple[str, int]]:
    """
    Genera las respuestas de un LOTE de mensajes en una sola llamada a
    ``generate`` (mucho más rápido en GPU que de uno en uno). Devuelve una lista
    de (texto, n_tokens_generados), en el mismo orden.

    Para generar en lote, el padding va a la IZQUIERDA (así todos los prompts
    terminan en la misma posición y la generación continúa desde ahí). Con
    decodificación greedy el resultado es el mismo que generando de a uno.
    """
    entradas = [_texto_entrada(m, tokenizer) for m in mensajes]

    lado_original = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(entradas, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = lado_original  # el entrenamiento usa "right"

    with torch.no_grad():
        salida = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            # temperature/top_p solo aplican al muestrear; con greedy se omiten
            # para que transformers no avise de parámetros sin efecto.
            **({"temperature": config.temperature, "top_p": config.top_p}
               if config.do_sample else {}),
            pad_token_id=tokenizer.pad_token_id,
        )

    largo_prompt = inputs.input_ids.shape[1]
    resultados = []
    for fila in salida:
        generados = fila[largo_prompt:]
        # Tokens reales generados (sin el padding que se agrega tras terminar).
        n_generados = int((generados != tokenizer.pad_token_id).sum())
        texto = tokenizer.decode(generados, skip_special_tokens=True)
        resultados.append((texto.strip(), n_generados))
    return resultados


def _f1_conjuntos(pred: List[str], real: List[str]) -> float:
    """F1 entre dos conjuntos de síntomas (normalizados a minúsculas)."""
    sp = {s.strip().lower() for s in pred}
    sr = {s.strip().lower() for s in real}
    if not sp and not sr:
        return 1.0  # ambos vacíos: acierto perfecto (p. ej. administrativo)
    if not sp or not sr:
        return 0.0
    inter = len(sp & sr)
    if inter == 0:
        return 0.0
    precision = inter / len(sp)
    recall = inter / len(sr)
    return 2 * precision * recall / (precision + recall)


def _macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Macro-F1 con sklearn sobre la unión de etiquetas (incluye el sentinel)."""
    from sklearn.metrics import f1_score

    etiquetas = sorted(set(y_true) | set(y_pred))
    return float(f1_score(y_true, y_pred, labels=etiquetas, average="macro", zero_division=0))


def evaluar(
    model,
    tokenizer,
    test_ejs: List[Ejemplo],
    config: Config,
    guardar_predicciones_en: Optional[str] = None,
) -> Metricas:
    """
    Corre la inferencia sobre `test_ejs` y calcula todas las métricas.
    Si se pasa `guardar_predicciones_en`, escribe un JSONL con cada predicción.
    """
    n = len(test_ejs)
    m = Metricas(n=n)

    # Acumuladores
    c_json_raw = c_esq_raw = c_json_post = c_esq_post = 0
    y_true = {campo: [] for campo in CAMPOS_CATEGORICOS}
    y_pred = {campo: [] for campo in CAMPOS_CATEGORICOS}
    sum_sintomas_f1 = 0.0
    exact_ok = 0
    subtriaje_casos = 0
    subtriaje_medibles = 0
    total_tokens = 0
    predicciones = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    lote = max(1, config.per_device_eval_batch_size)
    t0 = time.perf_counter()
    respuestas: List[Tuple[str, int]] = []
    for i in range(0, n, lote):
        mensajes = [ej.mensaje for ej in test_ejs[i:i + lote]]
        respuestas.extend(generar_respuestas(mensajes, model, tokenizer, config))
    segundos = time.perf_counter() - t0

    for ej, (texto, n_tok) in zip(test_ejs, respuestas):
        total_tokens += n_tok

        # Validez RAW (todo el texto debe ser JSON del esquema)
        v_raw = validar_salida(texto)
        raw_es_json = v_raw.error is None or "parsear JSON" not in (v_raw.error or "")
        c_json_raw += int(raw_es_json)
        c_esq_raw += int(bool(v_raw))

        # Validez POST (tras extraer el primer objeto JSON)
        post = extraer_objeto_json(texto)
        v_post = validar_salida(post)
        post_es_json = v_post.error is None or "parsear JSON" not in (v_post.error or "")
        c_json_post += int(post_es_json)
        c_esq_post += int(bool(v_post))

        pred = v_post.datos if v_post else None
        real = ej.salida

        # Campos categóricos (sentinel si la predicción es inválida)
        for campo in CAMPOS_CATEGORICOS:
            y_true[campo].append(real[campo])
            y_pred[campo].append(pred[campo] if pred else _INVALIDO)

        # Exact match de los 4 categóricos juntos
        if pred and all(pred[c] == real[c] for c in CAMPOS_CATEGORICOS):
            exact_ok += 1

        # Síntomas (F1 de conjuntos)
        sum_sintomas_f1 += _f1_conjuntos(pred["sintomas"] if pred else [], real["sintomas"])

        # Sub-triaje: solo medible si la prioridad predicha es válida.
        if pred and pred["prioridad"] in ORDEN_PRIORIDAD:
            subtriaje_medibles += 1
            # Número MAYOR = MENOS urgente. Sub-triaje = predijo menos urgente.
            if ORDEN_PRIORIDAD[pred["prioridad"]] > ORDEN_PRIORIDAD[real["prioridad"]]:
                subtriaje_casos += 1

        predicciones.append({
            "id": ej.id,
            "mensaje": ej.mensaje,
            "real": real,
            "prediccion_texto": texto,
            "prediccion_parseada": pred,
            "esquema_ok": bool(v_post),
        })

    # ── Consolidar métricas ──
    m.json_valido_raw = c_json_raw / n if n else 0.0
    m.esquema_ok_raw = c_esq_raw / n if n else 0.0
    m.json_valido_post = c_json_post / n if n else 0.0
    m.esquema_ok_post = c_esq_post / n if n else 0.0

    for campo in CAMPOS_CATEGORICOS:
        aciertos = sum(int(t == p) for t, p in zip(y_true[campo], y_pred[campo]))
        m.accuracy[campo] = aciertos / n if n else 0.0
        m.macro_f1[campo] = _macro_f1(y_true[campo], y_pred[campo])

    m.sintomas_f1 = sum_sintomas_f1 / n if n else 0.0
    m.exact_match_categoricos = exact_ok / n if n else 0.0
    m.tasa_subtriaje = subtriaje_casos / subtriaje_medibles if subtriaje_medibles else 0.0
    m.tasa_subtriaje_cobertura = subtriaje_medibles / n if n else 0.0

    m.segundos_inferencia_total = segundos
    m.segundos_por_ejemplo = segundos / n if n else 0.0
    m.tokens_por_segundo = total_tokens / segundos if segundos > 0 else 0.0
    m.vram_pico_inferencia_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )

    if guardar_predicciones_en:
        with open(guardar_predicciones_en, "w", encoding="utf-8") as f:
            for p in predicciones:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return m
