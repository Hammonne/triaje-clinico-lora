"""
Prompts y formato de chat para el triaje.
=========================================

Un solo lugar define el *system prompt* y cómo se arma el diálogo con el modelo.
El esquema que se inserta en el prompt viene de `schema.py` (fuente única), así
el modelo nunca ve un esquema distinto del que valida el evaluador.

En el código real usamos ``tokenizer.apply_chat_template`` (ver `train.py` y
`evaluate.py`), que aplica el formato propio de cada modelo (en Qwen2.5,
``<|im_start|>system ... <|im_end|>``). Aquí definimos los *mensajes* (lista de
dicts role/content) y un formateo manual de respaldo (estilo
``<|system|> ... <|user|> ... <|assistant|>``) para el smoke-test con modelos que
no traen chat template.
"""

from typing import Dict, List, Optional

from .schema import esquema_como_texto

# System prompt: define la tarea, exige SOLO JSON y pega el esquema vigente.
SYSTEM_PROMPT = (
    "Eres un asistente de triaje de una clínica en Perú. A partir del mensaje de "
    "un paciente o familiar, extrae la información clínica y responde ÚNICAMENTE "
    "con un objeto JSON (sin texto adicional, sin explicaciones, sin ```json) que "
    "siga EXACTAMENTE este esquema:\n"
    f"{esquema_como_texto()}\n"
    "Reglas: usa solo los valores permitidos de cada campo; 'prioridad' va de I "
    "(riesgo vital inmediato) a IV (no urgente); ante la duda entre dos "
    "prioridades, elige la más urgente; 'sintomas' es una lista de síntomas "
    "normalizados en minúsculas (vacía si es un trámite); 'resumen' tiene entre 5 "
    "y 25 palabras."
)


def construir_mensajes(mensaje_paciente: str) -> List[Dict[str, str]]:
    """
    Devuelve la lista de mensajes role/content para ``apply_chat_template``.
    Solo system + user; la respuesta del assistant la genera (o se añade en
    entrenamiento como target).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": mensaje_paciente.strip()},
    ]


def formatear_manual(mensaje_paciente: str, respuesta: Optional[str] = None) -> str:
    """
    Formateo manual del chat (estilo Phi), de respaldo para cuando el tokenizer no
    tiene chat template (p. ej. el modelo diminuto del smoke-test).

    Si se pasa ``respuesta`` (el JSON esperado), se incluye como turno del
    assistant (para entrenamiento). Si no, se deja abierto para que el modelo
    genere (para inferencia).
    """
    texto = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{mensaje_paciente.strip()}<|end|>\n"
        f"<|assistant|>\n"
    )
    if respuesta is not None:
        texto += f"{respuesta.strip()}<|end|>\n"
    return texto
