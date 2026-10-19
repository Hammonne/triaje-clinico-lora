"""
Esquema de salida del triaje clínico y su validación.
======================================================

Este módulo es la **fuente única de verdad** del proyecto: define las clases
válidas de cada campo, el esquema JSON esperado y el validador. Todo lo demás
(generación de datos, prompts, entrenamiento, evaluación) importa de aquí para
que no haya listas de clases duplicadas ni "números mágicos" repartidos.

Adaptado del validador de 8 pasos de `referencia_profe/validator.py`, pero con
nuestro esquema de triaje (ver `CLAUDE.md`, sección 3).

La tarea: a partir de un mensaje libre de un paciente/familiar a una clínica,
producir SOLO un objeto JSON con este esquema::

    {
      "especialidad": "<una de ESPECIALIDADES>",
      "prioridad": "<una de PRIORIDADES>",       # I..IV (NTS 042-MINSA)
      "accion": "<una de ACCIONES>",
      "grupo_edad": "<una de GRUPOS_EDAD>",
      "signos_alarma": true | false,
      "sintomas": ["lista", "de", "sintomas"],   # strings normalizados
      "resumen": "resumen clínico de 5 a 25 palabras"
    }
"""

import json
from typing import Any, Dict, List, Optional

# ── Clases válidas de cada campo categórico ─────────────────────────────────
# Usamos tuplas (orden estable para prompts/reportes) y derivamos sets para la
# validación. El orden de PRIORIDADES va de más a menos urgente: es importante
# para la "tasa de sub-triaje" (ver `ORDEN_PRIORIDAD`).

ESPECIALIDADES: tuple = (
    "medicina_general",
    "pediatria",
    "ginecologia_obstetricia",
    "traumatologia",
    "cardiologia",
    "neurologia",
    "gastroenterologia",
    "dermatologia",
    "salud_mental",
    "otorrinolaringologia",
    "administrativo",
)

# Niveles de prioridad de la NTS N° 042-MINSA/DGSP-V.01, de más a menos urgente.
PRIORIDADES: tuple = ("I", "II", "III", "IV")

ACCIONES: tuple = (
    "emergencia_inmediata",
    "atencion_mismo_dia",
    "cita_programada",
    "orientacion_remota",
    "gestion_administrativa",
)

GRUPOS_EDAD: tuple = (
    "lactante",
    "nino",
    "adolescente",
    "adulto",
    "adulto_mayor",
    "no_especificado",
)

# Mapa prioridad -> rango numérico (1 = más urgente). Lo usa la métrica de
# seguridad clínica (sub-triaje): predecir un número MAYOR que el real significa
# clasificar como MENOS urgente de lo que corresponde.
ORDEN_PRIORIDAD: Dict[str, int] = {p: i + 1 for i, p in enumerate(PRIORIDADES)}

# Campos categóricos que se evalúan con accuracy y macro-F1.
CAMPOS_CATEGORICOS: tuple = ("especialidad", "prioridad", "accion", "grupo_edad")

# Conjuntos para validación rápida.
_SET_ESPECIALIDADES = set(ESPECIALIDADES)
_SET_PRIORIDADES = set(PRIORIDADES)
_SET_ACCIONES = set(ACCIONES)
_SET_GRUPOS_EDAD = set(GRUPOS_EDAD)

# Claves obligatorias y exactas del objeto de salida.
CLAVES_REQUERIDAS = {
    "especialidad",
    "prioridad",
    "accion",
    "grupo_edad",
    "signos_alarma",
    "sintomas",
    "resumen",
}

# Límites del resumen (en palabras). Igual criterio que el validador del profe.
RESUMEN_MIN_PALABRAS = 5
RESUMEN_MAX_PALABRAS = 25


class ResultadoValidacion:
    """
    Resultado de validar una salida del modelo.

    Siempre se devuelve un objeto de este tipo. ``.valido`` indica si la salida
    cumple el esquema; ``.error`` describe el primer problema encontrado (o
    ``None`` si todo bien); ``.datos`` contiene el dict ya parseado si es válido.

    Se puede usar directamente en un ``if`` gracias a ``__bool__`` (igual que el
    ``ValidationResult`` del profe), p. ej. ``if validar_salida(txt): ...``.
    """

    def __init__(
        self,
        valido: bool,
        error: Optional[str] = None,
        datos: Optional[Dict[str, Any]] = None,
    ):
        self.valido = valido
        self.error = error
        self.datos = datos

    def __bool__(self) -> bool:
        return self.valido

    def __repr__(self) -> str:  # ayuda al depurar en notebooks
        estado = "OK" if self.valido else f"ERROR: {self.error}"
        return f"<ResultadoValidacion {estado}>"


def _validar_dict(data: Dict[str, Any]) -> ResultadoValidacion:
    """
    Valida un dict ya parseado contra el esquema. El 'por qué' de cada paso está
    en los comentarios: cada regla corresponde a una métrica o a un requisito
    clínico del enunciado.
    """
    # 1. Debe ser un objeto JSON (dict).
    if not isinstance(data, dict):
        return ResultadoValidacion(
            False, f"Se esperaba un objeto JSON, se obtuvo {type(data).__name__}"
        )

    # 2. Claves exactas: ni de más ni de menos. Así medimos cumplimiento estricto
    #    del esquema (métrica "% JSON que cumple el esquema").
    claves = set(data.keys())
    if claves != CLAVES_REQUERIDAS:
        extra = claves - CLAVES_REQUERIDAS
        faltan = CLAVES_REQUERIDAS - claves
        partes = []
        if extra:
            partes.append(f"claves de más: {sorted(extra)}")
        if faltan:
            partes.append(f"claves faltantes: {sorted(faltan)}")
        return ResultadoValidacion(False, "Claves incorrectas - " + ", ".join(partes))

    # 3. Campos categóricos: cada uno dentro de su conjunto permitido.
    if data["especialidad"] not in _SET_ESPECIALIDADES:
        return ResultadoValidacion(
            False, f"especialidad inválida: {data['especialidad']!r}"
        )
    if data["prioridad"] not in _SET_PRIORIDADES:
        return ResultadoValidacion(
            False, f"prioridad inválida: {data['prioridad']!r} (debe ser I..IV)"
        )
    if data["accion"] not in _SET_ACCIONES:
        return ResultadoValidacion(False, f"accion inválida: {data['accion']!r}")
    if data["grupo_edad"] not in _SET_GRUPOS_EDAD:
        return ResultadoValidacion(
            False, f"grupo_edad inválido: {data['grupo_edad']!r}"
        )

    # 4. signos_alarma debe ser booleano real (no "true" string, no 0/1). Es la
    #    bandera de seguridad clínica, debe ser inequívoca.
    if not isinstance(data["signos_alarma"], bool):
        return ResultadoValidacion(
            False,
            f"signos_alarma debe ser booleano, se obtuvo "
            f"{type(data['signos_alarma']).__name__}",
        )

    # 5. sintomas: lista de strings (puede estar vacía en casos administrativos).
    sintomas = data["sintomas"]
    if not isinstance(sintomas, list):
        return ResultadoValidacion(
            False, f"sintomas debe ser lista, se obtuvo {type(sintomas).__name__}"
        )
    if not all(isinstance(s, str) for s in sintomas):
        return ResultadoValidacion(False, "sintomas debe contener solo strings")

    # 6. resumen: string con cuenta de palabras entre los límites definidos.
    resumen = data["resumen"]
    if not isinstance(resumen, str):
        return ResultadoValidacion(
            False, f"resumen debe ser string, se obtuvo {type(resumen).__name__}"
        )
    n_palabras = len(resumen.split())
    if n_palabras < RESUMEN_MIN_PALABRAS or n_palabras > RESUMEN_MAX_PALABRAS:
        return ResultadoValidacion(
            False,
            f"resumen debe tener entre {RESUMEN_MIN_PALABRAS} y "
            f"{RESUMEN_MAX_PALABRAS} palabras, tiene {n_palabras}",
        )

    return ResultadoValidacion(True, datos=data)


def validar_salida(salida: Any) -> ResultadoValidacion:
    """
    Valida la salida del modelo contra el esquema de triaje.

    Acepta tanto un string (que debe ser JSON estricto: TODO el texto debe ser
    JSON, sin nada antes ni después, igual que el validador del profe) como un
    dict ya parseado (útil para validar los ejemplos del dataset).

    Devuelve un :class:`ResultadoValidacion`.
    """
    if isinstance(salida, str):
        texto = salida.strip()
        try:
            data = json.loads(texto)
        except json.JSONDecodeError as e:
            return ResultadoValidacion(False, f"Error al parsear JSON: {e}")
        return _validar_dict(data)

    if isinstance(salida, dict):
        return _validar_dict(salida)

    return ResultadoValidacion(
        False, f"Tipo de salida no soportado: {type(salida).__name__}"
    )


def esquema_como_texto() -> str:
    """
    Devuelve el esquema en texto plano, para insertarlo en el system prompt.
    Vive aquí para que el prompt y el validador nunca se desincronicen.
    """
    return (
        "{\n"
        f'  "especialidad": "{" | ".join(ESPECIALIDADES)}",\n'
        f'  "prioridad": "{" | ".join(PRIORIDADES)}",\n'
        f'  "accion": "{" | ".join(ACCIONES)}",\n'
        f'  "grupo_edad": "{" | ".join(GRUPOS_EDAD)}",\n'
        '  "signos_alarma": true | false,\n'
        '  "sintomas": ["lista", "de", "sintomas", "normalizados"],\n'
        f'  "resumen": "resumen clínico de {RESUMEN_MIN_PALABRAS} a '
        f'{RESUMEN_MAX_PALABRAS} palabras"\n'
        "}"
    )


def extraer_objeto_json(texto: str) -> str:
    """
    Extrae el primer objeto JSON completo de un texto, emparejando llaves.

    Reutiliza la heurística de `referencia_profe/evaluation.py`: sirve para la
    métrica "post-procesada" (cuánto rescatamos si el modelo agrega texto extra
    alrededor del JSON). Si no encuentra un objeto cerrado, devuelve el texto tal
    cual.
    """
    inicio = texto.find("{")
    if inicio == -1:
        return texto
    profundidad = 0
    for i in range(inicio, len(texto)):
        if texto[i] == "{":
            profundidad += 1
        elif texto[i] == "}":
            profundidad -= 1
            if profundidad == 0:
                return texto[inicio : i + 1]
    return texto  # llave sin cerrar: se devuelve como está
