"""
Carga, validación y splits del dataset de triaje.
=================================================

Los datos viven en `data/` como JSONL. Cada fila::

    {"id": "...", "mensaje": "...", "salida": {<esquema>}, "origen": "manual|real|sintetico"}

Este módulo:
- carga y valida todas las filas contra `schema.py`,
- arma splits train/val/test **estratificados por `prioridad`** con semilla fija,
- estratifica también por `origen`, para que cada split mantenga la misma
  proporción de ejemplos manuales/reales (~20%) que el dataset completo,
- garantiza que el test **nunca** se use para entrenar ni elegir hiperparámetros.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .schema import PRIORIDADES, validar_salida

# Orígenes escritos por humanos: se reparten proporcionalmente entre los splits
# para que el entrenamiento también aprenda de ellos (>=20% exigido).
ORIGENES_HUMANOS = ("manual", "real")


@dataclass
class Ejemplo:
    """Una fila del dataset ya validada."""

    id: str
    mensaje: str
    salida: Dict
    origen: str

    @property
    def prioridad(self) -> str:
        return self.salida["prioridad"]


def cargar_jsonl(ruta: Path) -> List[Ejemplo]:
    """Carga un archivo JSONL a lista de :class:`Ejemplo` (sin validar aún)."""
    ejemplos: List[Ejemplo] = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            obj = json.loads(linea)
            ejemplos.append(
                Ejemplo(
                    id=str(obj.get("id", f"{ruta.stem}_{n}")),
                    mensaje=obj["mensaje"],
                    salida=obj["salida"],
                    origen=obj.get("origen", "sintetico"),
                )
            )
    return ejemplos


def cargar_todos(data_dir: Path) -> List[Ejemplo]:
    """
    Carga y VALIDA todos los JSONL de `data/manual`, `data/real` y
    `data/sintetico`. Lanza ``ValueError`` si alguna fila no cumple el esquema o
    si hay IDs duplicados: preferimos fallar temprano antes de entrenar.
    """
    ejemplos: List[Ejemplo] = []
    for sub in ("manual", "real", "sintetico"):
        carpeta = data_dir / sub
        if not carpeta.exists():
            continue
        for ruta in sorted(carpeta.glob("*.jsonl")):
            ejemplos.extend(cargar_jsonl(ruta))

    # Validación de esquema e IDs únicos.
    vistos = set()
    for ej in ejemplos:
        res = validar_salida(ej.salida)
        if not res:
            raise ValueError(f"Ejemplo {ej.id!r} no cumple el esquema: {res.error}")
        if ej.id in vistos:
            raise ValueError(f"ID duplicado en el dataset: {ej.id!r}")
        vistos.add(ej.id)

    return ejemplos


@dataclass
class Splits:
    """Resultado de dividir el dataset."""

    train: List[Ejemplo]
    val: List[Ejemplo]
    test: List[Ejemplo]

    def resumen(self) -> str:
        return (
            f"train={len(self.train)} val={len(self.val)} test={len(self.test)} "
            f"(total={len(self.train) + len(self.val) + len(self.test)})"
        )


def dividir(
    ejemplos: List[Ejemplo],
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> Splits:
    """
    Divide en train/val/test **estratificando por `prioridad` y por `origen`**
    (humano = manual/real vs. sintético).

    Estrategia (determinística con `seed`):

    1. Se agrupa por (prioridad, humano/sintético).
    2. Cada estrato se baraja y se corta según las fracciones.

    Así los tres splits conservan el balance de prioridades y el ~20% de
    ejemplos escritos a mano, sin que ningún ejemplo caiga en dos splits.
    Las métricas por origen se pueden separar después con el campo `origen`.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
        "Las fracciones deben sumar 1.0"
    )
    rng = random.Random(seed)

    estratos: Dict[tuple, List[Ejemplo]] = defaultdict(list)
    for ej in ejemplos:
        estratos[(ej.prioridad, ej.origen in ORIGENES_HUMANOS)].append(ej)

    train: List[Ejemplo] = []
    val: List[Ejemplo] = []
    test: List[Ejemplo] = []

    # Recorremos los estratos en orden fijo (PRIORIDADES) para reproducibilidad.
    for prioridad in PRIORIDADES:
        for humano in (True, False):
            grupo = sorted(estratos.get((prioridad, humano), []), key=lambda e: e.id)
            if not grupo:
                continue
            rng.shuffle(grupo)

            n = len(grupo)
            n_test = round(n * frac_test)
            n_val = round(n * frac_val)
            test.extend(grupo[:n_test])
            val.extend(grupo[n_test : n_test + n_val])
            train.extend(grupo[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return Splits(train=train, val=val, test=test)


def guardar_splits(splits: Splits, splits_dir: Path) -> None:
    """Escribe train.jsonl / val.jsonl / test.jsonl en `splits_dir`."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    for nombre, datos in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        ruta = splits_dir / f"{nombre}.jsonl"
        with open(ruta, "w", encoding="utf-8") as f:
            for ej in datos:
                fila = {
                    "id": ej.id,
                    "mensaje": ej.mensaje,
                    "salida": ej.salida,
                    "origen": ej.origen,
                }
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def cargar_split(splits_dir: Path, nombre: str) -> List[Ejemplo]:
    """Carga un split ya generado (``train`` | ``val`` | ``test``)."""
    return cargar_jsonl(splits_dir / f"{nombre}.jsonl")
