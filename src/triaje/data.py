"""
Carga, validación y splits del dataset de triaje.
=================================================

Los datos viven en `data/` como JSONL. Cada fila::

    {"id": "...", "mensaje": "...", "salida": {<esquema>}, "origen": "manual|real|sintetico"}

Este módulo:
- carga y valida todas las filas contra `schema.py`,
- arma splits train/val/test **estratificados por `prioridad`** con semilla fija,
- prioriza los ejemplos `manual`/`real` para el **test** (CLAUDE.md §5),
- garantiza que el test **nunca** se use para entrenar ni elegir hiperparámetros.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .schema import PRIORIDADES, validar_salida

# Orígenes que priorizamos para el test (datos de mayor calidad / realismo).
ORIGENES_PREFERIDOS_TEST = ("manual", "real")


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
    Divide en train/val/test **estratificando por `prioridad`** y priorizando los
    ejemplos `manual`/`real` para el test.

    Estrategia (determinística con `seed`):

    1. Se agrupa por prioridad (estratos).
    2. Dentro de cada estrato, los ejemplos preferidos (manual/real) se ponen
       primero en la "cola" del test, hasta llenar la cuota de test del estrato;
       el resto de test se completa con sintéticos.
    3. El remanente se reparte entre train y val según las fracciones.

    Así el test queda estratificado por prioridad y cargado de datos de calidad,
    sin que ningún ejemplo caiga en dos splits.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
        "Las fracciones deben sumar 1.0"
    )
    rng = random.Random(seed)

    por_prioridad: Dict[str, List[Ejemplo]] = defaultdict(list)
    for ej in ejemplos:
        por_prioridad[ej.prioridad].append(ej)

    train: List[Ejemplo] = []
    val: List[Ejemplo] = []
    test: List[Ejemplo] = []

    # Recorremos los estratos en orden fijo (PRIORIDADES) para reproducibilidad.
    for prioridad in PRIORIDADES:
        grupo = por_prioridad.get(prioridad, [])
        if not grupo:
            continue

        preferidos = [e for e in grupo if e.origen in ORIGENES_PREFERIDOS_TEST]
        resto = [e for e in grupo if e.origen not in ORIGENES_PREFERIDOS_TEST]
        rng.shuffle(preferidos)
        rng.shuffle(resto)

        n = len(grupo)
        n_test = max(1, round(n * frac_test)) if n > 0 else 0

        # Test: primero los preferidos, luego sintéticos hasta la cuota.
        cola_test = preferidos + resto
        test_estrato = cola_test[:n_test]
        remanente = cola_test[n_test:]

        # Del remanente, separar val y train.
        rng.shuffle(remanente)
        n_val = round(n * frac_val)
        val_estrato = remanente[:n_val]
        train_estrato = remanente[n_val:]

        train.extend(train_estrato)
        val.extend(val_estrato)
        test.extend(test_estrato)

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
