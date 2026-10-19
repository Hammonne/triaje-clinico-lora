#!/usr/bin/env python
"""
Validador de ejemplos del dataset de triaje.
============================================

Valida los ejemplos escritos a mano (CSV de `data/plantilla_manual.csv`) o
cualquier archivo JSONL del dataset, contra el esquema de `src/triaje/schema.py`.
Reporta, por fila, el primer error encontrado, y al final un resumen con el
balance de clases (útil para no dejar clases vacías o desbalanceadas).

Uso::

    python scripts/validar_dataset.py data/plantilla_manual.csv
    python scripts/validar_dataset.py data/sintetico/lote01.jsonl
    python scripts/validar_dataset.py data/manual/*.csv

Formato esperado:

- **CSV** (plantilla manual): columnas `id, mensaje, especialidad, prioridad,
  accion, grupo_edad, signos_alarma, sintomas, resumen, origen[, notas]`.
  - `signos_alarma`: ``true``/``false`` (también se aceptan ``si``/``no``, ``1``/``0``).
  - `sintomas`: lista separada por ``;`` (p. ej. ``fiebre;tos``). Vacío = lista vacía.
  - Las filas con `origen == "ejemplo_borrar"` se ignoran (son muestras).
- **JSONL**: una fila por línea con `id`, `mensaje`, `salida` (dict con el
  esquema completo) y `origen`.

No tiene números mágicos propios: todas las clases y reglas vienen de
`src/triaje/schema.py`.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Permite ejecutar el script sin instalar el paquete (añade src/ al path).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from triaje.schema import (  # noqa: E402
    CAMPOS_CATEGORICOS,
    ESPECIALIDADES,
    PRIORIDADES,
    validar_salida,
)

# Orígenes válidos (ver CLAUDE.md §5). "ejemplo_borrar" es solo muestra.
ORIGENES_VALIDOS = {"manual", "real", "sintetico"}
_BOOL_TRUE = {"true", "verdadero", "si", "sí", "1"}
_BOOL_FALSE = {"false", "falso", "no", "0"}


def _parse_bool(valor: str) -> Any:
    """Convierte el texto de `signos_alarma` a bool; si no reconoce, lo deja igual
    para que el validador reporte el error de tipo."""
    v = str(valor).strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    return valor


def _parse_sintomas(valor: str) -> List[str]:
    """Separa por ';' y limpia espacios. Vacío -> []."""
    if valor is None:
        return []
    return [s.strip() for s in str(valor).split(";") if s.strip()]


def _fila_csv_a_ejemplo(fila: Dict[str, str]) -> Tuple[str, str, Dict[str, Any]]:
    """Convierte una fila del CSV a (id, origen, salida-dict)."""
    salida = {
        "especialidad": (fila.get("especialidad") or "").strip(),
        "prioridad": (fila.get("prioridad") or "").strip(),
        "accion": (fila.get("accion") or "").strip(),
        "grupo_edad": (fila.get("grupo_edad") or "").strip(),
        "signos_alarma": _parse_bool(fila.get("signos_alarma", "")),
        "sintomas": _parse_sintomas(fila.get("sintomas", "")),
        "resumen": (fila.get("resumen") or "").strip(),
    }
    return (fila.get("id", "").strip(), (fila.get("origen") or "").strip(), salida)


def _leer_csv(ruta: Path) -> List[Tuple[str, str, Dict[str, Any]]]:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        return [_fila_csv_a_ejemplo(fila) for fila in lector]


def _leer_jsonl(ruta: Path) -> List[Tuple[str, str, Dict[str, Any]]]:
    ejemplos = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            obj = json.loads(linea)
            ejemplos.append(
                (str(obj.get("id", f"linea_{n}")), obj.get("origen", ""), obj.get("salida", {}))
            )
    return ejemplos


def validar_archivo(ruta: Path) -> Tuple[int, int, Counter, Counter]:
    """
    Valida un archivo. Devuelve (n_ok, n_total_validados, contador_prioridad,
    contador_especialidad) e imprime los errores por fila.
    """
    if ruta.suffix.lower() == ".csv":
        ejemplos = _leer_csv(ruta)
    elif ruta.suffix.lower() in (".jsonl", ".json"):
        ejemplos = _leer_jsonl(ruta)
    else:
        raise ValueError(f"Extensión no soportada: {ruta.suffix} ({ruta})")

    n_ok = 0
    n_validados = 0
    conteo_prioridad: Counter = Counter()
    conteo_especialidad: Counter = Counter()
    ids_vistos: Counter = Counter()

    print(f"\n=== {ruta} ===")
    for id_, origen, salida in ejemplos:
        if origen == "ejemplo_borrar":
            continue  # fila de muestra de la plantilla
        n_validados += 1
        ids_vistos[id_] += 1

        # Validación de 'origen' (no lo cubre el esquema de salida).
        if origen not in ORIGENES_VALIDOS:
            print(f"  [ERROR] {id_}: origen inválido {origen!r} "
                  f"(debe ser uno de {sorted(ORIGENES_VALIDOS)})")
            continue

        res = validar_salida(salida)
        if res:
            n_ok += 1
            conteo_prioridad[salida["prioridad"]] += 1
            conteo_especialidad[salida["especialidad"]] += 1
        else:
            print(f"  [ERROR] {id_}: {res.error}")

    # IDs duplicados
    duplicados = [i for i, c in ids_vistos.items() if c > 1]
    if duplicados:
        print(f"  [ERROR] IDs duplicados: {duplicados}")

    return n_ok, n_validados, conteo_prioridad, conteo_especialidad


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Valida ejemplos del dataset de triaje.")
    parser.add_argument("rutas", nargs="+", help="Archivos CSV o JSONL a validar.")
    args = parser.parse_args(argv)

    total_ok = 0
    total = 0
    conteo_prioridad: Counter = Counter()
    conteo_especialidad: Counter = Counter()

    for patron in args.rutas:
        # Soporta comodines pasados como texto (p. ej. en shells que no expanden).
        rutas = sorted(Path().glob(patron)) if any(c in patron for c in "*?[") else [Path(patron)]
        for ruta in rutas:
            if not ruta.exists():
                print(f"[AVISO] No existe: {ruta}")
                continue
            ok, n, cp, ce = validar_archivo(ruta)
            total_ok += ok
            total += n
            conteo_prioridad += cp
            conteo_especialidad += ce

    # ── Resumen global ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Ejemplos válidos: {total_ok}/{total}")

    print("\nBalance por prioridad:")
    for p in PRIORIDADES:
        print(f"  {p:>3}: {conteo_prioridad.get(p, 0)}")

    print("\nBalance por especialidad:")
    for e in ESPECIALIDADES:
        marca = "" if conteo_especialidad.get(e, 0) > 0 else "  <-- vacía"
        print(f"  {e:<26}: {conteo_especialidad.get(e, 0)}{marca}")

    # Código de salida != 0 si hubo algún inválido (útil en CI / pre-commit).
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
