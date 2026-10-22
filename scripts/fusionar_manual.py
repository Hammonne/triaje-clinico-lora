#!/usr/bin/env python
"""
Fusiona los CSV de etiquetado manual de `data/manual_fuentes/` en uno solo.
===================================================================

Cada miembro entregó su CSV con formatos distintos (sin cabecera, con/sin BOM,
con/sin columna `notas`, comas sin comillas en `resumen`, booleanos en distintas
capitalizaciones, ids repetidos entre archivos...). Este script:

1. Lee cada CSV de forma tolerante y lo normaliza al formato de la plantilla
   (`id, mensaje, especialidad, prioridad, accion, grupo_edad, signos_alarma,
   sintomas, resumen, origen, notas`).
2. Descarta duplicados exactos (mismo mensaje), p. ej. un export que ya contiene
   otros archivos.
3. Valida cada fila con `triaje.schema.validar_salida` (fuente única de verdad).
4. Resuelve colisiones de `id` asignando ids nuevos (ver `REMAPEO` para los valores
   fuera del esquema que se convierten a clases válidas).
5. Escribe:
   - `data/manual_fusionado.csv`   filas válidas (para revisar/editar a mano)
   - `data/manual/manual.jsonl`    las mismas filas en el formato que carga el pipeline
   - `data/manual_ids_reasignados.csv`  id original -> id nuevo
   - `data/manual_pendientes.csv`  filas con errores de esquema (para decidir
     como equipo; NO se corrigen automáticamente porque son decisiones clínicas)

Uso::

    python scripts/fusionar_manual.py [carpeta_entrada]

Los avisos de coherencia prioridad vs accion son solo informativos.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from triaje.schema import ESPECIALIDADES, PRIORIDADES, validar_salida  # noqa: E402

CABECERA = ["id", "mensaje", "especialidad", "prioridad", "accion", "grupo_edad",
            "signos_alarma", "sintomas", "resumen", "origen", "notas"]
ORIGENES_VALIDOS = {"manual", "real", "sintetico"}
_BOOL = {"true": True, "verdadero": True, "si": True, "sí": True, "1": True,
         "false": False, "falso": False, "no": False, "0": False}

# Valores fuera del esquema que usó un miembro del equipo, y su equivalente.
# Decisión del equipo: las especialidades sin clase propia van a medicina_general
# (puerta de entrada por defecto, ver guia_etiquetado.md); ante la duda de acción
# se prefiere la más urgente (se prefiere sobre-triar que sub-triar).
REMAPEO = {
    "especialidad": {
        "psiquiatria": "salud_mental",
        "neumologia": "medicina_general",
        "urologia": "medicina_general",
        "oftalmologia": "medicina_general",
        "endocrinologia": "medicina_general",
    },
    "accion": {
        "derivar_urgencias": "emergencia_inmediata",
        "cita_prioritaria_48h": "cita_programada",   # 48 h no es "mismo día"
        "orientacion_autocuidado": "orientacion_remota",
    },
}

# Coherencia prioridad -> acciones típicas (guia_etiquetado.md). Solo informativo.
_ACCIONES_TIPICAS = {
    "I": {"emergencia_inmediata"},
    "II": {"emergencia_inmediata", "atencion_mismo_dia"},
    "III": {"atencion_mismo_dia", "cita_programada"},
    "IV": {"cita_programada", "orientacion_remota", "gestion_administrativa"},
}


def _clave_mensaje(texto: str) -> str:
    return " ".join(texto.lower().split())


def leer_csv(ruta: Path) -> List[Dict[str, str]]:
    """Lee un CSV tolerando cabecera ausente, BOM, 10/11 columnas y comas sueltas."""
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if filas and filas[0][0].strip().lower() == "id":
        cabecera, filas = [c.strip().lower() for c in filas[0]], filas[1:]
    else:
        cabecera = CABECERA  # archivo sin cabecera
    n = len(cabecera)
    salida = []
    for campos in filas:
        if len(campos) > n and campos[-1].strip() in ORIGENES_VALIDOS \
                and campos[n - 1].strip() not in ORIGENES_VALIDOS | {""}:
            # `resumen` con comas sin comillas: lo reunimos y movemos origen.
            # (solo aplica a archivos con cabecera de 10 columnas)
            idx_resumen = cabecera.index("resumen")
            campos = (campos[:idx_resumen]
                      + [",".join(campos[idx_resumen:-1]).strip(), campos[-1]])
        fila = dict(zip(cabecera, [c.strip() for c in campos]))
        salida.append({k: fila.get(k, "") for k in CABECERA})
    return salida


def fila_a_salida(fila: Dict[str, str]) -> dict:
    alarma = fila["signos_alarma"].strip().lower()
    return {
        "especialidad": fila["especialidad"],
        "prioridad": fila["prioridad"],
        "accion": fila["accion"],
        "grupo_edad": fila["grupo_edad"],
        "signos_alarma": _BOOL.get(alarma, fila["signos_alarma"]),
        "sintomas": [s.strip() for s in fila["sintomas"].split(";") if s.strip()],
        "resumen": fila["resumen"],
    }


def main(argv: List[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    carpeta = Path(argv[0]) if argv else _REPO / "data" / "manual_fuentes"
    archivos = sorted(carpeta.glob("*.csv"))
    print(f"Archivos en {carpeta}: {len(archivos)}")

    vistos: Dict[str, str] = {}          # mensaje normalizado -> "archivo:id"
    ids_usados: Counter = Counter()
    validas: List[Dict[str, str]] = []
    pendientes: List[Dict[str, str]] = []
    errores_por_archivo: Dict[str, Counter] = {}
    stats = {}

    for ruta in archivos:
        filas = leer_csv(ruta)
        s = Counter(leidas=len(filas))
        errores_por_archivo[ruta.name] = Counter()
        for fila in filas:
            clave = _clave_mensaje(fila["mensaje"])
            if clave in vistos:
                s["duplicadas"] += 1
                continue
            vistos[clave] = f"{ruta.name}:{fila['id']}"
            if not fila["origen"]:
                fila["origen"] = "manual"
            for campo, tabla in REMAPEO.items():
                if fila[campo] in tabla:
                    fila[campo] = tabla[fila[campo]]
                    s["remapeadas"] += 1
            fila["signos_alarma"] = str(
                _BOOL.get(fila["signos_alarma"].lower(), fila["signos_alarma"])).lower()

            error = None
            if fila["origen"] not in ORIGENES_VALIDOS:
                error = f"origen inválido {fila['origen']!r}"
            else:
                res = validar_salida(fila_a_salida(fila))
                if not res:
                    error = res.error
            if error:
                s["con_error"] += 1
                errores_por_archivo[ruta.name][error.split(":")[0]] += 1
                pendientes.append({**fila, "notas": f"[{ruta.name}] {error}; {fila['notas']}".strip("; ")})
                continue
            fila["_archivo"] = ruta.name
            validas.append(fila)
            s["validas"] += 1
        stats[ruta.name] = s

    # Resolver colisiones de id: primer archivo conserva el id, el resto recibe uno nuevo.
    for fila in validas + pendientes:
        ids_usados[fila["id"]] += 1
    siguiente = max(int(re.sub(r"\D", "", i) or 0) for i in ids_usados) + 1
    asignados: set = set()
    renumeradas = []
    for fila in validas + pendientes:
        if fila["id"] in asignados:
            nuevo = f"man_{siguiente:04d}"
            siguiente += 1
            renumeradas.append((fila["id"], nuevo))
            fila["id"] = nuevo
        asignados.add(fila["id"])

    with open(_REPO / "data" / "manual_ids_reasignados.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_original", "id_nuevo"])
        w.writerows(renumeradas)

    validas.sort(key=lambda r: r["id"])
    pendientes.sort(key=lambda r: r["id"])
    ruta_pend = _REPO / "data" / "manual_pendientes.csv"
    ruta_pend.unlink(missing_ok=True)  # solo existe si hay filas con errores
    salidas = [(_REPO / "data" / "manual_fusionado.csv", validas)]
    if pendientes:
        salidas.append((ruta_pend, pendientes))
    for ruta_salida, filas in salidas:
        with open(ruta_salida, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CABECERA, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)

    # JSONL que consume `triaje.data.cargar_todos` (data/manual/*.jsonl).
    with open(_REPO / "data" / "manual" / "manual.jsonl", "w", encoding="utf-8") as f:
        for r in validas:
            f.write(json.dumps({"id": r["id"], "mensaje": r["mensaje"],
                                "salida": fila_a_salida(r), "origen": r["origen"]},
                               ensure_ascii=False) + "\n")

    # ── Reporte ─────────────────────────────────────────────────────────────
    print("\nPor archivo:")
    for nombre, s in stats.items():
        print(f"  {nombre:<40} leídas={s['leidas']:<4} válidas={s['validas']:<4} "
              f"duplicadas={s['duplicadas']:<4} remapeadas={s['remapeadas']:<3} con_error={s['con_error']}")
        for tipo, n in errores_por_archivo[nombre].most_common():
            print(f"      - {tipo}: {n}")
    print(f"\nIds reasignados por colisión: {len(renumeradas)}")
    print(f"Válidas  -> data/manual_fusionado.csv  : {len(validas)}")
    print(f"Pendientes -> data/manual_pendientes.csv: {len(pendientes)}")

    print("\nBalance (válidas) por prioridad:")
    c = Counter(r["prioridad"] for r in validas)
    for p in PRIORIDADES:
        print(f"  {p:>3}: {c.get(p, 0)}")
    print("Balance (válidas) por especialidad:")
    c = Counter(r["especialidad"] for r in validas)
    for e in ESPECIALIDADES:
        print(f"  {e:<26}: {c.get(e, 0)}")

    avisos = [r for r in validas if r["accion"] not in _ACCIONES_TIPICAS[r["prioridad"]]]
    print(f"\nAvisos de coherencia prioridad vs accion: {len(avisos)}")
    for r in avisos:
        print(f"  {r['id']}: prioridad {r['prioridad']} con accion {r['accion']}")
    return 0 if not pendientes else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
