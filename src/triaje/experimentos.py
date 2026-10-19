"""
Runner del diseño experimental factorial, con reanudación.
==========================================================

Corre las configuraciones del diseño (LoRA y DoRA) sobre la misma GPU/datos,
guardando cada resultado apenas termina (robustez ante desconexiones de Colab).
El runner **se salta las corridas cuyo `run_id` ya está en `resultados.csv`**, así
se puede reanudar dándole play de nuevo.

Diseño (CLAUDE.md §6):
- Piloto con extremos: r ∈ {2, 128}.
- Factorial final: r ∈ {4, 16, 64} × (alpha/r) ∈ {1, 2} = 6 configuraciones.
- Réplicas: una semilla por réplica (42, 123).
Todo lo demás queda fijo entre corridas.
"""

import csv
import dataclasses
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, set_seed
from .data import Ejemplo, cargar_split
from .evaluate import Metricas, evaluar


def costo_simulado_usd(segundos: float, precio_hora: float) -> float:
    """Costo simulado = horas de GPU × precio por hora (configurable en Config)."""
    return (segundos / 3600.0) * precio_hora


def _grid_lora_dora(
    rs_piloto=(2, 128),
    rs_factorial=(4, 16, 64),
    ratios=(1, 2),
    semillas=(42, 123),
    incluir_dora=True,
    incluir_piloto=True,
) -> List[Dict]:
    """
    Devuelve la lista de combinaciones (como dicts) del diseño. Cada dict trae
    r, alpha, seed, use_dora. La semilla se asigna por réplica (ronda).
    """
    combos: List[Dict] = []
    tecnicas = [False, True] if incluir_dora else [False]

    for use_dora in tecnicas:
        for seed in semillas:
            if incluir_piloto:
                for r in rs_piloto:
                    combos.append({"r": r, "alpha": r * 2, "seed": seed, "use_dora": use_dora})
            for r in rs_factorial:
                for ratio in ratios:
                    combos.append(
                        {"r": r, "alpha": r * ratio, "seed": seed, "use_dora": use_dora}
                    )
    return combos


def _config_para_combo(base: Config, combo: Dict) -> Config:
    """Clona la Config base y le aplica los factores de una corrida."""
    cfg = deepcopy(base)
    cfg.lora_r = combo["r"]
    cfg.lora_alpha = combo["alpha"]
    cfg.seed = combo["seed"]
    cfg.use_dora = combo["use_dora"]
    return cfg


def _run_ids_existentes(ruta_csv: Path) -> set:
    """Lee los run_id ya presentes en resultados.csv (para reanudar)."""
    if not ruta_csv.exists():
        return set()
    with open(ruta_csv, encoding="utf-8", newline="") as f:
        return {fila["run_id"] for fila in csv.DictReader(f) if fila.get("run_id")}


def _fila_resultado(cfg: Config, metr: Metricas, res_train, costo_usd: float) -> Dict:
    """Aplana una corrida a una fila de resultados.csv."""
    fila = {
        "run_id": cfg.run_id(),
        "tecnica": "dora" if cfg.use_dora else "lora",
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "ratio_alpha_r": cfg.ratio_alpha_r,
        "seed": cfg.seed,
        # Costo
        "parametros_entrenables": res_train.parametros_entrenables if res_train else "",
        "segundos_entrenamiento": round(res_train.segundos_entrenamiento, 2) if res_train else "",
        "vram_entrenamiento_gb": round(res_train.vram_pico_gb, 3) if res_train else "",
        "loss_final": round(res_train.loss_final, 4) if res_train and res_train.loss_final else "",
        "costo_usd": round(costo_usd, 4),
        # Desempeño
        "esquema_ok_post": round(metr.esquema_ok_post, 4),
        "json_valido_post": round(metr.json_valido_post, 4),
        "exact_match_categoricos": round(metr.exact_match_categoricos, 4),
        "sintomas_f1": round(metr.sintomas_f1, 4),
        "tasa_subtriaje": round(metr.tasa_subtriaje, 4),
        "segundos_por_ejemplo": round(metr.segundos_por_ejemplo, 4),
        "tokens_por_segundo": round(metr.tokens_por_segundo, 2),
        "vram_inferencia_gb": round(metr.vram_pico_inferencia_gb, 3),
    }
    # Un campo por accuracy y macro-F1 de cada categórico.
    for campo, val in metr.accuracy.items():
        fila[f"acc_{campo}"] = round(val, 4)
    for campo, val in metr.macro_f1.items():
        fila[f"f1_{campo}"] = round(val, 4)
    return fila


def _append_csv(ruta_csv: Path, fila: Dict) -> None:
    """Agrega una fila al CSV, escribiendo el header si es la primera vez."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_csv.exists()
    # Unificamos columnas: si el archivo ya existe, respetamos su header.
    with open(ruta_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if not existe:
            writer.writeheader()
        writer.writerow(fila)


def correr_experimento(
    cfg: Config,
    train_ejs: List[Ejemplo],
    val_ejs: List[Ejemplo],
    test_ejs: List[Ejemplo],
    entrenar_lora: bool = True,
) -> Dict:
    """
    Corre UNA configuración completa: carga modelo, (opcional) entrena LoRA/DoRA,
    evalúa y devuelve la fila de resultados + guarda el JSON de detalle y las
    predicciones en `results_dir`.
    """
    # Import diferido: estos módulos requieren torch/transformers (solo en GPU/Colab
    # o smoke-test). Mantener el import aquí permite usar el resto sin GPU.
    from .modelo import aplicar_lora, cargar_modelo_y_tokenizer
    from .train import ResultadoEntrenamiento, entrenar, guardar_adaptador

    set_seed(cfg.seed)
    run_id = cfg.run_id() if entrenar_lora else f"base_s{cfg.seed}"

    model, tokenizer = cargar_modelo_y_tokenizer(cfg, para_entrenar=entrenar_lora)

    res_train: Optional[ResultadoEntrenamiento] = None
    if entrenar_lora:
        model = aplicar_lora(model, cfg)
        res_train = entrenar(model, tokenizer, train_ejs, val_ejs, cfg)
        guardar_adaptador(model, tokenizer, cfg)

    # Guardamos predicciones por corrida (para análisis cualitativo).
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    ruta_pred = cfg.results_dir / f"pred_{run_id}.jsonl"
    metr = evaluar(model, tokenizer, test_ejs, cfg, guardar_predicciones_en=str(ruta_pred))

    costo = costo_simulado_usd(
        res_train.segundos_entrenamiento if res_train else 0.0, cfg.precio_gpu_usd_hora
    )
    fila = _fila_resultado(cfg, metr, res_train, costo)
    fila["run_id"] = run_id
    if not entrenar_lora:
        # El baseline no tiene LoRA: no tiene sentido reportar r/alpha/tecnica.
        fila["tecnica"] = "base"
        fila["r"] = ""
        fila["alpha"] = ""
        fila["ratio_alpha_r"] = ""

    # JSON de detalle (métricas completas + config de la corrida).
    detalle = {
        "run_id": run_id,
        "config": dataclasses.asdict(cfg),
        "metricas": metr.to_dict(),
        "entrenamiento": dataclasses.asdict(res_train) if res_train else None,
    }
    with open(cfg.results_dir / f"detalle_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(detalle, f, ensure_ascii=False, indent=2, default=str)

    # Liberar memoria entre corridas (importante en la T4).
    try:
        import torch

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return fila


def correr_suite(
    base: Config,
    incluir_baseline: bool = True,
    combos: Optional[List[Dict]] = None,
    **kwargs_grid,
) -> None:
    """
    Corre todo el diseño con reanudación. Lee los splits desde `base.data_dir/splits`,
    y para cada corrida que NO esté ya en `resultados.csv`, la ejecuta y la agrega.

    `kwargs_grid` se pasan a `_grid_lora_dora` (p. ej. incluir_dora=False para
    una primera pasada solo-LoRA).
    """
    splits_dir = base.data_dir / "splits"
    train_ejs = cargar_split(splits_dir, "train")
    val_ejs = cargar_split(splits_dir, "val")
    test_ejs = cargar_split(splits_dir, "test")

    # En smoke-test recortamos a pocos ejemplos para que corra en CPU.
    if base.smoke_test:
        train_ejs = train_ejs[: base.smoke_n_ejemplos]
        val_ejs = val_ejs[: max(2, base.smoke_n_ejemplos // 2)]
        test_ejs = test_ejs[: base.smoke_n_ejemplos]

    ruta_csv = base.results_dir / "resultados.csv"
    ya_hechos = _run_ids_existentes(ruta_csv)

    tareas: List[Dict] = []
    if incluir_baseline:
        # Baseline: modelo sin LoRA (zero-shot con el mismo prompt), una semilla.
        tareas.append({"baseline": True, "seed": base.seed})

    for combo in combos or _grid_lora_dora(**kwargs_grid):
        tareas.append({"baseline": False, **combo})

    print(f"Diseño: {len(tareas)} corridas | ya hechas: {len(ya_hechos)}")
    for i, tarea in enumerate(tareas, 1):
        if tarea.get("baseline"):
            cfg = deepcopy(base)
            cfg.seed = tarea["seed"]
            run_id = f"base_s{cfg.seed}"
            entrenar_lora = False
        else:
            cfg = _config_para_combo(base, tarea)
            run_id = cfg.run_id()
            entrenar_lora = True

        if run_id in ya_hechos:
            print(f"  [{i}/{len(tareas)}] {run_id}: ya existe, se omite")
            continue

        print(f"  [{i}/{len(tareas)}] {run_id}: corriendo...")
        t0 = time.perf_counter()
        fila = correr_experimento(cfg, train_ejs, val_ejs, test_ejs, entrenar_lora=entrenar_lora)
        _append_csv(ruta_csv, fila)
        ya_hechos.add(run_id)
        print(f"      hecho en {time.perf_counter() - t0:.1f}s "
              f"(esquema_ok={fila['esquema_ok_post']}, subtriaje={fila['tasa_subtriaje']})")

    print(f"Suite terminada. Resultados en {ruta_csv}")
