"""
Análisis de los resultados experimentales.
===========================================

A partir de `results/resultados.csv` (lo genera `experimentos.py`), este módulo:

- **ANOVA de dos factores con réplicas** (factores: `r` y el ratio alpha/r;
  réplicas: las semillas) sobre una métrica de desempeño.
- **Regresión** del desempeño vs `r` (lineal, log(r) y polinómica) para hallar el
  punto de **rendimientos decrecientes**.
- **Gráficos** de desempeño, VRAM, tiempo y costo (LoRA vs DoRA, baseline).
- **Comparación de costo** contra resolver el test set con una API de pago.

Todo recibe un `DataFrame` ya cargado, para que los notebooks solo orquesten.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Métrica de desempeño por defecto para los análisis (cumplimiento del esquema).
METRICA_DEFECTO = "esquema_ok_post"


def cargar_resultados(ruta_csv: Path) -> pd.DataFrame:
    """Carga `resultados.csv` y añade columnas derivadas útiles para el análisis."""
    df = pd.read_csv(ruta_csv)
    # Separamos baseline de las corridas con LoRA/DoRA.
    df["es_baseline"] = df["tecnica"] == "base"
    return df


def _solo_factorial(
    df: pd.DataFrame, tecnica: str, rs_factorial=(4, 16, 64)
) -> pd.DataFrame:
    """
    Filtra a las corridas del factorial final de una técnica (excluye baseline y
    el piloto de extremos r∈{2,128}). Deja las réplicas (semillas) para el ANOVA.
    """
    sub = df[(df["tecnica"] == tecnica) & (df["r"].isin(rs_factorial))].copy()
    sub["r"] = sub["r"].astype(int)
    # ratio alpha/r como factor categórico (1 o 2).
    sub["ratio"] = (sub["alpha"].astype(int) / sub["r"]).round().astype(int)
    return sub


def anova_dos_factores(
    df: pd.DataFrame, tecnica: str = "lora", metrica: str = METRICA_DEFECTO
) -> pd.DataFrame:
    """
    ANOVA de dos factores con réplicas: ``metrica ~ C(r) * C(ratio)``.

    Devuelve la tabla ANOVA (suma de cuadrados tipo II) con los p-valores de cada
    factor y de su interacción. Requiere al menos 2 réplicas (semillas) por celda.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    sub = _solo_factorial(df, tecnica)
    if sub.empty:
        raise ValueError(f"No hay corridas factoriales para la técnica {tecnica!r}")

    modelo = ols(f"{metrica} ~ C(r) * C(ratio)", data=sub).fit()
    return sm.stats.anova_lm(modelo, typ=2)


@dataclass
class ResultadoRegresion:
    """Ajustes de regresión del desempeño vs r y el punto de rendim. decrecientes."""

    coef_lineal: Dict[str, float]
    coef_log: Dict[str, float]
    coef_poly2: Dict[str, float]
    r2: Dict[str, float]
    mejor_modelo: str
    r_rendimientos_decrecientes: Optional[float]


def regresion_desempeno_vs_r(
    df: pd.DataFrame,
    tecnica: str = "lora",
    metrica: str = METRICA_DEFECTO,
    umbral_pendiente: float = 0.001,
) -> ResultadoRegresion:
    """
    Ajusta 3 modelos de ``metrica`` vs ``r`` (promediando réplicas y ratios por r):
    lineal, logarítmico (en log(r)) y polinómico de grado 2. Reporta R² de cada uno,
    elige el de mejor R² y estima el **r de rendimientos decrecientes**: el valor de
    `r` a partir del cual la mejora marginal (derivada) cae por debajo de
    `umbral_pendiente` (por unidad de r).
    """
    sub = _solo_factorial(df, tecnica)
    agg = sub.groupby("r")[metrica].mean().reset_index().sort_values("r")
    r = agg["r"].to_numpy(dtype=float)
    y = agg[metrica].to_numpy(dtype=float)

    # Ajustes con numpy.polyfit (sobre r y sobre log(r)).
    lin = np.polyfit(r, y, 1)
    poly2 = np.polyfit(r, y, 2)
    logc = np.polyfit(np.log(r), y, 1)  # y = a*log(r) + b

    def _r2(y_real, y_pred):
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2 = {
        "lineal": _r2(y, np.polyval(lin, r)),
        "log": _r2(y, np.polyval(logc, np.log(r))),
        "poly2": _r2(y, np.polyval(poly2, r)),
    }
    mejor = max(r2, key=r2.get)

    # Rendimientos decrecientes: usamos el modelo logarítmico (derivada a/r), que
    # modela naturalmente la saturación. a/r < umbral  =>  r > a/umbral.
    a_log = logc[0]
    r_dec: Optional[float] = None
    if a_log > 0:
        r_dec = float(a_log / umbral_pendiente)

    return ResultadoRegresion(
        coef_lineal={"pendiente": float(lin[0]), "intercepto": float(lin[1])},
        coef_log={"a": float(logc[0]), "b": float(logc[1])},
        coef_poly2={"a": float(poly2[0]), "b": float(poly2[1]), "c": float(poly2[2])},
        r2=r2,
        mejor_modelo=mejor,
        r_rendimientos_decrecientes=r_dec,
    )


def comparar_costo_api(
    df: pd.DataFrame,
    n_test: int,
    tokens_in_prom: float,
    tokens_out_prom: float,
    precio_in_1m: float,
    precio_out_1m: float,
) -> Dict[str, float]:
    """
    Compara el costo de resolver el test set con una API de pago vs. el costo
    simulado de entrenar+inferir con nuestro LoRA (promedio de las corridas).

    Devuelve un dict con ambos costos para graficar/reportar.
    """
    costo_api = (
        n_test * tokens_in_prom / 1e6 * precio_in_1m
        + n_test * tokens_out_prom / 1e6 * precio_out_1m
    )
    costo_lora = float(df.loc[~df["es_baseline"], "costo_usd"].mean())
    return {
        "costo_api_usd": costo_api,
        "costo_lora_entrenamiento_prom_usd": costo_lora,
    }


# ── Gráficos ────────────────────────────────────────────────────────────────


def _preparar_figuras_dir(results_dir: Path) -> Path:
    figuras = results_dir / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    return figuras


def graficar_desempeno_vs_r(
    df: pd.DataFrame, results_dir: Path, metrica: str = METRICA_DEFECTO
) -> Path:
    """Desempeño vs r, una línea por técnica (promediando réplicas y ratios)."""
    import matplotlib.pyplot as plt

    figuras = _preparar_figuras_dir(results_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tecnica in ("lora", "dora"):
        sub = df[df["tecnica"] == tecnica]
        if sub.empty:
            continue
        agg = sub.groupby("r")[metrica].mean().reset_index().sort_values("r")
        ax.plot(agg["r"], agg[metrica], marker="o", label=tecnica.upper())
    # Línea del baseline si existe.
    base = df[df["es_baseline"]]
    if not base.empty:
        ax.axhline(base[metrica].mean(), ls="--", color="gray", label="baseline")
    ax.set_xlabel("r (rango del LoRA)")
    ax.set_ylabel(metrica)
    ax.set_title(f"Desempeño ({metrica}) vs r")
    ax.legend()
    fig.tight_layout()
    ruta = figuras / f"desempeno_vs_r_{metrica}.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def graficar_costo_recurso(
    df: pd.DataFrame, results_dir: Path, columna: str, titulo: str, ylabel: str
) -> Path:
    """Gráfico genérico de un recurso (VRAM, tiempo, costo) vs r por técnica."""
    import matplotlib.pyplot as plt

    figuras = _preparar_figuras_dir(results_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tecnica in ("lora", "dora"):
        sub = df[df["tecnica"] == tecnica]
        if sub.empty:
            continue
        agg = sub.groupby("r")[columna].mean().reset_index().sort_values("r")
        ax.plot(agg["r"], agg[columna], marker="s", label=tecnica.upper())
    ax.set_xlabel("r (rango del LoRA)")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    ruta = figuras / f"{columna}_vs_r.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def generar_todos_los_graficos(
    df: pd.DataFrame, results_dir: Path, metrica: str = METRICA_DEFECTO
) -> List[Path]:
    """Genera el set completo de figuras del análisis y devuelve sus rutas."""
    rutas = [
        graficar_desempeno_vs_r(df, results_dir, metrica),
        graficar_costo_recurso(
            df, results_dir, "vram_entrenamiento_gb",
            "VRAM de entrenamiento vs r", "VRAM pico (GB)",
        ),
        graficar_costo_recurso(
            df, results_dir, "segundos_entrenamiento",
            "Tiempo de entrenamiento vs r", "segundos",
        ),
        graficar_costo_recurso(
            df, results_dir, "costo_usd",
            "Costo simulado de entrenamiento vs r", "USD",
        ),
        graficar_costo_recurso(
            df, results_dir, "tasa_subtriaje",
            "Tasa de sub-triaje vs r (seguridad clínica)", "tasa de sub-triaje",
        ),
    ]
    return rutas
