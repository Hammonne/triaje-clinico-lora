"""
Análisis de los resultados experimentales.
===========================================

A partir de `results/resultados.csv` (lo genera `experimentos.py`), este módulo:

- **ANOVA de dos factores con réplicas** (factores: `r` y el ratio alpha/r;
  réplicas: las semillas) sobre una métrica de desempeño.
- **Regresión** del desempeño vs `r` (lineal, log(r) y polinómica) para hallar el
  punto de **rendimientos decrecientes**.
- **Gráficos** de desempeño, VRAM, tiempo y costo (LoRA vs DoRA, baseline).
- **Comparación de tecnologías:** ANOVA técnica × r (LoRA vs DoRA con el mismo r,
  alpha/r y semillas).
- **Análisis de errores** sobre las predicciones guardadas: matriz de confusión de
  prioridad, sub-triaje por nivel de urgencia, confusiones más frecuentes por campo
  y detección de signos de alarma.
- **Escenarios de costo** contra herramientas de pago: fine-tuning por API, GPU en
  la nube e inferencia por token, con el punto de equilibrio.

Todo recibe un `DataFrame` ya cargado, para que los notebooks solo orquesten.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import ORDEN_PRIORIDAD, PRIORIDADES

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


# ── Comparación de tecnologías (LoRA vs DoRA) ───────────────────────────────


def anova_tecnica_por_r(
    df: pd.DataFrame,
    metrica: str = METRICA_DEFECTO,
    ratio: int = 2,
    rs=(4, 16, 64),
) -> pd.DataFrame:
    """
    ANOVA ``metrica ~ C(tecnica) * C(r)`` sobre las corridas LoRA y DoRA con el
    mismo ratio alpha/r y los mismos r (réplicas = semillas).

    Responde objetivamente si DoRA cambia la métrica respecto de LoRA (factor
    `tecnica`) y si ese efecto depende del rango (interacción).
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    sub = df[
        df["tecnica"].isin(["lora", "dora"])
        & df["r"].isin(rs)
        & ((df["alpha"] / df["r"]).round() == ratio)
    ].copy()
    if sub["tecnica"].nunique() < 2:
        raise ValueError("Faltan corridas de LoRA o de DoRA con esa configuración")
    sub["r"] = sub["r"].astype(int)
    modelo = ols(f"{metrica} ~ C(tecnica) * C(r)", data=sub).fit()
    return sm.stats.anova_lm(modelo, typ=2)


def variabilidad_semillas(df: pd.DataFrame, metrica: str = METRICA_DEFECTO) -> pd.DataFrame:
    """
    Media, desviación estándar y coeficiente de variación (d.e. / media) de una
    métrica entre réplicas, por configuración. Mide qué tan reproducible es cada
    configuración frente al cambio de semilla.
    """
    sub = df[~df["es_baseline"]]
    tabla = sub.groupby(["tecnica", "r", "alpha"])[metrica].agg(["mean", "std", "count"])
    tabla["cv"] = tabla["std"] / tabla["mean"]
    return tabla


# ── Análisis de errores (sobre las predicciones guardadas) ──────────────────


def cargar_predicciones(results_dir: Path, run_id: str) -> List[Dict]:
    """Lee ``pred_<run_id>.jsonl`` (una predicción por ejemplo del test)."""
    with open(results_dir / f"pred_{run_id}.jsonl", encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def _validas(preds: List[Dict]) -> List[Dict]:
    """Solo las predicciones que cumplieron el esquema (las demás no tienen campos)."""
    return [p for p in preds if p.get("prediccion_parseada")]


def matriz_confusion_prioridad(preds: List[Dict]) -> pd.DataFrame:
    """Matriz real × predicha de `prioridad` (filas = real), en el orden I..IV."""
    ok = _validas(preds)
    real = pd.Categorical([p["real"]["prioridad"] for p in ok], categories=PRIORIDADES)
    pred = pd.Categorical(
        [p["prediccion_parseada"]["prioridad"] for p in ok], categories=PRIORIDADES
    )
    return pd.crosstab(
        pd.Series(real, name="real"), pd.Series(pred, name="predicha"), dropna=False
    )


def subtriaje_por_nivel(preds: List[Dict]) -> pd.DataFrame:
    """
    Tasa de sub-triaje desagregada por la prioridad REAL del caso.

    La tasa global mezcla casos que no se pueden sub-triar (un IV nunca puede
    recibir una prioridad menos urgente) con los urgentes, así que puede esconder
    que casi todo el error se concentra en las prioridades I y II.
    """
    filas = []
    for nivel in PRIORIDADES:
        casos = [p for p in _validas(preds) if p["real"]["prioridad"] == nivel]
        sub = sum(
            ORDEN_PRIORIDAD[p["prediccion_parseada"]["prioridad"]] > ORDEN_PRIORIDAD[nivel]
            for p in casos
        )
        filas.append({
            "prioridad_real": nivel,
            "casos": len(casos),
            "sub_triados": sub,
            "tasa_subtriaje": sub / len(casos) if casos else float("nan"),
        })
    return pd.DataFrame(filas).set_index("prioridad_real")


def confusiones_frecuentes(preds: List[Dict], campo: str, top: int = 5) -> pd.DataFrame:
    """Los pares (real → predicho) más frecuentes entre los errores de un campo."""
    ok = _validas(preds)
    errores = Counter(
        (p["real"][campo], p["prediccion_parseada"][campo])
        for p in ok
        if p["real"][campo] != p["prediccion_parseada"][campo]
    )
    filas = [
        {"real": r, "predicho": q, "casos": n, "pct_de_validas": n / len(ok)}
        for (r, q), n in errores.most_common(top)
    ]
    return pd.DataFrame(filas)


def metricas_signos_alarma(preds: List[Dict]) -> Dict[str, float]:
    """Sensibilidad (recall) y precisión de `signos_alarma` (la clase positiva = True)."""
    ok = _validas(preds)
    pares = [(p["real"]["signos_alarma"], p["prediccion_parseada"]["signos_alarma"]) for p in ok]
    vp = sum(1 for r, q in pares if r and q)
    fn = sum(1 for r, q in pares if r and not q)
    fp = sum(1 for r, q in pares if not r and q)
    return {
        "sensibilidad": vp / (vp + fn) if vp + fn else float("nan"),
        "precision": vp / (vp + fp) if vp + fp else float("nan"),
        "casos_con_alarma": vp + fn,
    }


# ── Escenarios de costo frente a herramientas de pago ───────────────────────


def escenarios_costo(
    df: pd.DataFrame,
    run_id_lora: str,
    tokens_in_prom: float,
    tokens_out_prom: float,
    tokens_entrenamiento: float,
    precio_gpu_hora: float,
    precio_api_in_1m: float,
    precio_api_out_1m: float,
    precio_finetuning_1m: float,
    horas_mes: float,
) -> Dict[str, float]:
    """
    Compara, en USD, entrenar y servir nuestro LoRA en una GPU alquilada contra
    usar herramientas de pago (fine-tuning por API e inferencia por token).

    - Entrenamiento: horas reales de GPU de la corrida × precio por hora, contra
      ``tokens_entrenamiento`` × precio de un servicio de fine-tuning.
    - Inferencia por cada 1000 mensajes: segundos por ejemplo medidos (generación
      en lotes) × precio por hora, contra tokens de entrada/salida × precio API.
    - Servir 24/7: una GPU dedicada cuesta ``horas_mes × precio_gpu_hora`` al mes,
      con independencia del volumen; el punto de equilibrio es el volumen mensual
      a partir del cual sale más barato que pagar por token.

    Todos los precios son supuestos configurables (ver `Config`), no cotizaciones.
    """
    fila = df[df["run_id"] == run_id_lora].iloc[0]
    horas_entrenamiento = fila["segundos_entrenamiento"] / 3600.0
    seg_por_mensaje = fila["segundos_por_ejemplo"]

    entrenar_gpu = horas_entrenamiento * precio_gpu_hora
    entrenar_api = tokens_entrenamiento / 1e6 * precio_finetuning_1m
    inferir_gpu_1000 = 1000 * seg_por_mensaje / 3600.0 * precio_gpu_hora
    inferir_api_1000 = 1000 * (
        tokens_in_prom / 1e6 * precio_api_in_1m + tokens_out_prom / 1e6 * precio_api_out_1m
    )
    gpu_dedicada_mes = horas_mes * precio_gpu_hora
    # Mensajes que la GPU dedicada puede atender en un mes a la velocidad medida.
    capacidad_mes = horas_mes * 3600.0 / seg_por_mensaje
    equilibrio_mes = gpu_dedicada_mes / (inferir_api_1000 / 1000)

    return {
        "entrenar_lora_gpu_usd": entrenar_gpu,
        "entrenar_finetuning_api_usd": entrenar_api,
        "inferencia_gpu_por_1000_usd": inferir_gpu_1000,
        "inferencia_api_por_1000_usd": inferir_api_1000,
        "gpu_dedicada_mes_usd": gpu_dedicada_mes,
        "capacidad_gpu_mensajes_mes": capacidad_mes,
        "equilibrio_mensajes_mes": equilibrio_mes,
    }


# ── Gráficos ────────────────────────────────────────────────────────────────


def _preparar_figuras_dir(results_dir: Path) -> Path:
    figuras = results_dir / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    return figuras


def graficar_desempeno_vs_r(
    df: pd.DataFrame, results_dir: Path, metrica: str = METRICA_DEFECTO
) -> Path:
    """
    Desempeño vs r, una línea por técnica y por ratio alpha/r (promediando
    réplicas; las barras son la desviación estándar entre semillas).

    No se promedian los ratios entre sí: el ratio tiene efecto significativo, y
    mezclarlo haría incomparables las técnicas (DoRA solo se corre con α/r = 2).
    """
    import matplotlib.pyplot as plt

    figuras = _preparar_figuras_dir(results_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    estilos = {("lora", 2): ("tab:blue", "o", "-"), ("lora", 1): ("tab:blue", "s", "--"),
               ("dora", 2): ("tab:orange", "D", "-"), ("dora", 1): ("tab:orange", "^", "--")}
    sub_ok = df[~df["es_baseline"]].copy()
    sub_ok["ratio"] = (sub_ok["alpha"] / sub_ok["r"]).round().astype(int)
    for (tecnica, ratio), sub in sub_ok.groupby(["tecnica", "ratio"]):
        agg = sub.groupby("r")[metrica].agg(["mean", "std"]).reset_index().sort_values("r")
        color, marcador, linea = estilos.get((tecnica, ratio), (None, "o", "-"))
        ax.errorbar(agg["r"], agg["mean"], yerr=agg["std"].fillna(0), color=color,
                    marker=marcador, ls=linea, capsize=3,
                    label=f"{tecnica.upper()} α/r = {ratio}")
    # Línea del baseline si existe.
    base = df[df["es_baseline"]]
    if not base.empty:
        ax.axhline(base[metrica].mean(), ls=":", color="gray", label="baseline (sin LoRA)")
    # r se duplica/cuadruplica entre niveles: en escala log2 los puntos quedan
    # equiespaciados y se ve mejor la saturación.
    rs = sorted(sub_ok["r"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(rs)
    ax.set_xticklabels([str(int(r)) for r in rs])
    ax.set_xlabel("r (rango del LoRA, escala log2)")
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


def graficar_subtriaje_por_nivel(
    results_dir: Path, run_ids: Dict[str, List[str]]
) -> Path:
    """
    Barras de tasa de sub-triaje por prioridad real, una serie por configuración.
    `run_ids` mapea una etiqueta (p. ej. "LoRA r=64") a sus réplicas; se suman los
    casos de todas las réplicas.
    """
    import matplotlib.pyplot as plt

    figuras = _preparar_figuras_dir(results_dir)
    niveles = [n for n in PRIORIDADES if n != "IV"]  # un IV no se puede sub-triar
    ancho = 0.8 / len(run_ids)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (etiqueta, ids) in enumerate(run_ids.items()):
        tablas = [subtriaje_por_nivel(cargar_predicciones(results_dir, rid)) for rid in ids]
        total = sum(t[["casos", "sub_triados"]] for t in tablas)
        tasas = (total["sub_triados"] / total["casos"]).loc[niveles]
        xs = np.arange(len(niveles)) + i * ancho
        ax.bar(xs, tasas.values, width=ancho, label=etiqueta)
    ax.set_xticks(np.arange(len(niveles)) + ancho * (len(run_ids) - 1) / 2)
    ax.set_xticklabels([f"Prioridad {n}" for n in niveles])
    ax.set_ylim(0, 1)
    ax.set_ylabel("tasa de sub-triaje")
    ax.set_title("Sub-triaje según la prioridad real del caso")
    ax.legend()
    fig.tight_layout()
    ruta = figuras / "subtriaje_por_nivel.png"
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
