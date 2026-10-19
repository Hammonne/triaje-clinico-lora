# Triaje clínico con LoRA — Hackatón 8 (equipo Las Grills)

Refinamiento de un LLM con **LoRA** (y comparación contra **DoRA**) para una tarea
de **triaje clínico**: a partir de un mensaje libre de un paciente o familiar a una
clínica en Perú, el modelo produce un objeto **JSON** estructurado con la
especialidad, prioridad, acción, grupo de edad, signos de alarma, síntomas y un
resumen.

Curso UTEC — docente Robert Buleje. El contexto completo del proyecto está en
[`CLAUDE.md`](CLAUDE.md) y el enunciado en [`docs/enunciado.md`](docs/enunciado.md).

> ⚠️ **Uso académico únicamente.** Este modelo **no** es una herramienta clínica
> real y **no** debe usarse para tomar decisiones médicas. Es un ejercicio de
> fine-tuning y diseño experimental.

## Esquema de salida

```json
{
  "especialidad": "medicina_general | pediatria | ... | administrativo",
  "prioridad": "I | II | III | IV",
  "accion": "emergencia_inmediata | atencion_mismo_dia | cita_programada | orientacion_remota | gestion_administrativa",
  "grupo_edad": "lactante | nino | adolescente | adulto | adulto_mayor | no_especificado",
  "signos_alarma": true,
  "sintomas": ["lista", "de", "sintomas", "normalizados"],
  "resumen": "resumen clínico de 5 a 25 palabras"
}
```

La **prioridad** sigue la NTS N° 042-MINSA/DGSP-V.01 (4 niveles). Las clases y la
validación viven en un único lugar: [`src/triaje/schema.py`](src/triaje/schema.py).
Los criterios de etiquetado con ejemplos están en
[`data/guia_etiquetado.md`](data/guia_etiquetado.md).

## Modelo y hardware

- **Modelo base:** `microsoft/Phi-3.5-mini-instruct` (3.8B), cargado en **4-bit NF4**
  (QLoRA) con bitsandbytes.
- **Hardware:** Google Colab **T4 (16 GB)**, la misma GPU para todos los experimentos.
- Ajustes T4: `fp16`, `compute_dtype=float16` (no hay bf16), `attn_implementation="eager"`
  (no hay flash-attention), `gradient_checkpointing`, `paged_adamw_8bit`.

## Instalación

### Local (para desarrollo y smoke-test en CPU)

```bash
pip install -e .
```

### Colab (con GPU)

```bash
git clone <URL-del-repo>
cd triaje-clinico-lora
pip install -r requirements.txt
pip install -e ".[gpu]"
# Montar Google Drive para guardar adaptadores y resultados
```

> Los adaptadores, checkpoints y `*.safetensors` **nunca** van a git: se guardan en
> Google Drive (ver `.gitignore`).

## Estructura

```
docs/enunciado.md            Enunciado del hackatón
referencia_profe/            Código de la clase (solo referencia, NO modificar)
data/
  guia_etiquetado.md         Criterios de etiquetado + 2 ejemplos por clase
  plantilla_manual.csv       Plantilla para escribir ejemplos a mano
  manual/ real/ sintetico/   Datos por origen (JSONL)
  splits/                    train / val / test (estratificados por prioridad)
src/triaje/                  Lógica del proyecto (config, schema, train, eval, ...)
scripts/                     Utilidades (validar dataset, generar sintético, ...)
notebooks/                   Orquestación reproducible (01..04)
results/                     CSV/JSON pequeños y figuras
```

## Validar el dataset

```bash
python scripts/validar_dataset.py data/plantilla_manual.csv
python scripts/validar_dataset.py data/sintetico/*.jsonl
```

Reporta errores por fila y el balance de clases (prioridad y especialidad).

## Estado

En construcción. Fases completadas:

- **Fase 0:** estructura del repo y dependencias fijadas.
- **Fase 1:** esquema (`schema.py`), configuración (`config.py`), guía de
  etiquetado, plantilla manual y validador de dataset.
- **Fase 2:** prompts, carga/validación de datos, splits estratificados y
  generación de ejemplos sintéticos.

## TO-DO del equipo (trabajo 100% humano)

Estas tareas las hace el equipo a mano; son parte de la nota y no deben
automatizarse:

- [ ] **Escribir a mano ≥20% del dataset** (`origen: manual`) usando
      `data/plantilla_manual.csv` y siguiendo `data/guia_etiquetado.md`.
      Priorizar casos **prioridad II** y ambiguos (jerga peruana, familiares,
      urgencias escondidas), que es donde más aporta el aporte humano.
- [ ] **Recolectar y anotar ejemplos reales** (`origen: real`), si se consiguen,
      anonimizados.
- [ ] **Revisar y corregir** los ejemplos sintéticos dudosos y resolver los casos
      marcados como "a revisar por el equipo" en la guía.
- [ ] **Ejecutar los notebooks en Google Colab con GPU T4** (forzar la T4 para que
      todos los experimentos sean comparables; montar Google Drive).
- [ ] **Lanzar los experimentos del diseño factorial** (LoRA y DoRA) y dejar que
      se guarden en Drive.
- [ ] **Redactar el análisis interpretativo**, las conclusiones y el PPT de la
      exposición (narrativa del equipo).
- [ ] **Tomar las decisiones clínicas finales** de etiquetado donde haya dudas
      (criterio del equipo, no inventar).
