#!/usr/bin/env python
"""
Generador de ejemplos SINTÉTICOS para el triaje.
================================================

Genera mensajes de pacientes con salida JSON etiquetada, marcados como
``origen: "sintetico"`` (NUNCA ``manual`` ni ``real``; esos los escribe el equipo).

Diseño para que las etiquetas sean coherentes por construcción:

- Hay un **banco de casos base** (`CASOS`), cada uno con sus etiquetas fijas
  (especialidad, prioridad, acción, grupo_edad, signos_alarma, síntomas, resumen)
  y una o varias redacciones base del mensaje.
- Sobre cada redacción se aplican **variaciones de estilo** que solo cambian la
  superficie del texto (saludos, jerga peruana, faltas de ortografía, emojis,
  registro formal/coloquial). Las etiquetas no se tocan → siempre son válidas.

Todo es determinístico con `--seed`. Uso::

    python scripts/generar_sintetico.py --n 1200 --salida data/sintetico/lote01.jsonl
    python scripts/generar_sintetico.py --n 30 --seed 7 --salida /tmp/prueba.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from triaje.schema import validar_salida  # noqa: E402

# ── Banco de casos base ─────────────────────────────────────────────────────
# Cada caso: etiquetas fijas + redacciones base (mensajes "neutros"). Las
# redacciones ya traen la info de edad cuando aplica, coherente con grupo_edad.


def _caso(esp, prio, accion, edad, alarma, sintomas, resumen, mensajes):
    return {
        "especialidad": esp,
        "prioridad": prio,
        "accion": accion,
        "grupo_edad": edad,
        "signos_alarma": alarma,
        "sintomas": sintomas,
        "resumen": resumen,
        "mensajes": mensajes,
    }


CASOS: List[Dict] = [
    # ── cardiologia ──
    _caso("cardiologia", "I", "emergencia_inmediata", "adulto", True,
          ["dolor de pecho", "sudoracion"],
          "Paciente adulto con dolor torácico irradiado y sudoración que sugiere emergencia cardiológica inmediata",
          ["Tengo un dolor fuerte en el pecho que se va al brazo izquierdo y estoy sudando frío",
           "Me aprieta el pecho desde hace media hora y me cuesta respirar, tengo miedo"]),
    _caso("cardiologia", "III", "cita_programada", "adulto_mayor", False,
          ["palpitaciones"],
          "Adulto mayor refiere palpitaciones al esfuerzo, requiere evaluación cardiológica programada no urgente",
          ["Mi papá de 68 años siente que el corazón le late rápido cuando camina rápido",
           "Últimamente me laten fuerte el corazón al subir escaleras, quisiera un chequeo"]),
    # ── pediatria ──
    _caso("pediatria", "II", "atencion_mismo_dia", "lactante", True,
          ["fiebre", "rechazo alimentario"],
          "Lactante con fiebre alta y rechazo del alimento, necesita atención pediátrica el mismo día",
          ["Mi bebé de 8 meses tiene 39 de fiebre y no quiere tomar nada",
           "Mi hijita de un año está con mucha fiebre y decaída, no come"]),
    _caso("pediatria", "IV", "cita_programada", "nino", False,
          ["tos"],
          "Niño con tos leve de varios días sin otros síntomas, puede verse en consulta programada",
          ["Mi hijo de 6 años tiene una tosecita hace unos días pero juega normal",
           "El niño tiene tos seca desde la semana pasada, nada más"]),
    # ── medicina_general ──
    _caso("medicina_general", "III", "atencion_mismo_dia", "adulto", False,
          ["fiebre", "malestar general"],
          "Adulto con fiebre y malestar general de inicio reciente, requiere evaluación médica el mismo día",
          ["Tengo fiebre y malestar general desde ayer, me siento muy mal",
           "Estoy con escalofríos, dolor de cuerpo y algo de fiebre hace un día"]),
    _caso("medicina_general", "IV", "orientacion_remota", "adulto", False,
          ["dolor de cabeza"],
          "Adulto con cefalea leve ocasional, consulta puede resolverse por orientación remota",
          ["Me duele un poco la cabeza en las tardes, ¿puedo tomar paracetamol?",
           "Tengo dolor de cabeza leve hace dos días, no sé si preocuparme"]),
    # ── ginecologia_obstetricia ──
    _caso("ginecologia_obstetricia", "I", "emergencia_inmediata", "adulto", True,
          ["sangrado vaginal"],
          "Gestante con sangrado vaginal activo, situación de emergencia obstétrica que requiere atención inmediata",
          ["Estoy embarazada de 7 meses y tengo sangrado, me asusté mucho",
           "Tengo 6 meses de embarazo y estoy botando sangre, ¿qué hago?"]),
    _caso("ginecologia_obstetricia", "IV", "cita_programada", "adulto", False,
          ["control prenatal"],
          "Gestante solicita cita de control prenatal de rutina, gestión no urgente programable",
          ["Quisiera una cita para mi control de embarazo de rutina",
           "Necesito agendar mi chequeo prenatal del mes"]),
    # ── traumatologia ──
    _caso("traumatologia", "I", "emergencia_inmediata", "adulto_mayor", True,
          ["fractura expuesta", "sangrado"],
          "Adulto mayor con fractura expuesta y sangrado tras caída, emergencia traumatológica inmediata",
          ["Mi papá se cayó y el hueso del brazo quedó salido y está sangrando",
           "Mi abuelo se resbaló y tiene el hueso de la pierna afuera, sangra mucho"]),
    _caso("traumatologia", "III", "atencion_mismo_dia", "adolescente", False,
          ["dolor de tobillo", "inflamacion"],
          "Adolescente con esguince de tobillo tras caída deportiva, requiere atención el mismo día",
          ["Mi hijo de 15 se torció el tobillo jugando fútbol y está hinchado",
           "Me caí jugando y no puedo apoyar bien el tobillo, está inflamado"]),
    # ── neurologia ──
    _caso("neurologia", "I", "emergencia_inmediata", "adulto_mayor", True,
          ["dificultad para hablar", "asimetria facial"],
          "Adulto mayor con habla alterada y desviación facial súbita, posible ACV, emergencia neurológica",
          ["Mi esposo de repente no puede hablar bien y se le torció la cara de un lado",
           "Mi mamá de 70 no mueve bien un brazo y habla raro desde hace un rato"]),
    _caso("neurologia", "III", "cita_programada", "adulto", False,
          ["cefalea"],
          "Adulto con cefaleas frecuentes de semanas de evolución, requiere evaluación neurológica programada",
          ["Tengo dolores de cabeza frecuentes en las tardes hace dos semanas",
           "Me dan jaquecas seguido últimamente, quisiera que me revisen"]),
    # ── gastroenterologia ──
    _caso("gastroenterologia", "I", "emergencia_inmediata", "adulto", True,
          ["vomito con sangre", "dolor abdominal"],
          "Adulto con vómito con sangre y dolor abdominal intenso, emergencia digestiva de atención inmediata",
          ["Estoy vomitando sangre desde la mañana y me duele mucho el estómago",
           "Boto sangre cuando vomito y tengo un dolor fuerte en la barriga"]),
    _caso("gastroenterologia", "III", "atencion_mismo_dia", "adulto", False,
          ["diarrea", "dolor abdominal"],
          "Adulto con diarrea de dos días y dolor abdominal moderado, requiere atención el mismo día",
          ["Tengo diarrea hace dos días y algo de dolor de barriga",
           "Ando con cólicos y diarrea desde el fin de semana"]),
    # ── dermatologia ──
    _caso("dermatologia", "III", "cita_programada", "adulto", False,
          ["lesion cutanea"],
          "Adulto con lesión cutánea que cambia de aspecto, requiere evaluación dermatológica programada",
          ["Me salió una mancha oscura que ha crecido y cambia de forma",
           "Tengo un lunar que se está poniendo raro y más grande"]),
    _caso("dermatologia", "IV", "cita_programada", "adolescente", False,
          ["acne"],
          "Adolescente con acné persistente solicita tratamiento dermatológico, consulta no urgente programable",
          ["Mi hija de 16 tiene mucho acné y quisiéramos un tratamiento",
           "Tengo acné desde hace meses y quiero que me vea un dermatólogo"]),
    # ── salud_mental ──
    _caso("salud_mental", "I", "emergencia_inmediata", "adulto", True,
          ["ideacion suicida"],
          "Adulto expresa ideación suicida, situación de riesgo vital que requiere atención en salud mental inmediata",
          ["Ya no aguanto más, tengo pensamientos de hacerme daño",
           "Siento que no vale la pena seguir, he pensado en quitarme la vida"]),
    _caso("salud_mental", "III", "cita_programada", "adulto", False,
          ["ansiedad", "insomnio"],
          "Adulto con ansiedad e insomnio de semanas, requiere evaluación en salud mental de forma programada",
          ["Me siento muy ansioso y no puedo dormir hace semanas",
           "Ando con mucha angustia y no logro dormir bien últimamente"]),
    # ── otorrinolaringologia ──
    _caso("otorrinolaringologia", "III", "atencion_mismo_dia", "nino", False,
          ["otalgia", "otorrea"],
          "Niño con dolor de oído y secreción, requiere evaluación otorrinolaringológica el mismo día",
          ["A mi hijo de 5 años le duele mucho el oído y le sale líquido",
           "El niño llora por dolor de oído y tiene como pus saliendo"]),
    _caso("otorrinolaringologia", "IV", "orientacion_remota", "adulto", False,
          ["odinofagia"],
          "Adulto con dolor de garganta leve de pocos días, consulta puede resolverse por orientación remota",
          ["Tengo la garganta irritada y me cuesta un poco tragar hace tres días",
           "Me duele la garganta desde el lunes, ¿qué me recomiendan tomar?"]),
    # ── administrativo ──
    _caso("administrativo", "IV", "gestion_administrativa", "no_especificado", False,
          [],
          "Consulta administrativa sobre horarios de atención de la clínica, sin motivo clínico",
          ["¿A qué hora atienden los sábados?",
           "Buenas, ¿cuál es el horario de atención de la clínica?"]),
    _caso("administrativo", "IV", "gestion_administrativa", "no_especificado", False,
          [],
          "Paciente solicita recoger resultados de análisis, trámite administrativo sin urgencia clínica",
          ["Quiero recoger los resultados de mi análisis de sangre",
           "¿Ya están listos mis exámenes? Quisiera pasar a recogerlos"]),
    _caso("administrativo", "IV", "gestion_administrativa", "no_especificado", False,
          [],
          "Paciente desea reprogramar una cita existente, gestión administrativa sin componente clínico",
          ["Necesito reprogramar mi cita del viernes para otro día",
           "¿Puedo cambiar la fecha de mi cita con el doctor?"]),
    # ── caso difícil: urgencia escondida en mensaje que parece leve ──
    _caso("cardiologia", "II", "atencion_mismo_dia", "adulto_mayor", True,
          ["dolor de pecho", "dificultad para respirar"],
          "Adulto mayor minimiza molestia pero describe dolor de pecho y falta de aire, requiere atención el mismo día",
          ["No creo que sea nada, pero mi mamá de 65 siente una molestia en el pecho y se agita al caminar",
           "Seguro es gastritis, pero me aprieta el pecho y me falta el aire a ratos"]),
]

# ── Variaciones de estilo (solo cambian la superficie del mensaje) ──────────

_SALUDOS = ["", "Hola, ", "Buenas, ", "Buenas tardes, ", "Doctor, ", "Disculpe, ",
            "Hola doc, ", "Buen día, "]
_CIERRES = ["", " gracias", " por favor", " porfa", " ayúdenme por favor",
            " ¿me pueden ayudar?", " agradezco su ayuda", " 🙏", " 😟", " 😢"]
_EMOJIS = ["", "", "", " 😷", " 🤒", " 😥", " 🙏", " ❗"]

# Faltas de ortografía típicas (jerga/tipeo peruano). Sustituciones suaves.
_TYPOS = {
    "que": "ke", "porque": "xq", "por favor": "xfa", "está": "esta",
    "años": "anios", "también": "tmb", "muchísimo": "muchisimo",
    "mi hijo": "mi ijo", "hace": "ase", "sangre": "sangere",
}


def _meter_typos(texto: str, rng: random.Random, prob: float = 0.5) -> str:
    """Aplica algunas faltas de ortografía al azar (determinístico con rng)."""
    for bien, mal in _TYPOS.items():
        if bien in texto and rng.random() < prob:
            texto = texto.replace(bien, mal, 1)
    return texto


def _variar(mensaje: str, rng: random.Random) -> str:
    """Devuelve una variante de superficie del mensaje."""
    texto = mensaje
    # A veces, faltas de ortografía / jerga.
    if rng.random() < 0.4:
        texto = _meter_typos(texto, rng)
    # Saludo inicial (respeta mayúscula si no hay saludo).
    saludo = rng.choice(_SALUDOS)
    if saludo:
        texto = saludo + texto[0].lower() + texto[1:]
    # Cierre y/o emoji.
    if rng.random() < 0.5:
        texto += rng.choice(_CIERRES)
    if rng.random() < 0.3:
        texto += rng.choice(_EMOJIS)
    return texto.strip()


def generar(n: int, seed: int) -> List[Dict]:
    """Genera `n` ejemplos sintéticos balanceados, determinísticos con `seed`."""
    rng = random.Random(seed)
    ejemplos: List[Dict] = []
    vistos_msg = set()  # evita mensajes repetidos exactos

    # Barajamos el orden de (caso, redacción base) para balancear clases.
    pares = []
    for caso in CASOS:
        for base in caso["mensajes"]:
            pares.append((caso, base))

    idx = 0
    intentos = 0
    max_intentos = n * 20
    while len(ejemplos) < n and intentos < max_intentos:
        intentos += 1
        caso, base = pares[idx % len(pares)]
        idx += 1
        mensaje = _variar(base, rng)
        if mensaje in vistos_msg:
            continue
        vistos_msg.add(mensaje)

        salida = {
            "especialidad": caso["especialidad"],
            "prioridad": caso["prioridad"],
            "accion": caso["accion"],
            "grupo_edad": caso["grupo_edad"],
            "signos_alarma": caso["signos_alarma"],
            "sintomas": list(caso["sintomas"]),
            "resumen": caso["resumen"],
        }
        # Garantía: cada ejemplo cumple el esquema antes de agregarse.
        if not validar_salida(salida):
            continue

        ejemplos.append({
            "id": f"syn_{len(ejemplos) + 1:05d}",
            "mensaje": mensaje,
            "salida": salida,
            "origen": "sintetico",
        })

    return ejemplos


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Genera ejemplos sintéticos de triaje.")
    parser.add_argument("--n", type=int, default=1200, help="Número de ejemplos.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla.")
    parser.add_argument("--salida", type=str, default="data/sintetico/lote01.jsonl")
    args = parser.parse_args(argv)

    ejemplos = generar(args.n, args.seed)
    ruta = Path(args.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for ej in ejemplos:
            f.write(json.dumps(ej, ensure_ascii=False) + "\n")

    print(f"Generados {len(ejemplos)} ejemplos sintéticos en {ruta}")
    if len(ejemplos) < args.n:
        print(f"[AVISO] Se pidieron {args.n} pero solo se pudieron generar "
              f"{len(ejemplos)} únicos. Amplía el banco CASOS o baja --n.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
