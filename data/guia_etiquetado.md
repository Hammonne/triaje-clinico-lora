# Guía de etiquetado — Triaje clínico

Esta guía define **cómo etiquetar** cada mensaje de paciente. Es obligatoria para
quien escriba ejemplos `manual` y para revisar los `sintetico`. El esquema y las
listas de clases son los de `src/triaje/schema.py` (fuente única de verdad); si
algo cambia ahí, esta guía se actualiza.

> **Importante:** este trabajo es **académico**, no es una herramienta clínica
> real. Los criterios buscan consistencia en el etiquetado, no reemplazar el
> juicio médico. **No inventes criterios clínicos: si un caso no está claro,
> etiquétalo lo mejor posible y márcalo en la columna `notas` como
> "a revisar por el equipo".**

## Esquema de salida

```json
{
  "especialidad": "...",
  "prioridad": "I | II | III | IV",
  "accion": "...",
  "grupo_edad": "...",
  "signos_alarma": true,
  "sintomas": ["..."],
  "resumen": "5 a 25 palabras"
}
```

## Reglas generales (orden de decisión)

1. **Primero la seguridad (`prioridad` y `signos_alarma`).** Ante la duda entre
   dos prioridades, elige la **más urgente** (es preferible sobre-triar que
   sub-triar; la métrica de seguridad penaliza el sub-triaje).
2. **Luego la `especialidad`** según el sistema/órgano dominante del motivo de
   consulta. Si es claramente no médico (cita, pago, resultado) →
   `administrativo`.
3. **`accion`** es consecuencia de la prioridad y del tipo de caso (ver tabla de
   coherencia abajo).
4. **`grupo_edad`** por la edad explícita o pistas ("mi bebé", "mi hijo de 15",
   "mi mamá de 70"). Si no hay ninguna pista → `no_especificado`.
5. **`sintomas`**: lista de síntomas **normalizados** (ver más abajo). Vacía `[]`
   en casos puramente administrativos.
6. **`resumen`**: 5 a 25 palabras, en tercera persona, sin datos personales.

### Coherencia prioridad ↔ acción (regla práctica, no rígida)

| prioridad | acción típica |
| --- | --- |
| I  | `emergencia_inmediata` |
| II | `emergencia_inmediata` o `atencion_mismo_dia` |
| III | `atencion_mismo_dia` o `cita_programada` |
| IV | `cita_programada` u `orientacion_remota` |
| (no médico) | `gestion_administrativa` |

### Normalización de `sintomas`

- En minúsculas, sin tildes opcionales pero consistentes, forma canónica:
  `"dolor de pecho"`, `"fiebre"`, `"dificultad para respirar"`, `"sangrado"`,
  `"vomitos"`, `"mareo"`, `"dolor abdominal"`.
- Un síntoma por elemento; no metas la causa ni la duración en el string.
- Traduce jerga a término estándar: *"me falta el aire"* → `"dificultad para respirar"`;
  *"bota sangre"* → `"sangrado"`.

### `signos_alarma`

`true` si el mensaje describe algún signo de alarma que exige evaluación urgente,
por ejemplo: dolor torácico opresivo, dificultad respiratoria, pérdida de
conciencia, déficit neurológico súbito (no habla, no mueve un lado), sangrado
abundante, convulsiones, fiebre alta en lactante, ideación suicida, trauma grave.
`false` en consultas leves, dudas o trámites.

---

## `prioridad` — NTS N° 042-MINSA/DGSP-V.01

Norma Técnica de Salud de los Servicios de Emergencia (RM N° 386-2006/MINSA).
Usamos **4 niveles** (algunas escalas hospitalarias modernas usan 5; lo
justificamos en el README/exposición).

- **I — Gravedad súbita extrema.** Riesgo de vida inminente, atención inmediata.
  *Ej.:* paro, inconsciencia, dolor torácico con sudoración, dificultad
  respiratoria severa, convulsión activa, sangrado masivo.
- **II — Emergencia.** Riesgo de vida que necesita atención dentro de un tiempo
  vital (no puede esperar horas). *Ej.:* dolor torácico sin inestabilidad, fiebre
  alta en lactante, fractura expuesta, deshidratación marcada.
- **III — Urgencia.** Necesita evaluación médica pero la vida no está
  comprometida. *Ej.:* fiebre de 2 días, dolor abdominal moderado, esguince,
  infección urinaria.
- **IV — Urgencia menor / no urgente.** Puede esperar o atenderse en consulta
  ambulatoria. *Ej.:* resfrío leve, control, receta, dolor crónico estable,
  consulta dermatológica no urgente.

> Los casos `administrativo` no llevan prioridad clínica real; por convención los
> etiquetamos como **IV** (no urgente) para mantener el campo siempre poblado.

---

## `especialidad` — criterios y 2 ejemplos por clase

> En los ejemplos se muestra el mensaje y los campos clave. La `accion`,
> `grupo_edad` y `resumen` se completan con las reglas de arriba.

### `medicina_general`
Molestias generales, inespecíficas o multisistémicas sin órgano dominante claro,
o puerta de entrada por defecto.
- *"Tengo fiebre y malestar general desde ayer, ¿debo preocuparme?"* →
  prioridad III, signos_alarma false, sintomas `["fiebre","malestar general"]`.
- *"Me siento cansado, con dolor de cuerpo y algo de tos hace 3 días."* →
  prioridad III, sintomas `["fatiga","dolor muscular","tos"]`.

### `pediatria`
El paciente es menor (lactante, niño, adolescente) y el motivo es clínico.
- *"Mi bebé de 8 meses tiene fiebre de 39 y no quiere comer."* → grupo_edad
  lactante, prioridad II, signos_alarma true, sintomas `["fiebre","rechazo alimentario"]`.
- *"Mi hija de 6 años tiene ronchas en la piel después de comer maní."* →
  grupo_edad nino, prioridad II, signos_alarma true, sintomas `["urticaria"]`.

### `ginecologia_obstetricia`
Embarazo, salud ginecológica, sangrados vaginales, controles obstétricos.
- *"Estoy embarazada de 7 meses y tengo sangrado, ¿qué hago?"* → prioridad I,
  signos_alarma true, sintomas `["sangrado vaginal"]`, grupo_edad adulto.
- *"Quisiera una cita para mi control de embarazo de rutina."* → prioridad IV,
  accion cita_programada, sintomas `[]` o `["control prenatal"]`.

### `traumatologia`
Lesiones musculoesqueléticas, fracturas, esguinces, golpes, dolor articular por
trauma.
- *"Me caí jugando fútbol y no puedo apoyar el tobillo, está hinchado."* →
  prioridad III, sintomas `["dolor de tobillo","inflamacion"]`.
- *"Mi papá se cayó y el hueso del brazo quedó salido, sangra."* → prioridad I,
  signos_alarma true, sintomas `["fractura expuesta","sangrado"]`.

### `cardiologia`
Síntomas cardiovasculares: dolor torácico, palpitaciones, presión alta, disnea de
esfuerzo.
- *"Siento un dolor fuerte en el pecho que va al brazo izquierdo y sudo frío."* →
  prioridad I, signos_alarma true, sintomas `["dolor de pecho","sudoracion"]`.
- *"Últimamente me late muy rápido el corazón cuando subo escaleras."* →
  prioridad III, sintomas `["palpitaciones"]`.

### `neurologia`
Cefaleas, convulsiones, déficits neurológicos, mareos de origen central.
- *"Mi esposo de repente no puede hablar bien y se le cayó la cara de un lado."* →
  prioridad I, signos_alarma true, sintomas `["dificultad para hablar","asimetria facial"]`.
- *"Tengo dolores de cabeza frecuentes en las tardes hace 2 semanas."* →
  prioridad III, sintomas `["cefalea"]`.

### `gastroenterologia`
Síntomas digestivos: dolor abdominal, diarrea, vómitos, sangrado digestivo.
- *"Vomito sangre desde esta mañana y me duele mucho el estómago."* → prioridad I,
  signos_alarma true, sintomas `["vomito con sangre","dolor abdominal"]`.
- *"Tengo diarrea hace 2 días y algo de dolor de barriga."* → prioridad III,
  sintomas `["diarrea","dolor abdominal"]`.

### `dermatologia`
Piel, lesiones cutáneas, erupciones, caída de cabello, acné (sin reacción
alérgica sistémica).
- *"Me salió una mancha oscura que ha crecido y cambia de forma."* → prioridad III,
  sintomas `["lesion cutanea"]`.
- *"Tengo acné desde hace meses y quisiera un tratamiento."* → prioridad IV,
  accion cita_programada, sintomas `["acne"]`.

### `salud_mental`
Ansiedad, depresión, crisis emocionales, ideación suicida, consumo.
- *"Ya no aguanto más, tengo pensamientos de hacerme daño."* → prioridad I,
  signos_alarma true, sintomas `["ideacion suicida"]`.
- *"Me siento muy ansioso y no puedo dormir hace semanas."* → prioridad III,
  sintomas `["ansiedad","insomnio"]`.

### `otorrinolaringologia`
Oído, nariz, garganta: otalgia, dolor de garganta, epistaxis, vértigo periférico.
- *"Me duele muchísimo el oído y me sale líquido."* → prioridad III,
  sintomas `["otalgia","otorrea"]`.
- *"Tengo la garganta irritada y me cuesta tragar hace 3 días."* → prioridad III,
  sintomas `["odinofagia"]`.

### `administrativo`
Casos **no médicos**: citas, resultados, pagos, horarios, seguros, reclamos.
- *"¿A qué hora atienden los sábados?"* → prioridad IV, accion
  gestion_administrativa, signos_alarma false, sintomas `[]`.
- *"Quiero recoger los resultados de mi análisis de sangre."* → prioridad IV,
  accion gestion_administrativa, sintomas `[]`.

---

## `accion` — criterios y 2 ejemplos por clase

### `emergencia_inmediata`
Requiere atención ya (prioridad I, a veces II).
- *"No puede respirar, se está poniendo morado."*
- *"Se desmayó y no responde."*

### `atencion_mismo_dia`
Debe verse hoy pero no es riesgo vital inmediato (prioridad II–III).
- *"Mi hijo tiene fiebre alta y vómitos desde temprano."*
- *"Me torcí el tobillo y está muy hinchado, ¿puedo ir hoy?"*

### `cita_programada`
Consulta que puede agendarse en los próximos días (prioridad III–IV).
- *"Quiero una cita con dermatología por mi acné."*
- *"Necesito control de mi presión con el cardiólogo."*

### `orientacion_remota`
Duda que puede resolverse por mensaje/teleconsulta, sin examen presencial urgente.
- *"¿Puedo tomar paracetamol si tengo solo dolor de cabeza leve?"*
- *"Tengo una duda sobre cómo tomar mi medicamento."*

### `gestion_administrativa`
Trámites no clínicos.
- *"¿Cuánto cuesta la consulta con pediatría?"*
- *"Necesito reprogramar mi cita del viernes."*

---

## `grupo_edad` — criterios y 2 ejemplos por clase

### `lactante` (menor de 2 años)
- *"Mi bebé de 10 meses..."*
- *"Recién nacido con..."*

### `nino` (2 a 11 años)
- *"Mi hijo de 6 años..."*
- *"La niña de 9 años..."*

### `adolescente` (12 a 17 años)
- *"Mi hija de 15 años..."*
- *"Un chico de 16..."*

### `adulto` (18 a 59 años)
- *"Tengo 34 años y..."*
- *"Mi esposo de 40..."*

### `adulto_mayor` (60 años a más)
- *"Mi mamá de 72 años..."*
- *"El abuelo de 80..."*

### `no_especificado`
No hay ninguna pista de edad.
- *"Me duele la cabeza hace días."*
- *"¿A qué hora atienden?"*
