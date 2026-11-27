# Protocolo para el reenvio del paper

Implementacion experimental del plan de reenvio, separada del protocolo legacy.
Incluye SisFall, UCI HAR y KFall. Los resultados KFall previos y el piloto de
15 epocas no sustituyen las nuevas ablaciones y replicas. No se han ejecutado aqui los entrenamientos completos
ni se han generado numeros para las tablas del articulo.

## Preparacion y ejecucion

Desde la raiz del repositorio, con el entorno PyTorch activado:

```bash
python run_paper_experiments.py \
  --sisfall-root data/SisFall_dataset \
  --ucihar-root "data/UCI HAR Dataset" \
  --kfall-root data/KFall \
  --label-dir data/labels_transitions \
  --output runs/paper_reenvio \
  --seeds 0 1 2 3 4 --epochs 60 --batch-size 32 --workers 0 --amp
```

Este comando **solo prepara** `manifest.json`: 125 entrenamientos por perfil:
25 SisFall, 25 UCI HAR y 75 KFall (tres ventanas), con cinco baselines y cinco
semillas. No usa la GPU.
Para ejecutar exactamente ese manifiesto en el servidor:

```bash
python run_paper_experiments.py \
  --sisfall-root data/SisFall_dataset \
  --ucihar-root "data/UCI HAR Dataset" \
  --kfall-root data/KFall \
  --label-dir data/labels_transitions \
  --output runs/paper_reenvio \
  --seeds 0 1 2 3 4 --epochs 60 --batch-size 32 --workers 0 --amp \
  --execute --resume
```

Se ejecutan calibraciones y entrenamientos secuenciales. El runner muestra el
comando/modelo actual, su posicion en la matriz y la salida del proceso en vivo,
incluidas las epocas; tambien conserva todo en `command_NNN.log`.
`--log-only` vuelve a ocultar la salida del hijo de la terminal, no los logs.
`completed_commands.json` permite omitir comandos terminados
al repetir `--execute --resume` con los mismos argumentos. Si un entrenamiento
interrumpido dejo una carpeta parcial, archivala antes de reintentar: el runner
no borra ni considera completo un checkpoint parcial.

Todos los entrenamientos utilizan `--skip-test`: el mejor checkpoint se escoge
solo con validacion. Por defecto, la evaluacion posterior tambien es de
validacion. Para incluir test al terminar toda la matriz, agrega
`--evaluate-test` **desde la preparacion del manifiesto**, conserva los demas
argumentos y congela previamente el protocolo. No cambies hiperparametros
despues de observar test. Una alternativa es evaluar checkpoints congelados:

```bash
python evaluate_paper_run.py --dataset sisfall --run-dir RUTA_RUN --split test
python evaluate_paper_run.py --dataset ucihar --run-dir RUTA_RUN --split test
python evaluate_paper_run.py --dataset kfall --run-dir RUTA_RUN --split test
```

`RUTA_RUN` contiene `config.json` y `checkpoints/best_model.pt`, no es la carpeta
superior de toda la matriz. Para un piloto usa `--pilot --seeds 0 --epochs 1`
y otra carpeta de salida. Un piloto no es evidencia publicable.

## Baselines y perfiles

| Variante | Entrada real al modelo | Activacion | Offset | Pasos SisFall / UCI |
|---|---|---|---|---|
| B1 | Senal nativa densa, sin Sigma-Delta | ReLU | No | 410 / 128 |
| B2 | Senal nativa densa, sin Sigma-Delta | LIF | No | 410 / 128 |
| B3 | Eventos Sigma-Delta interpolados | LIF | No | 2048 / 640 |
| B4 | Eventos Sigma-Delta interpolados | LIF | Si | 2048 / 640 |
| B5 | Eventos Sigma-Delta sin interpolacion | LIF | Si | 410 / 128 |

B1 **no** es ReLU sobre eventos. B2 paga MAC densos en la primera convolucion,
aunque la senal contenga ceros. B3 elimina completamente la rama offset.
B4/B5 comparten theta para aislar la interpolacion, sin recalibrar usando test.
Las entradas densas tienen seis canales; los eventos tienen doce canales al
separar polaridades. Se informa el numero real de parametros, no se supone que
ambas primeras convoluciones tengan igual tamano.

KFall compara B1/B2/B5 con 120/206/320 muestras nativas frente a B3/B4 con
1200/2060/3200 pasos, respectivamente; la arquitectura se comparte entre
variantes y cada ventana tiene calibracion TRAIN independiente.

SisFall se adquiere a **200 Hz**, UCI HAR a **50 Hz** y KFall a **100 Hz**. Los pasos adicionales son
interpolacion, no nuevas medidas ni una frecuencia de ADC de 10 kHz.
2048 es la longitud ya utilizada por el repositorio; no se presenta como 2050.
Los canales SisFall son acc XYZ y gyro XYZ en cuentas ADC; los UCI son
gyro XYZ en rad/s y total_acc XYZ en g. KFall usa acc XYZ en g y gyro XYZ en
grados/s; no se convierten sus giroscopios a rad/s silenciosamente.

### Ejecutar solo KFall

```bash
python run_paper_experiments.py --datasets kfall \
  --kfall-root data/KFall \
  --kfall-windows 120 206 320 \
  --output runs/paper_kfall --seeds 0 1 2 3 4 \
  --epochs 60 --amp --execute
```

Son **75 entrenamientos por perfil**. La seleccion y los resultados por defecto
usan validacion; agrega `--evaluate-test` desde la preparacion para evaluar test
una vez congelado el protocolo. Para incluir los tres datasets usa
`--datasets sisfall ucihar kfall` (tambien es el default).

KFall conserva:

- Split por sujetos **60/30/10**, como los runs anteriores; `--kfall-split`
  permite otro reparto sin alterar SisFall. El protocolo publicable exige los
  32 sujetos, sin mezclar sujetos entre train/val/test.
- Ventanas de 1.20/2.06/3.20 s nominales, strides 60/103/160 muestras.
- Etiquetas propias de `annotations/kfall_manual_labels.json` o su `.bak`
  si no existe el primero. `--kfall-annotation-file RUTA` fija otro JSON.
  `--kfall-label-dir` solo cambia las anotaciones **XLSX oficiales**, no el JSON.
- Exclusiones **T05 T13 T14 T16 T18 T19**. T15 no esta excluida por defecto:
  agregala explicitamente con `--kfall-exclude-tasks T05 T13 T14 T15 T16 T18 T19`
  si ese es el protocolo elegido. Sin valores, ese flag conserva todas las tareas.
- Cuatro clases de red: Fall, Dynamic, Static y PostImpact (1.2 s). Las seis
  actividades UCI de la GUI no sustituyen estas cuatro clases del entrenamiento.
  Los intervalos oficiales Fall/PostImpact prevalecen sobre etiquetas normales.
  Las grabaciones sin manual conservan el fallback advertido y registrado.

Estas reglas son el modo legacy por defecto. Con `--kfall-label-scheme extended`
se entrenan Fall, las nueve actividades individuales de `activity_intervals`
y PostImpact (11 clases); en este modo no se excluyen tareas por defecto y
se omiten las ventanas normales sin etiqueta detallada. La clase Fall y
PostImpact siguen teniendo prioridad. Para excluir tareas explicitamente usa
`--kfall-exclude-tasks T05 ...`. La calibracion y todas las variantes usan el
mismo esquema, y no deben mezclarse sus resultados con los del modo legacy.

Se exige un JSON manual en modo publicable. Cada variante/semilla/ventana usa
las mismas anotaciones y exclusiones; estas se guardan en la procedencia y
se verifican al reutilizar theta. Las posiciones FrameCounter se conservan,
tambien para ventanas con frames faltantes. B1/B2/B5 no interpolan la senal
nativa; B3/B4 y su calibracion respetan los tiempos fisicos de esos frames.
KFall no aporta gold temporal Stable/Transition: no se inventa una segunda
cabeza de postura ni una cohorte mayor SE equivalente a SisFall.

No se inventaron las configuraciones SCNsel/SCNacc/SCNminE. `reference` usa los
defaults actuales. Para comparar arquitecturas conocidas, entrega
`--profiles perfiles.json` al preparar y ejecutar. Ejemplo de formato, **no**
reproduccion de los perfiles del articulo:

```json
{
  "referencia": {},
  "reducida_ejemplo": {
    "conv_ch": [8, 8, 8],
    "kernels": [8, 5, 8],
    "strides": [4, 2, 1],
    "offset_hidden": 4,
    "p_drop": 0.2
  }
}
```

Claves permitidas: `conv_ch`, `kernels`, `strides`, `pool_kernels`,
`pool_strides`, `pool_paddings`, `tau`, `thresh`, `p_drop`, `offset_hidden`,
`hard_reset`. Los arrays convolucionales tienen tres entradas.

## Separacion y calibracion sin fuga

- SisFall: los 38 sujetos completos, split 64/16/20 antes de crear ventanas.
  Los 15 sujetos mayores SE01--SE15 deben seguir presentes. Son cinco semillas
  de holdout repetido, **no** cinco folds disjuntos.
- UCI: particion oficial, 21 sujetos de desarrollo y nueve de test; validacion
  separa sujetos del desarrollo. Se exigen 7352 segmentos de desarrollo y
  **2947** de test, a 50 Hz y 128 muestras, sin recortar segmentos extra.
- KFall: separacion por sujetos antes de crear ventanas, calibracion solo con
  TRAIN por cada longitud y mismas exclusiones en todas las variantes.
- Theta: calibracion balanceada por clases de TRAIN, seleccionando por canal
  MSE de reconstruccion normalizado mas penalizacion de tasa de eventos.
  El JSON guarda sujetos, muestras utilizadas, objetivo y procedencia.
- Postura: escalado, PCA, clasificadores y umbrales tau/m se ajustan solo con
  gold TRAIN. Si faltan ejemplos manuales claros de ambas clases en TRAIN,
  falla explicitamente; no aprende de validacion/test como alternativa.
- Los caches legacy se invalidan. Cambios en archivos/anotaciones invalidan
  caches SisFall; theta tambien comprueba sujetos, raiz y politica de ventanas.

La calibracion puede ejecutarse por separado:

```bash
python calibrate_paper_theta.py \
  --dataset sisfall --output results/theta_sisfall_s0.json \
  --event-out-win 2048 --per-class 32 -- \
  --dataset-root data/SisFall_dataset --label-dir data/labels_transitions \
  --seed 0 --manual-eval-only
```

El optimizador historico `optimize_theta.py` conserva su comportamiento legacy;
no es el calibrador train-only de este protocolo.

## Evaluacion y ablaciones

`evaluate_paper_run.py` produce `metrics/paper_val` o `paper_test` con:
accuracy, macro-F1, matriz de confusion de cuatro/seis clases, PNG de confusion,
reportes por clase, predicciones NPY, `by_subject.json`, parametros,
operaciones MAC/SOP, tasas de disparo y energia PE normalizada. Las redes LIF
tambien producen `firing_rates.png`.
Por defecto muestra las etapas (datos/cache, checkpoint, inferencia, reportes),
el dispositivo real CPU/CUDA y avance por lotes con muestras, velocidad y ETA.
Las actualizaciones aparecen cada cinco segundos y siempre tras el primer y
ultimo lote: `--progress-every 1` cambia el intervalo y `--quiet-progress` las
oculta. El API `evaluate(...)` tambien muestra este avance, por lo que funciona
con el bloque Python de evaluacion masiva; `progress=False` lo desactiva.
La ETA empieza tras completar el primer lote y no incluye carga ni figuras.
Ventanas largas LIF pueden tardar bastante por lote, especialmente en CPU.
Cambiar los archivos no modifica un proceso Python que ya esta ejecutandose.
En KFall se agregan `by_subject/SAxx/confusion_matrix.png`, reportes por sujeto
y `by_subject/summary.csv`. Las matrices y macro-F1 siempre mantienen las cuatro
clases aunque un sujeto no tenga muestras de alguna clase.

En SisFall, la metrica primaria de cuatro clases incluye todas las ventanas
Fall/Dynamic y **solo** Static con gold manual estable/transicion; nunca toma
autotags como verdad de test. Head2 se evalua exclusivamente sobre gold manual.
La cascada enruta mediante la **prediccion** de Head1, no usando su etiqueta real.
Los sujetos mayores retenidos en test tienen su agregado SE01--SE15; si no hay
mayores en ese split, queda `null`, no un cero inventado. Esto no es una
evaluacion externa de todos los mayores si algunos participaron en TRAIN.

Para comparar Head1+Head2 con una unica cabeza de cuatro clases, agrega
`--compare-single-head`; para omega=0.4/0.6/0.8/1.0, agrega `--omega-sweep`.
Ambos agregan experimentos B4 a la matriz y usan el mismo gold en evaluacion.

### Inferencia continua

No conviertas el numero de ventanas en horas ADL. Este comando recorre las
grabaciones completas de los sujetos test con stride uniforme, sin seleccionar
ventanas por el pico de la caida:

```bash
python evaluate_paper_continuous.py \
  --run-dir RUTA_RUN_SISFALL --truth eventos_test.json \
  --stride 50 --batch-size 32 --refractory-seconds 2
```

Se requiere verdad temporal explicita; no se infieren intervalos de caida a
partir del maximo de aceleracion. Esquema:

```json
{
  "recordings": {
    "D01_SA01_R01": {"duration_seconds": 100.0, "is_adl": true},
    "F01_SA01_R01": {"duration_seconds": 15.0, "is_adl": false}
  },
  "fall_events": [
    {"recording_id": "F01_SA01_R01", "start_seconds": 5.0, "end_seconds": 6.2}
  ]
}
```

Son valores ilustrativos: reemplazalos por anotaciones verificadas de **todas**
las grabaciones test. Las duraciones se contrastan con muestras/200 Hz.
Una alarma tiene timestamp en el ultimo sample observado; matching 1:1 y periodo
refractario se aplican por grabacion. Se guarda sensibilidad por evento y falsas
alarmas por hora ADL fisica, incluyendo explicitamente el calentamiento inicial
de una ventana. No se mide anticipacion clinica ni monitorizacion real en hardware.
El umbral opcional `--fall-threshold` debe haberse fijado en validacion.

Para KFall los intervalos oficiales ya existen en XLSX; no hace falta inventar
ni volver a marcar los eventos:

```bash
python evaluate_paper_continuous.py --dataset kfall \
  --run-dir RUTA_RUN_KFALL --stride 25 --refractory-seconds 2
```

Se recorren todas las ventanas de las grabaciones test retenidas, no solamente
las seleccionadas por las anotaciones durante el entrenamiento. Se deriva un
evento onset--impact por caida y se guarda `truth_used.json`. Las horas ADL
se obtienen de la duracion FrameCounter/100 Hz de grabaciones ADL completas;
las fases normales de un ensayo Fall no se cuentan como horas ADL adicionales.
La sensibilidad por evento admite alarmas hasta impacto; no es una medida de
anticipacion estrictamente pre-impacto. `--truth` permite una verdad externa
explicita, si el protocolo la requiere.

### Robustez antes del encoder

Ejemplo `perturbacion.json` para SisFall:

```json
{"units": "ADC counts", "sample_rate_hz": 200, "noise_std": 5.0, "seed": 123}
```

```bash
python evaluate_paper_run.py --dataset sisfall --run-dir RUTA_RUN --split test \
  --perturbation perturbacion.json --tag ruido_5
```

Opciones del JSON: `noise_std`, `noise_channels`, `drift_per_second`,
`zero_channels`, `seed`, mas `units` y `sample_rate_hz` obligatorios. Un canal
anulado usa indices 0--5. UCI exige 50 Hz; sus canales mezclan rad/s y g, por lo
que elige `noise_channels` y amplitudes fisicamente adecuadas por modalidad.
No se reajusta theta ni se reentrena. Se perturban ventanas nativas antes de
interpolar/codificar, incluido su offset; ruido y deriva son **locales a cada
ventana**, no una simulacion de deriva acumulada durante horas. Las etiquetas
permanecen fijas para aislar el deterioro del modelo.
Los resultados se separan en `paper_test_ruido_5`; no sobrescriben la evaluacion
limpia. `energy_layers.json` permite comparar tasa de eventos/operaciones.
Para KFall usa `--dataset kfall`, `sample_rate_hz: 100` y unidades g/grados por
segundo adecuadas a `noise_channels`. El offset y los tiempos FrameCounter se
mantienen coherentes con la senal perturbada.

## Estadistica y energia

Cuando el runner termine, genera `results.csv`:

```bash
python report_paper_results.py runs/paper_reenvio/results.csv \
  --output-dir results/paper_reenvio --energy energy_pe_norm
```

El reporte exige al menos cinco semillas distintas por grupo, usa desviacion
muestral (`ddof=1`), rechaza duplicados/datos parciales y genera CSV, JSON con
procedencia, Pareto y figuras. `--descriptive` permite explorar grupos pequenos,
pero marca toda la salida `NON_PUBLISHABLE`. No mezcles protocolos, scopes de
metricas ni modelos energeticos distintos. El CSV identifica el perfil por
`model_id`; `protocol` es comun a val/test para poder emparejarlos.

`--scn-variants B2 B3 B4 B5` habilita seleccion SCNacc/SCNminE exclusivamente
desde validacion y transfiere el modelo elegido a test. Requiere pares val/test
completos. SCNminE minimiza energia sin imponer un piso de accuracy: no equivale
a un perfil historico si aquel usaba otra regla. SCNsel necesita su definicion
exacta y no se asigna automaticamente.

La energia actual es **normalizada a un MAC denso**, no joules ni una medicion
de GPU. El modelo v2 incluye MAC densos para entrada cruda y acceso a memoria
tambien para capas densas. No compares directamente con reportes legacy sin
recalcularlos. El estimador es operacional aproximado: no cubre toda la
plataforma, BatchNorm, pooling, radio, sensores ni entrenamiento. GRU y SRNN
siguen rechazandose porque sus costes no estan modelados.

Para energia total, `evaluation.paper_statistics.system_energy` exige energia
de red en **J**, duracion en **s**, potencias totales ADC/Sigma-Delta en **W** y
fuentes explicitas de cada componente/topologia. Calcula
`E_total = E_red + (P_ADC + P_SD) * duracion`. No asigna conversiones ni valores
publicados arbitrarios. El contrato completo de columnas esta en
`evaluation/paper_statistics.py`; PE normalizada nunca se suma a joules.

## Pendiente antes de reenviar

La infraestructura cubre el entrenamiento/evaluacion, no sustituye el trabajo
experimental ni editorial:

1. Proveer las configuraciones reales SCNsel/SCNacc/SCNminE y congelar criterios.
2. Ejecutar la matriz completa en GPU y resolver cobertura de gold manual.
3. Anotar/verificar intervalos de caida SisFall para la evaluacion continua;
   KFall utiliza sus intervalos oficiales XLSX.
4. Ejecutar niveles de ruido/deriva/canales y agregar cada condicion con cinco semillas.
5. Incorporar potencias y conversion energetica documentadas con fuentes verificadas.
6. Actualizar tablas 1--9, contribucion, relacionados/SOTA y narrativa con el
   manuscrito original y los resultados nuevos. No se encontro un manuscrito
   editable ni se fabricaron resultados o referencias del PDF.
