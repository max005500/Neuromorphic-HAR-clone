# Entrenamiento de la SCN recurrente en dos etapas

`--training-strategy staged` permite entrenar la representación SCN y la memoria
temporal por separado. El objetivo es evitar ejecutar y retropropagar el
backbone convolucional en cada época del entrenamiento recurrente. El modo
anterior continúa disponible con `--training-strategy joint`, que es el valor
predeterminado.

La arquitectura compacta conserva **22.156 parámetros**. El entrenamiento por
etapas cambia cómo se aprenden sus pesos, sin añadir una rama permanente al
modelo final ni cambiar el encoder Sigma-Delta o las etiquetas.

## Qué ocurre en cada etapa

| Etapa | Entrada | Qué se entrena |
|---|---|---|
| 1. Preentrenamiento | Ventanas de eventos y offsets | SCN convolucional, rama de offsets y un clasificador temporalmente independiente |
| Extracción | Las mismas ventanas, con el mejor backbone de validación | Nada; se guardan las características en FP32 |
| 2. Memoria temporal | Secuencias de características guardadas | LIF recurrentes y readout temporal |
| Ajuste final, opcional | Ventanas de eventos y offsets | Último bloque convolucional, memoria y readout |

En la primera etapa, cada ventana se presenta de forma independiente. Un
cabezal provisional supervisa las cuatro clases y se descarta después. Los
pesos de la memoria y del readout final no se entrenan en esta etapa. Se
conserva el mejor backbone según validación, con separación de sujetos.

La extracción utiliza ese backbone en modo evaluación: BatchNorm emplea sus
estadísticas guardadas y dropout queda desactivado. En la configuración
compacta, cada ventana produce **40 valores: 32 características convolucionales
y 8 de offsets**. Las características de train y validación se guardan una sola
vez y se reutilizan durante las épocas de la segunda etapa. Esta extracción no
ejecuta ni guarda el estado recurrente.

En la segunda etapa, el backbone permanece congelado y en modo evaluación,
incluidas las estadísticas de BatchNorm. El entrenamiento procesa directamente
vectores `[batch, K, 40]`: no ejecuta convoluciones ni la proyección de offsets.
La memoria recurrente y el clasificador reúnen **2.468 parámetros entrenables**
en el modelo compacto de 32 LIF, salida plana y sin cabezal de límites. Otros
tamaños, la jerarquía o las pérdidas auxiliares pueden cambiar ese conteo.

## Orden temporal, estado y particiones

El preentrenamiento y la extracción utilizan solo train y validación. Test no
se usa para seleccionar el backbone, preparar características o decidir entre
la segunda etapa y el ajuste final. `--validation-only` también omite la
evaluación final de test.

Las secuencias conservan los extremos de ventana, etiquetas, máscaras de
ambigüedad y orden de las grabaciones del entrenamiento temporal. Se puede
barajar el orden de las grabaciones en train, manteniendo el orden cronológico
interno de cada una. No se crean pares temporales entre grabaciones ni se
transfiere estado entre sujetos.

La memoria se reinicia al comenzar otra grabación y se congela en posiciones de
padding. Con `--sequence-length 20`, se truncan los gradientes cada 20 ventanas,
pero la membrana persiste entre bloques de la misma grabación. La ruta de
características respeta la misma semántica de estado que la ruta de eventos.

La caché de características queda asociada a los pesos, configuración y dinámica
del backbone, datos, encoder, particiones y extremos de ventana seleccionados.
Una caché incompatible se rechaza. Si cambias esas condiciones y quieres
reconstruirla, utiliza `--refresh-features`; el programa no reutiliza en silencio
características de otro backbone. Los arrays de eventos originales permanecen
disponibles para inferencia y para el ajuste final.

## Comando inicial

Desde la raíz del repositorio:

```bash
python train_kfall_temporal.py \
  --architecture compact --training-strategy staged \
  --window-seconds 1.00 --stride 10 --resample-hz 200 \
  --theta-file configs/kfall_theta_200hz.json \
  --label-policy spikes --post-impact-seconds 1.20 \
  --pretrain-epochs 20 --pretrain-batch-size 128 \
  --feature-batch-size 128 \
  --sequence-length 20 --epochs 100 --batch-size 8 \
  --workers 0 --device auto --validation-only \
  --run kfall_compact_staged
```

El ejemplo utiliza el JSON incluido, que existe en el repositorio. Para usar tu
ajuste manual del giroscopio, sustituye la ruta por
`configs/kfall_theta_gyro_custom.json` únicamente después de guardarlo desde
`tune_kfall_gyro.py`, o indica la ruta real del archivo que guardaste.

`--pretrain-epochs 20` corresponde a 20 épocas de ventanas independientes;
`--epochs 100` corresponde a la segunda etapa y no incluye las épocas previas.
El preentrenamiento y la memoria conservan su selección por validación y
parada temprana. Estos números son puntos de partida, no ajustes óptimos ya
demostrados.

Los tamaños de lote tienen significados distintos:

- `--pretrain-batch-size 128`: hasta 128 ventanas independientes por lote.
- `--feature-batch-size 128`: hasta 128 ventanas durante la extracción sin gradientes.
- `--batch-size 8`: hasta ocho grabaciones paralelas en el entrenamiento temporal.

`--pretrain-lr` permite elegir una tasa de aprendizaje para la primera etapa;
si se omite, se utiliza `--lr`. `--feature-cache-dir RUTA` permite cambiar el
directorio de características; de forma predeterminada se utiliza `RUN/features`,
con particiones train y val separadas. Las barras de progreso indican la etapa
que está ejecutándose; `--no-progress` las desactiva.

Las características, la segunda etapa y el ajuste final utilizan **FP32**.
Si solicitas `--amp`, se aplica únicamente al preentrenamiento del backbone.
La configuración conserva esa elección como `pretrain_amp`, mientras que el
modelo temporal final se evalúa en FP32. Esto evita mezclar estados iniciales
o características con precisiones distintas entre el entrenamiento sobre
caché y la inferencia sobre eventos.

## Ajuste final del último bloque

Para añadir cinco épocas de ajuste después de la memoria, agrega:

```bash
--finetune-epochs 5 --finetune-lr 1e-4
```

El valor predeterminado es `--finetune-epochs 0`. Al activarlo, se vuelve a la
entrada de eventos. Solo el tercer bloque convolucional del backbone se
descongela; los bloques primero y segundo y la rama de offsets mantienen pesos,
BatchNorm y dropout congelados. La memoria y los readouts siguen entrenándose.
No se usan características antiguas para entrenar pesos que ya cambiaron.

El mejor checkpoint temporal de la segunda etapa se compara con el mejor del
ajuste final usando la misma pérdida de validación online: antes del ajuste se
reevalúa una vez el modelo de la segunda etapa sobre eventos, para evitar que
el redondeo de convoluciones con otro tamaño de lote sesgue la comparación. El
checkpoint final conserva
la mejor de esas alternativas: no se adopta un ajuste que empeore ese criterio.
Esto no garantiza que todas las métricas, como macro-F1 o latencia, mejoren a la
vez; también deben revisarse. La pérdida del cabezal provisional de la primera
etapa no se compara directamente con las pérdidas de las etapas temporales.

## Archivos y evaluación

En la carpeta de la ejecución encontrarás:

- `stages/pretrain`: checkpoint, índices, historial y gráficos del preentrenamiento.
- `features/train` y `features/val`: características FP32, índices y manifiestos de caché.
- `stages/temporal/checkpoints/best_model.pt`: mejor checkpoint de la segunda etapa.
- `stages/finetune`: resultados del ajuste final, si se solicita.
- `checkpoints/best_model.pt`: checkpoint final elegido por validación.
- `metrics`: métricas y curvas del entrenamiento temporal.
- `training_stages.json`: resultados, tiempos y conteos de las etapas.

La inferencia online conserva la cadena completa:

```text
Eventos + offsets → SCN → 40 características → LIF recurrentes → cuatro clases
```

El ahorro de convoluciones corresponde al entrenamiento de la segunda etapa;
no se presenta como una eliminación del backbone al inferir señales nuevas.
Los evaluadores y la interfaz temporal existentes leen el checkpoint final:

```bash
python evaluate_kfall_temporal.py --run-dir runs/NOMBRE_DEL_RUN --split val --device auto
python animate_kfall_temporal.py --run-dir runs/NOMBRE_DEL_RUN --device auto
```

Después de seleccionar la configuración con validación, puedes evaluar test
con `evaluate_kfall_temporal.py --split test`. Conserva los mismos sujetos,
encoder, ventanas, avance y criterio de etiquetas al comparar contra el
entrenamiento conjunto.

## Qué se puede esperar del rendimiento

El LIF convolucional actual itera por **pasos temporales**, no por cada spike
individual; las neuronas de cada paso se calculan en paralelo en el lote.
Reducir parámetros conserva esos pasos y los lanzamientos de kernels pequeños.
La traza anterior mostró operaciones de GPU cercanas a un microsegundo y
lanzamientos de varios microsegundos: es una explicación posible de la baja
utilización, pero no demuestra por sí sola cuál es el cuello de botella total.

El entrenamiento conjunto mantiene las activaciones y el grafo del backbone
para las K ventanas del bloque. En la segunda etapa, ese grafo se reduce a la
memoria y al clasificador; la extracción paga el costo del backbone una vez.
El preentrenamiento y el ajuste final siguen ejecutando los LIF
convolucionales. En ejecuciones muy cortas, el costo inicial de preentrenar y
extraer puede superar el ahorro posterior.

Con la ventana actual de **1.00 s**, KFall aporta **100 muestras originales**.
Un avance de 10 muestras produce **90 % de solapamiento**. El 91,67 % corresponde
a una ventana de 120 muestras con ese mismo avance. La interpolación a 200 Hz
no cambia la frecuencia de adquisición ni la interpretación de `--stride`.
No se cambia el solapamiento automáticamente porque también cambiaría la
frecuencia de decisiones y el protocolo experimental.

Para comparar una RTX 3050 y una RTX 5090, utiliza exactamente la misma
configuración, versiones y datos, y separa los tiempos de preentrenamiento,
extracción, memoria y ajuste final en `training_stages.json`. Esta ruta evita
repetir el backbone durante las épocas temporales; el factor de aceleración y
la precisión deben medirse en cada equipo.

Para aislar la ejecución de GPU de la carga de datos, ejecuta en ambos equipos:

```bash
python benchmark_kfall_temporal.py \
  --architecture compact --window-seconds 1.00 \
  --batch-size 8 --sequence-length 20 --pretrain-batch-size 128 \
  --device cuda:0 --warmup 3 --steps 10 \
  --output results/benchmark_temporal_gpu.json
```

El JSON identifica la tarjeta, versiones, configuración y las fases `joint`,
`window_pretraining` y `cached_temporal`. Compara ventanas/s y conserva
`windows_per_step`: el preentrenamiento utiliza 128 ventanas por actualización,
mientras que las fases temporales utilizan hasta 160 con B=8 y K=20. Son bloques
sintéticos completos, sin padding y con memoria reiniciada entre actualizaciones.
La extracción de características, lectura de disco y DataLoader están fuera del
intervalo medido. El factor entre fases no es una medida de aceleración de todo
el entrenamiento por etapas.

## Respaldo

El código anterior está respaldado en
`/home/max008800/proyecto/seba/backups/before_staged_training_20261103_231419.zip`,
con manifiesto SHA-256 verificado. No incluye datasets ni runs ignorados por
Git. La estrategia `joint` permite conservar el flujo de entrenamiento anterior.
