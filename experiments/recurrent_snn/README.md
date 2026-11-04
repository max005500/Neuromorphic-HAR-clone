# Experimento de SNN recurrente para KFall

Este experimento prepara la alternativa **SCN + 32 LIF recurrentes** elegida por
el usuario. Es independiente de `train_kfall.py`: la clasificación plana y el
experimento jerárquico anterior siguen disponibles. El informe original está en
[proposal.md](proposal.md), y [backup.json](backup.json) identifica el ZIP previo.
La inclusión de la jerarquía no se presupone beneficiosa: se mantiene opcional
hasta comparar los resultados de validación.

## Arquitectura y estados

El mismo encoder Sigma-Delta y backbone Conv/BN/LIF/Pool más offsets entregan
`conv_channels[-1] + offset_hidden` características por ventana (72 por defecto).
`extract_features` expone esa representación. La memoria recurrente implementa:

```text
corriente = W_input * features + W_recurrent * spikes_previos + bias
membrana = tau * membrana_previa + corriente
spike = 1[membrana > umbral]
membrana = membrana - spike * umbral       (soft reset)
```

También respeta el hard reset si se solicita. Usa el gradiente surrogate del
repositorio, incluida la optimización que evita lecturas CUDA escalares por
paso. La salida de clasificación es lineal como en la configuración actual;
no hay GRU. `--recurrent-hidden 32` y `64` permiten comparar tamaños.
`--memory-tau` y `--memory-threshold` son parámetros neuronales; no son los theta
del encoder.

Cada secuencia pertenece a una sola grabación y un solo sujeto. Se baraja el
orden de grabaciones, conservando su orden interno. El estado se inicia en cero
al comenzar una grabación, continúa entre sus bloques y se reinicia al cambiar
de episodio. Los gradientes se truncan cada K ventanas (TBPTT): **K no limita
la memoria de inferencia ni reinicia la membrana**. El backbone LIF sí mantiene
su semántica original de reinicio dentro de cada ventana. Se extraen features
por posición temporal: no se mezcla K en BatchNorm, evitando estadísticas de
ventanas futuras. El padding no entra en BatchNorm ni modifica la memoria.

## Datos y frecuencia real

KFall se adquiere a **100 Hz**. `--resample-hz 200` interpola dentro de la ventana;
no cambia los frames originales ni la escala de `--stride`. Un avance de 10
muestras originales son 0.10 s / 10 decisiones por segundo. Para 20 decisiones/s,
usa `--stride 5`.

Se generan etiquetas completas antes de segmentar y se separan sujetos primero.
El modo `spikes` utiliza solo el acelerómetro para Dynamic/Static; caída y
postimpacto conservan sus intervalos. Las pseudoetiquetas por grupos de spikes
son offline; esto no permite información futura en la entrada del modelo.
Se excluye `legacy`, que descartaba fases no anotadas en ensayos de caída.

La caché antigua puede añadir puntos extra junto a anotaciones. El nuevo
constructor filtra esos puntos y conserva una rejilla uniforme independiente de
onset/impact, para que la programación de inferencia no dependa de las etiquetas.
Se guardan los índices originales de caché en cada predicción. **No compares sus
métricas directamente con las de un protocolo que incluya otros extremos**:
el experimento A usa exactamente la misma rejilla que B/C/D/F.

Con ventanas nominales de 1.00 s, un bloque de K=20 cubre 2.90 s y K=32 cubre
4.10 s si `stride=10`; con `stride=5`, cubren 1.95 y 2.55 s. El span entre la
primera y última muestra es 0.01 s menor. Son horizontes nominales del bloque de
entrenamiento; la memoria puede conservar información desde el inicio de la
misma grabación. Grabaciones demasiado cortas no se rellenan con señales
inventadas; la primera decisión requiere una ventana completa.

## Generar datos y entrenar primero B

Preparación sin entrenamiento:

```bash
python train_kfall_temporal.py --preprocess-only \
  --memory lif --recurrent-hidden 32 --sequence-length 20 \
  --window-seconds 1.00 --stride 10 --resample-hz 200 \
  --theta-file configs/kfall_theta_gyro_custom.json \
  --label-policy spikes --post-impact-seconds 1.20 \
  --workers 0
```

El JSON es el ajuste de giroscopio guardado por el usuario; si tiene otro nombre,
cambia la ruta. Para comparar con modelos previos que usaban contexto 410,
`--endpoint-context 410` lo conserva, pero excluye el inicio de algunos episodios.
Para estudiar episodios lo más completos posible, el contexto predeterminado es
el necesario para la propia ventana. El informe de eventos cuenta explícitamente
caídas no observables antes de la primera ventana.

Entrenamiento recurrente sin pérdidas auxiliares:

```bash
python train_kfall_temporal.py \
  --memory lif --recurrent-hidden 32 --sequence-length 20 \
  --window-seconds 1.00 --stride 10 --resample-hz 200 \
  --theta-file configs/kfall_theta_gyro_custom.json \
  --label-policy spikes --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 8 --workers 0 --device auto \
  --validation-only --run kfall_snn_recurrente
```

Aquí **batch-size cuenta grabaciones paralelas**, no ventanas individuales. Con
8 grabaciones y K=20 puede haber hasta 160 ventanas por bloque. El padding se
acota al bloque actual; los eventos viajan como uint8. El pipeline recorta
memmaps directamente y requiere workers=0, sin reutilizar la semántica del
DataLoader de ventanas. Estos defaults se eligen por memoria y corrección, no
por haber sido optimizados para la RTX 5090.

`--validation-only` evita inferencias de test durante la selección. Repite con
K=32 o `--recurrent-hidden 64` como experimentos separados.

## Transiciones y límites

Se activan por separado:

```bash
--lambda-transition 0.1
--lambda-boundary 0.1
```

Los valores son puntos de partida para validar. La matriz de costos se calcula
**solo con train**, con suavizado Laplace, costos negativos logarítmicos
normalizados y diagonal cero. No hay transiciones prohibidas ni una secuencia
física impuesta: recuperación Postimpact→Dynamic sigue siendo posible. La
pérdida esperada es `p_prev^T M p_current` sobre pares consecutivos válidos.
El cabezal de límites usa BCE para `y_t != y_prev`, con peso positivo estimado
solo en train. También supervisa pares que cruzan un bloque TBPTT dentro de la
misma grabación, nunca pares entre grabaciones.

`--boundary-ignore-seconds 0.05` permite excluir de CE y pérdidas de pares los
extremos cercanos a cambios de etiqueta. Por defecto vale 0: no se declara
ambigüedad adicional. Las ventanas excluidas siguen pasando por la memoria;
se enmascara su supervisión, no se elimina tiempo del episodio.

## Ablaciones

El comparador primero crea un plan, sin entrenar:

```bash
python compare_kfall_temporal.py --mode plan \
  --experiments A B C D --sequence-lengths 20 32 \
  --window-seconds 1.00 --stride 10 --resample-hz 200 \
  --theta-file configs/kfall_theta_gyro_custom.json \
  --label-policy spikes --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 8 --workers 0 --device auto --validation-only
```

Cambia `--mode plan` por `--mode train` para ejecutarlo secuencialmente.
Recomendamos empezar con `--experiments A B --sequence-lengths 20` y después
introducir C/D. El comparador comprueba que todos los runs tengan el mismo
cohort de extremos y etiquetas de validación antes de guardar la comparación.

| Caso | Memoria | Transición | Límites | Salida |
|---|---|---|---|---|
| A | Ninguna | 0 | 0 | 4 clases |
| B | LIF recurrente | 0 | 0 | 4 clases |
| C | LIF recurrente | λ_trans | 0 | 4 clases |
| D | LIF recurrente | λ_trans | λ_bound | 4 clases |
| F, opcional | LIF recurrente | 0 | 0 | Jerárquica |

Se conserva el nombre B como primera memoria experimental, sustituyendo la GRU
del documento por la preferencia del usuario. No se implementa una GRU ni un
HMM/CRF en esta versión. F se solicita explícitamente con `--experiments F` o
`--head-mode hierarchical` al entrenar. No forma parte del plan predeterminado.

Para comprobar la condición de no empeorar, sin usar test:

```bash
python compare_kfall_validation.py \
  --baseline runs/RUN_REFERENCIA --candidate runs/RUN_CANDIDATO
```

El informe requiere cohort, sujetos, encoder y parámetros de entrenamiento
comparables. Marca `non_degrading` cuando macro-F1 y recall de Fall de validación
no disminuyen (tolerancia predeterminada cero). Es una comparación puntual; no
demuestra superioridad estadística. No adopta automáticamente ninguna variante.
Los resultados del clasificador jerárquico anterior todavía deben aportarse o
evaluarse antes de asumir que también conviene incorporarlo a la memoria.

## Evaluar y visualizar

Después de seleccionar con validación:

```bash
python evaluate_kfall_temporal.py --run-dir runs/NOMBRE_TEMPORAL --device auto
python animate_kfall_temporal.py --run-dir runs/NOMBRE_TEMPORAL --device auto
```

El reproductor comparte la interfaz de señales/probabilidades/etiquetas reales,
pero mantiene el estado durante toda la grabación. Opciones: `--recording`,
`--subject`, `--paused`, `--sensor gyro`, `--time 5`,
`--save-frame results/recurrent_preview.png`. La rejilla es la guardada al
entrenar; no se altera la frecuencia recurrente al reproducir.

Los informes se guardan en `metrics/validation` y `metrics/test_best`:

- Métricas de ventana, macro-F1, matrices de confusión y CSV con índices de caché.
- `episode_predictions.npz`: todas las decisiones, incluida la máscara incierta.
- `event_metrics.json`: recall por episodio, observabilidad, falsas alarmas/hora
  de tiempo Normal observado, fragmentación y latencia firmada de detección.
- Cambios de fase: matching uno a uno de transiciones con la misma dirección
  dentro de una tolerancia descriptiva de 1.00 s; MAE y retraso medio.
- Parámetros, tiempo de inferencia por decisión, throughput, pico de VRAM en CUDA
  y spikes recurrentes. El estimado de eventos sinápticos comprende solo las
  conexiones recurrentes densas; no mide energía ni incluye el backbone.
- `temporal_loss.png`: CE y términos auxiliares; las lambdas constan en config.

Una alerta Fall se empareja con una caída si el intervalo durante el que se
mantiene esa predicción intersecta `[onset, impact)`. Su inicio puede ser anterior
al onset, por lo que la latencia firmada puede ser negativa. Alertas no
emparejadas que solapan tiempo Normal cuentan como falsas alarmas. Las caídas
sin observación previa suficiente siguen contando en el denominador del recall.
Estas definiciones están guardadas en JSON: no mezclar con otros protocolos.

Los informes jerárquicos conservan la decisión raíz→rama y las probabilidades
conjuntas. `evaluate_kfall.py` rechaza checkpoints temporales y dirige al nuevo
evaluador para evitar reiniciar memoria accidentalmente en cada ventana.

## Recuperación

La copia previa está fuera del repositorio, indicada en `backup.json`, con
SHA-256 de los archivos y commit de Git. Incluye código y configuración no
ignorados, no los datasets/runs ignorados. Copia también el ZIP al trasladar el
experimento a otro equipo. Para descartar este experimento basta con volver a
`train_kfall.py`; los comandos originales siguen disponibles. Si necesitas
recuperar archivos, extrae el ZIP en una carpeta aparte y compara antes de
sobrescribir cambios posteriores.
