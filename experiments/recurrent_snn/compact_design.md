# SCN compacta con memoria LIF persistente

Esta variante implementa la propuesta compacta como un experimento separado.
Conserva el encoder Sigma-Delta de seis canales y dos polaridades, la rama de
offsets y las cuatro clases `Fall`, `Dynamic`, `Static` y `Postimpact`. La
precisión, la actividad de spikes y el costo en hardware deben medirse; la
reducción de parámetros por sí sola no demuestra una mejora de velocidad ni
de energía.

## Arquitectura

| Componente | Estándar recurrente | Compacta |
|---|---|---|
| Canales de las tres convoluciones | 32 / 64 / 64 | 16 / 32 / 32 |
| Kernels | 32 / 32 / 8 | 16 / 16 / 8 |
| Bloques | Conv1D → BN → LIF → MaxPool | Conv1D → BN → LIF → MaxPool |
| Rama de offsets | 6 → 8 LIF | 6 → 8 LIF |
| Características fusionadas | 72 | 40 |
| Memoria entre decisiones | 32 LIF recurrentes | 32 LIF recurrentes |
| Reinicio recurrente predeterminado | Inmediato | Diferido |
| Salida | Lineal, cuatro clases | Lineal, cuatro clases |
| Parámetros, sin cabezal de límites | 114.476 | 22.156 |

El conteo compacto está dentro del objetivo de diseño de 20.000–30.000
parámetros: representa una reducción del **80,6 %** frente a la SCN recurrente
estándar y del **80,1 %** frente a los 111.276 parámetros de SCNsel. Son
arquitecturas distintas, por lo que esos porcentajes no comparan precisión.
Activar la jerarquía, el cabezal de límites o cambiar tamaños modifica el conteo.

`--architecture standard` conserva los valores previos. El preset
`--architecture compact` resuelve los tamaños de la tabla y el reinicio diferido;
los argumentos explícitos permiten experimentar con otros tamaños. La
configuración resuelta queda en `config.json` y en el checkpoint, para que el
evaluador reconstruya exactamente el modelo entrenado. Los pesos de la red
grande no se cargan directamente en la pequeña: las formas de las matrices son
distintas y hace falta un entrenamiento nuevo.

Los kernels más cortos cambian el campo receptivo dentro de la ventana. Para
aislar la contribución de cada cambio, compara también:

```bash
--architecture compact --kernels 32 32 8 --memory-reset immediate
```

Ese caso reduce canales conservando los kernels y el tipo de reinicio del
modelo anterior. Después puedes cambiar solo los kernels y finalmente probar
el reinicio diferido. Mantén los mismos sujetos, encoder, ventanas, avance y
parámetros de entrenamiento en las comparaciones.

## Memoria y reinicio

Con reinicio diferido suave, la memoria recurrente calcula:

```text
u_t = beta * u_anterior + W * features_t + R * s_anterior + bias
      - umbral * s_anterior
s_t = 1[u_t >= umbral]
```

El estado explícito contiene la membrana y los spikes anteriores. El gradiente
sustituto permite entrenar las matrices de entrada y recurrencia. Este modo no
es numéricamente equivalente al reinicio inmediato cuando `beta < 1`; no se
debe cambiar el modo al evaluar pesos entrenados con otro reinicio.

El estado persiste entre decisiones y entre bloques TBPTT de la misma
grabación. Se reinicia al comenzar otra grabación, nunca se transfiere entre
sujetos y se congela para posiciones de padding. `--sequence-length 20` trunca
los gradientes cada 20 ventanas, sin reiniciar la memoria de inferencia. Los
LIF del backbone conservan su reinicio dentro de cada ventana; la persistencia
entre decisiones corresponde a la capa recurrente final. La inferencia utiliza
solo ventanas ya observadas, sin contexto futuro.

## Entrenamiento

Desde la raíz del repositorio:

```bash
python train_kfall_temporal.py \
  --architecture compact --sequence-length 20 \
  --window-seconds 1.00 --stride 10 --resample-hz 200 \
  --theta-file configs/kfall_theta_gyro_custom.json \
  --label-policy spikes --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 8 --workers 0 --device auto \
  --validation-only --run kfall_compact_lif32
```

Usa la ruta real del JSON que guardaste al ajustar el giroscopio. Si solo quieres
preparar y revisar los datos, añade `--preprocess-only`. El preset no altera los
umbrales ni el criterio de etiquetas: Dynamic/Static siguen usando los spikes
del acelerómetro, mientras que Fall/Postimpact conservan las anotaciones.

`batch-size` cuenta grabaciones paralelas. KFall conserva sus frames originales
a 100 Hz; `--stride 10` equivale a 10 decisiones/s aunque se interpole a 200 Hz.
La reducción de canales puede ahorrar cómputo y activaciones, pero mantiene los
bucles de pasos temporales y sus lanzamientos de kernels. Hay que medir tiempo
por época y throughput en la GPU utilizada.

Para comparar con la referencia usando solo validación:

```bash
python compare_kfall_validation.py \
  --baseline runs/RUN_ESTANDAR --candidate runs/RUN_COMPACTO \
  --allow-architecture-changes \
  --output results/kfall_compact_vs_standard.json
```

La opción autoriza diferencias de arquitectura; los datos, las etiquetas, los
extremos de ventana y el protocolo de entrenamiento deben seguir siendo
comparables. Revisa macro-F1, recall de Fall, falsas alarmas, fragmentación y
latencia, además de accuracy. No elijas la variante utilizando test.

## BatchNorm y precisión de los pesos

El entrenamiento conserva BatchNorm y pesos FP32, salvo que se active AMP de
forma explícita. La fusión se aplica en una copia en modo evaluación: integra
las estadísticas guardadas de BN en la convolución o proyección lineal
precedente. Incluye la rama de offsets. No vuelve a estimar estadísticas en
validación o test ni modifica el checkpoint original.

Evalúa primero las tres variantes en validación, con carpetas distintas:

```bash
python evaluate_kfall_temporal.py --run-dir runs/RUN_COMPACTO \
  --split val --device auto \
  --output-dir results/compact_val_fp32

python evaluate_kfall_temporal.py --run-dir runs/RUN_COMPACTO \
  --split val --device auto --fuse-bn \
  --output-dir results/compact_val_fused

python evaluate_kfall_temporal.py --run-dir runs/RUN_COMPACTO \
  --split val --device auto --fuse-bn --weight-bits 8 \
  --output-dir results/compact_val_weights8
```

Las evaluaciones transformadas usan FP32 aunque el entrenamiento haya usado
AMP. La evaluación sin opciones conserva la configuración AMP guardada; para
aislar la sensibilidad a INT8, compara el modo `--fuse-bn` con
`--fuse-bn --weight-bits 8`, que utilizan ambos la misma aritmética FP32.
Si no indicas `--output-dir`, los modos transformados también reciben carpetas
distintas dentro de `metrics`, sin reemplazar la validación original.

La fusión utiliza operaciones equivalentes en aritmética real. El redondeo de
punto flotante puede afectar spikes que estén muy cerca del umbral; contrasta
las predicciones y las métricas del modelo fusionado con las originales.

`--weight-bits 8` **simula cuantización de pesos** simétrica por canal de salida:
redondea los pesos a una rejilla INT8 y los devuelve a FP32 para ejecutar el
modelo. Permite medir sensibilidad de las predicciones, no velocidad nativa
INT8, memoria de un artefacto desplegado ni consumo energético. Abarca las
convoluciones y capas lineales, incluidas recurrencia y readout. Los estados de
membrana, activaciones y operaciones permanecen en punto flotante.

Una implementación de precisión fija completa requiere definir el hardware de
destino, la escala y saturación de estados, el formato de `beta` y umbrales, el
ancho de acumuladores y el tratamiento de offsets. Esa implementación y una
exportación ejecutable INT8 no quedan demostradas por esta simulación. Si la
sensibilidad de pesos es alta, habrá que explorar entrenamiento consciente de
cuantización o mayor precisión antes del despliegue.

Tras elegir la arquitectura y precisión con validación, evalúa test una vez
usando el modo seleccionado, por ejemplo:

```bash
python evaluate_kfall_temporal.py --run-dir runs/RUN_COMPACTO \
  --split test --device auto --fuse-bn --weight-bits 8 \
  --output-dir results/compact_test_selected
```

La interfaz temporal existente permite revisar señales, etiquetas y
predicciones con los pesos FP32 guardados:

```bash
python animate_kfall_temporal.py --run-dir runs/RUN_COMPACTO --device auto
```

## Respaldo

La copia de código anterior a esta variante está en
`/home/max008800/proyecto/seba/backups/before_compact_snn_20261103_224620.zip`, con
un manifiesto SHA-256 verificado. No contiene datasets ni entrenamientos
ignorados por Git. El modo estándar y sus checkpoints anteriores permanecen
disponibles.
