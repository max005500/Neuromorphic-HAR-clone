# KFall: jerarquía experimental

Se activa con `--head-mode hierarchical`. El modo predeterminado continúa siendo
`single`, la clasificación plana original. Los datos, etiquetas, encoder y
particiones por sujetos pueden compartirse entre ambos experimentos.

```text
SNN de eventos + offsets
├── Group: Normal / Evento
├── Normal: Dynamic / Static
└── Event: Fall / Postimpact
```

Las tres cabezas reciben la representación compartida de la SCN. Por defecto
son lineales; `--spiking-heads` utiliza bloques LIF como salidas. No corresponden
a la antigua segunda cabeza de posturas de SisFall, que sigue deshabilitada en
el modo `dual` de KFall.

Las etiquetas originales se conservan: 0=Fall, 1=Dynamic, 2=Static, 3=Postimpact.
Normal agrupa 1/2 y Evento agrupa 0/3. La pérdida suma la clasificación del grupo
y la clasificación de la rama correcta. Cada muestra solo supervisa su rama.
Si hay ponderación balanceada, los pesos se calculan exclusivamente con train y
se aplican a la suma por muestra. El mejor peso se selecciona por pérdida de val.
Se requieren cuatro clases, con `--post-impact-seconds` positivo.

La decisión hace primero argmax de Normal/Evento, después argmax de la rama
elegida. Las probabilidades finales guardadas son conjuntas:
`P(clase) = P(grupo) * P(clase|grupo)`. Por la decisión en dos etapas, la clase
seleccionada puede diferir del argmax global de esas cuatro probabilidades.
Entrenamiento, evaluación, CSV de predicciones y reproductores usan la misma
decisión por rutas. Los informes incluyen también las probabilidades de cada
cabeza. Las métricas de las ramas se calculan sobre el grupo verdadero; las
métricas finales de cuatro clases incluyen los errores de la raíz.

## Entrenar

```bash
python train_kfall.py \
  --head-mode hierarchical \
  --window-seconds 1.00 --resample-hz 200 --endpoint-context 410 \
  --theta-file configs/kfall_theta_gyro_custom.json \
  --label-policy spikes --spike-label-hz 200 \
  --spike-min-duration-seconds 0.30 --spike-max-gap-seconds 0.10 \
  --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 128 --workers 4 --device auto \
  --run kfall_1s_hierarchical
```

El archivo gyro_custom es el ajuste manual del usuario; si tiene otro nombre,
sustituir la ruta. Para una comparación de ventanas, el comparador admite
`--head-mode hierarchical`. Usa carpetas de runs/results separadas del experimento
plano y los mismos seed, umbrales, sujetos, batch y precisión para comparar.
La identidad de las cachés depende de los datos/encoder, no del número de salidas.

## Verificación y resultados

```bash
python evaluate_kfall.py --run-dir runs/NOMBRE_HIERARCHICAL --device auto
python animate_kfall.py --run-dir runs/NOMBRE_HIERARCHICAL --device auto
```

Cada entrenamiento guarda métricas de cuatro clases, `hierarchy_accuracy.png`,
`hierarchy_loss.png`, y carpetas `metrics/hierarchy_validation` y
`metrics/hierarchy_test`. Cada carpeta contiene `hierarchy_metrics.json`,
matrices de confusión PNG de group/normal/event y probabilidades NPY por salida.
La evaluación independiente los guarda en `metrics/test_best/hierarchy`.
El reproductor indica la ruta, probabilidad de grupo y probabilidad condicional.
Los pesos antiguos se cargan con su arquitectura original.

## Copia previa y vuelta al experimento original

Antes de modificar el código se guardó el estado de los archivos versionados y
no ignorados en:

`/home/max008800/proyecto/seba/backups/before_hierarchical_20261103_211510.zip`

El ZIP incluye `BACKUP_MANIFEST.json` con SHA-256 por archivo y el commit de Git.
El archivo está fuera del repositorio; conservar una copia al trasladar el
experimento a otro equipo. No contiene datos/runs ignorados por Git.

Para dejar de usar la jerarquía basta con entrenar con `--head-mode single`.
Para recuperar el código previo, extrae el ZIP en una carpeta aparte y compara
los archivos antes de copiarlos sobre el repositorio; así puedes preservar
cambios posteriores. Los módulos nuevos del experimento son
`models/hierarchical.py` y `evaluation/kfall_hierarchy.py`, junto con sus pruebas.
