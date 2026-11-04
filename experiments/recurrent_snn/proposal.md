# Propuesta de modificaciones a la SNN para detección temporal de caídas y postimpacto

**Documento base:** *Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks*, Valdivia, Vera y Yunge.

**Objetivo:** extender el clasificador neuromórfico para aprender la continuidad temporal de las fases **Estático**, **Dinámico**, **Caída** y **Postimpacto**, reduciendo predicciones fragmentadas sin imponer secuencias físicas excesivamente rígidas.

> **Nota de alcance:** el paper describe una clasificación principal de tres clases (*Fall*, *Dynamic*, *Static*) y un segundo cabezal para refinar posturas estáticas. La clase **Postimpacto** pertenece a la extensión mostrada en el experimento del gráfico, no al modelo original descrito en el paper. Las modificaciones presentadas aquí son propuestas para una versión nueva y no resultados experimentales ya demostrados.

## 1. Análisis de la arquitectura actual

Según las secciones 4 y 5 del paper, el sistema utiliza:

1. **Encoder ΣΔ:** codifica seis canales IMU como eventos de polaridad positiva y negativa.
2. **Backbone SNN:** tres bloques `Conv1D → BatchNorm → LIF → MaxPool`.
3. **Firing-rate pooling:** promedia temporalmente las activaciones de la última capa convolucional para obtener 64 características.
4. **Rama de offset:** incorpora una referencia inercial absoluta mediante ocho neuronas LIF.
5. **Fusión de características:** concatena ambas ramas en un vector de 72 dimensiones.
6. **Doble cabezal original:** uno clasifica `Fall / Dynamic / Static`; el otro distingue `Stable posture / Postural transition` para ventanas estáticas válidas.

El modelo seleccionado `SCNsel` tiene **111.276 parámetros** y alcanza **94,27 % de accuracy en el cabezal principal** y **89,88 % en el de refinamiento** (tabla 4).

### Limitaciones para el problema actual

**A. No hay memoria explícita entre ventanas.** El backbone extrae información temporal *dentro* de cada ventana, pero el clasificador descrito no mantiene un estado de secuencia que relacione directamente la ventana anterior con la siguiente. Esto puede permitir cambios como:

`Dinámico → Caída → Dinámico → Caída → Postimpacto`

**B. La definición original de caída no aprende el episodio completo.** En la sección 3, las ventanas de las grabaciones de caída se seleccionan alrededor del máximo pico de aceleración; se excluyen periodos anteriores y posteriores. Una versión que pretenda predecir **Postimpacto** debe incluir esos intervalos y disponer de anotaciones temporalmente coherentes.

**C. Las clases no representan exactamente el mismo nivel semántico.** `Estático` y `Dinámico` describen movimiento, mientras que `Caída` y `Postimpacto` describen fases de un evento. Conviene considerar esta distinción en el diseño de los cabezales.

## 2. Modificación principal: agregar memoria temporal

Se propone mantener el encoder ΣΔ, la rama de offset y los bloques convolucionales LIF, e insertar un módulo recurrente tras la fusión de características:

```text
IMU (6 canales)
   │
   ├── Encoder ΣΔ ──> 3 × (Conv1D + BN + LIF + Pool) ──> 64 features ─┐
   │                                                                  ├─> 72 features por ventana
   └── Offset ─────────────> Rama LIF ─────────────────> 8 features ──┘
                                                                            │
                                                     Secuencia de K ventanas ordenadas
                                                                            │
                                                                  Memoria temporal
                                                             (GRU o SNN recurrente)
                                                                            │
                                                              Head de 4 clases
                                                  Estático / Dinámico / Caída / Postimpacto
```

Una formalización sencilla es:

$$
z_t = f_{\mathrm{SCN}}(E_t, x_{t,0})
$$

$$
h_t = f_{\mathrm{temporal}}(z_t, h_{t-1})
$$

$$
P(y_t\mid x_{\leq t}) = \operatorname{softmax}(W h_t+b)
$$

Aquí, $z_t\in\mathbb{R}^{72}$ es el vector por ventana y $h_t$ conserva el contexto temporal previo.

### Alternativas de memoria

| Módulo | Ventajas | Limitaciones |
|---|---|---|
| **GRU de 32 unidades** | Baseline simple, eficiente y fácil de entrenar. | Incluye operaciones densas, por lo que no mantiene una implementación estrictamente neuromórfica. |
| **SNN recurrente con 32–64 LIF** | Más alineada con el procesamiento por eventos. | Exige manejar correctamente estados de membrana, reinicios y entrenamiento temporal. |
| **CRF/HMM sobre las salidas** | Modela transiciones con pocos parámetros adicionales. | No reemplaza por sí mismo la memoria de características IMU; según el algoritmo puede requerir contexto futuro. |

**Propuesta inicial:** entrenar primero **SCN + GRU(32)** como referencia, y luego contrastar con una **SNN recurrente** y su costo energético.

## 3. Modificar la función de pérdida

La pérdida original combina *cross-entropy* del cabezal principal con *binary cross-entropy* enmascarada del refinamiento de postura (sección 5.1). No incluye una penalización explícita para cambios improbables entre predicciones consecutivas.

Para la versión de cuatro clases, se propone experimentar con:

$$
\boxed{\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{trans}}\mathcal{L}_{\mathrm{trans}} + \lambda_{\mathrm{bound}}\mathcal{L}_{\mathrm{bound}}}
$$

- **$\mathcal{L}_{\mathrm{CE}}$:** clasificación por ventana, con máscaras para etiquetas ambiguas o no disponibles.
- **$\mathcal{L}_{\mathrm{trans}}$:** penalización de transiciones poco plausibles.
- **$\mathcal{L}_{\mathrm{bound}}$:** supervisión explícita de los cambios de fase para evitar que una excesiva suavización retrase la detección.

### 3.1. Pérdida de transiciones

Sea $p_t\in[0,1]^4$ el vector de probabilidades de cada ventana y $M\in\mathbb{R}^{4\times4}$ una matriz de costos de transición:

$$
\mathcal{L}_{\mathrm{trans}} = \frac{1}{T-1}\sum_{t=2}^{T}p_{t-1}^{\mathsf T} M p_t
$$

Los valores grandes de $M_{ij}$ penalizan pasar del estado $i$ al estado $j$. Es preferible estimar los patrones de transición en el conjunto de **entrenamiento** y ajustar las penalizaciones en **validación**, en vez de declarar todas las transiciones como imposibles por intuición.

**Precaución:** `Postimpacto → Dinámico` puede ser válido si la persona se reincorpora. `Caída → Estático` también puede aparecer en registros con límites de anotación inciertos o donde el postimpacto fue breve. Una restricción demasiado rígida podría ocultar casos reales.

### 3.2. Detección de límites

Se puede añadir un cabezal binario supervisado con:

$$
b_t = \mathbb{1}[y_t\neq y_{t-1}]
$$

Se entrena con BCE sobre pares de ventanas con anotaciones válidas. Esto permite que el sistema aprenda cuándo debe cambiar de fase, compensando la tendencia de una penalización temporal a mantener etiquetas demasiado tiempo.

## 4. Preparación de datos: requisito prioritario

### 4.1. Usar episodios completos

Para aprender `Caída → Postimpacto`, el conjunto de entrenamiento debe contener grabaciones completas con las cuatro fases delimitadas temporalmente. **No basta con reentrenar usando ventanas centradas únicamente en el impacto**, como en la formulación original del paper.

Se recomienda:

- Conservar el orden cronológico de las ventanas de cada grabación.
- Definir criterios de anotación consistentes para el inicio y término de `Caída` y `Postimpacto`.
- Permitir segmentos inciertos cerca de los límites de fase, aplicando máscaras de pérdida o un protocolo de anotación claro.
- Evitar combinar en una misma secuencia artificial ventanas pertenecientes a distintas grabaciones.

### 4.2. Contexto de las ventanas actuales

El gráfico muestra ventanas de **120 muestras** y un avance de **10 muestras**. Si los datos mantienen la frecuencia de **200 Hz** de SisFall, esto corresponde a:

- **0,60 s** por ventana.
- **0,05 s** entre comienzos consecutivos (20 decisiones por segundo).

Para $K$ ventanas consecutivas, la cobertura temporal total es:

$$
T_{\mathrm{contexto}}=\frac{120+(K-1)\times10}{200}\;\mathrm{s}
$$

| Ventanas $K$ | Contexto cubierto |
|---:|---:|
| 8 | 0,95 s |
| 16 | 1,35 s |
| 20 | 1,55 s |
| 32 | 2,15 s |

**Experimento inicial:** probar secuencias de 20 y 32 ventanas. Los valores son puntos de partida para una ablación, no hiperparámetros optimizados.

### 4.3. Evitar fuga de información (*data leakage*)

El paper indica una división aleatoria 80/10/10 **a nivel de ventana**. Cuando existen ventanas altamente solapadas, ese esquema puede dejar segmentos casi idénticos en entrenamiento y test.

Para la nueva versión, conviene:

1. Separar primero las grabaciones completas y, preferentemente, los **sujetos** entre entrenamiento, validación y test.
2. Generar ventanas y secuencias **después** de definir las particiones.
3. Ajustar las matrices de transición, los umbrales y los pesos de pérdida sin utilizar el test.
4. Reiniciar la memoria de la GRU/SNN en los límites correctos de grabación y controlar que no se transmita estado entre sujetos o episodios independientes.

Este punto es decisivo para obtener métricas representativas del rendimiento ante sujetos y movimientos no vistos.

## 5. Alternativa: clasificación jerárquica

El paper reporta mejores resultados para una formulación de dos cabezales que para una de cuatro clases de un solo cabezal en **su tarea original** (tabla 6). Ese resultado no demuestra automáticamente que la misma ventaja se mantendrá con tu clase `Postimpacto`, pero justifica probar una descomposición jerárquica.

Diseño experimental sugerido:

```text
                 SCN + MEMORIA TEMPORAL
                          │
                 Contexto de actividad
                    /           \
                 Normal      Evento de caída
                /    \          /      \
           Estático Dinámico   Caída  Postimpacto
```

- Un componente distingue actividad normal de episodio relacionado con caída.
- Otro refina la actividad normal en `Estático` o `Dinámico`.
- Otro refina el episodio en `Caída` o `Postimpacto`.

Esta división podría ayudar cuando el movimiento postimpacto se parece a una actividad estática, pero el historial temporal indica que acaba de ocurrir una caída.

## 6. Plan de ablación

Se recomienda introducir una modificación por vez, usando el **mismo conjunto de prueba** y protocolos comparables:

| Experimento | Modelo | Hipótesis |
|---|---|---|
| **A** | SCN actual de 4 clases | Establecer referencia. |
| **B** | SCN + GRU(32) | La memoria temporal reduce cambios espurios. |
| **C** | B + pérdida de transiciones | Mayor coherencia en la secuencia de estados. |
| **D** | C + supervisión de límites | Mejora el equilibrio entre estabilidad y latencia. |
| **E** | SCN + SNN recurrente | Mantiene una solución más próxima al hardware neuromórfico. |
| **F** | SCN temporal jerárquica | Reduce confusiones entre contexto de actividad y fase de caída. |

### Métricas recomendadas

**A nivel de ventana:** precisión, recall, F1 por clase y macro-F1.

**A nivel de evento:** recall de episodios de caída, falsas alarmas por hora, fragmentación de un evento, error en el instante de transición entre fases y latencia de detección.

**A nivel de hardware:** parámetros, operaciones sinápticas/eventos, memoria y consumo energético por segundo de operación. La tasa de **20 inferencias/s** inferida del gráfico debe considerarse al comparar energía con el protocolo del paper.

**Comparación online/offline:** una GRU causal funciona con información pasada; un Viterbi global o un suavizado bidireccional pueden utilizar información futura y deben reportarse separadamente.

## 7. Ejemplo de integración con PyTorch

El siguiente ejemplo presupone que la SNN existente expone un método `extract_features` capaz de entregar las **72 características fusionadas** por ventana. Este método y el manejo de estados LIF se deben implementar según el código concreto del proyecto.

```python
import torch
import torch.nn as nn


class TemporalSCN(nn.Module):
    def __init__(self, backbone, hidden_size=32):
        super().__init__()

        self.backbone = backbone

        self.temporal = nn.GRU(
            input_size=72,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_size, 4)

    def forward(self, events, offsets, h=None):
        # events:  [B, K, ...]
        # offsets: [B, K, 6]
        B, K = events.shape[:2]

        # Extraer características de cada ventana con la SCN.
        events_flat = events.flatten(0, 1)
        offsets_flat = offsets.flatten(0, 1)

        z = self.backbone.extract_features(
            events_flat, offsets_flat
        )  # [B*K, 72]

        z = z.reshape(B, K, 72)
        out, h = self.temporal(z, h)
        logits = self.head(out)  # [B, K, 4]

        return logits, h
```

**Atención:** si el backbone almacena estado de membrana LIF, ese estado debe reiniciarse o administrarse explícitamente de acuerdo con la semántica de cada ventana. El código anterior es un **esqueleto arquitectónico**, no una implementación lista para entrenar sin adaptar.

## 8. Recomendación final

La ruta de menor riesgo experimental es:

1. **Corregir la estructura del dataset:** secuencias completas, anotaciones de `Postimpacto` y divisiones por grabación/sujeto.
2. **Conservar el encoder ΣΔ y el backbone SCN original.**
3. **Agregar una GRU causal pequeña** para establecer un baseline con memoria temporal.
4. **Incorporar gradualmente una penalización de transiciones y un cabezal de límites.**
5. **Comparar contra una SNN recurrente**, midiendo energía y latencia, además de exactitud.

La hipótesis principal de investigación sería:

> Un clasificador neuromórfico con memoria temporal entre ventanas y supervisión de transiciones puede mejorar la coherencia y detección de episodios completos de caída sin incrementar de forma excesiva la latencia o el costo energético.

Esto debe demostrarse con experimentos; no es una conclusión establecida por el paper original.

## 9. Fuente principal

Valdivia, S., Vera, E. y Yunge, D. *Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks*. Documento proporcionado por el usuario; consultar especialmente:

- **Sección 3:** definición de las clases, ventanas y selección de caídas.
- **Sección 4:** encoder ΣΔ y rama de offset.
- **Sección 5, figura 6:** arquitectura de doble cabezal y entrenamiento.
- **Tablas 4 y 6:** resultados del modelo seleccionado y comparación de formulaciones.

Repositorio mencionado en el paper: https://github.com/s-valdivia-vasquez/Neuromorphic-HAR
