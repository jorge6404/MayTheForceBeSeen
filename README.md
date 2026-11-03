# visual_normal_force

Estima la **fuerza normal que la red ejerce sobre el ROV** a partir de la imagen de la cámara.

```
sensor_msgs/CompressedImage  ──►  [ normal_force_node ]  ──►  std_msgs/Float64  (N)
                                    ConvNeXt-T · ONNX
                                    TensorRT / CUDA / CPU
```

- Salida en newtons: **positivo = el robot empuja la red**, **negativo = tira de ella**.
- Solo procesa la imagen más reciente: si la inferencia va más lenta que la cámara, se saltan imágenes en lugar de acumular retraso.
- ROS 2 Humble · C++ · ONNX Runtime.

## Parámetros

Están en `config/normal_force.yaml`:

| Parámetro      | Por defecto                                                | Qué hace                                          |
|----------------|------------------------------------------------------------|---------------------------------------------------|
| `input_topic`  | `/cirtesub/stereo_left_camera/camera/image_raw/compressed` | Imagen de entrada (`CompressedImage`)             |
| `output_topic` | `/visual_normal_force/force`                               | Fuerza normal en N (`Float64`)                    |
| `engine`       | `auto`                                                     | Motor de inferencia: `auto`, `tensorrt`, `cuda` o `cpu` |
| `rate_hz`      | `5.0`                                                      | Inferencias por segundo                           |

Con `auto` se usa el mejor motor disponible: **TensorRT** (Jetson) → **CUDA** (PC con NVIDIA) → **CPU**.
Si pides un motor que no está disponible, el nodo pasa al siguiente y avisa en el log.

## Docker

Necesitas Docker con Compose y, para la variante GPU, `nvidia-container-toolkit`.

```bash
cd docker
docker compose up --build gpu    # PC con gráfica NVIDIA
docker compose up --build cpu    # cualquier PC
```

`--build` crea la imagen; las siguientes veces basta con `docker compose up gpu`.

- **Cambiar parámetros**: edita `config/normal_force.yaml` y reinicia el contenedor; no hace falta reconstruir.
- **Cambiar el código o el modelo**: vuelve a lanzar con `--build`.
- La red es la del host y el `ROS_DOMAIN_ID` es `4`. Para otro dominio: `ROS_DOMAIN_ID=0 docker compose up gpu`.
- En gráficas RTX 50xx el primer arranque tarda unos 2 min (compila los kernels de CUDA). Queda en caché para las siguientes veces.

## Sin Docker

Descarga ONNX Runtime en cualquier carpeta (no instala nada en el sistema) y compila:

```bash
mkdir -p ~/onnxruntime
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
  | tar -xz -C ~/onnxruntime --strip-components=1        # con NVIDIA: onnxruntime-linux-x64-gpu-1.23.2.tgz

colcon build --packages-select visual_normal_force --cmake-args -DONNXRUNTIME_ROOT=$HOME/onnxruntime
source install/setup.bash
ros2 launch visual_normal_force normal_force.launch.py
```

Tests: `colcon test --packages-select visual_normal_force && colcon test-result --verbose`

## Modelo

ConvNeXt-Tiny entrenado en Stonefish (semilla 2 de `entrenamiento_fuerza_normal`, kursk): **MAE 0,64 N en test**.
Ya devuelve newtons, con la normalización de la imagen incluida.

Está en Hugging Face: [jorge6404/visual-normal-force](https://huggingface.co/jorge6404/visual-normal-force).
**Se descarga solo al compilar** (106 MB, una vez), fijado a un commit y comprobado por SHA-256.
Sin internet: copia `normal_force.onnx` en `models/` antes de compilar.

Para regenerarlo desde un `modelo.pt`:

```bash
python tools/export_onnx.py modelo.pt models/normal_force.onnx
```
