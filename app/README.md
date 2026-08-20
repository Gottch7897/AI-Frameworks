# EcoSort — App móvil (Ionic + ONNX)

App de reciclaje que **clasifica el material de un objeto desde la cámara** en vivo:
**plástico / vidrio / papel**. Los modelos corren **100% en el dispositivo** con
ONNX Runtime Web (WebAssembly) — sin servidor ni internet.

- **Framework:** Ionic 8 + Angular 18.
- **Cámara:** `getUserMedia` (WebRTC) sobre canvas.
- **Inferencia:** `onnxruntime-web` con los modelos exportados desde TensorFlow y
  PyTorch (intercambiables desde la UI). Ver `../export_onnx.py`.
- **Empaquetado nativo:** Capacitor → APK Android.

## Cómo funciona
1. Se abre la cámara y se muestra un cuadro-guía central.
2. Al presionar **"Analizar objeto"** se captura el frame, se recorta al centro,
   se redimensiona a 224×224 y se corre en los 3 detectores ONNX.
3. Se muestra el material más probable (con umbral + margen; si no hay uno claro,
   dice "Acerca un objeto reciclable").
4. Botón **TF ↔ PyTorch** para comparar ambos modelos en vivo.

Lógica de inferencia: `src/app/ecosort.service.ts`.

## Correr en el navegador (desarrollo)
```bash
npm install
npm start            # http://localhost:8100
```
Ábrelo en un equipo con webcam (o en el teléfono con la IP de la red).

## Compilar el APK (Android)

Requiere **JDK 21** y **Android SDK** (Capacitor 7 usa Java 21, compileSdk 36).

```bash
# 1) construir la web y sincronizar con Android
npm run build
npx cap sync android

# 2) compilar el APK (ajusta JAVA_HOME / ANDROID_HOME a tu instalación)
cd android
JAVA_HOME=/ruta/al/jdk-21 ANDROID_HOME=/ruta/al/android-sdk ./gradlew assembleDebug
```
El APK queda en `android/app/build/outputs/apk/debug/app-debug.apk`.

> Alternativa sencilla: abrir la carpeta `android/` en **Android Studio** y
> *Build → Build APK(s)* (trae JDK/SDK/Gradle).

## Estructura
```
src/app/ecosort.service.ts   # carga los 6 modelos ONNX + inferencia (TF/PyTorch)
src/app/home/home.page.ts    # cámara, loop de preview, botón Analizar, switch
src/assets/models/{tf,torch} # modelos .onnx (offline)
src/assets/ort/              # runtime WebAssembly de onnxruntime-web
android/                     # proyecto Capacitor para el APK
```
