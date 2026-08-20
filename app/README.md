# ionic-opencv-metrics

App Ionic + Angular + OpenCV.js que captura la cámara y calcula **FPS** y **Sharpness** en tiempo real.

## Stack

| Librería           | Versión  | Rol                                    |
|--------------------|----------|----------------------------------------|
| Node.js            | ≥ 22     | Runtime                                |
| Angular            | 18.x     | Framework frontend                     |
| Ionic Angular      | 8.x      | UI components + routing                |
| OpenCV.js          | 4.x      | Cálculo de nitidez (Laplaciano WASM)   |
| TypeScript         | 5.4      | Lenguaje                               |
| WebRTC             | nativo   | Captura de cámara (getUserMedia)       |

## Instalación y ejecución

```bash
# Clonar o descomprimir el proyecto
cd ionic-opencv-metrics

# Instalar dependencias (requiere Node 22)
npm install

# Servidor de desarrollo en http://localhost:8100
npm start
```

> La primera carga descarga opencv.js (~9 MB) desde el CDN de OpenCV.
> Verás "OpenCV Cargando…" hasta que el módulo WASM esté listo.

## Cómo funciona

### FPS
Calculado con `performance.now()` entre frames:
```
instantFps = 1000 / deltaMs
fpsSmoothed = 0.85 * fpsSmoothed + 0.15 * instantFps
```
El suavizado exponencial evita oscilaciones bruscas en la lectura.

### Sharpness (Nitidez)
Varianza del Laplaciano — método estándar en visión computacional:
```
frame → escala de grises → filtro Laplaciano → varianza del resultado
```
Mayor varianza = más bordes detectados = imagen más nítida.

Rangos orientativos:
- `< 50`   → Borrosa
- `50–150` → Aceptable  
- `150–300`→ Nítida
- `> 300`  → Muy nítida

### Gestión de memoria OpenCV
OpenCV.js usa WASM con heap manual. Cada `cv.Mat` debe liberarse:
```typescript
const src = cv.imread(canvas);
// ... procesar ...
src.delete(); // obligatorio — no hay GC automático
```

### requestAnimationFrame fuera de NgZone
El loop de frames corre fuera de la detección de cambios de Angular
para no disparar re-renders en cada frame (~60/seg):
```typescript
this.zone.runOutsideAngular(() => this.processLoop());
// Solo actualiza la UI cada N frames via zone.run()
```

## Estructura del proyecto

```
src/
├── app/
│   ├── home/
│   │   ├── home.page.ts    ← lógica principal
│   │   ├── home.page.html  ← template con canvas + métricas
│   │   └── home.page.scss  ← estilos
│   ├── opencv.service.ts   ← carga OpenCV.js y calcula sharpness
│   ├── app.component.ts
│   ├── app.config.ts
│   └── app.routes.ts
└── index.html              ← carga opencv.js desde CDN
```

## Despliegue en dispositivo móvil (Capacitor)

```bash
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init
npx cap add android
npm run build
npx cap sync
npx cap open android
```
