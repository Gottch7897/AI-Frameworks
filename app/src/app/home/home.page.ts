import {
  Component, OnInit, OnDestroy, AfterViewInit,
  ElementRef, ViewChild, NgZone,
  ChangeDetectionStrategy, ChangeDetectorRef,
  signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

import {
  IonHeader, IonToolbar, IonTitle, IonContent,
  IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle,
  IonCardContent, IonButton, IonIcon,
  IonSpinner, IonChip, IonLabel,
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  videocam, videocamOff, colorWand, lockClosedOutline,
  speedometerOutline, eyeOutline, informationCircleOutline,
  alertCircleOutline, refreshOutline,
} from 'ionicons/icons';

import { OpenCvService } from '../opencv.service';
import { EcosortService, Framework } from '../ecosort.service';

type AppState = 'idle' | 'requesting' | 'active' | 'error';
const SHARPNESS_EVERY  = 5;
const INFER_EVERY      = 10;   // correr la inferencia cada N frames (fluidez)
const FPS_ALPHA        = 0.85;

@Component({
  selector:        'app-home',
  standalone:      true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    IonHeader, IonToolbar, IonTitle, IonContent,
    IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle,
    IonCardContent, IonButton, IonIcon,
    IonSpinner, IonChip, IonLabel,
  ],
  templateUrl: './home.page.html',
  styleUrls:   ['./home.page.scss'],
})
export class HomePage implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('videoEl')   videoRef!:  ElementRef<HTMLVideoElement>;
  @ViewChild('canvasLive') canvasRef!: ElementRef<HTMLCanvasElement>;

  readonly Math = Math;

  appState    = signal<AppState>('idle');
  fps         = signal(0);
  sharpness   = signal(0);
  opencvReady = signal(false);
  errorMsg    = signal('');
  frameCount  = signal(0);

  // --- EcoSort (clasificación de material con ONNX) ---
  material    = signal<string | null>(null);
  confidence  = signal(0);
  probs       = signal<Record<string, number>>({});
  framework   = signal<Framework>('tf');
  modelReady  = signal(false);
  modelError  = signal('');
  analizando  = signal(false);
  readonly MATERIALS = ['plastico', 'vidrio', 'papel'];

  isActive    = computed(() => this.appState() === 'active');
  isRequesting= computed(() => this.appState() === 'requesting');
  isError     = computed(() => this.appState() === 'error');

  sharpnessLabel = computed(() => {
    const s = this.sharpness();
    if (s === 0)   return '—';
    if (s < 50)    return 'Borrosa';
    if (s < 150)   return 'Aceptable';
    if (s < 300)   return 'Nítida';
    return 'Muy nítida';
  });

  sharpnessColor = computed(() => {
    const s = this.sharpness();
    if (s < 50)  return 'danger';
    if (s < 150) return 'warning';
    return 'success';
  });

  fpsColor = computed(() => {
    const f = this.fps();
    if (f < 15) return 'danger';
    if (f < 25) return 'warning';
    return 'success';
  });

  private stream:       MediaStream | null = null;
  private rafId:        number | null      = null;
  private lastTime      = 0;
  private fpsSmoothed   = 0;
  private frameCounter  = 0;
  private viewReady     = false;
  private inferring     = false;
  private cvSub!:       Subscription;

  constructor(
    private opencv:  OpenCvService,
    private ecosort: EcosortService,
    private zone:    NgZone,
    private cdr:     ChangeDetectorRef,
  ) {
    addIcons({
      videocam, videocamOff, colorWand, lockClosedOutline,
      speedometerOutline, eyeOutline, informationCircleOutline,
      alertCircleOutline, refreshOutline,
    });
  }

  ngOnInit(): void {
    this.cvSub = this.opencv.ready$.subscribe(ready => {
      this.opencvReady.set(ready);
      this.cdr.markForCheck();
    });

    // Cargar los 3 modelos ONNX (TF por defecto) una sola vez.
    this.ecosort.load('tf')
      .then(() => { this.modelReady.set(true); this.modelError.set(''); this.cdr.markForCheck(); })
      .catch(err => {
        console.error('Error cargando modelos ONNX:', err);
        this.modelError.set(String(err?.message ?? err));
        this.cdr.markForCheck();
      });
  }

  /** Captura el frame actual y lo clasifica UNA vez (modo foto: no bloquea el preview). */
  async analizar(): Promise<void> {
    if (!this.modelReady() || !this.isActive() || this.analizando()) return;
    this.analizando.set(true);
    this.cdr.markForCheck();
    try {
      const result = await this.ecosort.analyze(this.canvasRef.nativeElement);
      this.material.set(result.material);
      this.confidence.set(result.confidence);
      this.probs.set(result.probs);
    } catch (err) {
      console.error('Error al analizar:', err);
    }
    this.analizando.set(false);
    this.cdr.markForCheck();
  }

  /** Alterna entre los modelos de TensorFlow y PyTorch (recarga las sesiones). */
  async cambiarFramework(): Promise<void> {
    const next: Framework = this.framework() === 'tf' ? 'torch' : 'tf';
    this.modelReady.set(false);
    this.framework.set(next);
    this.material.set(null);
    this.probs.set({});
    this.cdr.markForCheck();
    try {
      await this.ecosort.load(next);
      this.modelReady.set(true);
    } catch (err) {
      console.error('No se pudo cargar el framework', next, err);
    }
    this.cdr.markForCheck();
  }

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    canvas.width  = 640;
    canvas.height = 480;
    this.viewReady = true;
  }

  ngOnDestroy(): void {
    this.releaseCamera();
    this.cvSub?.unsubscribe();
  }

  async solicitarCamara(): Promise<void> {
    if (!this.viewReady) return;
    this.errorMsg.set('');
    this.appState.set('requesting');
    this.cdr.markForCheck();

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width:  { ideal: 1280, min: 320 },
          height: { ideal: 720,  min: 240 },
        },
      });
      await this.iniciarCaptura();
    } catch (err: any) {
      const msg = this.mensajeError(err);
      this.appState.set('error');
      this.errorMsg.set(msg);
      this.cdr.markForCheck();
    }
  }

  async iniciarCaptura(): Promise<void> {
    if (!this.stream) return;

    const video  = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;

    video.srcObject = this.stream;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = (e) => reject(e);
      setTimeout(() => reject(new Error('Timeout esperando video')), 5000);
    });

    await video.play();

    const settings = this.stream.getVideoTracks()[0].getSettings();
    canvas.width  = settings.width  ?? video.videoWidth  ?? 640;
    canvas.height = settings.height ?? video.videoHeight ?? 480;

    this.lastTime     = performance.now();
    this.fpsSmoothed  = 0;
    this.frameCounter = 0;

    this.appState.set('active');
    this.cdr.markForCheck();

    this.zone.runOutsideAngular(() => this.frameLoop());
  }

  detenerCaptura(): void {
    this.releaseCamera();

    const canvas = this.canvasRef?.nativeElement;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width  = 640;
      canvas.height = 480;
    }

    this.zone.run(() => {
      this.appState.set('idle');
      this.fps.set(0);
      this.sharpness.set(0);
      this.frameCount.set(0);
      this.cdr.markForCheck();
    });
  }

  reintentar(): void {
    this.appState.set('idle');
    this.errorMsg.set('');
    this.cdr.markForCheck();
  }

  private frameLoop(): void {
    if (this.appState() !== 'active') return;

    const video  = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;

    if (video.readyState < 2) {
      this.rafId = requestAnimationFrame(() => this.frameLoop());
      return;
    }

    const now   = performance.now();
    const delta = now - this.lastTime;
    this.lastTime = now;

  
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Guía: cuadrado central donde poner el objeto (coincide con el crop del modelo).
    const guideSide = Math.min(canvas.width, canvas.height);
    const gx = (canvas.width - guideSide) / 2;
    const gy = (canvas.height - guideSide) / 2;
    ctx.strokeStyle = this.material() ? '#2dd36f' : 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 4;
    ctx.strokeRect(gx + 2, gy + 2, guideSide - 4, guideSide - 4);

    // En esta zona se integra el tema que el frame sea analizado por su flujo de modelos
    // Les recomiendo crear un servicio para despues las modificaciones no interfieran en el
    // codigo que funciona
    /**
     * ONNX Runtime Web, bauticen su servicio con el nombre apropiado a su funcion 
     *
     * PASO 1 — Instalar
     *   npm install onnxruntime-web
     *
     * PASO 2 — Copiar los archivos WASM a src/assets/ort/
     *   cp node_modules/onnxruntime-web/dist/*.wasm src/assets/ort/
     *
     * PASO 3 — Copiar el modelo
     *   cp mi_modelo.onnx src/assets/modelo.onnx
     *
     * PASO 4 — Indicar la ruta de los WASM (antes de crear la sesión)
     *   ort.env.wasm.wasmPaths = 'assets/ort/';
     *
     * PASO 5 — Cargar el modelo una sola vez <-- importante!!! para optimizar funcionamiento
     *   session = await ort.InferenceSession.create('assets/modelo.onnx');
     *
     * PASO 6 — Inspeccionar el modelo con netron.app
     *   Verificar: nombre del input, shape esperado, nombre del output.
     *   También por código: console.log(session.inputNames, session.outputNames)
     *
     * PASO 7 — Preprocesar: canvas → Float32Array
     *   canvas (RGBA HWC) → resize → RGB CHW → normalizar → Float32Array
     *   Si el modelo usa ImageNet: aplicar mean/std por canal.
     *
     * PASO 8 — Ejecutar la sesión
     *   const feeds   = { [session.inputNames[0]]: new ort.Tensor('float32', data, [1,3,H,W]) };
     *   const results = await session.run(feeds);
     *   const logits  = results[session.outputNames[0]].data as Float32Array;
     *
     * PASO 9 — Postprocesar: logits → resultado
     *   Clasificación : argmax(logits) → clase,  softmax(logits) → confianza
     *   Detección     : filtrar por confianza → NMS → bounding boxes
     *
     * PASO 10 — Llamar desde frameLoop() sin bloquear
     *   this.inference.analyze(canvas).then(result => {
     *     this.zone.run(() => this.prediction.set(result));
     *   });
     *
     * PROBLEMAS FRECUENTES
     *   "Failed to fetch ort-wasm.wasm"  → verificar PASO 2 y PASO 4
     *   "SharedArrayBuffer is not defined"  → usar executionProviders: ['wasm']
     *   Shape mismatch  → revisar con Netron (PASO 6) y preprocesamiento (PASO 7)
     *   Video trabado   → usar .then() en lugar de await (PASO 10)
     */


    if (delta > 0) {
      const inst = 1000 / delta;
      this.fpsSmoothed = this.fpsSmoothed === 0
        ? inst
        : FPS_ALPHA * this.fpsSmoothed + (1 - FPS_ALPHA) * inst;
    }

    this.frameCounter++;

    // Sharpness cada N frames
    let newSharpness = this.sharpness();
    if (this.opencvReady() && this.frameCounter % SHARPNESS_EVERY === 0) {
      try { newSharpness = this.opencv.calcSharpness(canvas); } catch { /* ignorar */ }
    }

    // Actualizar UI cada N frames
    if (this.frameCounter % SHARPNESS_EVERY === 0) {
      this.zone.run(() => {
        this.fps.set(Math.round(this.fpsSmoothed));
        this.sharpness.set(newSharpness);
        this.frameCount.set(this.frameCounter);
        this.cdr.markForCheck();
      });
    }

    this.rafId = requestAnimationFrame(() => this.frameLoop());
  }

  private releaseCamera(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }

  private mensajeError(err: any): string {
    switch (err?.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Permiso de cámara denegado. Ve a Configuración del navegador → Privacidad → Cámara y permite el acceso.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No se encontró ninguna cámara en este dispositivo.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'La cámara está en uso por otra aplicación. Ciérrala e intenta de nuevo.';
      case 'OverconstrainedError':
        return 'La resolución solicitada no es compatible con esta cámara.';
      default:
        return `Error de cámara: ${err?.message ?? String(err)}`;
    }
  }
}
