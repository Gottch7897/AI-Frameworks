/**
 * OpenCvService
 * =============
 * Envuelve la carga asíncrona de OpenCV.js.
 * El script se carga en index.html y dispara window.dispatchEvent(new Event('opencv-ready'))
 * cuando el módulo WASM está completamente inicializado.
 *
 * Uso:
 *   constructor(private opencv: OpenCvService) {}
 *   ngOnInit() {
 *     this.opencv.ready$.subscribe(ready => { if (ready) this.doOpenCvStuff(); });
 *   }
 */
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

// Declaración global para que TypeScript no proteste
declare var cv: any;

@Injectable({ providedIn: 'root' })
export class OpenCvService {

  private _ready$ = new BehaviorSubject<boolean>(false);

  /** Observable que emite `true` cuando cv está listo. */
  get ready$(): Observable<boolean> { return this._ready$.asObservable(); }

  /** Acceso directo al módulo cv (undefined si aún no está listo). */
  get cv(): any { return typeof cv !== 'undefined' ? cv : null; }

  constructor() {
    // Ya estaba listo antes de que el servicio se construyera (ej. recarga en dev)
    if (typeof cv !== 'undefined' && (window as any)['_opencvReady']) {
      this._ready$.next(true);
      return;
    }
    // Esperar el evento que dispara onOpenCvReady en index.html
    window.addEventListener('opencv-ready', () => {
      this._ready$.next(true);
    }, { once: true });
  }

  /** Convierte un HTMLCanvasElement a cv.Mat RGBA y lo retorna.
   *  Responsabilidad del caller hacer .delete() sobre el resultado. */
  imread(canvas: HTMLCanvasElement): any {
    return cv.imread(canvas);
  }

  /** Calcula la nitidez de un canvas usando la varianza del Laplaciano.
   *  Método estándar: mayor valor = imagen más nítida.
   *
   *  Rangos orientativos:
   *    < 50   → borrosa
   *    50-300 → aceptable
   *    > 300  → nítida
   */
  calcSharpness(canvas: HTMLCanvasElement): number {
    const src    = cv.imread(canvas);
    const gray   = new cv.Mat();
    const lap    = new cv.Mat();
    const mean   = new cv.Mat();
    const stddev = new cv.Mat();

    try {
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      // Laplaciano: detecta cambios bruscos de intensidad (bordes = nitidez)
      cv.Laplacian(gray, lap, cv.CV_64F, 1, 1, 0, cv.BORDER_DEFAULT);
      // Varianza = stddev²  — más varianza → más bordes → más nítido
      cv.meanStdDev(lap, mean, stddev);
      return Math.round(stddev.doubleAt(0, 0) ** 2);
    } finally {
      // Liberar memoria WASM (obligatorio — no hay GC automático)
      src.delete(); gray.delete(); lap.delete();
      mean.delete(); stddev.delete();
    }
  }
}
