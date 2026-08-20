import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';

export type Framework = 'tf' | 'torch';

export interface Analysis {
  material: string | null;         // material detectado (o null si nada claro)
  confidence: number;              // probabilidad del material top
  probs: Record<string, number>;   // probabilidad de cada material
  timeMs: number;                  // tiempo de inferencia (ms)
}

export const MATERIALS = ['plastico', 'vidrio', 'papel'];
const SIZE = 224;

/**
 * Inferencia EcoSort con ONNX Runtime Web. Carga los modelos de AMBOS frameworks
 * (TF y PyTorch) para poder compararlos sobre la misma foto.
 *   - TF   : input NHWC [1,224,224,3], salida = probabilidad (sigmoid incluido).
 *   - Torch: input NCHW [1,3,224,224], salida = logit -> se aplica sigmoid aquí.
 */
@Injectable({ providedIn: 'root' })
export class EcosortService {
  private sessions: Record<Framework, Record<string, ort.InferenceSession>> = { tf: {}, torch: {} };
  private offscreen = document.createElement('canvas');
  private lastStd = 0;
  ready = false;

  // Umbral suave: igual mostramos las probabilidades reales de cada modelo.
  private readonly THRESHOLD = 0.5;
  private readonly MARGIN = 0.10;
  private readonly STD_MIN = 0.05;   // foto casi vacía/plana -> no clasificar

  /** Carga los 3 modelos de cada framework (una sola vez). */
  async load(): Promise<void> {
    this.ready = false;
    ort.env.wasm.wasmPaths = '/assets/ort/';
    ort.env.wasm.numThreads = 1;
    for (const fw of ['tf', 'torch'] as Framework[]) {
      for (const material of MATERIALS) {
        this.sessions[fw][material] = await ort.InferenceSession.create(
          `/assets/models/${fw}/ecosort_${material}.onnx`,
          { executionProviders: ['wasm'] },
        );
      }
    }
    this.ready = true;
  }

  private preprocess(canvas: HTMLCanvasElement, framework: Framework): ort.Tensor {
    this.offscreen.width = SIZE;
    this.offscreen.height = SIZE;
    const ctx = this.offscreen.getContext('2d', { willReadFrequently: true })!;
    // Center-crop al cuadrado central + resize a 224.
    const side = Math.min(canvas.width, canvas.height);
    const sx = (canvas.width - side) / 2;
    const sy = (canvas.height - side) / 2;
    ctx.drawImage(canvas, sx, sy, side, side, 0, 0, SIZE, SIZE);
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const n = SIZE * SIZE;
    const arr = new Float32Array(3 * n);

    // "Contenido" del frame (std de luminancia): fondo plano -> std bajo.
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
      const g = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
      sum += g; sumSq += g * g;
    }
    this.lastStd = Math.sqrt(Math.max(0, sumSq / n - (sum / n) ** 2));

    if (framework === 'tf') {
      for (let i = 0; i < n; i++) {
        arr[i * 3 + 0] = data[i * 4 + 0] / 255;
        arr[i * 3 + 1] = data[i * 4 + 1] / 255;
        arr[i * 3 + 2] = data[i * 4 + 2] / 255;
      }
      return new ort.Tensor('float32', arr, [1, SIZE, SIZE, 3]);
    }
    for (let i = 0; i < n; i++) {
      arr[0 * n + i] = data[i * 4 + 0] / 255;
      arr[1 * n + i] = data[i * 4 + 1] / 255;
      arr[2 * n + i] = data[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', arr, [1, 3, SIZE, SIZE]);
  }

  /** Corre los 3 modelos del framework indicado y devuelve el resultado + tiempo. */
  async analyze(canvas: HTMLCanvasElement, framework: Framework): Promise<Analysis> {
    if (!this.ready) return { material: null, confidence: 0, probs: {}, timeMs: 0 };
    const input = this.preprocess(canvas, framework);
    const probs: Record<string, number> = {};
    const t0 = performance.now();
    for (const material of MATERIALS) {
      const session = this.sessions[framework][material];
      const output = await session.run({ [session.inputNames[0]]: input });
      let value = (output[session.outputNames[0]].data as Float32Array)[0];
      if (framework === 'torch') value = 1 / (1 + Math.exp(-value));
      probs[material] = value;
    }
    const timeMs = performance.now() - t0;

    if (this.lastStd < this.STD_MIN) {
      return { material: null, confidence: 0, probs, timeMs };
    }
    const ranked = MATERIALS.map(m => ({ m, p: probs[m] })).sort((a, b) => b.p - a.p);
    const top = ranked[0];
    const second = ranked[1]?.p ?? 0;
    const material = (top.p >= this.THRESHOLD && top.p - second >= this.MARGIN) ? top.m : null;
    return { material, confidence: top.p, probs, timeMs };
  }
}
