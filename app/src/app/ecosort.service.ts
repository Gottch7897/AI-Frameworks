import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';

export type Framework = 'tf' | 'torch';

export interface MaterialResult {
  material: string | null;        // material detectado (o null si ninguno supera el umbral)
  confidence: number;             // probabilidad del material detectado
  probs: Record<string, number>;  // probabilidad de cada material
}

const MATERIALS = ['plastico', 'vidrio', 'papel'];
const SIZE = 224;

/**
 * Servicio de inferencia EcoSort con ONNX Runtime Web.
 * Corre los 3 detectores de material (plástico/vidrio/papel) sobre un frame.
 * Intercambiable entre los modelos de TensorFlow y PyTorch (misma interfaz ONNX):
 *   - TF   : input NHWC [1,224,224,3], la salida ya es probabilidad (sigmoid incluido).
 *   - Torch: input NCHW [1,3,224,224], la salida es un logit -> se aplica sigmoid aquí.
 */
@Injectable({ providedIn: 'root' })
export class EcosortService {
  private sessions: Record<string, ort.InferenceSession> = {};
  private framework: Framework = 'tf';
  private offscreen = document.createElement('canvas');
  ready = false;

  /** Carga (una sola vez) los 3 modelos del framework elegido. */
  async load(framework: Framework = 'tf'): Promise<void> {
    this.ready = false;
    this.framework = framework;
    ort.env.wasm.wasmPaths = '/assets/ort/';  // PASO 4: ruta ABSOLUTA (la app corre en /home)
    ort.env.wasm.numThreads = 1;              // 1 hilo: evita necesitar SharedArrayBuffer (COOP/COEP)

    const sessions: Record<string, ort.InferenceSession> = {};
    for (const material of MATERIALS) {
      sessions[material] = await ort.InferenceSession.create(
        `/assets/models/${framework}/ecosort_${material}.onnx`,
        { executionProviders: ['wasm'] },     // 'wasm' evita el problema de SharedArrayBuffer
      );
    }
    this.sessions = sessions;
    this.framework = framework;
    this.ready = true;
  }

  getFramework(): Framework {
    return this.framework;
  }

  /** Canvas -> Float32Array en el layout que espera el framework activo. */
  private preprocess(canvas: HTMLCanvasElement): ort.Tensor {
    this.offscreen.width = SIZE;
    this.offscreen.height = SIZE;
    const ctx = this.offscreen.getContext('2d', { willReadFrequently: true })!;
    // Center-crop al cuadrado central antes de redimensionar: reduce el fondo y
    // se parece más al training (objeto centrado que llena la imagen).
    const side = Math.min(canvas.width, canvas.height);
    const sx = (canvas.width - side) / 2;
    const sy = (canvas.height - side) / 2;
    ctx.drawImage(canvas, sx, sy, side, side, 0, 0, SIZE, SIZE);
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);  // RGBA plano
    const n = SIZE * SIZE;
    const arr = new Float32Array(3 * n);

    if (this.framework === 'tf') {
      // NHWC: por pixel R,G,B
      for (let i = 0; i < n; i++) {
        arr[i * 3 + 0] = data[i * 4 + 0] / 255;
        arr[i * 3 + 1] = data[i * 4 + 1] / 255;
        arr[i * 3 + 2] = data[i * 4 + 2] / 255;
      }
      return new ort.Tensor('float32', arr, [1, SIZE, SIZE, 3]);
    }
    // NCHW: planos R, G, B
    for (let i = 0; i < n; i++) {
      arr[0 * n + i] = data[i * 4 + 0] / 255;
      arr[1 * n + i] = data[i * 4 + 1] / 255;
      arr[2 * n + i] = data[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', arr, [1, 3, SIZE, SIZE]);
  }

  /** Corre los 3 modelos y devuelve el material más probable. */
  async analyze(canvas: HTMLCanvasElement): Promise<MaterialResult> {
    if (!this.ready) {
      return { material: null, confidence: 0, probs: {} };
    }
    const input = this.preprocess(canvas);
    const probs: Record<string, number> = {};

    for (const material of MATERIALS) {
      const session = this.sessions[material];
      const output = await session.run({ [session.inputNames[0]]: input });
      let value = (output[session.outputNames[0]].data as Float32Array)[0];
      if (this.framework === 'torch') {
        value = 1 / (1 + Math.exp(-value));   // logit -> probabilidad
      }
      probs[material] = value;
    }

    // Umbral + margen: solo se declara un material si su probabilidad es alta Y
    // le saca ventaja clara al segundo. Si no, material = null ("acerca un objeto").
    const ranked = MATERIALS.map(m => ({ m, p: probs[m] })).sort((a, b) => b.p - a.p);
    const top = ranked[0];
    const second = ranked[1]?.p ?? 0;
    const THRESHOLD = 0.6;
    const MARGIN = 0.15;
    const material = (top.p >= THRESHOLD && top.p - second >= MARGIN) ? top.m : null;
    return { material, confidence: top.p, probs };
  }
}
