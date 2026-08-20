import {
  Component, OnInit, OnDestroy, ElementRef, ViewChild,
  ChangeDetectionStrategy, ChangeDetectorRef, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';

import {
  IonHeader, IonToolbar, IonTitle, IonContent,
  IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle, IonCardContent,
  IonButton, IonIcon, IonSpinner, IonChip, IonLabel,
  IonSegment, IonSegmentButton,
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  camera, images, colorWand, refreshOutline, eyeOutline,
  informationCircleOutline, alertCircleOutline, checkmarkCircle, closeCircle,
} from 'ionicons/icons';

import { OpenCvService } from '../opencv.service';
import { EcosortService, Framework, Analysis, MATERIALS } from '../ecosort.service';

@Component({
  selector: 'app-home',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    IonHeader, IonToolbar, IonTitle, IonContent,
    IonCard, IonCardHeader, IonCardTitle, IonCardSubtitle, IonCardContent,
    IonButton, IonIcon, IonSpinner, IonChip, IonLabel,
    IonSegment, IonSegmentButton,
  ],
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
})
export class HomePage implements OnInit, OnDestroy {
  @ViewChild('photoCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('fileCamera') fileCameraRef!: ElementRef<HTMLInputElement>;
  @ViewChild('fileGallery') fileGalleryRef!: ElementRef<HTMLInputElement>;

  readonly Math = Math;
  readonly MATERIALS = MATERIALS;

  opencvReady = signal(false);
  modelReady  = signal(false);
  modelError  = signal('');
  imageReady  = signal(false);
  sharpness   = signal(0);
  analizando  = signal(false);
  resultTf    = signal<Analysis | null>(null);
  resultTorch = signal<Analysis | null>(null);
  modo        = signal<'tf' | 'torch' | 'compare'>('compare');

  private cvSub!: Subscription;

  constructor(
    private opencv: OpenCvService,
    private ecosort: EcosortService,
    private cdr: ChangeDetectorRef,
  ) {
    addIcons({
      camera, images, colorWand, refreshOutline, eyeOutline,
      informationCircleOutline, alertCircleOutline, checkmarkCircle, closeCircle,
    });
  }

  ngOnInit(): void {
    this.cvSub = this.opencv.ready$.subscribe(ready => {
      this.opencvReady.set(ready);
      this.cdr.markForCheck();
    });
    this.ecosort.load()
      .then(() => { this.modelReady.set(true); this.cdr.markForCheck(); })
      .catch(err => { this.modelError.set(String(err?.message ?? err)); this.cdr.markForCheck(); });
  }

  ngOnDestroy(): void {
    this.cvSub?.unsubscribe();
  }

  abrirCamara(): void { this.fileCameraRef.nativeElement.click(); }
  abrirGaleria(): void { this.fileGalleryRef.nativeElement.click(); }

  /** Carga la foto (cámara/galería) al canvas, mide nitidez con OpenCV y limpia resultados. */
  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      const canvas = this.canvasRef.nativeElement;
      const scale = Math.min(1, 900 / img.width);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext('2d', { willReadFrequently: true })!.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(img.src);

      // Nitidez con OpenCV (requisito): Laplaciano sobre la foto.
      let s = 0;
      if (this.opencvReady()) { try { s = this.opencv.calcSharpness(canvas); } catch { /* ignore */ } }
      this.sharpness.set(Math.round(s));

      this.resultTf.set(null);
      this.resultTorch.set(null);
      this.imageReady.set(true);
      this.cdr.markForCheck();
    };
    img.src = URL.createObjectURL(file);
    input.value = '';   // permite volver a elegir el mismo archivo
  }

  /** Cambia de modelo (TF / PyTorch / Comparar) y reanaliza si ya hay foto. */
  cambiarModo(ev: CustomEvent): void {
    this.modo.set(ev.detail.value);
    if (this.imageReady()) this.analizar();
    else this.cdr.markForCheck();
  }

  /** Analiza la foto según el modo: un solo framework o ambos (comparar). */
  async analizar(): Promise<void> {
    if (!this.modelReady() || !this.imageReady() || this.analizando()) return;
    this.analizando.set(true);
    this.cdr.markForCheck();
    try {
      const canvas = this.canvasRef.nativeElement;
      const modo = this.modo();
      this.resultTf.set(
        modo === 'tf' || modo === 'compare' ? await this.ecosort.analyze(canvas, 'tf') : null);
      this.resultTorch.set(
        modo === 'torch' || modo === 'compare' ? await this.ecosort.analyze(canvas, 'torch') : null);
    } catch (err) {
      console.error('Error al analizar:', err);
    }
    this.analizando.set(false);
    this.cdr.markForCheck();
  }

  // --- helpers de UI ---
  // Nitidez = varianza del Laplaciano (OpenCV). Informativa: en fotos de un
  // objeto sobre fondo liso los valores son bajos aunque el foco sea perfecto.
  sharpnessLabel(): string {
    const s = this.sharpness();
    if (!s) return '—';
    if (s < 10) return 'Baja';
    if (s < 40) return 'Media';
    return 'Alta';
  }
  sharpnessColor(): string {
    const s = this.sharpness();
    if (s < 10) return 'warning';
    if (s < 40) return 'primary';
    return 'success';
  }

  esComparar(): boolean { return this.modo() === 'compare'; }

  /** ¿TF y PyTorch coinciden en el material? (solo en modo comparar) */
  coinciden(): boolean {
    const a = this.resultTf()?.material;
    const b = this.resultTorch()?.material;
    return a != null && a === b;
  }

  /** Cards a renderizar según el modo (uno o ambos frameworks). */
  resultados(): { label: string; r: Analysis | null }[] {
    const out: { label: string; r: Analysis | null }[] = [];
    if (this.resultTf())    out.push({ label: 'TensorFlow', r: this.resultTf() });
    if (this.resultTorch()) out.push({ label: 'PyTorch', r: this.resultTorch() });
    return out;
  }
}
