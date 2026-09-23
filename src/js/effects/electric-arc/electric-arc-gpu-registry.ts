/**
 * Small registry connecting the `/webgpu` entry point (WebGPU-only code)
 * to the backend-neutral `createElectricArc` factory, following the same
 * opt-in pattern used by the particle TSL material factories.
 *
 * @module
 */
import type { ElectricArcGPUFactory } from './electric-arc-types.js';

let gpuFactory: ElectricArcGPUFactory | null = null;
let gpuRenderer: unknown | null = null;

export function registerElectricArcGPUFactory(
  factory: ElectricArcGPUFactory | null,
  renderer?: unknown
): void {
  gpuFactory = factory;
  gpuRenderer = renderer === undefined ? null : renderer;
}

export function getElectricArcGPUFactory(): ElectricArcGPUFactory | null {
  return gpuFactory;
}

/** Renderer that was active when the GPU factory was registered (§3 probe). */
export function getElectricArcGPURenderer(): unknown | null {
  return gpuRenderer;
}
