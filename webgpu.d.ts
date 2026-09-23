/**
 * WebGPU entry point type declarations for @cyberluke/three-particles/webgpu.
 *
 * Hand-written because automatic DTS generation fails on TSL node types
 * (Three.js TSL Fn return types resolve to `unknown` in the type system).
 */
import type { Material, Blending } from 'three';
import type { RendererType } from '@cyberluke/three-particles';

/** Renderer configuration for material creation. */
export interface RendererConfig {
  transparent: boolean;
  blending: Blending;
  depthTest: boolean;
  depthWrite: boolean;
}

/** Creates a TSL NodeMaterial for the main particle system (non-trail). */
export declare function createTSLParticleMaterial(
  rendererType: RendererType,
  sharedUniforms: Record<string, { value: unknown }>,
  rendererConfig: RendererConfig,
  gpuCompute?: boolean
): Material;

/** Creates a TSL NodeMaterial for the trail ribbon renderer. */
export declare function createTSLTrailMaterial(
  trailUniforms: Record<string, { value: unknown }>,
  rendererConfig: RendererConfig
): Material;

/** Creates the GPU compute pipeline for particle simulation. */
export declare function createComputePipeline(
  maxParticles: number,
  instanced: boolean,
  normalizedConfig: unknown,
  particleSystemId: number,
  forceFieldCount: number,
  collisionPlaneCount?: number
): unknown;

/** Writes init data for a newly emitted particle into modifier storage buffers. */
export declare function writeParticleToModifierBuffers(
  buffers: unknown,
  index: number,
  data: Record<string, unknown>
): void;

/** Deactivates a particle in the modifier storage buffers. */
export declare function deactivateParticleInModifierBuffers(
  buffers: unknown,
  index: number
): void;

/** Flushes pending init data to the GPU. Call once per frame before compute dispatch. */
export declare function flushEmitQueue(buffers: unknown): number;

/** Registers the curve data length for a buffer. Called once during pipeline creation. */
export declare function registerCurveDataLength(
  buffers: unknown,
  curveDataLength: number
): void;

/** Packs force field configs into a flat Float32Array for GPU upload. */
export declare function encodeForceFieldsForGPU(
  forceFields: ReadonlyArray<unknown>,
  particleSystemId: number,
  systemLifetimePercentage: number
): Float32Array;

/** Packs collision plane configs into a flat Float32Array for GPU upload. */
export declare function encodeCollisionPlanesForGPU(
  planes: ReadonlyArray<unknown>
): Float32Array;

/**
 * Convenience function that registers all WebGPU TSL material factories,
 * GPU compute helpers, and the Electric Arc GPU factory in a single call.
 *
 * Call this **once** after `await renderer.init()` of each
 * `THREE.WebGPURenderer` before creating effects.
 *
 * Pass your renderer to get automatic capability detection: when the
 * renderer cannot dispatch compute shaders (e.g. `THREE.WebGLRenderer`),
 * registration is skipped with a console warning and all particle systems
 * / electric arcs keep their CPU execution path (which is fully supported).
 *
 * Backend semantics for `simulationBackend`:
 *   AUTO -> GPU compute where supported, CPU otherwise.
 *   CPU  -> always CPU simulation.
 *   GPU  -> requests GPU compute, falls back to CPU when compute is unavailable.
 *
 * @param renderer - Optional Three.js renderer used for capability detection.
 * @returns `true` when the WebGPU path was registered, `false` when the
 *   provided renderer is not compute-capable and registration was skipped.
 *
 * @example
 * ```typescript
 * import { enableWebGPU } from '@cyberluke/three-particles/webgpu';
 * const renderer = new THREE.WebGPURenderer();
 * await renderer.init();
 * const gpuEnabled = enableWebGPU(renderer);
 * ```
 */
export declare function enableWebGPU(renderer?: unknown): boolean;

/** Solver discriminator accepted by `createFluidSimPipeline`. */
export type FluidSolverId = 'MLS-MPM' | 'SPH';

/** GPU storage + kernels of one fluid solver. */
export interface FluidSimPipeline {
  computeNodes: unknown[];
  passNames: string[];
  passLayouts: Array<{
    name: string;
    storageBindings: number;
    uniformBindings: number;
  }>;
  buffers: Record<string, unknown>;
  /** Host-written scalars (`boxWidthRatio` = animated `z` squeeze). */
  uniforms: Record<string, { value: unknown }>;
  gridCount: number;
  numParticles: number;
}

/**
 * Builds one ocean-style fluid solver pipeline (MLS-MPM or SPH) on top of the
 * base modifier pool's `position` / `velocity` storage, seeds the shared
 * arrays with the reference dambreak lattice and returns the kernels in strict
 * dispatch order together with the real per-pass binding budgets.
 */
export declare function createFluidSimPipeline(
  solver: FluidSolverId,
  shared: {
    position: { array: Float32Array };
    velocity: { array: Float32Array };
  },
  maxParticles: number,
  normalizedConfig: unknown
): FluidSimPipeline;

/** Single-pass metaball fluid material (`RendererType.FLUID` fallback). */
export declare function createFluidTSLMaterial(
  sharedUniforms: Record<string, { value: unknown }>,
  rendererConfig: RendererConfig,
  gpuCompute?: boolean,
  stretch?: number,
  absorption?: number,
  ior?: number
): Material;

/** Screen-space depth map pass material (`depthMap.wgsl`). */
export declare function createFluidDepthTSLMaterial(config?: unknown): Material;

/** Additive thickness-map pass material (`thicknessMap.wgsl`). */
export declare function createFluidThicknessTSLMaterial(config?: unknown): Material;

/** One bilateral up-sample iteration (levels 1..4 of the depth map). */
export declare function createFluidBilateralTSLMaterial(
  level: number,
  sourceRadius: number,
  sourceTexture: unknown,
  iterationCount: number
): Material;

/** One separable Gaussian blur axis (`1` = x, `0` = y). */
export declare function createFluidGaussianTSLMaterial(
  textureIn: unknown,
  axisWeight: 1 | 0
): Material;

/** Final Beer-Lambert / Fresnel shading pass (`fluid.wgsl`). */
export declare function createFluidShadingTSLMaterial(
  sources: Record<string, unknown>,
  config?: unknown
): Material;

/** Direct per-particle sphere debug shading (`sphere.wgsl`). */
export declare function createFluidSphereTSLMaterial(config?: unknown): Material;

/**
 * Assembles the whole screen-space fluid pass chain. The returned `material`
 * is attached to the visible mesh, and `passNodes` are late-bound with the
 * scene camera by the host (`updateWorld`).
 */
export declare function buildFluidScreenSpacePasses(
  config?: unknown,
  envMap?: unknown,
  camera?: unknown
): { material: Material; passNodes: Array<{ camera: unknown }> };
