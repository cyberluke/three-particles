import { registerTSLMaterialFactory } from '@newkrok/three-particles';
import { encodeCollisionPlanesForGPU } from './js/effects/three-particles/webgpu/compute-collision-planes.js';
import { encodeForceFieldsForGPU } from './js/effects/three-particles/webgpu/compute-force-fields.js';
import {
  writeParticleToModifierBuffers,
  deactivateParticleInModifierBuffers,
  flushEmitQueue,
  registerCurveDataLength,
} from './js/effects/three-particles/webgpu/compute-modifiers.js';
import {
  createTSLParticleMaterial,
  createTSLTrailMaterial,
  createComputePipeline,
} from './js/effects/three-particles/webgpu/tsl-materials.js';

// Re-export all individual functions for power users
export {
  createTSLParticleMaterial,
  createTSLTrailMaterial,
  createComputePipeline,
  writeParticleToModifierBuffers,
  deactivateParticleInModifierBuffers,
  flushEmitQueue,
  registerCurveDataLength,
  encodeCollisionPlanesForGPU,
  encodeForceFieldsForGPU,
};

/**
 * Convenience function that registers all WebGPU TSL material factories
 * and GPU compute helpers in a single call.
 *
 * Call this **once** before creating any particle systems that use WebGPU rendering.
 *
 * Pass your renderer to get automatic capability detection: when the
 * renderer cannot dispatch compute shaders (e.g. `THREE.WebGLRenderer`),
 * registration is skipped with a console warning and all particle systems
 * keep using the CPU/GLSL path. Without a renderer argument, registration
 * is unconditional (previous behavior) — only do that when you know a
 * WebGPU-capable renderer is in use.
 *
 * @param renderer - Optional Three.js renderer used for capability detection.
 * @returns `true` when the WebGPU path was registered, `false` when the
 *   provided renderer is not compute-capable and registration was skipped.
 *
 * @example
 * ```typescript
 * import { enableWebGPU } from '@newkrok/three-particles/webgpu';
 * const renderer = new THREE.WebGPURenderer();
 * const gpuEnabled = enableWebGPU(renderer); // false with a WebGLRenderer
 * ```
 */
export function enableWebGPU(renderer?: unknown): boolean {
  // The TSLMaterialFactory interface deliberately uses wider parameter types
  // (Record<string,…>) to avoid pulling WebGPU-specific imports into the
  // main DTS output. The concrete functions are fully type-safe at their
  // own definition sites; the cast here bridges the two type worlds.

  const factory: Parameters<typeof registerTSLMaterialFactory>[0] = {
    createTSLParticleMaterial: createTSLParticleMaterial as any,
    createTSLTrailMaterial: createTSLTrailMaterial as any,
    createComputePipeline,
    writeParticleToModifierBuffers,
    deactivateParticleInModifierBuffers,
    flushEmitQueue,
    registerCurveDataLength,
    encodeForceFieldsForGPU,
    encodeCollisionPlanesForGPU,
  };
  return registerTSLMaterialFactory(
    factory,
    renderer !== undefined ? { renderer } : undefined
  );
}
