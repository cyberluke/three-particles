import { registerTSLMaterialFactory } from '@cyberluke/three-particles';
import {
  assertNamed,
  normalizeVector2Value,
  normalizeTextureValue,
  normalizeDepthTextureValue,
  normalizeBackgroundToVector3,
  resolveWebGPUEffectiveRendererType,
} from '@cyberluke/three-particles';
import { encodeCollisionPlanesForGPU } from './js/effects/three-particles/webgpu/compute-collision-planes.js';
import { encodeForceFieldsForGPU } from './js/effects/three-particles/webgpu/compute-force-fields.js';
import {
  pcgRawU32,
  pcg01,
  mixBirthSeed,
  randomChannel,
  nextSystemSeed,
  CH,
} from './js/effects/three-particles/webgpu/compute-modifiers.js';
import {
  createTSLParticleMaterial,
  createTSLTrailMaterial,
  createComputePipeline,
  createModifierStorageBuffers,
  createSubEmitterInitUpdate,
  createSubEmitterFifoAttribute,
  createTrailRibbonUpdate,
  encodeShapeEmitParams,
  subEmitterWindowSize,
} from './js/effects/three-particles/webgpu/tsl-materials.js';

// Re-export individual functions for power users who do not want the factory.
export {
  createTSLParticleMaterial,
  createTSLTrailMaterial,
  createComputePipeline,
  createModifierStorageBuffers,
  createSubEmitterInitUpdate,
  createSubEmitterFifoAttribute,
  createTrailRibbonUpdate,
  encodeShapeEmitParams,
  subEmitterWindowSize,
  encodeForceFieldsForGPU,
  encodeCollisionPlanesForGPU,
  pcgRawU32,
  pcg01,
  mixBirthSeed,
  randomChannel,
  nextSystemSeed,
  CH,
  assertNamed,
  normalizeVector2Value,
  normalizeTextureValue,
  normalizeDepthTextureValue,
  normalizeBackgroundToVector3,
  resolveWebGPUEffectiveRendererType,
};

/**
 * Registers the TSL material + compute-pipeline factories against a specific
 * `WebGPURenderer` instance. Returns `true` when registration took place;
 * `false` when the supplied renderer is not compute-capable (e.g. WebGL2
 * fallback), in which case every subsequent `createParticleSystem(...)` call
 * throws a clear "no CPU fallback" error.
 */
export function enableWebGPU(renderer?: unknown): boolean {
  const factory: Parameters<typeof registerTSLMaterialFactory>[0] = {
    createTSLParticleMaterial: createTSLParticleMaterial as never,
    createTSLTrailMaterial: createTSLTrailMaterial as never,
    createComputePipeline: createComputePipeline as never,
    encodeForceFieldsForGPU,
    encodeCollisionPlanesForGPU,
    createSubEmitterFifoAttribute: createSubEmitterFifoAttribute as never,
    createSubEmitterInitUpdate: createSubEmitterInitUpdate as never,
    createTrailRibbonUpdate: createTrailRibbonUpdate as never,
    encodeShapeEmitParams: encodeShapeEmitParams as never,
  };
  return registerTSLMaterialFactory(
    factory,
    renderer !== undefined ? { renderer } : undefined
  );
}
