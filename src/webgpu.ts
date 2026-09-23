import { registerTSLMaterialFactory } from '@cyberluke/three-particles';
import {
  assertNamed,
  normalizeVector2Value,
  normalizeTextureValue,
  normalizeDepthTextureValue,
  normalizeBackgroundToVector3,
  resolveWebGPUEffectiveRendererType,
  registerElectricArcGPUFactory,
} from '@cyberluke/three-particles';
import { createElectricArcGPU } from './js/effects/electric-arc/webgpu/electric-arc-webgpu.js';
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
  MLS_MPM_WORKGROUP_SIZE,
  MLS_MPM_FIXED_POINT_MULTIPLIER,
  MLS_MPM_SUBSTEPS,
  MLS_MPM_MAX_GRID_DIM,
  MLS_MPM_PARTICLE_SPACING,
  MLS_MPM_CELL_WORDS,
  MLS_MPM_C_WORDS,
  MLS_MPM_DEFAULTS,
  MLS_MPM_WALL,
  encodeFixedPoint,
  decodeFixedPoint,
  computeMLSMPMGridDims,
  computeMLSMPMGridCount,
  mlsmpmQuadraticWeights,
  mlsmpmCellIndex,
  mlsmpmCellWordBase,
  initMLSMPMDambreak,
  countMLSMPMDambreak,
  createMLSMPMBuffers,
  resolveMLSMPMParams,
  createMLSMPMPipeline,
} from './js/effects/three-particles/webgpu/fluid-mpm.js';
import {
  SPH_WORKGROUP_SIZE,
  SPH_SUBSTEPS,
  SPH_DEFAULT_KERNEL_RADIUS,
  SPH_CELL_SIZE_FACTOR,
  SPH_SENTINEL_CELLS,
  SPH_MAX_HALF_BOX,
  SPH_WALL_STIFFNESS,
  SPH_R2_EPSILON,
  SPH_SCAN_CHUNK,
  SPH_SCAN_STAGES,
  SPH_SLAB_RADIUS,
  SPH_DEFAULTS,
  SPH_LATTICE_FACTOR,
  SPH_LATTICE_MARGIN,
  computeSPHGridDims,
  computeSPHGridCount,
  computeSPHOffset,
  sphCellId,
  computeSPHScanBlocks,
  computeSPHScanInnerSteps,
  sphKernelPowers,
  sphDensityKernelScale,
  sphNearDensityKernelScale,
  sphDensityGradientScale,
  sphViscosityLaplacianScale,
  initSPHDambreak,
  countSPHDambreak,
  createSPHBuffers,
  resolveSPHParams,
  createSPHPipeline,
} from './js/effects/three-particles/webgpu/fluid-sph.js';
import { createFluidTSLMaterial } from './js/effects/three-particles/webgpu/tsl-fluid-metaball-material.js';
import {
  GAUSSIAN_WEIGHTS,
  bilinearWeight,
  BILATERAL_GRID_LEN,
  DEPTH_LEVEL_RADII,
  DEPTH_LEVEL_MIPS,
  FLUID_SHADING_DEFAULTS,
  beerLambert,
  fresnelCoefficient,
  createFluidAttributes,
  createFluidUniforms,
  createFluidDepthTSLMaterial,
  createFluidThicknessTSLMaterial,
  createFluidBilateralTSLMaterial,
  createFluidGaussianTSLMaterial,
  createFluidShadingTSLMaterial,
  createFluidSphereTSLMaterial,
  buildFluidScreenSpacePasses,
} from './js/effects/three-particles/webgpu/tsl-fluid-screen-space-material.js';
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
  createFluidSimPipeline,
} from './js/effects/three-particles/webgpu/tsl-materials.js';
import type {
  FluidSimPipeline,
  FluidSolverId,
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
  // Fluid (ocean) family: solvers + screen-space pass-chain materials.
  createFluidSimPipeline,
  MLS_MPM_WORKGROUP_SIZE,
  MLS_MPM_FIXED_POINT_MULTIPLIER,
  MLS_MPM_SUBSTEPS,
  MLS_MPM_MAX_GRID_DIM,
  MLS_MPM_PARTICLE_SPACING,
  MLS_MPM_CELL_WORDS,
  MLS_MPM_C_WORDS,
  MLS_MPM_DEFAULTS,
  MLS_MPM_WALL,
  encodeFixedPoint,
  decodeFixedPoint,
  computeMLSMPMGridDims,
  computeMLSMPMGridCount,
  mlsmpmQuadraticWeights,
  mlsmpmCellIndex,
  mlsmpmCellWordBase,
  initMLSMPMDambreak,
  countMLSMPMDambreak,
  createMLSMPMBuffers,
  resolveMLSMPMParams,
  createMLSMPMPipeline,
  SPH_WORKGROUP_SIZE,
  SPH_SUBSTEPS,
  SPH_DEFAULT_KERNEL_RADIUS,
  SPH_CELL_SIZE_FACTOR,
  SPH_SENTINEL_CELLS,
  SPH_MAX_HALF_BOX,
  SPH_WALL_STIFFNESS,
  SPH_R2_EPSILON,
  SPH_SCAN_CHUNK,
  SPH_SCAN_STAGES,
  SPH_SLAB_RADIUS,
  SPH_DEFAULTS,
  SPH_LATTICE_FACTOR,
  SPH_LATTICE_MARGIN,
  computeSPHGridDims,
  computeSPHGridCount,
  computeSPHOffset,
  sphCellId,
  computeSPHScanBlocks,
  computeSPHScanInnerSteps,
  sphKernelPowers,
  sphDensityKernelScale,
  sphNearDensityKernelScale,
  sphDensityGradientScale,
  sphViscosityLaplacianScale,
  initSPHDambreak,
  countSPHDambreak,
  createSPHBuffers,
  resolveSPHParams,
  createSPHPipeline,
  GAUSSIAN_WEIGHTS,
  bilinearWeight,
  BILATERAL_GRID_LEN,
  DEPTH_LEVEL_RADII,
  DEPTH_LEVEL_MIPS,
  FLUID_SHADING_DEFAULTS,
  beerLambert,
  fresnelCoefficient,
  createFluidAttributes,
  createFluidUniforms,
  createFluidDepthTSLMaterial,
  createFluidThicknessTSLMaterial,
  createFluidBilateralTSLMaterial,
  createFluidGaussianTSLMaterial,
  createFluidShadingTSLMaterial,
  createFluidSphereTSLMaterial,
  buildFluidScreenSpacePasses,
  createFluidTSLMaterial,
};

export type { FluidSimPipeline, FluidSolverId };

/**
 * Registers the TSL material + compute-pipeline factories AND the Electric
 * Arc GPU factory against a specific `WebGPURenderer` instance. Returns
 * `true` when registration took place; `false` when the supplied renderer
 * is not compute-capable (e.g. WebGL2 fallback), in which case particle
 * systems and Electric Arc effects keep their CPU execution path.
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
  const registered = registerTSLMaterialFactory(
    factory,
    renderer !== undefined ? { renderer } : undefined
  );
  // Electric Arc GPU path: registered against the same compute-capable
  // renderer; skipped (CPU fallback keeps working) otherwise.
  if (renderer !== undefined && registered) {
    registerElectricArcGPUFactory({ create: createElectricArcGPU }, renderer);
  } else {
    registerElectricArcGPUFactory(null);
  }
  return registered;
}
