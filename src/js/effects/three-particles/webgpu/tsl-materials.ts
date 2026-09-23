/**
 * TSL material factory for all particle renderer types + the GPU-only compute
 * pipeline (emission kernel + simulation kernel sharing one storage pool).
 *
 * @module
 */
import { StorageBufferAttribute } from 'three/webgpu';
import { sRGBToLinear } from '../color-utils.js';
import { RendererType } from '../three-particles-enums.js';
import { isLifeTimeCurve } from '../three-particles-utils.js';

import {
  createModifierStorageBuffers,
  createModifierComputeUpdate,
  createSubEmitterInitUpdate,
  createSubEmitterFifoAttribute,
  createTrailRibbonUpdate,
  SUB_EMITTER_EVENT_STRIDE,
  subEmitterWindowSize,
  type ModifierComputePipeline,
  type ModifierFlags,
  type ShapeEmitParams,
  type SubEmitterFifo,
  type SubEmitterInitPipeline,
  type TrailHistoryDesc,
  type TrailRibbonDesc,
  type TrailRibbonPipeline,
} from './compute-modifiers.js';

export {
  createModifierStorageBuffers,
  createModifierComputeUpdate,
  createSubEmitterInitUpdate,
  createSubEmitterFifoAttribute,
  createTrailRibbonUpdate,
  SUB_EMITTER_EVENT_STRIDE,
  subEmitterWindowSize,
};
export type {
  ModifierComputePipeline,
  ModifierFlags,
  ShapeEmitParams,
  SubEmitterFifo,
  SubEmitterInitPipeline,
  TrailHistoryDesc,
  TrailRibbonDesc,
  TrailRibbonPipeline,
};
import { bakeParticleSystemCurves } from './curve-bake.js';
import {
  createMLSMPMPipeline,
  resolveMLSMPMParams,
  initMLSMPMDambreak,
  MLS_MPM_DEFAULTS,
  type MLSMPMPipeline,
} from './fluid-mpm.js';
import {
  createSPHPipeline,
  resolveSPHParams,
  initSPHDambreak,
  SPH_DEFAULTS,
  type SPHPipeline,
} from './fluid-sph.js';
import { createFluidTSLMaterial } from './tsl-fluid-metaball-material.js';
import {
  createFluidSphereTSLMaterial,
  createFluidDepthTSLMaterial,
  createFluidThicknessTSLMaterial,
  createFluidBilateralTSLMaterial,
  createFluidGaussianTSLMaterial,
  createFluidShadingTSLMaterial,
  buildFluidScreenSpacePasses,
} from './tsl-fluid-screen-space-material.js';
import { createInstancedBillboardTSLMaterial } from './tsl-instanced-billboard-material.js';
import { createMeshParticleTSLMaterial } from './tsl-mesh-particle-material.js';
import { createPointSpriteTSLMaterial } from './tsl-point-sprite-material.js';
import {
  createTrailRibbonTSLMaterial,
  type TrailUniforms,
} from './tsl-trail-ribbon-material.js';
import type { SharedUniforms } from './tsl-shared.js';
import type {
  FluidConfig,
  NormalizedParticleSystemConfig,
  PassLayout,
  ShapeConfig,
} from '../types.js';
import type * as THREE from 'three';

/** Solver discriminator accepted by {@link createFluidSimPipeline}. */
export type FluidSolverId = 'MLS-MPM' | 'SPH';

/** GPU storage + kernels of one fluid solver (see `fluid-mpm` / `fluid-sph`). */
export type FluidSimPipeline = {
  /** Solver kernels in strict dispatch order. */
  computeNodes: unknown[];
  /** Semantic pass names aligned with {@link FluidSimPipeline.computeNodes}. */
  passNames: string[];
  /** Real per-pass storage / uniform budgets (`<= 8` storages each). */
  passLayouts: PassLayout[];
  /** Solver-owned storage plus the two shared position / velocity handles. */
  buffers: Record<string, unknown>;
  /** Host-written scalars (`boxWidthRatio` = animated `z` squeeze). */
  uniforms: Record<string, { value: unknown }>;
  /** Lattice size of the solver grid. */
  gridCount: number;
  /** Particle count actually filled by the dambreak initialisation. */
  numParticles: number;
};

export type { TrailUniforms };

export {
  createFluidSphereTSLMaterial,
  createFluidTSLMaterial,
  createFluidDepthTSLMaterial,
  createFluidThicknessTSLMaterial,
  createFluidBilateralTSLMaterial,
  createFluidGaussianTSLMaterial,
  createFluidShadingTSLMaterial,
  buildFluidScreenSpacePasses,
};

export type RendererConfig = {
  transparent: boolean;
  blending: THREE.Blending;
  depthTest: boolean;
  depthWrite: boolean;
};

/**
 * Creates a TSL NodeMaterial for the main particle system (non-trail).
 */
export function createTSLParticleMaterial(
  rendererType: RendererType,
  sharedUniforms: SharedUniforms,
  rendererConfig: RendererConfig,
  gpuCompute = false,
  particleGeometry?: THREE.BufferGeometry
): THREE.Material {
  switch (rendererType) {
    case RendererType.INSTANCED:
      return createInstancedBillboardTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
    case RendererType.MESH:
      return createMeshParticleTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
    case RendererType.FLUID: {
      const fluidCfg: FluidConfig = {
        stretch: readFluidScalar(sharedUniforms, 'fluidStretch', 1),
        absorption: readFluidScalar(sharedUniforms, 'fluidAbsorption', 1.44),
        ior: readFluidScalar(sharedUniforms, 'fluidIor', 1.33),
        sphereSize: readFluidScalar(sharedUniforms, 'fluidSphereSize', 1.2),
        density: readFluidScalar(sharedUniforms, 'fluidDensity', 0.7),
        waterColor: readFluidWaterColor(sharedUniforms),
        sphereRender: readFluidFlag(sharedUniforms, 'fluidSphereRender'),
      };
      // `sphereRender` short-circuits into the direct sphere shading; the rest
      // is the reference pass chain (depth / bilateral / thickness / gaussian /
      // shading). The `pass()` nodes get their camera late-bound per frame.
      // `particleGeometry` (the instanced pool) is forwarded so the depth and
      // thickness passes draw the real billboards, as in the upstream
      // `draw(6, numParticles)`; the image-space stages keep the NDC triangle.
      const chain = buildFluidScreenSpacePasses(
        fluidCfg,
        (sharedUniforms as Record<string, { value: unknown } | undefined>)
          .envMap?.value ?? null,
        undefined,
        particleGeometry
      );
      (
        chain.material as unknown as {
          __fluidPassNodes?: Array<{ camera: unknown }>;
          __fluidPassGeometry?: THREE.BufferGeometry;
        }
      ).__fluidPassNodes = chain.passNodes;
      if (chain.geometry) {
        (
          chain.material as unknown as {
            __fluidPassGeometry?: THREE.BufferGeometry;
          }
        ).__fluidPassGeometry = chain.geometry;
      }
      return chain.material;
    }
    case RendererType.POINTS:
    default:
      return createPointSpriteTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
  }
}

/**
 * Reads an optional scalar fluid parameter from the shared uniform table.
 * The main `createParticleSystem` writes `fluidStretch`, `fluidAbsorption`,
 * and `fluidIor` when `renderer.fluid` is present; missing entries fall back
 * to the documented defaults.
 */
const readFluidScalar = (
  sharedUniforms: SharedUniforms,
  key: string,
  fallback: number
): number => {
  const raw = (
    sharedUniforms as Record<string, { value: unknown } | undefined>
  )[key]?.value;
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : fallback;
};

/** Boolean reader for the `fluidSphereRender` style flags. */
const readFluidFlag = (
  sharedUniforms: SharedUniforms,
  key: string
): boolean => {
  const raw = (
    sharedUniforms as Record<string, { value: unknown } | undefined>
  )[key]?.value;
  return raw === true;
};

/** `rgb` tuple reader with the `fluid.wgsl` water tint as the fallback. */
const readFluidWaterColor = (
  sharedUniforms: SharedUniforms
): [number, number, number] => {
  const raw = (sharedUniforms as Record<string, { value: unknown } | undefined>)
    .fluidWaterColor?.value;
  if (Array.isArray(raw) && raw.length === 3) {
    return [Number(raw[0]) || 0, Number(raw[1]) || 0, Number(raw[2]) || 0];
  }
  return [0.0, 0.7375, 0.95];
};

/**
 * Builds one ocean-style fluid solver pipeline on top of an existing particle
 * pool. `shared` must be the base pipeline's `position` / `velocity` storage
 * attributes so the render material reads the integrated state directly.
 *
 * @param solver - `'MLS-MPM'` (grid, 2 sub-steps) or `'SPH'` (neighbour search).
 * @param shared - Shared position / velocity storage of the base pipeline.
 * @param maxParticles - Particle capacity of the base pool.
 * @param normalizedConfig - Merged system config (`renderer.mlsMpm` / `.sph`).
 * @returns Kernels, per-pass budgets, buffers, live uniforms and lattice size.
 */
export function createFluidSimPipeline(
  solver: FluidSolverId,
  shared: {
    position: { array: Float32Array };
    velocity: { array: Float32Array };
  },
  maxParticles: number,
  normalizedConfig: NormalizedParticleSystemConfig
): FluidSimPipeline {
  // `NormalizedParticleSystemConfig` is `Required<ParticleSystemConfig>`, so
  // `renderer` is non-nullable and `Renderer` already carries the `sph` /
  // `mlsMpm` optional sub-blocks.
  const renderer = normalizedConfig.renderer;
  const isSPH = solver === 'SPH';
  const sharedPair = {
    position: shared.position as never,
    velocity: shared.velocity as never,
  };
  // Optional spherical boundary (`domain: { kind: 'sphere' }`, upstream
  // WaterBall): constrains the CPU seed AND the GPU wall kernels. Radius 0 /
  // missing block keeps the classic box domain.
  const domain = renderer.fluid?.domain;
  const seedSphere =
    domain && domain.kind === 'sphere' && domain.radius > 0
      ? {
          center: (domain.center ?? [0, 0, 0]) as readonly [
            number,
            number,
            number,
          ],
          radius: domain.radius,
        }
      : undefined;

  if (isSPH) {
    const cfg = renderer.sph;
    const halfBox: [number, number, number] = [
      ...(cfg?.halfBoxSize ?? SPH_DEFAULTS.halfBoxSize),
    ] as [number, number, number];
    const ratio =
      typeof cfg?.boxWidthRatio === 'number' &&
      Number.isFinite(cfg.boxWidthRatio)
        ? (cfg.boxWidthRatio as number)
        : 1;
    // Seed the shared pos / vec4 storage with the reference dambreak lattice.
    const state = initSPHDambreak(
      halfBox,
      Math.max(1, maxParticles),
      undefined,
      Math.random,
      seedSphere
    );
    shared.position.array.set(state.position);
    shared.velocity.array.set(state.velocity);
    const sph: SPHPipeline = createSPHPipeline(
      state.count,
      resolveSPHParams(cfg, halfBox),
      [halfBox[0], halfBox[1], halfBox[2] * ratio],
      sharedPair
    );
    return {
      computeNodes: sph.computeNodes,
      passNames: sph.passNames,
      passLayouts: sph.passLayouts,
      buffers: sph.buffers as unknown as Record<string, unknown>,
      uniforms: sph.uniforms as unknown as Record<string, { value: unknown }>,
      gridCount: sph.gridCount,
      numParticles: sph.numParticles,
    };
  }

  const cfg = renderer.mlsMpm;
  const box: [number, number, number] = [
    ...(cfg?.boxSize ?? MLS_MPM_DEFAULTS.boxSize),
  ] as [number, number, number];
  const ratio =
    typeof cfg?.boxWidthRatio === 'number' && Number.isFinite(cfg.boxWidthRatio)
      ? (cfg.boxWidthRatio as number)
      : 1;
  const state = initMLSMPMDambreak(
    box,
    Math.max(1, maxParticles),
    undefined,
    Math.random,
    seedSphere
  );
  shared.position.array.set(state.position);
  shared.velocity.array.set(state.velocity);
  const mls: MLSMPMPipeline = createMLSMPMPipeline(
    state.count,
    resolveMLSMPMParams(cfg, box),
    [box[0], box[1], box[2] * ratio],
    sharedPair
  );
  return {
    computeNodes: mls.computeNodes,
    passNames: mls.passNames,
    passLayouts: mls.passLayouts,
    buffers: mls.buffers as unknown as Record<string, unknown>,
    uniforms: mls.uniforms as unknown as Record<string, { value: unknown }>,
    gridCount: mls.gridCount,
    numParticles: mls.numParticles,
  };
}

export function createTSLTrailMaterial(
  trailUniforms: TrailUniforms,
  rendererConfig: RendererConfig
): THREE.Material {
  return createTrailRibbonTSLMaterial(trailUniforms, rendererConfig);
}

/** Pull a scalar pair out of either a `number` or `{ min, max }` shape. */
const pair = (v: unknown): [number, number] => {
  if (typeof v === 'number') return [v, v];
  if (v && typeof v === 'object') {
    const o = v as Record<string, number>;
    return [Number(o.min) || 0, Number(o.max) || 0];
  }
  return [0, 0];
};

/** Map the public `Shape` string onto the GPU kernel kind. */
const shapeKindOf = (t: ShapeConfig['shape']): 0 | 1 | 2 | 3 | 4 => {
  switch (t) {
    case 'SPHERE':
      return 0;
    case 'CONE':
      return 1;
    case 'CIRCLE':
      return 2;
    case 'RECTANGLE':
      return 3;
    case 'BOX':
      return 4;
    default:
      return 0; // SPHERE, matching DEFAULT_PARTICLE_SYSTEM_CONFIG.shape.shape
  }
};

/** Map `EmitFrom` onto the box-emission kernel code. */
const boxEmitFromOf = (e: string | undefined): 0 | 1 | 2 =>
  e === 'SHELL' ? 1 : e === 'EDGE' ? 2 : 0;

/**
 * Decodes the nested public `ShapeConfig` (+ start values / sheet frame) into
 * the flat scalar table consumed by the compute kernels. Exported so the
 * sub-emitter init kernel can reuse the identical encoding.
 */
export function encodeShapeEmitParams(
  normalizedConfig: NormalizedParticleSystemConfig,
  particleSystemId: number
): ShapeEmitParams {
  const bakedCurves = bakeParticleSystemCurves(
    normalizedConfig,
    particleSystemId
  );
  const pairLocal = pair;
  const [lifeMin, lifeMax] = pairLocal(normalizedConfig.startLifetime);
  const [spdMin, spdMax] = pairLocal(normalizedConfig.startSpeed);
  const [szMin, szMax] = pairLocal(normalizedConfig.startSize);
  const [rotMin, rotMax] = pairLocal(normalizedConfig.startRotation);
  const [opMin, opMax] = pairLocal(normalizedConfig.startOpacity);
  const cMin = (
    normalizedConfig.startColor as {
      min?: { r: number; g: number; b: number };
    }
  ).min || { r: 1, g: 1, b: 1 };
  const cMax = (
    normalizedConfig.startColor as {
      max?: { r: number; g: number; b: number };
    }
  ).max || { r: 1, g: 1, b: 1 };
  const sf =
    (normalizedConfig.textureSheetAnimation &&
      (normalizedConfig.textureSheetAnimation as { startFrame?: unknown })
        .startFrame) ||
    0;
  const sfPair = pairLocal(sf);
  const shp = normalizedConfig.shape;
  const sph = shp.sphere;
  const cone = shp.cone;
  const circ = shp.circle;
  const rect = shp.rectangle;
  const bx = shp.box;
  const num = (v: unknown, fallback: number): number =>
    typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  const kind = shapeKindOf(shp.shape);
  return {
    shapeKind: kind,
    radius:
      shp.shape === 'CONE'
        ? num(cone?.radius, 1)
        : shp.shape === 'CIRCLE'
          ? num(circ?.radius, 1)
          : num(sph?.radius, 1),
    radiusThickness:
      shp.shape === 'CONE'
        ? num(cone?.radiusThickness, 1)
        : shp.shape === 'CIRCLE'
          ? num(circ?.radiusThickness, 1)
          : num(sph?.radiusThickness, 1),
    arcDeg:
      shp.shape === 'CONE'
        ? num(cone?.arc, 360)
        : shp.shape === 'CIRCLE'
          ? num(circ?.arc, 360)
          : num(sph?.arc, 360),
    coneAngleDeg: num(cone?.angle, 90),
    rectangleRotXDeg: num(rect?.rotation?.x, 0),
    rectangleRotYDeg: num(rect?.rotation?.y, 0),
    rectangleScaleX: num(rect?.scale?.x, 1),
    rectangleScaleY: num(rect?.scale?.y, 1),
    boxScaleX: num(bx?.scale?.x, 1),
    boxScaleY: num(bx?.scale?.y, 1),
    boxScaleZ: num(bx?.scale?.z, 1),
    boxEmitFrom: boxEmitFromOf(bx?.emitFrom),
    speedMin: spdMin,
    speedMax: spdMax,
    sizeMin: szMin,
    sizeMax: szMax,
    rotMin: rotMin,
    rotMax: rotMax,
    opacityMin: opMin,
    opacityMax: opMax,
    lifeMin: lifeMin,
    lifeMax: lifeMax,
    colorRMin: cMin.r,
    colorRMax: cMax.r,
    colorGMin: cMin.g,
    colorGMax: cMax.g,
    colorBMin: cMin.b,
    colorBMax: cMax.b,
    startFrameMin: sfPair[0],
    startFrameMax: sfPair[1],
    // Separate rotationOverLifetime range (never the startRotation pair).
    rotOverLifeMin: (() => {
      const rol = normalizedConfig.rotationOverLifetime as unknown as
        { min?: number; max?: number } | undefined;
      return typeof rol?.min === 'number' && Number.isFinite(rol.min)
        ? rol.min
        : 0;
    })(),
    rotOverLifeMax: (() => {
      const rol = normalizedConfig.rotationOverLifetime as unknown as
        { min?: number; max?: number } | undefined;
      return typeof rol?.max === 'number' && Number.isFinite(rol.max)
        ? rol.max
        : 0;
    })(),
    noiseOctaves: num(normalizedConfig.noise?.octaves, 1),
    noiseUseRandomOffset: !!normalizedConfig.noise?.useRandomOffset,
    rotationCurveActive: normalizedConfig.rotationOverLifetime.isActive,
    rotationalXCurve:
      (bakedCurves as unknown as Record<string, number>).orbitalVelX ?? -1,
    rotationalYCurve:
      (bakedCurves as unknown as Record<string, number>).orbitalVelY ?? -1,
    rotationalZCurve:
      (bakedCurves as unknown as Record<string, number>).orbitalVelZ ?? -1,
    linearXCurve:
      (bakedCurves as unknown as Record<string, number>).linearVelX ?? -1,
    linearYCurve:
      (bakedCurves as unknown as Record<string, number>).linearVelY ?? -1,
    linearZCurve:
      (bakedCurves as unknown as Record<string, number>).linearVelZ ?? -1,
  };
}

export function createComputePipeline(
  maxParticles: number,
  instanced: boolean,
  normalizedConfig: NormalizedParticleSystemConfig,
  particleSystemId: number,
  forceFieldCount: number,
  collisionPlaneCount = 0,
  subFifos?: SubEmitterFifo[],
  trailDesc?: TrailHistoryDesc
): ModifierComputePipeline {
  const bakedCurves = bakeParticleSystemCurves(
    normalizedConfig,
    particleSystemId
  );
  const v = normalizedConfig.velocityOverLifetime;

  const flags: ModifierFlags = {
    sizeOverLifetime: normalizedConfig.sizeOverLifetime.isActive,
    opacityOverLifetime: normalizedConfig.opacityOverLifetime.isActive,
    colorOverLifetime: normalizedConfig.colorOverLifetime.isActive,
    rotationOverLifetime: normalizedConfig.rotationOverLifetime.isActive,
    linearVelocity:
      v.isActive &&
      (isLifeTimeCurve(v.linear.x ?? 0) ||
        isLifeTimeCurve(v.linear.y ?? 0) ||
        isLifeTimeCurve(v.linear.z ?? 0) ||
        v.linear.x !== 0 ||
        v.linear.y !== 0 ||
        v.linear.z !== 0),
    orbitalVelocity:
      v.isActive &&
      (isLifeTimeCurve(v.orbital.x ?? 0) ||
        isLifeTimeCurve(v.orbital.y ?? 0) ||
        isLifeTimeCurve(v.orbital.z ?? 0) ||
        v.orbital.x !== 0 ||
        v.orbital.y !== 0 ||
        v.orbital.z !== 0),
    noise: normalizedConfig.noise.isActive,
    forceFields: forceFieldCount > 0,
    collisionPlanes: collisionPlaneCount > 0,
  };

  // Shared decoder (also used for sub-emitter children) - nested ShapeConfig
  // branches + start values, arcs / angles in degrees.
  const shapeParams: ShapeEmitParams = encodeShapeEmitParams(
    normalizedConfig,
    particleSystemId
  );

  // Raw axis values (constants / random ranges / curves) for the seed-based
  // per-particle axis derivation in the kernels (oracle-parity randomness).
  const velocityValues = {
    linear: [v.linear.x as never, v.linear.y as never, v.linear.z as never],
    orbital: [v.orbital.x as never, v.orbital.y as never, v.orbital.z as never],
  };

  // Trail ring integer metadata (atomic<u32>): two cursor/count words.
  if (trailDesc && !trailDesc.meta) {
    trailDesc.meta = new StorageBufferAttribute(
      new Uint32Array(Math.max(1, maxParticles) * 2),
      1
    );
  }

  const built = createModifierStorageBuffers(
    maxParticles,
    instanced,
    bakedCurves.data,
    flags.forceFields,
    flags.collisionPlanes,
    trailDesc ? trailDesc.length : 0
  );
  if (trailDesc && built.buffers.trailMeta) {
    trailDesc.meta = built.buffers.trailMeta;
  }

  return createModifierComputeUpdate(
    built.buffers,
    maxParticles,
    bakedCurves,
    flags,
    shapeParams,
    forceFieldCount,
    collisionPlaneCount,
    subFifos ?? [],
    trailDesc,
    velocityValues
  );
}
