/**
 * TSL material factory for all particle renderer types + the GPU-only compute
 * pipeline (emission kernel + simulation kernel sharing one storage pool).
 *
 * @module
 */
import { RendererType } from '../three-particles-enums.js';
import { isLifeTimeCurve } from '../three-particles-utils.js';
import { sRGBToLinear } from '../color-utils.js';
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
import { createInstancedBillboardTSLMaterial } from './tsl-instanced-billboard-material.js';
import { createMeshParticleTSLMaterial } from './tsl-mesh-particle-material.js';
import { createPointSpriteTSLMaterial } from './tsl-point-sprite-material.js';
import {
  createTrailRibbonTSLMaterial,
  type TrailUniforms,
} from './tsl-trail-ribbon-material.js';
import type { SharedUniforms } from './tsl-shared.js';
import type {
  NormalizedParticleSystemConfig,
  ShapeConfig,
} from '../types.js';

import type * as THREE from 'three';

export type { TrailUniforms };

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
  gpuCompute = false
): THREE.Material {
  switch (rendererType) {
    case RendererType.INSTANCED:
      return createInstancedBillboardTSLMaterial(sharedUniforms, rendererConfig, gpuCompute);
    case RendererType.MESH:
      return createMeshParticleTSLMaterial(sharedUniforms, rendererConfig, gpuCompute);
    case RendererType.POINTS:
    default:
      return createPointSpriteTSLMaterial(sharedUniforms, rendererConfig, gpuCompute);
  }
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
    case 'SPHERE': return 0;
    case 'CONE': return 1;
    case 'CIRCLE': return 2;
    case 'RECTANGLE': return 3;
    case 'BOX': return 4;
    default: return 0; // SPHERE, matching DEFAULT_PARTICLE_SYSTEM_CONFIG.shape.shape
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
  const bakedCurves = bakeParticleSystemCurves(normalizedConfig, particleSystemId);
  const pairLocal = pair;
  const [lifeMin, lifeMax] = pairLocal(normalizedConfig.startLifetime);
  const [spdMin, spdMax] = pairLocal(normalizedConfig.startSpeed);
  const [szMin, szMax] = pairLocal(normalizedConfig.startSize);
  const [rotMin, rotMax] = pairLocal(normalizedConfig.startRotation);
  const [opMin, opMax] = pairLocal(normalizedConfig.startOpacity);
  const cMin =
    (
      normalizedConfig.startColor as {
        min?: { r: number; g: number; b: number };
      }
    ).min || { r: 1, g: 1, b: 1 };
  const cMax =
    (
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
    coneAngleDeg: num(cone?.angle, 25),
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

  const bakedCurves = bakeParticleSystemCurves(normalizedConfig, particleSystemId);
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


  const built = createModifierStorageBuffers(
    maxParticles,
    instanced,
    bakedCurves.data,
    flags.forceFields,
    flags.collisionPlanes
  );

  return createModifierComputeUpdate(
    built.buffers,
    maxParticles,
    bakedCurves,
    flags,
    shapeParams,
    forceFieldCount,
    collisionPlaneCount,
    subFifos ?? [],
    trailDesc
  );
}

