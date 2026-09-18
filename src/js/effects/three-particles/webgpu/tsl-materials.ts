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
  type ModifierComputePipeline,
  type ModifierFlags,
  type ShapeEmitParams,
} from './compute-modifiers.js';

export {
  createModifierStorageBuffers,
  createModifierComputeUpdate,
};
export type { ModifierComputePipeline, ModifierFlags, ShapeEmitParams };

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
  ShapeType,
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

/** Map the 5 shape enum values into the {cone, sphere, axis-aligned box} kernel kinds. */
const shapeKind = (t: ShapeType | undefined): 0 | 1 | 2 => {
  switch (t) {
    case 'SPHERE': return 1;
    case 'CONE':
    default: return 0;
  }
};

export function createComputePipeline(
  maxParticles: number,
  instanced: boolean,
  normalizedConfig: NormalizedParticleSystemConfig,
  particleSystemId: number,
  forceFieldCount: number,
  collisionPlaneCount = 0
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

  const [lifeMin, lifeMax] = pair(normalizedConfig.startLifetime);
  const [spdMin, spdMax] = pair(normalizedConfig.startSpeed);
  const [szMin,  szMax]  = pair(normalizedConfig.startSize);
  const [rotMin, rotMax] = pair(normalizedConfig.startRotation);
  const [opMin,  opMax]  = pair(normalizedConfig.startOpacity);
  const cMin = (normalizedConfig.startColor as { min?: { r: number; g: number; b: number } }).min || { r: 1, g: 1, b: 1 };
  const cMax = (normalizedConfig.startColor as { max?: { r: number; g: number; b: number } }).max || { r: 1, g: 1, b: 1 };

  const sf = (normalizedConfig.textureSheetAnimation && (normalizedConfig.textureSheetAnimation as { startFrame?: unknown }).startFrame) || 0;
  const sfPair = pair(sf);

  const shp = normalizedConfig.shape;
  const shapeParams: ShapeEmitParams = {
    shapeKind: shapeKind(shp.shapeType),
    radius: shp.radius ?? 1,
    length: shp.length ?? 0,
    arc: shp.arc ?? 360,
    spreadX: (shp as unknown as Record<string, number>).spreadX ?? 0,
    spreadY: (shp as unknown as Record<string, number>).spreadY ?? 0,
    spreadZ: (shp as unknown as Record<string, number>).spreadZ ?? 0,
    speedMin: spdMin, speedMax: spdMax,
    sizeMin: szMin, sizeMax: szMax,
    rotMin: rotMin, rotMax: rotMax,
    opacityMin: opMin, opacityMax: opMax,
    lifeMin: lifeMin, lifeMax: lifeMax,
    colorRMin: cMin.r, colorRMax: cMax.r,
    colorGMin: cMin.g, colorGMax: cMax.g,
    colorBMin: cMin.b, colorBMax: cMax.b,
    startFrameMin: sfPair[0], startFrameMax: sfPair[1],
    rotationCurveActive: normalizedConfig.rotationOverLifetime.isActive,
    rotationalXCurve: (bakedCurves as unknown as Record<string, number>).orbitalVelX ?? -1,
    rotationalYCurve: (bakedCurves as unknown as Record<string, number>).orbitalVelY ?? -1,
    rotationalZCurve: (bakedCurves as unknown as Record<string, number>).orbitalVelZ ?? -1,
    linearXCurve: (bakedCurves as unknown as Record<string, number>).linearVelX ?? -1,
    linearYCurve: (bakedCurves as unknown as Record<string, number>).linearVelY ?? -1,
    linearZCurve: (bakedCurves as unknown as Record<string, number>).linearVelZ ?? -1,
  };

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
    collisionPlaneCount
  );
}
