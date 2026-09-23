/**
 * Electric Arc configuration normalization:
 *   - resolves Point3D/Vector3 endpoints,
 *   - resolves the `chaos` macro into concrete shape parameters (§11),
 *   - applies quality tiers (§43) while explicit user values win,
 *   - exposes the structural-field set (§25).
 *
 * @module
 */
import * as THREE from 'three';
import { SimulationBackend } from '../three-particles/three-particles-enums.js';
import { ELECTRIC_ARC_TIERS } from './electric-arc-defaults.js';
import {
  chaosAmplitude,
  chaosFlickerHz,
  clamp,
  lerp,
  mixSeedScalar,
  pcg01Scalar,
} from './electric-arc-math.js';
import type {
  ElectricArcConfig,
  NormalizedElectricArcConfig,
} from './electric-arc-types.js';

/** Structural properties (§25): a change requires an internal rebuild. */
export const ELECTRIC_ARC_STRUCTURAL_FIELDS = [
  'segments',
  'quality',
  'simulationBackend',
  'chaosAlgorithm',
] as const;

const toVec3 = (
  p: { x?: number; y?: number; z?: number } | THREE.Vector3 | undefined,
  out: THREE.Vector3
): THREE.Vector3 => {
  if (!p) return out.set(0, 0, 0);
  return out.set(p.x ?? 0, p.y ?? 0, p.z ?? 0);
};

const isPlainObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null;

/** Resolve a persistent endpoint Euler rotation with finite-number defaults. */
const toEuler = (
  r: { pitch?: number; yaw?: number; roll?: number } | undefined
): { pitch: number; yaw: number; roll: number } => ({
  pitch: Number.isFinite(r?.pitch) ? Number(r?.pitch) : 0,
  yaw: Number.isFinite(r?.yaw) ? Number(r?.yaw) : 0,
  roll: Number.isFinite(r?.roll) ? Number(r?.roll) : 0,
});

/**
 * In-place per-component merge of a persistent offset: finite components win,
 * unspecified/NaN components keep the previous value (partial live patches).
 */
const mergeVec3 = (
  p: { x?: number; y?: number; z?: number } | THREE.Vector3 | undefined,
  out: THREE.Vector3
): void => {
  if (!p) return;
  const q = p as { x?: number; y?: number; z?: number };
  if (Number.isFinite(q.x)) out.x = Number(q.x);
  if (Number.isFinite(q.y)) out.y = Number(q.y);
  if (Number.isFinite(q.z)) out.z = Number(q.z);
};

/** Generate a stable 24-bit seed matching the seed machinery style (§10). */
export const nextElectricArcSeed = (): number =>
  pcg01Scalar(
    mixSeedScalar(Date.now(), Math.floor(Math.random() * 0x1000000) || 1, 3)
  ) * 0x00ffffff;

export function normalizeElectricArcConfig(
  config: ElectricArcConfig
): NormalizedElectricArcConfig {
  const tier = config.quality ? ELECTRIC_ARC_TIERS[config.quality] : null;

  const segments = Math.round(config.segments ?? (tier ? tier.segments : 96));

  const chaosity = clamp(config.chaos ?? 0.35, 0, 1);

  const start = toVec3(config.start, new THREE.Vector3());
  const end = toVec3(config.end, new THREE.Vector3());
  const distance = Math.max(end.distanceTo(start), 1e-4);

  // Persistent user transforms (§ Phase 2–4): non-optional in the normalized
  // form, zero-defaulted so zero-transform behavior matches the legacy engine.
  const startOffset = toVec3(config.startOffset, new THREE.Vector3());
  const endOffset = toVec3(config.endOffset, new THREE.Vector3());
  const startRotation = toEuler(config.startRotation);
  const endRotation = toEuler(config.endRotation);

  const color = new THREE.Color(config.color ?? 0xbaff63);
  const coreColor = new THREE.Color(config.coreColor ?? 0xfffde0);

  const seed =
    config.seed !== undefined && Number.isFinite(config.seed)
      ? Math.trunc(config.seed) >>> 0
      : nextElectricArcSeed() >>> 0;

  const sparksLifetime =
    Array.isArray(config.sparks?.lifetime) &&
    config.sparks.lifetime.length === 2
      ? (config.sparks.lifetime as [number, number])
      : ([0.08, 0.25] as [number, number]);
  const sparksSpeed =
    Array.isArray(config.sparks?.speed) && config.sparks.speed.length === 2
      ? (config.sparks.speed as [number, number])
      : ([0.6, 2.8] as [number, number]);
  const sparksSize =
    Array.isArray(config.sparks?.size) && config.sparks.size.length === 2
      ? (config.sparks.size as [number, number])
      : ([0.05, 0.3] as [number, number]);

  const branchLength =
    Array.isArray(config.branches?.length) &&
    config.branches.length!.length === 2
      ? (config.branches.length as [number, number])
      : ([0.08, 0.28] as [number, number]);
  const branchThickness =
    Array.isArray(config.branches?.thicknessScale) &&
    config.branches.thicknessScale.length === 2
      ? (config.branches.thicknessScale as [number, number])
      : ([0.18, 0.42] as [number, number]);

  const glowWidth = Math.max(1, config.glow?.width ?? 6);

  const normalized: NormalizedElectricArcConfig = {
    simulationBackend: config.simulationBackend ?? SimulationBackend.AUTO,
    start,
    end,
    color: colorToNumber(color),
    coreColor: colorToNumber(coreColor),
    thickness: config.thickness ?? 0.04,
    chaos: chaosity,
    chaosAlgorithm:
      config.chaosAlgorithm === 'pulse'
        ? 'pulse'
        : config.chaosAlgorithm === 'organic'
          ? 'organic'
          : 'linear',
    speed: config.speed ?? 1.0,
    segments: clamp(segments, 8, 512),
    seed,
    flickerHz: config.flickerHz ?? chaosFlickerHz(chaosity),
    intensity: config.intensity ?? 10,
    endpointPinning: clamp(config.endpointPinning ?? 0.72, 0, 4),
    rotationZ: Number.isFinite(config.rotationZ) ? Number(config.rotationZ) : 0,
    startOffset,
    endOffset,
    startRotation,
    endRotation,
    glow: {
      enabled: config.glow?.enabled ?? true,
      width: glowWidth,
      intensity: config.glow?.intensity ?? 1.2,
      profile: config.glow?.profile === 'triangle' ? 'triangle' : 'gaussian',
    },
    contact: {
      enabled: config.contact?.enabled ?? true,
      radius: config.contact?.radius ?? 0.07,
      intensity: config.contact?.intensity ?? 15,
    },
    lighting: {
      enabled: isPlainObj(config.lighting) ? !!config.lighting.enabled : false,
      endpointIntensity: config.lighting?.endpointIntensity ?? 1,
      midpointIntensity: config.lighting?.midpointIntensity ?? 0.45,
      distance: config.lighting?.distance ?? 3,
      decay: config.lighting?.decay ?? 2,
    },
    sparks: {
      enabled: isPlainObj(config.sparks)
        ? (config.sparks.enabled ?? (!!tier && tier.sparks > 0))
        : !!tier && tier.sparks > 0,
      rate: config.sparks?.rate ?? (tier && tier.sparks > 0 ? tier.sparks : 6),
      lifetime: sparksLifetime,
      speed: sparksSpeed,
      size: sparksSize,
    },
    branches: {
      enabled: isPlainObj(config.branches)
        ? (config.branches.enabled ?? (!!tier && tier.branches))
        : !!tier && tier.branches,
      maxCount: Math.round(config.branches?.maxCount ?? 3),
      probability: config.branches?.probability ?? 0.15,
      length: branchLength,
      thicknessScale: branchThickness,
    },
    amplitude: chaosAmplitude(distance, chaosity),
    coarseKnots: Math.round(lerp(4, 20, chaosity)),
    microFrequency: lerp(15, 75, chaosity),
    brightnessVariation: lerp(0.05, 0.38, chaosity),
    branchProbability: Math.pow(chaosity, 2) * 0.32,
  };

  if (!normalized.branches.enabled) normalized.branches.probability = 0;
  // chaos-driven branch probability overrides only when it adds branches.
  if (
    normalized.branches.enabled &&
    config.branches?.probability === undefined
  ) {
    normalized.branches.probability =
      normalized.branchProbability || normalized.branches.probability;
  }
  return normalized;
}

/** Linear (non-sRGB-encoded) numeric color for shader uniforms. */
export function colorToNumber(c: THREE.Color): number {
  // pack to 24-bit; components already linear in c.r/g/b
  const r = Math.round(clamp(c.r, 0, 1) * 255);
  const g = Math.round(clamp(c.g, 0, 1) * 255);
  const b = Math.round(clamp(c.b, 0, 1) * 255);
  return (r << 16) | (g << 8) | b;
}

/** Finite-number coercion with fallback (pair arrays from the editor). */
const fin = (v: unknown, fb: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fb;
};

/** True when the partial config touches a structural field (§25). */
export function touchesStructuralField(
  config: Partial<ElectricArcConfig>
): boolean {
  return (
    config.segments !== undefined ||
    config.quality !== undefined ||
    config.simulationBackend !== undefined ||
    config.chaosAlgorithm !== undefined
  );
}

/** Merge a Partial config into a normalized config in place (live updates). */
export function mergeLiveConfig(
  target: NormalizedElectricArcConfig,
  patch: Partial<ElectricArcConfig>
): void {
  if (patch.start !== undefined) toVec3(patch.start, target.start);
  if (patch.end !== undefined) toVec3(patch.end, target.end);
  if (patch.color !== undefined)
    target.color = colorToNumber(new THREE.Color(patch.color));
  if (patch.coreColor !== undefined)
    target.coreColor = colorToNumber(new THREE.Color(patch.coreColor));
  if (patch.thickness !== undefined) target.thickness = patch.thickness;
  if (patch.speed !== undefined) target.speed = patch.speed;
  if (patch.flickerHz !== undefined) target.flickerHz = patch.flickerHz;
  if (patch.intensity !== undefined) target.intensity = patch.intensity;
  if (patch.endpointPinning !== undefined)
    target.endpointPinning = clamp(patch.endpointPinning, 0, 4);
  if (patch.rotationZ !== undefined)
    target.rotationZ = Number.isFinite(patch.rotationZ)
      ? Number(patch.rotationZ)
      : target.rotationZ;
  // Persistent transforms merge in place; partial patches keep unspecified
  // components, NaN components fall back to the current value.
  if (patch.startOffset !== undefined)
    mergeVec3(patch.startOffset, target.startOffset);
  if (patch.endOffset !== undefined)
    mergeVec3(patch.endOffset, target.endOffset);
  if (patch.startRotation !== undefined) {
    if (Number.isFinite(patch.startRotation.pitch))
      target.startRotation.pitch = Number(patch.startRotation.pitch);
    if (Number.isFinite(patch.startRotation.yaw))
      target.startRotation.yaw = Number(patch.startRotation.yaw);
    if (Number.isFinite(patch.startRotation.roll))
      target.startRotation.roll = Number(patch.startRotation.roll);
  }
  if (patch.endRotation !== undefined) {
    if (Number.isFinite(patch.endRotation.pitch))
      target.endRotation.pitch = Number(patch.endRotation.pitch);
    if (Number.isFinite(patch.endRotation.yaw))
      target.endRotation.yaw = Number(patch.endRotation.yaw);
    if (Number.isFinite(patch.endRotation.roll))
      target.endRotation.roll = Number(patch.endRotation.roll);
  }
  if (patch.glow?.intensity !== undefined)
    target.glow.intensity = patch.glow.intensity;
  if (patch.glow?.enabled !== undefined)
    target.glow.enabled = patch.glow.enabled;
  if (patch.glow?.width !== undefined)
    target.glow.width = Math.max(1, patch.glow.width);
  if (patch.glow?.profile !== undefined)
    target.glow.profile =
      patch.glow.profile === 'triangle' ? 'triangle' : 'gaussian';
  if (patch.contact?.enabled !== undefined)
    target.contact.enabled = patch.contact.enabled;
  if (patch.contact?.radius !== undefined)
    target.contact.radius = patch.contact.radius;
  if (patch.contact?.intensity !== undefined)
    target.contact.intensity = patch.contact.intensity;
  if (patch.lighting?.enabled !== undefined)
    target.lighting.enabled = patch.lighting.enabled;
  if (patch.lighting?.endpointIntensity !== undefined)
    target.lighting.endpointIntensity = patch.lighting.endpointIntensity;
  if (patch.lighting?.midpointIntensity !== undefined)
    target.lighting.midpointIntensity = patch.lighting.midpointIntensity;
  if (patch.lighting?.distance !== undefined)
    target.lighting.distance = patch.lighting.distance;
  if (patch.lighting?.decay !== undefined)
    target.lighting.decay = patch.lighting.decay;
  if (patch.sparks?.enabled !== undefined)
    target.sparks.enabled = patch.sparks.enabled;
  if (patch.sparks?.rate !== undefined) target.sparks.rate = patch.sparks.rate;
  if (patch.sparks) {
    const sp = patch.sparks;
    if (Array.isArray(sp.lifetime) && sp.lifetime.length === 2)
      target.sparks.lifetime = [
        fin(sp.lifetime[0], target.sparks.lifetime[0]),
        fin(sp.lifetime[1], target.sparks.lifetime[1]),
      ];
    if (Array.isArray(sp.speed) && sp.speed.length === 2)
      target.sparks.speed = [
        fin(sp.speed[0], target.sparks.speed[0]),
        fin(sp.speed[1], target.sparks.speed[1]),
      ];
    if (Array.isArray(sp.size) && sp.size.length === 2)
      target.sparks.size = [
        fin(sp.size[0], target.sparks.size[0]),
        fin(sp.size[1], target.sparks.size[1]),
      ];
  }
  if (patch.chaosAlgorithm !== undefined)
    target.chaosAlgorithm =
      patch.chaosAlgorithm === 'pulse'
        ? 'pulse'
        : patch.chaosAlgorithm === 'organic'
          ? 'organic'
          : 'linear';
  if (patch.simulationBackend !== undefined)
    target.simulationBackend = patch.simulationBackend;

  if (patch.chaos !== undefined && Number.isFinite(patch.chaos)) {
    const c = clamp(patch.chaos, 0, 1);
    target.chaos = c;
    const distance = Math.max(target.end.distanceTo(target.start), 1e-4);
    target.amplitude = chaosAmplitude(distance, c);
    target.coarseKnots = Math.round(lerp(4, 20, c));
    target.microFrequency = lerp(15, 75, c);
    target.brightnessVariation = lerp(0.05, 0.38, c);
    target.branchProbability = Math.pow(c, 2) * 0.32;
    if (patch.flickerHz === undefined) target.flickerHz = chaosFlickerHz(c);
  }
}
