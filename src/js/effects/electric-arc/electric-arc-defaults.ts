/**
 * Defaults + quality tiers for the Electric Arc subsystem.
 *
 * Quality-critical constants are exported here (not hidden in shader
 * source) so applications can tune the effect explicitly.
 *
 * @module
 */
import type {
  ElectricArcConfig,
  ElectricArcQuality,
} from './electric-arc-types.js';

/** Centerline samples per tier (§43). */
export const ELECTRIC_ARC_TIER_SEGMENTS: Record<ElectricArcQuality, number> = {
  low: 32,
  medium: 64,
  high: 96,
  cinematic: 128,
};

/** Tier feature flags (§43). */
export const ELECTRIC_ARC_TIERS: Record<
  ElectricArcQuality,
  {
    segments: number;
    contacts: boolean;
    lighting: boolean;
    sparks: number; // 0 = off, else sparks rate
    branches: boolean;
  }
> = {
  low: {
    segments: 32,
    contacts: true,
    lighting: false,
    sparks: 0,
    branches: false,
  },
  medium: {
    segments: 64,
    contacts: true,
    lighting: true,
    sparks: 4,
    branches: false,
  },
  high: {
    segments: 96,
    contacts: true,
    lighting: true,
    sparks: 6,
    branches: true,
  },
  cinematic: {
    segments: 128,
    contacts: true,
    lighting: true,
    sparks: 7,
    branches: true,
  },
};

/** Screenshot-quality preset used by the examples application (§42). */
export const ELECTRIC_ARC_PRESET_CINEMATIC: Readonly<
  Partial<ElectricArcConfig>
> = {
  color: '#baff63',
  coreColor: '#fffde0',

  thickness: 0.04,
  chaos: 0.19,
  speed: 1,

  segments: 128,

  intensity: 12,
  flickerHz: 24,

  endpointPinning: 0.72,

  glow: {
    enabled: true,
    width: 7.5,
    intensity: 1.4,
    profile: 'gaussian',
  },

  contact: {
    enabled: true,
    radius: 0.075,
    intensity: 15,
  },

  lighting: {
    enabled: true,
    endpointIntensity: 30,
    midpointIntensity: 14,
    distance: 1.6,
    decay: 2,
  },
};

/** Scalar defaults + quality tiers (§§ 1–4, 42, 43). */
export const ELECTRIC_ARC_BASE = {
  color: 0xbaff63,
  coreColor: 0xfffde0,
  thickness: 0.04,
  chaos: 0.35,
  speed: 1.0,
  segments: 96,
  flickerHzDefault: lerpChaosToFlicker(0.35),
  intensity: 10,
  endpointPinning: 0.72,
  glow: { enabled: true, width: 6, intensity: 1.2, profile: 'gaussian' },
  contact: { enabled: true, radius: 0.07, intensity: 15 },
  lighting: {
    enabled: false,
    endpointIntensity: 1,
    midpointIntensity: 0.45,
    distance: 3,
    decay: 2,
  },
  sparks: {
    enabled: false,
    rate: 6,
    lifetime: [0.08, 0.25] as [number, number],
    speed: [0.6, 2.8] as [number, number],
    size: [0.05, 0.3] as [number, number],
  },
  branches: {
    enabled: false,
    maxCount: 3,
    probability: 0.15,
    length: [0.08, 0.28] as [number, number],
    thicknessScale: [0.18, 0.42] as [number, number],
  },
};

function lerpChaosToFlicker(c: number): number {
  return 8 + (42 - 8) * Math.min(1, Math.max(0, c));
}
