/**
 * Electric Arc subsystem entry (CPU-capable code only; the WebGPU
 * implementation is injected via `@cyberluke/three-particles/webgpu`).
 *
 * @module
 */
export { createElectricArc } from './electric-arc.js';
export {
  ELECTRIC_ARC_BASE,
  ELECTRIC_ARC_PRESET_CINEMATIC,
  ELECTRIC_ARC_TIERS,
  ELECTRIC_ARC_TIER_SEGMENTS,
} from './electric-arc-defaults.js';
export {
  ORGANIC_HOLD_SPAN,
  ORGANIC_HOLD_START,
  PULSE_DECAY,
  PULSE_DEG_GAIN,
  PULSE_DEG_WIDTH,
  PULSE_MAX_SLOTS,
  PULSE_RISE,
  PULSE_THIN_MIN,
  chaosAmplitude,
  chaosFlickerHz,
  coarseOffset,
  dischargeHash,
  globalFlicker,
  organicOffset,
  pulseEnvelope,
  pulseOffset,
  rotateZ2,
  widthFactor,
} from './electric-arc-math.js';
export {
  colorToNumber,
  mergeLiveConfig,
  nextElectricArcSeed,
  normalizeElectricArcConfig,
  touchesStructuralField,
} from './electric-arc-config.js';
export * from './electric-arc-types.js';
export {
  getElectricArcGPUFactory,
  getElectricArcGPURenderer,
  registerElectricArcGPUFactory,
} from './electric-arc-gpu-registry.js';
