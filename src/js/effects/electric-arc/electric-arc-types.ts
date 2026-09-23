/**
 * Electric Arc subsystem — public types.
 *
 * A 3D procedural electrical discharge (plasma filament) rendered as a
 * camera-facing ribbon through a deterministically generated centerline.
 *
 * Backend semantics (identical to {@link SimulationBackend}):
 *
 * ```text
 * AUTO  -> GPU compute where supported, CPU otherwise.
 * CPU   -> always CPU simulation.
 * GPU   -> requests GPU compute, falls back to CPU when compute is unavailable.
 * ```
 *
 * @module
 */
import type { SimulationBackend } from '../three-particles/three-particles-enums.js';
import type { CycleData, Point3D } from '../three-particles/types.js';
import type * as THREE from 'three';

/** Runtime quality tier (§43). Explicit user values override tier values. */
export type ElectricArcQuality = 'low' | 'medium' | 'high' | 'cinematic';

/**
 * Chaos model selection (§11):
 *   - `linear`:  classical piecewise-linear lattice kinks (default);
 *   - `pulse`:   non-linear asymmetric slot model — non-uniform slot holds,
 *                hard gaps (~22% of slots), fast-attack / slow-decay spike
 *                envelope ("zap zap bzzz zap zap");
 *   - `organic`: smooth cosine-eased lattice kinks with random pauses
 *                (organic breathing line with occasional flat rests).
 */
export type ChaosAlgorithm = 'linear' | 'pulse' | 'organic';

/**
 * Cross-section shape of the filament ribbon.
 *   - `gaussian`: smooth exp falloff (classic plasma look);
 *   - `triangle`: sharp linear tent — crisp knife-edge core, no soft skirt.
 */
export type ArcProfileKind = 'gaussian' | 'triangle';

/** Large soft corona around the plasma filament. */
export type ElectricArcGlowConfig = {
  enabled?: boolean;
  /** Half-span of the outer halo layer, as a multiple of the core. @default 6 */
  width?: number;
  /** Halo intensity multiplier. @default 1.2 */
  intensity?: number;
  /** Cross-section shape. @default 'gaussian' */
  profile?: ArcProfileKind;
};

/** Hot endpoint contact plasma. */
export type ElectricArcContactConfig = {
  enabled?: boolean;
  /** Contact billboard radius in world units. @default 0.07 */
  radius?: number;
  /** HDR contact intensity. @default 15 */
  intensity?: number;
};

/** Local point-light illumination owned by the arc. */
export type ElectricArcLightingConfig = {
  /** @default false (cinematic preset enables). */
  enabled?: boolean;
  /** Intensity of the two endpoint lights. @default 1 (multiplied by `intensity`) */
  endpointIntensity?: number;
  /** Intensity of the midpoint light. @default 0.45 */
  midpointIntensity?: number;
  /** PointLight distance. @default 3 */
  distance?: number;
  /** PointLight decay. @default 2 */
  decay?: number;
};

/** Tiny sparks emitted through the existing particle engine. */
export type ElectricArcSparksConfig = {
  enabled?: boolean;
  /** Sparks per second per emitter. @default 6 */
  rate?: number;
  lifetime?: [number, number];
  speed?: [number, number];
  size?: [number, number];
};

/** Optional branch discharges (batched). */
export type ElectricArcBranchesConfig = {
  /** @default false */
  enabled?: boolean;
  /** @default 3 */
  maxCount?: number;
  /** @default 0.04 */
  probability?: number;
  /** Fraction of main length. @default [0.08, 0.28] */
  length?: [number, number];
  /** Multiplier of the main thickness. @default [0.18, 0.42] */
  thicknessScale?: [number, number];
};

/**
 * Public per-endpoint Euler rotation, in DEGREES, applied to the endpoint
 * frame of the arc.
 *
 * Frame convention (per endpoint): `normal`/`binormal`/`tangent` are the
 * endpoint frame axes — for `Object3D` bindings these are the object's own
 * world-space axes (tangent = local +Z, normal = local +X, binormal =
 * local +Y); for plain points the deterministic chord frame is used.
 * `pitch` rotates the endpoint offset around `normal`, `yaw` around
 * `binormal`, `roll` around `tangent`. With a zero offset the rotation
 * leaves the anchor unchanged (documented, observable only via the frame
 * read-back). These are LOCAL endpoint-frame adjustments; the global
 * whole-arc roll remains `rotationZ`.
 *
 * Legacy input `{x, y, z}` is accepted at the input boundary and normalized
 * 1:1 to `pitch = x`, `yaw = y`, `roll = z` (see `toEuler`).
 */
export type EndpointRotation = {
  /** Elevation around the frame `normal` axis, degrees. @default 0 */
  pitch?: number;
  /** Azimuth around the frame `binormal` axis, degrees. @default 0 */
  yaw?: number;
  /** Roll around the `tangent` axis, degrees. @default 0 */
  roll?: number;
};

/**
 * Full configuration accepted by `createElectricArc`.
 *
 * All fields are optional; defaults come from `electric-arc-defaults.ts`
 * (see `ELECTRIC_ARC_PRESET_CINEMATIC` for the screenshot-quality preset).
 */
export type ElectricArcConfig = {
  start: Point3D | THREE.Vector3;
  end: Point3D | THREE.Vector3;

  /**
   * Backend preference. AUTO uses GPU compute when a registered
   * ElectricArc GPU factory is active, otherwise CPU.
   */
  simulationBackend?: SimulationBackend;

  color?: THREE.ColorRepresentation;
  coreColor?: THREE.ColorRepresentation;

  /** Full visible core diameter in world units. @default 0.04 */
  thickness?: number;

  /** 0..1 macro control over spatial irregularity. @default 0.35 */
  chaos?: number;

  /** Chaos model. @default 'linear' */
  chaosAlgorithm?: ChaosAlgorithm;

  /** Overall temporal speed multiplier. @default 1 */
  speed?: number;

  /** Number of centerline samples. @default 96 */
  segments?: number;

  seed?: number;

  /** Topology replacements per second. @default derived from `chaos` */
  flickerHz?: number;

  /** HDR core intensity. @default 10 */
  intensity?: number;

  /** Power of the sin(pi*t) endpoint-pinning envelope. @default 0.72 */
  endpointPinning?: number;

  /**
   * Rotation of the whole arc around the Z axis, in degrees (front-vs-side
   * view difference). @default 0
   *
   * Legacy semantics preserved: applied as the FINAL step, rotating the
   * effective endpoint positions around the world Z axis
   * (`rotateZ2(scratchStart), rotateZ2(scratchEnd)`), exactly like the
   * pre-4.1.2 implementation. Composition order:
   *
   * ```text
   * 1. resolve base anchors (config start/end, or live binding)
   * 2. resolve the per-endpoint frame (object axes / chord frame)
   * 3. rotate the endpoint-local offset by startRotation/endRotation
   * 4. effective = base + frame * rotatedOffset
   * 5. rotationZ rotates both effective endpoints around world Z
   * ```
   */
  rotationZ?: number;

  /**
   * Persistent user offset of the START endpoint (since 4.1.2). Expressed in
   * the ENDPOINT_FRAME_LOCAL basis: `x` along the frame normal, `y` along the
   * binormal, `z` along the tangent. For `Object3D` bindings this is the
   * object's own frame; otherwise the deterministic chord frame
   * (tangent = normalized end - start).
   * @default { x: 0, y: 0, z: 0 }
   */
  startOffset?: Point3D | THREE.Vector3;
  /** Persistent user offset of the END endpoint. @default { x: 0, y: 0, z: 0 } */
  endOffset?: Point3D | THREE.Vector3;
  /**
   * Persistent Euler adjustment of the START endpoint frame, in degrees.
   * Applied to the offset vector within the endpoint frame before it is
   * mapped to world axes. Consumed by `mergeLiveConfig` (since 4.1.2).
   */
  startRotation?: EndpointRotation;
  /** Persistent Euler adjustment of the END endpoint frame, in degrees. */
  endRotation?: EndpointRotation;

  quality?: ElectricArcQuality;

  glow?: ElectricArcGlowConfig;
  contact?: ElectricArcContactConfig;
  lighting?: ElectricArcLightingConfig;
  sparks?: ElectricArcSparksConfig;
  branches?: ElectricArcBranchesConfig;
};

/** One endpoint of a bound pair: a scene object or object + local offset. */
export type ElectricArcEndpointRef =
  THREE.Object3D | { object: THREE.Object3D; offset?: THREE.Vector3 };

/** Live endpoint binding. */
export type ElectricArcBinding = {
  start: ElectricArcEndpointRef;
  end: ElectricArcEndpointRef;
};

/** Orthonormal endpoint frame in world space (derived, read-only). */
export type ElectricArcFrame = {
  tangent: Point3D;
  normal: Point3D;
  binormal: Point3D;
  /** Which basis was actually used for the offset composition. */
  frameSource: 'object' | 'chord';
};

/** Explicit endpoint mode of a live arc. */
export type ElectricArcEndpointMode = 'standalone' | 'bound';

/**
 * Runtime-derived readback (§ Phase 10): base ribbon/config anchors,
 * effective endpoints after persistent transforms, and the derived endpoint
 * frames. Read-only diagnostics; never written back into the config.
 */
export type ElectricArcRuntimeEndpoints = {
  /** `bound` while a live binding is active, `standalone` otherwise. */
  mode: ElectricArcEndpointMode;
  /** Host identity of the binding (object name), when available. */
  sourceId?: string;
  baseStart: Point3D;
  baseEnd: Point3D;
  effectiveStart: Point3D;
  effectiveEnd: Point3D;
  startFrame: ElectricArcFrame;
  endFrame: ElectricArcFrame;
};

/** Numeric (fully-resolved) form of a live electric-arc configuration. */
export type NormalizedElectricArcConfig = {
  simulationBackend: SimulationBackend;
  start: THREE.Vector3;
  end: THREE.Vector3;
  color: number;
  coreColor: number;
  thickness: number;
  chaos: number;
  chaosAlgorithm: ChaosAlgorithm;
  speed: number;
  segments: number;
  seed: number;
  flickerHz: number;
  intensity: number;
  endpointPinning: number;
  /** Whole-arc rotation around Z in degrees (applied to endpoints). */
  rotationZ: number;
  /** Persistent START offset (non-optional, defaults to zero). */
  startOffset: THREE.Vector3;
  /** Persistent END offset (non-optional, defaults to zero). */
  endOffset: THREE.Vector3;
  /** Persistent START frame Euler rotation, resolved (roll defaults to 0). */
  startRotation: { pitch: number; yaw: number; roll: number };
  /** Persistent END frame Euler rotation, resolved (roll defaults to 0). */
  endRotation: { pitch: number; yaw: number; roll: number };
  glow: {
    enabled: boolean;
    width: number;
    intensity: number;
    profile: ArcProfileKind;
  };
  contact: { enabled: boolean; radius: number; intensity: number };
  lighting: {
    enabled: boolean;
    endpointIntensity: number;
    midpointIntensity: number;
    distance: number;
    decay: number;
  };
  sparks: {
    enabled: boolean;
    rate: number;
    lifetime: [number, number];
    speed: [number, number];
    size: [number, number];
  };
  branches: {
    enabled: boolean;
    maxCount: number;
    probability: number;
    length: [number, number];
    thicknessScale: [number, number];
  };
  /** Derived from the `chaos` macro (§11). */
  amplitude: number;
  coarseKnots: number;
  microFrequency: number;
  brightnessVariation: number;
  branchProbability: number;
};

/**
 * Backend contract implemented by both the CPU fallback and the WebGPU
 * implementation. `update()` is allocation-free after construction.
 */
export type ElectricArcBackendInstance = {
  /** Root node owned by this backend (nested inside the public Group). */
  root: THREE.Group;
  /** @returns the global discharge flicker for this frame (0.75..1.05). */
  update: (
    cycle: CycleData,
    start: THREE.Vector3,
    end: THREE.Vector3
  ) => number;
  updateLive: (config: Partial<ElectricArcConfig>) => void;
  readonly backend: SimulationBackend.CPU | SimulationBackend.GPU;
  /** null on CPU. */
  computeNode: unknown | unknown[] | null;
  dispose: () => void;
};

/** Factory registered by the `/webgpu` entry for compute-capable paths. */
export type ElectricArcGPUFactory = {
  create: (config: NormalizedElectricArcConfig) => ElectricArcBackendInstance;
};

/**
 * Public ElectricArc handle. Deliberately resembles `ParticleSystem`:
 * `instance` / `update()` / `computeNode` / `dispose()`.
 */
export type ElectricArc = {
  instance: THREE.Group;
  update: (cycle: CycleData) => void;
  updateConfig: (config: Partial<ElectricArcConfig>) => void;
  setEndpoints: (
    start: Point3D | THREE.Vector3,
    end: Point3D | THREE.Vector3
  ) => void;
  bindEndpoints: (binding: ElectricArcBinding) => void;
  clearEndpointBinding: () => void;
  /**
   * Read-only diagnostics: runtime anchors + effective endpoints + derived
   * endpoint frames for the current/last resolved state. The editor must not
   * write these values back; persistent transforms live in the config.
   */
  getRuntimeEndpoints: () => ElectricArcRuntimeEndpoints;
  readonly backend: SimulationBackend.CPU | SimulationBackend.GPU;
  /** null on CPU. */
  computeNode: unknown | unknown[] | null;
  dispose: () => void;
};
