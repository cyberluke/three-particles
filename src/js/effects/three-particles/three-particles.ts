import { ObjectUtils } from '@newkrok/three-utils';
import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import { FBM } from 'three-noise/build/three-noise.module.js';
import { rgbSRGBToLinear, sRGBToLinear } from './color-utils.js';
import InstancedParticleFragmentShader from './shaders/instanced-particle-fragment-shader.glsl.js';
import InstancedParticleVertexShader from './shaders/instanced-particle-vertex-shader.glsl.js';
import MeshParticleFragmentShader from './shaders/mesh-particle-fragment-shader.glsl.js';
import MeshParticleVertexShader from './shaders/mesh-particle-vertex-shader.glsl.js';
import ParticleSystemFragmentShader from './shaders/particle-system-fragment-shader.glsl.js';
import ParticleSystemVertexShader from './shaders/particle-system-vertex-shader.glsl.js';
import TrailFragmentShader from './shaders/trail-fragment-shader.glsl.js';
import TrailVertexShader from './shaders/trail-vertex-shader.glsl.js';
import { removeBezierCurveFunction } from './three-particles-bezier.js';
import { applyCollisionPlanes } from './three-particles-collision.js';
import {
  SCALAR_STRIDE,
  S_IS_ACTIVE,
  S_LIFETIME,
  S_START_LIFETIME,
  S_START_FRAME,
  S_SIZE,
  S_ROTATION,
  S_COLOR_R,
  S_COLOR_G,
  S_COLOR_B,
  S_COLOR_A,
} from './three-particles-constants.js';
import {
  CollisionPlaneMode,
  EmitFrom,
  ForceFieldFalloff,
  ForceFieldType,
  LifeTimeCurve,
  RendererType,
  Shape,
  SimulationBackend,
  SimulationSpace,
  SubEmitterTrigger,
  TimeMode,
} from './three-particles-enums';
import { applyForceFields } from './three-particles-forces.js';
import { applyModifiers } from './three-particles-modifiers.js';
import { isComputeCapableRenderer } from './three-particles-renderer-detect.js';
import {
  calculateRandomPositionAndVelocityOnBox,
  calculateRandomPositionAndVelocityOnCircle,
  calculateRandomPositionAndVelocityOnCone,
  calculateRandomPositionAndVelocityOnRectangle,
  calculateRandomPositionAndVelocityOnSphere,
  calculateValue,
  getCurveFunctionFromConfig,
  isLifeTimeCurve,
  createDefaultMeshTexture,
  createDefaultParticleTexture,
} from './three-particles-utils.js';
import {
  CollisionPlaneConfig,
  Constant,
  CurveFunction,
  CycleData,
  FluidTelemetry,
  ForceFieldConfig,
  GeneralData,
  LifetimeCurve,
  MappedAttributes,
  NormalizedCollisionPlaneConfig,
  NormalizedForceFieldConfig,
  NormalizedParticleSystemConfig,
  ParticleSystem,
  ParticleSystemConfig,
  ParticleSystemInstance,
  Point3D,
  RandomBetweenTwoConstants,
  ShapeConfig,
  SubEmitterConfig,
  MeshConfig,
  TrailConfig,
} from './types.js';
import {
  createFluidSimPipeline,
  type FluidSimPipeline,
  type FluidSolverId,
} from './webgpu/tsl-materials.js';

/**
 * `resolveWebGPUEffectiveRendererType` — canonical mapping between the five
 * requested `rendererType` values and the five effective GPU render paths
 * (native runtime classes in parentheses).
 *
 *   requested POINTS    -> effective POINTS   (billboard quad + `THREE.Points`).
 *     POINTS IS a supported runtime class in this build: the billboard quad
 *     is drawn as a non-instanced `THREE.Points`; the TSL point material uses
 *     `pointUV` (r186 provides it for `PointsNodeMaterial`).
 *   requested INSTANCED -> effective INSTANCED (quad/box + `THREE.Mesh` with
 *     `InstancedBufferGeometry`).
 *   requested TRAIL     -> effective TRAIL    (ribbon strip + `THREE.Mesh`).
 *   requested MESH      -> effective MESH     (mesh/reused geometry +
 *     `THREE.Mesh`).
 *   requested FLUID     -> effective FLUID    (instanced quad + `THREE.Mesh`,
 *     i.e. the same attribute contract as INSTANCED; both ocean solvers
 *     (`renderer.fluid.solver` = `'MLS-MPM' | 'SPH'`) share this path and only
 *     differ in the compute kernels appended after `emit` / `simulate`).
 *
 * A missing / unknown request resolves to POINTS because `POINTS` is the
 * default value of `renderer.rendererType` in the merged default config.
 */
export function resolveWebGPUEffectiveRendererType(
  requested: RendererType | string | undefined
): RendererType {
  switch (requested) {
    case RendererType.INSTANCED:
      return RendererType.INSTANCED;
    case RendererType.TRAIL:
      return RendererType.TRAIL;
    case RendererType.MESH:
      return RendererType.MESH;
    case RendererType.FLUID:
      return RendererType.FLUID;
    case RendererType.POINTS:
    default:
      return RendererType.POINTS;
  }
}

export * from './types.js';

const normalizeTrailCurve = (
  curve: LifetimeCurve | undefined,
  defaultCurve: LifetimeCurve
): LifetimeCurve => {
  if (!curve) return defaultCurve;
  const raw = curve as Record<string, unknown>;
  if (!raw.type && Array.isArray(raw.bezierPoints)) {
    return { type: LifeTimeCurve.BEZIER, ...raw } as LifetimeCurve;
  }
  return curve;
};

// Re-export so downstream consumers can access stride constants from the main module.
export {
  SCALAR_STRIDE,
  S_IS_ACTIVE,
  S_LIFETIME,
  S_START_LIFETIME,
  S_START_FRAME,
  S_SIZE,
  S_ROTATION,
  S_COLOR_R,
  S_COLOR_G,
  S_COLOR_B,
  S_COLOR_A,
} from './three-particles-constants.js';

let _particleSystemId = 0;
let createdParticleSystems: Array<ParticleSystemInstance> = [];

// ????????? GPU Compute Uniform Helpers ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
// Centralise the `as unknown as` casts for setting TSL uniform values.
// The TSL uniform nodes expose `.value` at runtime but their TypeScript
// type (`ShaderNodeObject<Node>`) does not declare it.

/** Sets a TSL float uniform's value. */
const setUniformFloat = (u: unknown, v: number): void => {
  (u as { value: number }).value = v;
};

/** Sets a TSL vec3 uniform's value. */
const setUniformVec3 = (u: unknown, x: number, y: number, z: number): void => {
  (
    u as { value: { set: (x: number, y: number, z: number) => void } }
  ).value.set(x, y, z);
};

// ????????? WebGPU TSL Material Support (opt-in via registerTSLMaterialFactory) ???????????????

type TSLMaterialFactory = {
  createTSLParticleMaterial: (
    rendererType: RendererType,
    sharedUniforms: Record<string, { value: unknown }>,
    rendererConfig: {
      transparent: boolean;
      blending: THREE.Blending;
      depthTest: boolean;
      depthWrite: boolean;
    },
    gpuCompute?: boolean,
    particleGeometry?: THREE.BufferGeometry
  ) => THREE.Material;
  createTSLTrailMaterial: (
    trailUniforms: Record<string, { value: unknown }>,
    rendererConfig: {
      transparent: boolean;
      blending: THREE.Blending;
      depthTest: boolean;
      depthWrite: boolean;
    }
  ) => THREE.Material;
  // GPU compute functions ??? use opaque types to avoid pulling WebGPU/TSL
  // types into the DTS output.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createComputePipeline?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  writeParticleToModifierBuffers?: (...args: any[]) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  deactivateParticleInModifierBuffers?: (...args: any[]) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  flushEmitQueue?: (...args: any[]) => number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  registerCurveDataLength?: (...args: any[]) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  encodeForceFieldsForGPU?: (...args: any[]) => Float32Array;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  encodeCollisionPlanesForGPU?: (...args: any[]) => Float32Array;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createSubEmitterFifoAttribute?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createSubEmitterInitUpdate?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createTrailRibbonUpdate?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  encodeShapeEmitParams?: (...args: any[]) => any;
};

let _tslMaterialFactory: TSLMaterialFactory | null = null;
let _rendererBackendIsGPU = true;
/** One-shot note that the serialized `'CPU'` preference maps to the GPU kernel. */
let _cpuPreferenceWarned = false;
const _cpuPreferencePreferenceWarn = (): void => {
  _cpuPreferenceWarned = true;
  // eslint-disable-next-line no-console
  console.warn(
    "three-particles: simulationBackend 'CPU' maps to the GPU kernel in 4.0.0 (GPU-only build)."
  );
};
const _rendererBackendIsWebGPU = (): boolean => _rendererBackendIsGPU;

/**
 * Registers the TSL (Three Shading Language) material factory for WebGPU support.
 *
 * Call this **once** before creating any particle systems that use WebGPU rendering.
 * The factory functions are imported from the `@cyberluke/three-particles/webgpu` sub-module.
 *
 * When registered, all particle systems will use TSL-based `NodeMaterial` (compiles to WGSL)
 * instead of GLSL `ShaderMaterial`. If the factory also includes the GPU compute functions
 * (`createComputePipeline`, `writeParticleToModifierBuffers`, etc.), particle systems with
 * `simulationBackend: 'AUTO'` or `'GPU'` will run physics and modifiers on the GPU.
 *
 * @param factory - Object containing TSL material creators and optional GPU compute helpers.
 *
 * @example
 * ```typescript
 * import { registerTSLMaterialFactory } from '@cyberluke/three-particles';
 * import {
 *   createTSLParticleMaterial,
 *   createTSLTrailMaterial,
 *   createComputePipeline,
 *   writeParticleToModifierBuffers,
 *   deactivateParticleInModifierBuffers,
 *   flushEmitQueue,
 *   registerCurveDataLength,
 *   encodeForceFieldsForGPU,
 * } from '@cyberluke/three-particles/webgpu';
 *
 * registerTSLMaterialFactory({
 *   createTSLParticleMaterial,
 *   createTSLTrailMaterial,
 *   createComputePipeline,
 *   writeParticleToModifierBuffers,
 *   deactivateParticleInModifierBuffers,
 *   flushEmitQueue,
 *   registerCurveDataLength,
 *   encodeForceFieldsForGPU,
 * });
 * ```
 */
export const registerTSLMaterialFactory = (
  factory: TSLMaterialFactory,
  options?: { renderer?: unknown }
): boolean => {
  // When a renderer is provided, verify it can actually run TSL materials +
  // compute dispatches. Registering the factory alongside a plain
  // WebGLRenderer would produce NodeMaterials and a compute pipeline that
  // can never be dispatched.
  if (
    options &&
    'renderer' in options &&
    !isComputeCapableRenderer(options.renderer)
  ) {
    // eslint-disable-next-line no-console
    console.warn(
      'three-particles: registerTSLMaterialFactory skipped ??? the provided ' +
        'renderer does not support compute dispatches (expected ' +
        'THREE.WebGPURenderer). Particle systems will use the CPU/GLSL path.'
    );
    return false;
  }
  _tslMaterialFactory = factory;
  if (options && 'renderer' in options) {
    _rendererBackendIsGPU = !!(
      options.renderer as { backend?: { isWebGPUBackend?: boolean } }
    )?.backend?.isWebGPUBackend;
  } else {
    _rendererBackendIsGPU = true;
  }
  return true;
};

// Pre-allocated objects for updateParticleSystemInstance to avoid GC pressure
const _subEmitterPosition = new THREE.Vector3();
const _subLocalPosition = new THREE.Vector3();
const _shadowOrbitalEuler = new THREE.Euler(0, 0, 0, 'XYZ');
const _lastWorldPositionSnapshot = new THREE.Vector3();
// Force field local-space conversion helpers (reused across frames)
const _localForceFieldPos = new THREE.Vector3();
const _localForceFieldDir = new THREE.Vector3();
const _inverseQuat = new THREE.Quaternion();

// ─── §10 TSL-uniform boundary normalization (single source of truth) ──────
// The TSL material factory is the only real material path in the v4 GPU-only
// build, so every value crossing into it must already be exact. These module
// helpers implement the canonical shapes; the first mismatch throws one named
// error and construction stops immediately (no empty / vague messages).

/** Throw the first fatal, named (never empty) normalization error. */
export const assertNamed = (cond: unknown, message: string): void => {
  if (!cond) {
    throw new Error(`three-particles: ${message}`);
  }
};

/**
 * Canonical `Vector2` input: `THREE.Vector2 | [x,y] | [u,v] | {x,y} | {u,v}`
 * (or undefined/null => fallback). Anything else throws a labeled error.
 */
export const normalizeVector2Value = (
  raw: unknown,
  fallback: [number, number],
  label: string
): THREE.Vector2 => {
  if (raw === undefined || raw === null) {
    return new THREE.Vector2(fallback[0], fallback[1]);
  }
  if (raw instanceof THREE.Vector2) return raw;
  let n1: number | undefined;
  let n2: number | undefined;
  if (Array.isArray(raw)) {
    n1 = Number((raw as number[])[0]);
    n2 = Number((raw as number[])[1]);
  } else if (typeof raw === 'object') {
    const o = raw as { x?: number; y?: number; u?: number; v?: number };
    n1 =
      o.x !== undefined
        ? Number(o.x)
        : o.u !== undefined
          ? Number(o.u)
          : undefined;
    n2 =
      o.y !== undefined
        ? Number(o.y)
        : o.v !== undefined
          ? Number(o.v)
          : undefined;
  }
  assertNamed(
    n1 !== undefined &&
      n2 !== undefined &&
      Number.isFinite(n1) &&
      Number.isFinite(n2),
    `${label} must be one of: Vector2, [x,y], [u,v], {x,y} or {u,v}`
  );
  return new THREE.Vector2(n1 as number, n2 as number);
};

/** Canonical map slot: `null` (no map -> white dummy in the material) or a
 *  texture object with `.image`. Anything else throws a labeled error. */
export const normalizeTextureValue = (
  raw: unknown,
  label: string
): THREE.Texture | null => {
  if (raw === undefined || raw === null) return null;
  assertNamed(
    typeof raw === 'object' && 'image' in (raw as object),
    `${label} must be null or a texture object with .image (got ${String(raw)})`
  );
  return raw as THREE.Texture;
};

/** `null` (absent) or a texture object with `.image` — nothing else. */
export const normalizeDepthTextureValue = (
  raw: unknown,
  label: string
): THREE.Texture | null => {
  if (raw === undefined || raw === null) return null;
  assertNamed(
    typeof raw === 'object' && 'image' in (raw as object),
    `${label} must be a texture object with .image when set (got ${String(raw)})`
  );
  return raw as THREE.Texture;
};

/**
 * Background color from serialized data: `{r,g,b,a?}` object, `0x` number,
 * `#rgb`/`#rrggbb` string, or `[r,g,b]` array -> `Vector3`.
 */
export const normalizeBackgroundToVector3 = (
  raw: unknown,
  label: string
): THREE.Vector3 => {
  if (raw === undefined || raw === null) return new THREE.Vector3(1, 1, 1);
  if (typeof raw === 'number') {
    const c = new THREE.Color(raw);
    return new THREE.Vector3(c.r, c.g, c.b);
  }
  if (typeof raw === 'string') {
    const s = raw.trim();
    const c = new THREE.Color(s.startsWith('#') ? s : `#${s}`);
    assertNamed(
      Number.isFinite(c.r) && Number.isFinite(c.g) && Number.isFinite(c.b),
      `${label} is not a valid hex color string`
    );
    return new THREE.Vector3(c.r, c.g, c.b);
  }
  if (Array.isArray(raw)) {
    const [r, g, b] = raw as number[];
    assertNamed(
      Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b),
      `${label} array must contain three finite numbers`
    );
    return new THREE.Vector3(r, g, b);
  }
  const o = raw as { r?: number; g?: number; b?: number };
  assertNamed(
    Number.isFinite(Number(o.r)) &&
      Number.isFinite(Number(o.g)) &&
      Number.isFinite(Number(o.b)),
    `${label} object must provide finite r/g/b`
  );
  return new THREE.Vector3(Number(o.r), Number(o.g), Number(o.b));
};

/**
 * Compares two typed-array slices for value equality.
 * Used to gate GPU buffer `needsUpdate` flagging so we only re-upload when
 * the encoded data actually changed. See the force-field / collision-plane
 * upload block in `updateParticleSystemInstance` for the reasoning.
 */
const arraySlicesEqual = (
  a: Float32Array,
  aOffset: number,
  b: Float32Array,
  bOffset: number,
  length: number
): boolean => {
  for (let i = 0; i < length; i++) {
    if (a[aOffset + i] !== b[bOffset + i]) return false;
  }
  return true;
};
let _localForceFields: Array<NormalizedForceFieldConfig> = [];
// Collision plane local-space conversion helpers (reused across frames)
const _localCollisionPlanePos = new THREE.Vector3();
const _localCollisionPlaneNormal = new THREE.Vector3();
let _localCollisionPlanes: Array<NormalizedCollisionPlaneConfig> = [];
// Trail ribbon helpers (reused across frames to avoid allocations)
const _trailDir = new THREE.Vector3();
const _trailPerp = new THREE.Vector3();
const _trailToCam = new THREE.Vector3();
const _distanceStep = { x: 0, y: 0, z: 0 };
const _tempPosition = { x: 0, y: 0, z: 0 };
// Aggregated needsUpdate flags filled by applyModifiers ??? the attribute
// version counter is bumped once per frame instead of once per particle.
const _modifierUpdateFlags = { position: false, quat: false };
const _modifierParams = {
  delta: 0,
  generalData: null as unknown as GeneralData,
  normalizedConfig: null as unknown as NormalizedParticleSystemConfig,
  attributes: null as unknown as MappedAttributes,
  scalarArray: null as unknown as Float32Array,
  particleLifetimePercentage: 0,
  particleIndex: 0,
  updateFlags: _modifierUpdateFlags,
};
// Reusable parameter objects for the per-particle force-field / collision
// hot loops (avoids one or two object allocations per particle per frame).
const _forceFieldParams = {
  particleSystemId: 0,
  forceFields: null as unknown as Array<NormalizedForceFieldConfig>,
  velocity: null as unknown as THREE.Vector3,
  positionArr: null as unknown as THREE.TypedArray,
  positionIndex: 0,
  delta: 0,
  systemLifetimePercentage: 0,
};
const _collisionParams = {
  collisionPlanes: null as unknown as Array<NormalizedCollisionPlaneConfig>,
  velocity: null as unknown as THREE.Vector3,
  positionArr: null as unknown as THREE.TypedArray,
  positionIndex: 0,
  scalarArr: null as unknown as Float32Array,
  scalarBase: 0,
  deactivateParticle: null as unknown as (particleIndex: number) => void,
  particleIndex: 0,
};
// Scratch vector for onBeforeRender viewport queries (avoids a Vector2
// allocation every rendered frame).
const _viewportSize = new THREE.Vector2();
// Timestamp of the frame currently being processed by
// updateParticleSystemInstance ??? read by the per-system killParticle
// callbacks so they don't need a per-frame closure.
let _frameNow = 0;

/**
 * Converts a plain {x, y, z} object to a THREE.Vector3, using the fallback if undefined.
 */
const toVector3 = (
  v: { x?: number; y?: number; z?: number } | undefined,
  fallback: THREE.Vector3
): THREE.Vector3 =>
  v ? new THREE.Vector3(v.x ?? 0, v.y ?? 0, v.z ?? 0) : fallback.clone();

/**
 * Normalizes raw force field configs into the internal representation with THREE.Vector3 fields.
 */
const normalizeForceFields = (
  rawForceFields: Array<ForceFieldConfig> | undefined
): Array<NormalizedForceFieldConfig> =>
  (rawForceFields ?? []).map((ff: ForceFieldConfig) => ({
    isActive: ff.isActive ?? true,
    type: ff.type ?? ForceFieldType.POINT,
    position: toVector3(ff.position, new THREE.Vector3(0, 0, 0)),
    direction: toVector3(ff.direction, new THREE.Vector3(0, 1, 0)).normalize(),
    strength: ff.strength ?? 1,
    range: Math.max(0, ff.range ?? Infinity),
    falloff: ff.falloff ?? ForceFieldFalloff.LINEAR,
  }));

/**
 * Normalizes raw collision plane configs into the internal representation with THREE.Vector3 fields.
 */
const normalizeCollisionPlanes = (
  rawPlanes: Array<CollisionPlaneConfig> | undefined
): Array<NormalizedCollisionPlaneConfig> =>
  (rawPlanes ?? []).map((cp: CollisionPlaneConfig) => ({
    isActive: cp.isActive ?? true,
    position: toVector3(cp.position, new THREE.Vector3(0, 0, 0)),
    normal: toVector3(cp.normal, new THREE.Vector3(0, 1, 0)).normalize(),
    mode: cp.mode ?? CollisionPlaneMode.KILL,
    dampen: Math.max(0, Math.min(1, cp.dampen ?? 0.5)),
    lifetimeLoss: Math.max(0, Math.min(1, cp.lifetimeLoss ?? 0)),
  }));

/**
 * Mapping of blending mode string identifiers to Three.js blending constants.
 *
 * Used for converting serialized particle system configurations (e.g., from JSON)
 * to actual Three.js blending mode constants.
 *
 * @example
 * ```typescript
 * import { blendingMap } from '@cyberluke/three-particles';
 *
 * // Convert string to Three.js constant
 * const blending = blendingMap['THREE.AdditiveBlending'];
 * // blending === THREE.AdditiveBlending
 * ```
 */
export const blendingMap = {
  'THREE.NoBlending': THREE.NoBlending,
  'THREE.NormalBlending': THREE.NormalBlending,
  'THREE.AdditiveBlending': THREE.AdditiveBlending,
  'THREE.SubtractiveBlending': THREE.SubtractiveBlending,
  'THREE.MultiplyBlending': THREE.MultiplyBlending,
};

/**
 * Normalizes a blending value from either a numeric `THREE.Blending` constant
 * or the serialized string form used by the JSON example configs
 * (`"THREE.AdditiveBlending"` / `"AdditiveBlending"`).
 */
const toBlendingConstant = (v: unknown): THREE.Blending => {
  if (typeof v === 'number') return v as THREE.Blending;
  if (typeof v === 'string') {
    const key = v.startsWith('THREE.') ? v : `THREE.${v}`;
    const mapped = (blendingMap as unknown as Record<string, THREE.Blending>)[
      key
    ];
    if (mapped !== undefined) return mapped;
  }
  return THREE.NormalBlending;
};

/**
 * Returns a deep copy of the default particle system configuration.
 *
 * This is useful when you want to start with default settings and modify specific properties
 * without affecting the internal default configuration object.
 *
 * @returns A new object containing all default particle system settings
 *
 * @example
 * ```typescript
 * import { getDefaultParticleSystemConfig, createParticleSystem } from '@cyberluke/three-particles';
 *
 * // Get default config and modify it
 * const config = getDefaultParticleSystemConfig();
 * config.emission.rateOverTime = 100;
 * config.startColor.min = { r: 1, g: 0, b: 0 };
 *
 * const { instance } = createParticleSystem(config);
 * scene.add(instance);
 * ```
 */
export const getDefaultParticleSystemConfig = () =>
  JSON.parse(JSON.stringify(DEFAULT_PARTICLE_SYSTEM_CONFIG));

const DEFAULT_PARTICLE_SYSTEM_CONFIG: ParticleSystemConfig = {
  transform: {
    position: new THREE.Vector3(),
    rotation: new THREE.Vector3(),
    scale: new THREE.Vector3(1, 1, 1),
  },
  duration: 5.0,
  looping: true,
  startDelay: 0,
  startLifetime: 5.0,
  startSpeed: 1.0,
  startSize: 1.0,
  startOpacity: 1.0,
  startRotation: 0.0,
  startColor: {
    min: { r: 1.0, g: 1.0, b: 1.0 },
    max: { r: 1.0, g: 1.0, b: 1.0 },
  },
  gravity: 0.0,
  simulationSpace: SimulationSpace.LOCAL,
  simulationBackend: SimulationBackend.AUTO,
  maxParticles: 100.0,
  emission: {
    rateOverTime: 10.0,
    rateOverDistance: 0.0,
    bursts: [],
  },
  shape: {
    shape: Shape.SPHERE,
    sphere: {
      radius: 1.0,
      radiusThickness: 1.0,
      arc: 360.0,
    },
    cone: {
      angle: 25.0,
      radius: 1.0,
      radiusThickness: 1.0,
      arc: 360.0,
    },
    circle: {
      radius: 1.0,
      radiusThickness: 1.0,
      arc: 360.0,
    },
    rectangle: {
      rotation: { x: 0.0, y: 0.0, z: 0.0 },
      scale: { x: 1.0, y: 1.0 },
    },
    box: {
      scale: { x: 1.0, y: 1.0, z: 1.0 },
      emitFrom: EmitFrom.VOLUME,
    },
  },
  map: undefined,
  renderer: {
    blending: THREE.NormalBlending,
    discardBackgroundColor: false,
    backgroundColorTolerance: 1.0,
    backgroundColor: { r: 1.0, g: 1.0, b: 1.0 },
    transparent: true,
    depthTest: true,
    depthWrite: false,
    softParticles: {
      enabled: false,
      intensity: 1.0,
    },
  },
  velocityOverLifetime: {
    isActive: false,
    linear: {
      x: 0,
      y: 0,
      z: 0,
    },
    orbital: {
      x: 0,
      y: 0,
      z: 0,
    },
  },
  sizeOverLifetime: {
    isActive: false,
    lifetimeCurve: {
      type: LifeTimeCurve.BEZIER,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 1, y: 1, percentage: 1 },
      ],
    },
  },
  colorOverLifetime: {
    isActive: false,
    r: {
      type: LifeTimeCurve.BEZIER,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 },
      ],
    },
    g: {
      type: LifeTimeCurve.BEZIER,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 },
      ],
    },
    b: {
      type: LifeTimeCurve.BEZIER,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 },
      ],
    },
  },
  opacityOverLifetime: {
    isActive: false,
    lifetimeCurve: {
      type: LifeTimeCurve.BEZIER,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 1, y: 1, percentage: 1 },
      ],
    },
  },
  rotationOverLifetime: {
    isActive: false,
    min: 0.0,
    max: 0.0,
  },
  noise: {
    isActive: false,
    useRandomOffset: false,
    strength: 1.0,
    frequency: 0.5,
    octaves: 1,
    positionAmount: 1.0,
    rotationAmount: 0.0,
    sizeAmount: 0.0,
  },
  textureSheetAnimation: {
    tiles: new THREE.Vector2(1.0, 1.0),
    timeMode: TimeMode.LIFETIME,
    fps: 30.0,
    startFrame: 0,
  },
  forceFields: [],
  collisionPlanes: [],
};

const calculatePositionAndVelocity = (
  generalData: GeneralData,
  { shape, sphere, cone, circle, rectangle, box }: ShapeConfig,
  startSpeed: Constant | RandomBetweenTwoConstants | LifetimeCurve,
  position: THREE.Vector3,
  velocity: THREE.Vector3
) => {
  const calculatedStartSpeed = calculateValue(
    generalData.particleSystemId,
    startSpeed,
    generalData.normalizedLifetimePercentage
  );

  switch (shape) {
    case Shape.SPHERE:
      calculateRandomPositionAndVelocityOnSphere(
        position,
        generalData.wrapperQuaternion,
        velocity,
        calculatedStartSpeed,
        sphere as Required<NonNullable<ShapeConfig['sphere']>>
      );
      break;

    case Shape.CONE:
      calculateRandomPositionAndVelocityOnCone(
        position,
        generalData.wrapperQuaternion,
        velocity,
        calculatedStartSpeed,
        cone as Required<NonNullable<ShapeConfig['cone']>>
      );
      break;

    case Shape.CIRCLE:
      calculateRandomPositionAndVelocityOnCircle(
        position,
        generalData.wrapperQuaternion,
        velocity,
        calculatedStartSpeed,
        circle as Required<NonNullable<ShapeConfig['circle']>>
      );
      break;

    case Shape.RECTANGLE:
      calculateRandomPositionAndVelocityOnRectangle(
        position,
        generalData.wrapperQuaternion,
        velocity,
        calculatedStartSpeed,
        rectangle as Required<NonNullable<ShapeConfig['rectangle']>>
      );
      break;

    case Shape.BOX:
      calculateRandomPositionAndVelocityOnBox(
        position,
        generalData.wrapperQuaternion,
        velocity,
        calculatedStartSpeed,
        box as Required<NonNullable<ShapeConfig['box']>>
      );
      break;
  }
};

const destroyParticleSystem = (particleSystem: THREE.Points | THREE.Mesh) => {
  createdParticleSystems = createdParticleSystems.filter(
    ({
      particleSystem: savedParticleSystem,
      trailMesh,
      generalData: { particleSystemId },
    }) => {
      if (savedParticleSystem !== particleSystem) {
        return true;
      }

      removeBezierCurveFunction(particleSystemId);

      // Dispose trail mesh if present
      if (trailMesh) {
        trailMesh.geometry.dispose();
        if (Array.isArray(trailMesh.material))
          trailMesh.material.forEach((m) => m.dispose());
        else trailMesh.material.dispose();
        if (trailMesh.parent) trailMesh.parent.remove(trailMesh);
      }

      savedParticleSystem.geometry.dispose();
      if (Array.isArray(savedParticleSystem.material))
        savedParticleSystem.material.forEach((material) => material.dispose());
      else savedParticleSystem.material.dispose();

      if (savedParticleSystem.parent)
        savedParticleSystem.parent.remove(savedParticleSystem);
      return false;
    }
  );
};

/**
 * Creates a new particle system with the specified configuration.
 *
 * This is the primary function for instantiating particle effects. It handles the complete
 * setup of a particle system including geometry creation, material configuration, shader setup,
 * and initialization of all particle properties.
 *
 * @param config - Configuration object for the particle system. If not provided, uses default settings.
 *                 See {@link ParticleSystemConfig} for all available options.
 * @param externalNow - Optional custom timestamp in milliseconds. If not provided, uses `Date.now()`.
 *                      Useful for synchronized particle systems or testing.
 *
 * @returns A {@link ParticleSystem} object containing:
 *   - `instance`: The THREE.Object3D that should be added to your scene
 *   - `resumeEmitter()`: Function to resume particle emission
 *   - `pauseEmitter()`: Function to pause particle emission
 *   - `dispose()`: Function to clean up resources and remove the particle system
 *
 * @example
 * ```typescript
 * import { createParticleSystem, updateParticleSystems } from '@cyberluke/three-particles';
 *
 * // Create a basic particle system with default settings
 * const { instance, dispose } = createParticleSystem();
 * scene.add(instance);
 *
 * // Create a custom fire effect
 * const fireEffect = createParticleSystem({
 *   duration: 2.0,
 *   looping: true,
 *   startLifetime: { min: 0.5, max: 1.5 },
 *   startSpeed: { min: 2, max: 4 },
 *   startSize: { min: 0.5, max: 1.5 },
 *   startColor: {
 *     min: { r: 1.0, g: 0.3, b: 0.0 },
 *     max: { r: 1.0, g: 0.8, b: 0.0 }
 *   },
 *   emission: { rateOverTime: 50 },
 *   shape: {
 *     shape: Shape.CONE,
 *     cone: { angle: 10, radius: 0.2 }
 *   }
 * });
 * scene.add(fireEffect.instance);
 *
 * // In your animation loop
 * function animate(time) {
 *   updateParticleSystems({ now: time, delta: deltaTime, elapsed: elapsedTime });
 *   renderer.render(scene, camera);
 * }
 *
 * // Clean up when done
 * fireEffect.dispose();
 * ```
 *
 * @see {@link updateParticleSystems} - Required function to call in your animation loop
 * @see {@link ParticleSystemConfig} - Complete configuration options
 */
// ?? GPU-only helper functions (small scalar writers; NO per-particle loops) ??
type _ReusableArrayHelper = {
  src: Array<NormalizedForceFieldConfig>;
  dst: Array<NormalizedForceFieldConfig>;
  _pos: THREE.Vector3;
  _dir: THREE.Vector3;
};

const _uploadFFAndCollisionTails = (
  pipeline: any,
  generalData: GeneralData,
  config: NormalizedParticleSystemConfig,
  scratch: _ReusableArrayHelper
): void => {
  const info = pipeline.forceFieldInfo;
  const cinfo = pipeline.collisionPlaneInfo;
  if (!info && !cinfo) return;
  // Packed read-mostly f32 table (uniform buffer, non-atomic).
  const arr = pipeline.buffers.packedData as Float32Array;
  const cd = pipeline.packedDataNode as {
    addUpdateRange(start: number, count: number): void;
    needsUpdate: boolean;
  };
  if (info && config.forceFields.length > 0) {
    const encoded = _tslMaterialFactory!.encodeForceFieldsForGPU!(
      scratch.src,
      generalData.particleSystemId,
      generalData.normalizedLifetimePercentage
    );
    const off = info.offset;
    let changed = false;
    for (let i = 0; i < encoded.length; i++)
      if (arr[off + i] !== encoded[i]) {
        changed = true;
        break;
      }
    if (changed) {
      arr.set(encoded, off);
      cd.addUpdateRange(off, encoded.length);
      cd.needsUpdate = true;
    }
    (info.countUniform as { value: number }).value = config.forceFields.length;
  }
  if (cinfo && config.collisionPlanes.length > 0) {
    const encoded2 = _tslMaterialFactory!.encodeCollisionPlanesForGPU!(
      config.collisionPlanes as never
    );
    const off2 = cinfo.offset;
    let changed2 = false;
    for (let i = 0; i < encoded2.length; i++)
      if (arr[off2 + i] !== encoded2[i]) {
        changed2 = true;
        break;
      }
    if (changed2) {
      arr.set(encoded2, off2);
      cd.addUpdateRange(off2, encoded2.length);
      cd.needsUpdate = true;
    }
    (cinfo.countUniform as { value: number }).value =
      config.collisionPlanes.length;
  }
};

// After the sim kernel writes the GPU-side storage buffers, we simply flag
// the CPU-side typed arrays so three.js WebGPU backend will upload them
// before the render pass. (The compute kernels themselves run against these
// same GPU buffers; the upload is only needed on the FIRST frame.)
const _markBufferAttributeUploads = (
  buffers: Record<string, THREE.BufferAttribute>
): void => {};

// Default texture fallback (single white 1?1 pixel; matches upstream @newkrok).
let _defaultTexture: THREE.Texture | null = null;
const getDefaultTexture = (): THREE.Texture | null => {
  if (_defaultTexture) return _defaultTexture;
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 1, 1);
  }
  _defaultTexture = new THREE.Texture(canvas as unknown as TexImageSource);
  _defaultTexture.needsUpdate = true;
  return _defaultTexture;
};
/**
 * Prefill the eight per-particle storage buffers for the FLUID solver so the
 * first rendered frame already shows the dambreak lattice. Mirrors the CPU
 * writes of the base emit kernel: `color / particleState / startValues /
 * startColorsExt / orbitalIsActive`, with the two shared pos/vec4 already
 * written by `initMLSMPMDambreak` / `initSPHDambreak`. Slots beyond the
 * solver's `count` stay zeroed (material's `aColor.w > 0` guard makes them
 * invisible).
 */
export const prefillFluidState = (
  buffers: Record<string, unknown>,
  count: number,
  cfg: NormalizedParticleSystemConfig
): void => {
  const pairOf = (v: unknown, fb: number): [number, number] => {
    if (typeof v === 'number' && Number.isFinite(v)) return [v, v];
    if (v && typeof v === 'object') {
      const m = (v as { min?: unknown; max?: unknown }).min;
      const x = (v as { min?: unknown; max?: unknown }).max;
      const mn = typeof m === 'number' && Number.isFinite(m) ? m : fb;
      const mx = typeof x === 'number' && Number.isFinite(x) ? x : fb;
      return [mn, mx];
    }
    return [fb, fb];
  };
  const mid = (p: [number, number]): number => (p[0] + p[1]) * 0.5;
  const lifeMid = mid(pairOf(cfg.startLifetime, 5)) * 1000; // ms
  const sizeMid = mid(pairOf(cfg.startSize, 1));
  const rotMid = mid(pairOf(cfg.startRotation, 0));
  const opMid = mid(pairOf(cfg.startOpacity, 1));
  const cMin = cfg.startColor?.min ?? { r: 1, g: 1, b: 1 };
  const cMax = cfg.startColor?.max ?? { r: 1, g: 1, b: 1 };
  const cr = ((cMin.r ?? 1) + (cMax.r ?? 1)) * 0.5;
  const cg = ((cMin.g ?? 1) + (cMax.g ?? 1)) * 0.5;
  const cb = ((cMin.b ?? 1) + (cMax.b ?? 1)) * 0.5;
  // Some pools (e.g. minimal test stubs) omit the optional extension stacks,
  // so every write is guarded on the array being present.
  const w = (a: unknown): Float32Array | null =>
    (a as { array?: Float32Array } | undefined)?.array ?? null;
  const col = w(buffers.color);
  const ps = w(buffers.particleState);
  const sv = w(buffers.startValues);
  const ex = w(buffers.startColorsExt);
  const oi = w(buffers.orbitalIsActive);
  for (let i = 0; i < count; i++) {
    const b = i * 4;
    if (col) {
      col[b] = cr;
      col[b + 1] = cg;
      col[b + 2] = cb;
      col[b + 3] = opMid;
    }
    if (ps) {
      ps[b] = 0;
      ps[b + 1] = sizeMid;
      ps[b + 2] = rotMid;
      ps[b + 3] = 0;
    }
    if (sv) {
      sv[b] = lifeMid;
      sv[b + 1] = sizeMid;
      sv[b + 2] = opMid;
      sv[b + 3] = cr;
    }
    if (ex) {
      ex[b] = cg;
      ex[b + 1] = cb;
      ex[b + 2] = 0;
      ex[b + 3] = i; // stable per-particle seed (integer 0..2^24-1)
    }
    if (oi) {
      // The solver owns all slots from frame #1 (emitCount is forced to 0).
      oi[b] = 1;
    }
  }
};
/**
 * Writes the spherical-domain and pointer-force uniforms of an FLUID solver
 * from the live `renderer.fluid` block (WaterBall contract). `radius: 0`
 * keeps the classic box domain / disables the pointer force. Scalar-only:
 * no pool reconstruction is required, so the sim keeps running.
 */
export const writeFluidDomainUniforms = (
  uniforms: Record<string, { value: unknown } | undefined>,
  fluid: {
    domain?: {
      kind: 'box' | 'sphere';
      center?: readonly [number, number, number];
      radius?: number;
    };
    pointer?: {
      position: readonly [number, number, number];
      velocity: readonly [number, number, number];
      radius: number;
    };
  } | undefined
): void => {
  if (!fluid) return;
  const setV4 = (slot: { value: unknown } | undefined, v: readonly number[]) => {
    const target = slot?.value as
      | { set?: (x: number, y: number, z: number, w: number) => void }
      | undefined;
    if (!target || typeof target.set !== 'function') return;
    target.set(
      Number(v[0]) || 0,
      Number(v[1]) || 0,
      Number(v[2]) || 0,
      Number(v[3]) || 0
    );
  };
  const domain = fluid.domain;
  setV4(
    uniforms.fluidSphereDomain,
    domain && domain.kind === 'sphere'
      ? [
          domain.center?.[0] ?? 0,
          domain.center?.[1] ?? 0,
          domain.center?.[2] ?? 0,
          domain.radius ?? 0,
        ]
      : [0, 0, 0, 0]
  );
  const ptr = fluid.pointer;
  setV4(
    uniforms.fluidPointerPos,
    ptr ? [ptr.position[0], ptr.position[1], ptr.position[2], ptr.radius] : [0, 0, 0, 0]
  );
  setV4(
    uniforms.fluidPointerVel,
    ptr ? [ptr.velocity[0], ptr.velocity[1], ptr.velocity[2], 0] : [0, 0, 0, 0]
  );
};
/**
 * Create a new particle system (GPU-only). Every per-particle slot lives
 * on the GPU inside the 8 storage buffers that the WebGPU compute kernels
 * read/write. The CPU only:
 *   - merges the incoming config,
 *   - creates the TSL material + pipeline,
 *   - writes ~12 scalar uniforms per frame,
 *   - dispatches [emitNode, simNode] via renderer.compute(...).
 */
export const createParticleSystem = (
  config: ParticleSystemConfig = DEFAULT_PARTICLE_SYSTEM_CONFIG,
  externalNow?: number
): ParticleSystem => {
  const now = externalNow || Date.now();

  // ?? 1. strict GPU-only guards ??
  const useTSL = _tslMaterialFactory !== null;
  if (!useTSL) {
    throw new Error(
      'three-particles: WebGPU TSL material factory not registered. ' +
        'Call enableWebGPU(renderer) immediately after creating a WebGPURenderer. ' +
        '@cyberluke/three-particles 4.0.0 is GPU-only - no CPU fallback path exists.'
    );
  }
  if (!_rendererBackendIsGPU) {
    throw new Error(
      'three-particles: renderer is not a native WebGPU backend. ' +
        'This build has no WebGL2 fallback. Use a new THREE.WebGPURenderer().'
    );
  }
  const factory = _tslMaterialFactory!;
  if (!factory.createComputePipeline) {
    throw new Error(
      'three-particles: active WebGPU renderer does not provide a complete TSL compute pipeline ' +
        '(createComputePipeline missing). No CPU fallback exists; install a WebGPU-capable backend.'
    );
  }

  const maxParticles =
    config.maxParticles || DEFAULT_PARTICLE_SYSTEM_CONFIG.maxParticles!;
  const normalizedConfig = ObjectUtils.deepMerge(
    DEFAULT_PARTICLE_SYSTEM_CONFIG as unknown as NormalizedParticleSystemConfig,
    config,
    { applyToFirstObject: false, skippedProperties: [] }
  ) as NormalizedParticleSystemConfig;

  // The v4 build is GPU-only, so a serialized `'CPU'` preference is accepted as
  // the (identical) GPU path instead of failing.
  if ((normalizedConfig.simulationBackend as string) === 'CPU') {
    if (!_cpuPreferenceWarned) {
      _cpuPreferencePreferenceWarn();
    }
    normalizedConfig.simulationBackend =
      SimulationBackend.GPU as typeof normalizedConfig.simulationBackend;
  }

  const requestedRendererType =
    normalizedConfig.renderer.rendererType || RendererType.POINTS;
  // `resolveWebGPUEffectiveRendererType`: the GPU-only v4 renderer builds 4
  // material/geometry classes. POINTS is NOT a runtime point-sprite path
  // (PointsNodeMaterial + THREE.Points would sample pointUV where WGSL has
  // no `gl_PointCoord`); it is the BILLBOARD quad implementation, while
  // INSTANCED selects `InstancedBufferGeometry` + instanced TSL material.
  const effectiveRendererType = resolveWebGPUEffectiveRendererType(
    requestedRendererType
  );
  const rrType = effectiveRendererType;
  const useInstancing =
    effectiveRendererType === RendererType.INSTANCED ||
    effectiveRendererType === RendererType.MESH ||
    effectiveRendererType === RendererType.FLUID;

  // ?? 1b. Trail history ring (GPU-native, filled by the simulation kernel) ??
  const trailConfig = normalizedConfig.renderer.trail;
  const trailLength = Math.max(2, Math.round(trailConfig?.length ?? 20));
  const trailHistoryAttribute: StorageBufferAttribute | null =
    rrType === RendererType.TRAIL
      ? new StorageBufferAttribute(
          new Float32Array(maxParticles * (trailLength + 1) * 4),
          4
        )
      : null;
  const trailDesc: {
    attribute: StorageBufferAttribute;
    meta: StorageBufferAttribute | null;
    length: number;
    minVertexDistance: number;
    maxTime: number;
  } | null = trailHistoryAttribute
    ? {
        attribute: trailHistoryAttribute,
        meta: null,
        length: trailLength,
        minVertexDistance: trailConfig?.minVertexDistance ?? 0,
        maxTime: (trailConfig?.maxTime ?? 0) * 1000,
      }
    : null;

  // ?? 1c. Sub-emitter event FIFOs (integer counters + plain f32 payloads) ??
  const subEmitterConfigs = normalizedConfig.subEmitters ?? [];
  type FifoEntry = {
    counter: StorageBufferAttribute;
    payload: StorageBufferAttribute;
    trigger: 0 | 1;
    capacity: number;
    windowSize: number;
  };
  const fifos: FifoEntry[] = subEmitterConfigs.map((se) => {
    const capacity = Math.max(1, Math.round(se.maxInstances ?? 32));
    const f = factory.createSubEmitterFifoAttribute!(capacity) as FifoEntry;
    f.trigger = (se.trigger === 'BIRTH' ? 0 : 1) as 0 | 1;
    return f;
  });
  // One shared ping-pong window size per system (single `fifoBase` index).
  const fifoBaseStride = fifos.reduce((m, f) => Math.max(m, f.windowSize), 0);

  const forceFields = normalizeForceFields(normalizedConfig.forceFields);
  const collisionPlanes = normalizeCollisionPlanes(
    normalizedConfig.collisionPlanes
  );

  // ?? 2. GPU compute pipeline (emit + simulation kernels) ??
  const pipeline: NonNullable<ParticleSystemInstance['computePipeline']> =
    factory.createComputePipeline(
      maxParticles,
      useInstancing,
      normalizedConfig,
      _particleSystemId, // pre-increment inside generalData below would be off by 1; use the raw next id
      forceFields.length,
      collisionPlanes.length,
      fifos,
      trailDesc ?? undefined
    );

  // ?? 2a. FLUID solver extension (opt-in SPH, MLS-MPM otherwise).
  //
  // Shares the base pipeline's `position` / `velocity` storage attributes so
  // the FLUID render material's `instanceOffset` / `instanceVelocity` see the
  // integrated result, seeds them with the dambreak lattice, merges the
  // solver's own buffers into the first-frame upload set, and appends its
  // per-substep compute nodes to the dispatch list. Each per-pass budget
  // stays <= 8 (2 shared + up-to-6 solver-owned, per the solver layout).
  // Solver-derived snapshots consumed by the `generalData` block below.
  let fluidHighWater = 0;
  let fluidSolverId: FluidSolverId | null = null;
  let fluidBoxWidthRatioValue = 1;
  let fluidGridCount = 0;
  let fluidSolverPassNames: string[] = [];
  let fluidScreenSpacePasses = 0;
  if (rrType === RendererType.FLUID) {
    type SB =
      | StorageBufferAttribute
      | import('three/webgpu').StorageInstancedBufferAttribute;
    const sharedPos = pipeline.buffers.position as unknown as SB;
    const sharedVel = pipeline.buffers.velocity as unknown as SB;
    const pb = pipeline.buffers as unknown as Record<string, unknown>;
    const solverName = String(
      (
        normalizedConfig.renderer.fluid as unknown as
          { solver?: string } | undefined
      )?.solver ?? ''
    )
      .trim()
      .toUpperCase();
    const isSPHSolver = solverName === 'SPH';
    const solverId: FluidSolverId = isSPHSolver ? 'SPH' : 'MLS-MPM';
    const solverPrefix = isSPHSolver ? 'sph' : 'mlsmpm';
    const solver: FluidSimPipeline = createFluidSimPipeline(
      solverId,
      {
        position: sharedPos as unknown as { array: Float32Array },
        velocity: sharedVel as unknown as { array: Float32Array },
      },
      maxParticles,
      normalizedConfig
    );
    // Merge the solver-owned scratch into the base pool so the one-shot
    // first-frame upload (`_lastUploadStampMap` in the update loop) covers it.
    if (isSPHSolver) {
      pb.fluidForceDensity = solver.buffers.forceDensity;
      pb.fluidSortedPosition = solver.buffers.sortedPosition;
      pb.fluidSortedVelocity = solver.buffers.sortedVelocity;
      pb.fluidSortedForceDensity = solver.buffers.sortedForceDensity;
      pb.fluidCellCounts = solver.buffers.cellCounts;
      pb.fluidPrefixSums = solver.buffers.prefixSums;
      pb.fluidParticleCellOffsets = solver.buffers.particleCellOffsets;
      pb.fluidBlockPartials = solver.buffers.blockPartials;
      pb.fluidBlockInclusive = solver.buffers.blockInclusive;
      pb.fluidBlockOffsets = solver.buffers.blockOffsets;
    } else {
      pb.fluidCoefficients = solver.buffers.coefficients;
      pb.fluidCells = solver.buffers.cells;
    }
    fluidHighWater = solver.numParticles;
    fluidSolverId = solverId;
    fluidGridCount = solver.gridCount;
    fluidBoxWidthRatioValue = isSPHSolver
      ? (normalizedConfig.renderer.sph?.boxWidthRatio ?? 1)
      : (normalizedConfig.renderer.mlsMpm?.boxWidthRatio ?? 1);
    prefillFluidState(pipeline.buffers, solver.numParticles, normalizedConfig);
    // The solver owns positions/velocities after pre-seeding; disable the
    // base shape-emission pass (every slot is alive from frame #1) so the
    // Euler step cannot overwrite the solver's result downstream.
    if (pipeline.uniforms.emitCount) {
      (pipeline.uniforms.emitCount as { value: number }).value = 0;
    }
    // Live `z` squeeze of the simulation box (`changeBoxSize` upstream),
    // refreshed from `normalizedConfig` every frame like the other scalars.
    (pipeline.uniforms as Record<string, unknown>).fluidBoxWidthRatio =
      solver.uniforms.boxWidthRatio;
    // Aliases of the spherical-domain / pointer-force uniforms (WaterBall
    // contract): the per-frame update writes them from `renderer.fluid`.
    (pipeline.uniforms as Record<string, unknown>).fluidSphereDomain =
      solver.uniforms.sphereDomain;
    (pipeline.uniforms as Record<string, unknown>).fluidPointerPos =
      solver.uniforms.pointerPos;
    (pipeline.uniforms as Record<string, unknown>).fluidPointerVel =
      solver.uniforms.pointerVel;
    writeFluidDomainUniforms(
      pipeline.uniforms as Record<string, { value: unknown } | undefined>,
      normalizedConfig.renderer.fluid
    );
    const fl = pipeline as unknown as {
      computeNodes?: unknown[];
      passNames?: string[];
      passLayouts?: Array<{
        name: string;
        storageBindings: number;
        uniformBindings: number;
      }>;
    };
    fl.computeNodes = [
      ...(fl.computeNodes ??
        (pipeline.emitNode && pipeline.simNode
          ? [pipeline.emitNode, pipeline.simNode]
          : [])),
      ...solver.computeNodes,
    ];
    fl.passNames = [
      ...(fl.passNames ?? ['emit', 'simulate']),
      ...solver.passNames.map((n) => `${solverPrefix}:${n}`),
    ];
    fluidSolverPassNames = fl.passNames as string[];
    fl.passLayouts = [
      ...(fl.passLayouts ?? []),
      ...solver.passLayouts.map((p) => ({
        ...p,
        name: `${solverPrefix}:${p.name}`,
      })),
    ];
  }

  // ?? 2b. Trail ribbon expansion kernel (history ring -> ribbon vertices) ??
  const ribbonPipeline: {
    ribbonNode: unknown;
    passLayouts: Array<{
      name: string;
      storageBindings: number;
      uniformBindings: number;
    }>;
    uniforms: Record<string, { value: unknown }>;
    buffers: Record<string, THREE.BufferAttribute>;
  } | null = trailDesc
    ? factory.createTrailRibbonUpdate!({
        position: new StorageBufferAttribute(
          new Float32Array(maxParticles * trailLength * 2 * 4),
          4
        ),
        next: new StorageBufferAttribute(
          new Float32Array(maxParticles * trailLength * 2 * 4),
          4
        ),
        uvColorA: new StorageBufferAttribute(
          new Float32Array(maxParticles * trailLength * 2 * 4),
          4
        ),
        colorB: new StorageBufferAttribute(
          new Float32Array(maxParticles * trailLength * 2 * 4),
          4
        ),
        history: trailDesc.attribute,
        meta: trailDesc.meta as StorageBufferAttribute,
        particleColor: pipeline.buffers.color as StorageBufferAttribute,
        curveFns: {
          width: trailConfig?.widthOverTrail
            ? getCurveFunctionFromConfig(
                _particleSystemId,
                trailConfig.widthOverTrail
              )
            : undefined,
          opacity: trailConfig?.opacityOverTrail
            ? getCurveFunctionFromConfig(
                _particleSystemId,
                trailConfig.opacityOverTrail
              )
            : undefined,
          colorR: trailConfig?.colorOverTrail?.isActive
            ? getCurveFunctionFromConfig(
                _particleSystemId,
                trailConfig.colorOverTrail.r as unknown as LifetimeCurve
              )
            : undefined,
          colorG: trailConfig?.colorOverTrail?.isActive
            ? getCurveFunctionFromConfig(
                _particleSystemId,
                trailConfig.colorOverTrail.g as unknown as LifetimeCurve
              )
            : undefined,
          colorB: trailConfig?.colorOverTrail?.isActive
            ? getCurveFunctionFromConfig(
                _particleSystemId,
                trailConfig.colorOverTrail.b as unknown as LifetimeCurve
              )
            : undefined,
        },
        width: trailConfig?.width ?? 1,
        length: trailLength,
        maxTime: trailDesc.maxTime,
        maxParticles,
      })
    : null;

  // ?? 2c. Sub-emitter child pools (event-driven, GPU-owned) ??
  type SubEntry = {
    fifo: FifoEntry;
    pipeline: NonNullable<ParticleSystemInstance['computePipeline']>;
    init: {
      commandBuildNode: unknown;
      childInitNode: unknown;
      counterClearNode: unknown;
      commandBuffer: StorageBufferAttribute;
      passLayouts: Array<{
        name: string;
        storageBindings: number;
        uniformBindings: number;
      }>;
      passName: string;
      counterClearPassName: string;
      uniforms: Record<string, { value: unknown }>;
    };
    instanced: boolean;
    requestedRendererType: RendererType | undefined;
    effectiveRendererType: RendererType;
    cfg: NormalizedParticleSystemConfig;
    object: THREE.Points | THREE.Mesh | null;
    perEvent: number;
    gravity: number;
    noise: GeneralData['noise'] | null;
    rate: number;
    acc: number;
    lastEmit: number;
    poseFrom: 'parent' | 'self';
    selfPose: {
      x: number;
      y: number;
      z: number;
      qx: number;
      qy: number;
      qz: number;
      qw: number;
      sx: number;
      sy: number;
      sz: number;
      isWorld: 0 | 1;
    };
  };
  const subEntries: SubEntry[] = [];
  for (let fi = 0; fi < subEmitterConfigs.length; fi++) {
    const se = subEmitterConfigs[fi];
    const fifo = fifos[fi];
    const childCfg = ObjectUtils.deepMerge(
      getDefaultParticleSystemConfig() as unknown as NormalizedParticleSystemConfig,
      (se.config ?? {}) as unknown as NormalizedParticleSystemConfig,
      { applyToFirstObject: false, skippedProperties: [] }
    ) as NormalizedParticleSystemConfig;
    // Particles created per event: first burst count (max bound), else 1.
    const firstBurst = childCfg.emission?.bursts?.[0];
    const burstCount = firstBurst
      ? Math.max(
          1,
          Math.ceil(
            calculateValue(_particleSystemId + 1 + fi, firstBurst.count, 0) *
              (firstBurst.cycles ?? 1)
          )
        )
      : 1;
    const perEvent = Math.min(burstCount, fifo.capacity);
    const childMax = Math.max(2, Math.min(perEvent * fifo.capacity, 65536));
    // Child renderer classes are resolved from the CHILD config itself
    // (recursive `resolveWebGPUEffectiveRendererType`, §2): material, object
    // type, InstancedBufferGeometry choice and the compute instancing flag
    // all use the child's effective type.
    const childRequestedRendererType = childCfg.renderer?.rendererType;
    const childEffectiveRendererType = resolveWebGPUEffectiveRendererType(
      childRequestedRendererType
    );
    const childInstanced =
      childEffectiveRendererType === RendererType.INSTANCED ||
      childEffectiveRendererType === RendererType.MESH;
    const childPipeline = factory.createComputePipeline(
      childMax,
      childInstanced,
      childCfg,
      _particleSystemId + 1 + fi,
      0,
      0,
      [],
      undefined
    ) as NonNullable<ParticleSystemInstance['computePipeline']>;
    const childShapeParams = factory.encodeShapeEmitParams!(
      childCfg,
      _particleSystemId + 1 + fi
    );
    const childVel = childCfg.velocityOverLifetime;
    const init = factory.createSubEmitterInitUpdate!(
      childPipeline.buffers,
      childMax,
      childShapeParams,
      pipeline.buffers,
      maxParticles,
      fifo,
      se.inheritVelocity ?? 0,
      perEvent,
      {
        linear: [
          childVel?.linear?.x as never,
          childVel?.linear?.y as never,
          childVel?.linear?.z as never,
        ],
        orbital: [
          childVel?.orbital?.x as never,
          childVel?.orbital?.y as never,
          childVel?.orbital?.z as never,
        ],
      }
    ) as {
      commandBuildNode: unknown;
      childInitNode: unknown;
      counterClearNode: unknown;
      commandBuffer: StorageBufferAttribute;
      passLayouts: Array<{
        name: string;
        storageBindings: number;
        uniformBindings: number;
      }>;
      passName: string;
      counterClearPassName: string;
      uniforms: Record<string, { value: unknown }>;
    };

    // Child render object is built after `rendererConfig` exists (below).
    subEntries.push({
      fifo,
      pipeline: childPipeline,
      init,
      instanced: childInstanced,
      requestedRendererType: childRequestedRendererType,
      effectiveRendererType: childEffectiveRendererType,
      cfg: childCfg,
      object: null,
      perEvent,
      gravity: childCfg.gravity,
      noise: childCfg.noise?.isActive
        ? {
            isActive: true,
            strength: childCfg.noise.strength,
            noisePower: 0.15 * childCfg.noise.strength,
            frequency: childCfg.noise.frequency,
            positionAmount: childCfg.noise.positionAmount,
            rotationAmount: childCfg.noise.rotationAmount,
            sizeAmount: childCfg.noise.sizeAmount,
            fbmMax: 2 - Math.pow(2, -childCfg.noise.octaves),
          }
        : null,
      rate: childCfg.emission?.rateOverTime
        ? calculateValue(
            _particleSystemId + 1 + fi,
            childCfg.emission.rateOverTime,
            0
          )
        : 0,
      acc: 0,
      lastEmit: 0,
      poseFrom: 'self',
      selfPose: {
        x: 0,
        y: 0,
        z: 0,
        qx: 0,
        qy: 0,
        qz: 0,
        qw: 1,
        sx: 1,
        sy: 1,
        sz: 1,
        isWorld: childCfg.simulationSpace === SimulationSpace.WORLD ? 1 : 0,
      },
    });
  }

  // ?? 3. shared uniform table (TSL) ??
  // §10 normalization happens through the module-level helpers
  // (`normalizeVector2Value` etc.) defined above `createParticleSystem`.
  const cameraNearFarSource = (
    normalizedConfig.renderer as unknown as Record<string, unknown>
  ).cameraNearFar;
  const tilesSource = normalizedConfig.textureSheetAnimation?.tiles;
  const elapsedUniform: { value: number } = { value: 0 };
  const sharedUniforms: { [k: string]: { value: unknown } } = {
    elapsed: elapsedUniform,
    viewportHeight: { value: 720 },
    cameraNearFar: {
      value: normalizeVector2Value(
        cameraNearFarSource,
        [0.1, 1000],
        'renderer.cameraNearFar'
      ),
    },
    useInstancing: { value: useInstancing },
    softParticlesEnabled: {
      value: !!normalizedConfig.renderer.softParticles?.enabled,
    },
    softParticlesIntensity: {
      value: Math.max(
        normalizedConfig.renderer.softParticles?.intensity ?? 1,
        0.001
      ),
    },
    sceneDepthTexture: {
      value: normalizeDepthTextureValue(
        normalizedConfig.renderer.softParticles?.depthTexture,
        'renderer.softParticles.depthTexture'
      ),
    },
    discardBackgroundColor: {
      value: !!normalizedConfig.renderer.discardBackgroundColor,
    },
    backgroundColor: { value: new THREE.Color(0xffffff) },
    backgroundColorTolerance: {
      value: normalizedConfig.renderer.backgroundColorTolerance ?? 0,
    },
    map: {
      value: normalizeTextureValue(
        normalizedConfig.map ?? getDefaultTexture(),
        'map'
      ),
    },
    startLifetime: { value: 0 },
    startSize: { value: 1 },
    startRotation: { value: 0 },
    startOpacity: { value: 1 },
    startColor: { value: new THREE.Color(1, 1, 1) },
    lifetime: { value: 0 },
    color: { value: new THREE.Color(1, 1, 1) },
    // Sprite-sheet animation fields consumed by tsl-shared.createParticleUniforms.
    fps: { value: normalizedConfig.textureSheetAnimation?.fps || 30.0 },
    useFPSForFrameIndex: {
      value: normalizedConfig.textureSheetAnimation?.timeMode === TimeMode.FPS,
    },
    tiles: {
      // The ONLY normalizer: `tiles` reaches the TSL factory as a Vector2
      // (also {u,v} pairs are accepted per §10). The engine's own default is
      // already (1,1) via the merged default config.
      value: normalizeVector2Value(
        tilesSource,
        [1, 1],
        'textureSheetAnimation.tiles'
      ),
    },
    // FLUID metaball renderer parameters (consumed by createFluidTSLMaterial).
    fluidStretch: {
      value:
        typeof normalizedConfig.renderer.fluid?.stretch === 'number' &&
        Number.isFinite(normalizedConfig.renderer.fluid.stretch)
          ? (normalizedConfig.renderer.fluid.stretch as number)
          : 1,
    },
    fluidAbsorption: {
      value:
        typeof normalizedConfig.renderer.fluid?.absorption === 'number' &&
        Number.isFinite(normalizedConfig.renderer.fluid.absorption)
          ? (normalizedConfig.renderer.fluid.absorption as number)
          : 1.44,
    },
    fluidIor: {
      value:
        typeof normalizedConfig.renderer.fluid?.ior === 'number' &&
        Number.isFinite(normalizedConfig.renderer.fluid.ior)
          ? (normalizedConfig.renderer.fluid.ior as number)
          : 1.33,
    },
    // Extra knobs consumed by the screen-space pass chain
    // (`tsl-fluid-screen-space-material.ts`). Missing entries fall back to
    // the documented {@link FluidConfig} defaults; all three are stored as
    // plain scalars / fixed-length tuples so the TSL side can read them
    // without per-frame normalization.
    fluidSphereSize: {
      value:
        typeof normalizedConfig.renderer.fluid?.sphereSize === 'number' &&
        Number.isFinite(normalizedConfig.renderer.fluid.sphereSize)
          ? (normalizedConfig.renderer.fluid.sphereSize as number)
          : 1.2,
    },
    fluidDensity: {
      value:
        typeof normalizedConfig.renderer.fluid?.density === 'number' &&
        Number.isFinite(normalizedConfig.renderer.fluid.density)
          ? (normalizedConfig.renderer.fluid.density as number)
          : 0.7,
    },
    fluidWaterColor: {
      value:
        Array.isArray(normalizedConfig.renderer.fluid?.waterColor) &&
        (normalizedConfig.renderer.fluid!.waterColor as readonly number[])
          .length === 3
          ? [
              Number(
                (
                  normalizedConfig.renderer.fluid!
                    .waterColor as readonly number[]
                )[0]
              ) || 0,
              Number(
                (
                  normalizedConfig.renderer.fluid!
                    .waterColor as readonly number[]
                )[1]
              ) || 0,
              Number(
                (
                  normalizedConfig.renderer.fluid!
                    .waterColor as readonly number[]
                )[2]
              ) || 0,
            ]
          : [0.0, 0.7375, 0.95],
    },
    fluidSphereRender: {
      value: !!normalizedConfig.renderer.fluid?.sphereRender,
    },
  };
  const bgVec = normalizeBackgroundToVector3(
    normalizedConfig.renderer.backgroundColor,
    'renderer.backgroundColor'
  );
  (sharedUniforms.backgroundColor.value as THREE.Color).setRGB(
    bgVec.x,
    bgVec.y,
    bgVec.z
  );

  const rendererConfig: {
    transparent: boolean;
    blending: THREE.Blending;
    depthTest: boolean;
    depthWrite: boolean;
  } = {
    transparent: !!normalizedConfig.renderer.transparent,
    blending: toBlendingConstant(normalizedConfig.renderer.blending),
    depthTest: normalizedConfig.renderer.depthTest !== false,
    depthWrite: normalizedConfig.renderer.depthWrite !== false,
  };
  // The TSL material is created *after* the geometry (below): the FLUID pass
  // chain needs the final instanced pool for its depth / thickness passes.

  // ?? 4. GPU-backed geometry ??
  const buffers = pipeline.buffers as Record<string, THREE.BufferAttribute>;
  let geometry: THREE.BufferGeometry | THREE.InstancedBufferGeometry;
  if (useInstancing) {
    const g = new THREE.InstancedBufferGeometry();
    const meshGeometry = normalizedConfig.renderer.mesh?.geometry;
    const baseGeometry =
      rrType === RendererType.MESH && meshGeometry
        ? meshGeometry
        : new THREE.BufferGeometry();
    if (rrType !== RendererType.MESH || !meshGeometry) {
      const quad = new Float32Array([
        -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
      ]);
      const quadUV = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
      const quadNormal = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
      const idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
      baseGeometry.setAttribute('position', new THREE.BufferAttribute(quad, 3));
      // MESH node materials sample the sprite map via `uv` and read `normal`.
      baseGeometry.setAttribute('uv', new THREE.BufferAttribute(quadUV, 2));
      baseGeometry.setAttribute(
        'normal',
        new THREE.BufferAttribute(quadNormal, 3)
      );
      baseGeometry.setIndex(new THREE.BufferAttribute(idx, 1));
    }
    g.setAttribute('position', baseGeometry.getAttribute('position'));
    if (baseGeometry.index !== null) g.setIndex(baseGeometry.index);
    g.instanceCount = maxParticles;
    // Exact TSL attribute contract of the instanced / mesh particle materials
    // (`attribute('<name>')` resolves against `geometry.getAttribute(<name>)`).
    g.setAttribute('instanceOffset', buffers.position);
    g.setAttribute('instanceColor', buffers.color);
    g.setAttribute('instanceParticleState', buffers.particleState);
    g.setAttribute('instanceStartValues', buffers.startValues);
    // FLUID also reads the per-particle velocity (vec4: xyz + w padding) for
    // the velocity-stretched metaball quad. Only bound for the fluid class so
    // the other renderers keep their existing 4-attribute contract.
    if (rrType === RendererType.FLUID) {
      g.setAttribute('instanceVelocity', buffers.velocity);
    }
    geometry = g;
  } else {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', buffers.position);
    g.setAttribute('color', buffers.color);
    g.setAttribute('particleState', buffers.particleState);
    g.setAttribute('startValues', buffers.startValues);
    g.setDrawRange(0, maxParticles);
    geometry = g;
    (g as unknown as { instanceCount: number }).instanceCount = maxParticles;
  }

  const material = factory.createTSLParticleMaterial(
    rrType,
    sharedUniforms,
    rendererConfig,
    true,
    geometry as THREE.BufferGeometry
  );
  if (rrType === RendererType.FLUID) {
    // Screen-space chain length (0 for the sphere debug mode): depth +
    // bilateral x4 + thickness + blurX + blurY = 8 `pass()` nodes.
    fluidScreenSpacePasses =
      (material as unknown as { __fluidPassNodes?: unknown[] })
        .__fluidPassNodes?.length ?? 0;
  }

  // ?? 4b. TRAIL renderer: indexed ribbon geometry + ribbon TSL material ??
  let trailGeometry: THREE.BufferGeometry | null = null;
  if (ribbonPipeline && trailDesc) {
    const rb = ribbonPipeline.buffers as unknown as Record<
      string,
      THREE.BufferAttribute
    >;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', rb.position);
    g.setAttribute('trailNext', rb.next);
    g.setAttribute('trailUVColor', rb.uvColorA);
    g.setAttribute('trailColorBA', rb.colorB);
    const idx = new Uint32Array(maxParticles * (trailLength - 1) * 6);
    let o = 0;
    for (let pIdx = 0; pIdx < maxParticles; pIdx++) {
      for (let s = 0; s < trailLength - 1; s++) {
        const b = pIdx * trailLength * 2 + s * 2;
        idx[o++] = b;
        idx[o++] = b + 1;
        idx[o++] = b + 2;
        idx[o++] = b + 1;
        idx[o++] = b + 3;
        idx[o++] = b + 2;
      }
    }
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.setDrawRange(0, maxParticles * trailLength * 2);
    trailGeometry = g;
  }

  // Main render object: TRAIL uses the ribbon geometry + ribbon material.
  const trailMaterial = trailGeometry
    ? factory.createTSLTrailMaterial(
        {
          map: {
            value: (normalizedConfig.map ??
              getDefaultTexture()) as unknown as THREE.Texture,
          },
          useMap: { value: !!normalizedConfig.map },
          discardBackgroundColor: {
            value: !!normalizedConfig.renderer.discardBackgroundColor,
          },
          backgroundColor: {
            value: normalizedConfig.renderer.backgroundColor ?? {
              r: 1,
              g: 1,
              b: 1,
            },
          },
          backgroundColorTolerance: {
            value: normalizedConfig.renderer.backgroundColorTolerance ?? 0,
          },
          softParticlesEnabled: {
            value: !!normalizedConfig.renderer.softParticles?.enabled,
          },
          softParticlesIntensity: {
            value: Math.max(
              normalizedConfig.renderer.softParticles?.intensity ?? 1,
              0.001
            ),
          },
          sceneDepthTexture: {
            value: (normalizedConfig.renderer.softParticles?.depthTexture ??
              null) as unknown as THREE.Texture,
          },
          cameraNearFar: { value: new THREE.Vector2(0.1, 1000) },
        },
        {
          transparent: !!normalizedConfig.renderer.transparent,
          blending: toBlendingConstant(normalizedConfig.renderer.blending),
          depthTest: normalizedConfig.renderer.depthTest !== false,
          depthWrite: normalizedConfig.renderer.depthWrite !== false,
        }
      )
    : null;
  // FLUID's visible object is the fullscreen `fluid.wgsl` pass; the instanced
  // pool keeps driving the pass chain through the shared attribute handles.
  const fluidPassGeometry = (
    material as unknown as { __fluidPassGeometry?: THREE.BufferGeometry }
  ).__fluidPassGeometry;
  const particleSystem: THREE.Points | THREE.Mesh = trailGeometry
    ? new THREE.Mesh(trailGeometry, trailMaterial as THREE.Material)
    : useInstancing
      ? new THREE.Mesh(fluidPassGeometry ?? geometry, material)
      : new THREE.Points(geometry, material);
  particleSystem.frustumCulled = false;

  // ?? 4c. Sub-emitter child render objects ??
  for (const e of subEntries) {
    const cb = e.pipeline.buffers as Record<string, THREE.BufferAttribute>;
    const childMax = (e.pipeline.allocatorCount as number) - 1;
    const childGeometry: THREE.BufferGeometry | THREE.InstancedBufferGeometry =
      e.instanced
        ? (() => {
            const g = new THREE.InstancedBufferGeometry();
            const quad = new Float32Array([
              -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
            ]);
            const idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
            g.setAttribute('position', new THREE.BufferAttribute(quad, 3));
            g.setIndex(new THREE.BufferAttribute(idx, 1));
            g.instanceCount = childMax;
            g.setAttribute('instanceOffset', cb.position);
            g.setAttribute('instanceColor', cb.color);
            g.setAttribute('instanceParticleState', cb.particleState);
            g.setAttribute('instanceStartValues', cb.startValues);
            if (e.effectiveRendererType === RendererType.FLUID) {
              g.setAttribute('instanceVelocity', cb.velocity);
            }
            return g;
          })()
        : (() => {
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', cb.position);
            g.setAttribute('color', cb.color);
            g.setAttribute('particleState', cb.particleState);
            g.setAttribute('startValues', cb.startValues);
            g.setDrawRange(0, childMax);
            return g;
          })();
    const childUniforms: { [k: string]: { value: unknown } } = {
      ...sharedUniforms,
      useInstancing: { value: e.instanced },
    };
    const childMaterial = factory.createTSLParticleMaterial(
      e.effectiveRendererType,
      childUniforms,
      rendererConfig,
      true
    );
    const childObject: THREE.Points | THREE.Mesh = e.instanced
      ? new THREE.Mesh(childGeometry, childMaterial)
      : new THREE.Points(childGeometry, childMaterial);
    childObject.frustumCulled = false;
    particleSystem.add(childObject);
    e.object = childObject;
  }

  // ?? Construction-time attribute-contract assertion (development only) ????
  // Runs once per particle system; no per-frame work and no particle scan.
  if (
    (import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV !== false
  ) {
    const required = useInstancing
      ? [
          'position', // quad / mesh vertex positions
          'instanceOffset', // GPU particle position
          'instanceColor', // GPU particle RGBA
          'instanceParticleState', // GPU packed state vec4
          'instanceStartValues', // GPU packed initial-state vec4
        ]
      : ['position', 'color', 'particleState', 'startValues'];
    for (const name of required) {
      if (!geometry.getAttribute(name)) {
        throw new Error(
          'three-particles: ' +
            (useInstancing ? 'instanced' : 'POINTS') +
            ' geometry ' +
            name +
            ' is missing its required contract attribute.'
        );
      }
    }
    // Attribute identity: the renderer attributes must point at the SAME
    // GPU storage buffers the compute kernels own (no mirror copies).
    const contractIdentity: Array<[string, unknown]> = useInstancing
      ? [
          ['instanceOffset', buffers.position],
          ['instanceColor', buffers.color],
          ['instanceParticleState', buffers.particleState],
          ['instanceStartValues', buffers.startValues],
        ]
      : [
          ['position', buffers.position],
          ['color', buffers.color],
          ['particleState', buffers.particleState],
          ['startValues', buffers.startValues],
        ];
    for (const [name, buf] of contractIdentity) {
      if (geometry.getAttribute(name) !== buf) {
        throw new Error(
          `three-particles: attribute "${name}" is not the compute-owned storage buffer.`
        );
      }
    }
    const kind = (pipeline.shapeUniforms as Record<string, { value: number }>)
      .shapeKind.value;
    if (!(kind >= 0 && kind <= 4)) {
      throw new Error(
        `three-particles: gpuShapeKind ${kind} outside 0..4 (SPHERE..BOX).`
      );
    }
    if (!(maxParticles > 0)) {
      throw new Error('three-particles: maxParticles must be > 0.');
    }
    if (pipeline.allocatorCount !== maxParticles + 1) {
      throw new Error(
        'three-particles: allocator capacity must equal maxParticles + 1.'
      );
    }
    // Hard per-pass storage-budget assertion. The WebGPU guaranteed per-stage
    // limit is `maxStorageBuffersPerShaderStage = 8` (portable guarantee; we
    // never request 16). After the pass decomposition, every compute pass
    // reports its REAL budget from its own bound resources — no synthetic
    // totals. The base emit/sim pool stays at exactly bindings 1..8
    // (position, velocity, color, particleState, startValues,
    // startColorsExt, orbitalIsActive, allocator); trail and the sub-emitter
    // FIFO resources live in their own dedicated passes.
    type PassLayoutT = {
      name: string;
      storageBindings: number;
      uniformBindings: number;
    };
    const passLayouts = [
      ...((pipeline.passLayouts ?? []) as unknown as PassLayoutT[]),
      ...((ribbonPipeline?.passLayouts ?? []) as unknown as PassLayoutT[]),
      ...subEntries.flatMap((e) => [
        ...((e.init.passLayouts ?? []) as unknown as PassLayoutT[]),
        ...((e.pipeline.passLayouts ?? []) as unknown as PassLayoutT[]).map(
          (p) => ({ ...p, name: `child:${p.name}` })
        ),
      ]),
    ];
    for (const pass of passLayouts) {
      if (pass.storageBindings > 8) {
        throw new Error(
          `${pass.name}: ${pass.storageBindings} storage buffers > guaranteed limit 8`
        );
      }
    }
    if (trailDesc && trailDesc.meta !== pipeline.trailMeta) {
      throw new Error('three-particles: trail ring meta buffer mismatch.');
    }
    for (const f of fifos) {
      const n = (f.counter.array as Uint32Array).length;
      if (n !== 2) {
        throw new Error(
          'three-particles: sub-emitter FIFO must expose exactly 2 ping-pong counter slots.'
        );
      }
      const p = (f.payload.array as Float32Array).length;
      if (p !== 2 * 6 * f.capacity) {
        throw new Error(
          'three-particles: sub-emitter FIFO payload length must be 2 * 6 * capacity.'
        );
      }
    }
  }

  // ?? Emitter transform ??
  // Matches the CPU oracle: position passthrough, rotation in DEGREES, scale
  // passthrough. Missing components fall back to the defaults of the shared
  // default config (0 / 0 / 0 / 1). In LOCAL simulation space the object matrix
  // carries the transform; in WORLD simulation space matrixWorld is identity and
  // the pose is delivered to the kernels through the emitter-pose uniforms.
  const _numOr = (v: unknown, d: number): number =>
    typeof v === 'number' && Number.isFinite(v) ? v : d;
  const xform = normalizedConfig.transform as unknown as
    | {
        position?: Partial<THREE.Vector3>;
        rotation?: Partial<THREE.Vector3>;
        scale?: Partial<THREE.Vector3>;
      }
    | undefined;
  if (xform?.position) {
    particleSystem.position.set(
      _numOr(xform.position.x, 0),
      _numOr(xform.position.y, 0),
      _numOr(xform.position.z, 0)
    );
  }
  if (xform?.rotation) {
    particleSystem.rotation.set(
      THREE.MathUtils.degToRad(_numOr(xform.rotation.x, 0)),
      THREE.MathUtils.degToRad(_numOr(xform.rotation.y, 0)),
      THREE.MathUtils.degToRad(_numOr(xform.rotation.z, 0))
    );
  }
  if (xform?.scale) {
    particleSystem.scale.set(
      _numOr(xform.scale.x, 1),
      _numOr(xform.scale.y, 1),
      _numOr(xform.scale.z, 1)
    );
  }
  particleSystem.updateMatrix();
  particleSystem.updateMatrixWorld(true);

  if (normalizedConfig.simulationSpace === SimulationSpace.WORLD) {
    particleSystem.matrixWorldAutoUpdate = false;
    particleSystem.matrixWorld.identity();
  }

  // ?? 5. general data ??
  const generalData: GeneralData = {
    particleSystemId: _particleSystemId++,
    normalizedLifetimePercentage: 0,
    distanceFromLastEmitByDistance: 0,
    lastWorldPosition: new THREE.Vector3(-99999),
    currentWorldPosition: new THREE.Vector3(-99999),
    worldPositionChange: new THREE.Vector3(),
    sourceWorldMatrix: new THREE.Matrix4(),
    worldQuaternion: new THREE.Quaternion(),
    wrapperQuaternion: new THREE.Quaternion(),
    worldScale: new THREE.Vector3(1, 1, 1),
    worldEuler: new THREE.Euler(),
    gravityVelocity: new THREE.Vector3(0, 0, 0),
    startValues: {},
    linearVelocityData: undefined,
    orbitalVelocityData: undefined,
    lifetimeValues: {},
    creationTimes: new Float32Array(0),
    cpuDirtyParticleWatermark: -1,
    highWaterIndex: fluidHighWater,
    // FLUID solver snapshots (§2a); `null` keeps the single-pass metaball path.
    fluidSolver: fluidSolverId,
    fluidBoxWidthRatio: fluidBoxWidthRatioValue,
    noise: {
      isActive: normalizedConfig.noise.isActive,
      strength: normalizedConfig.noise.strength,
      // Oracle `0.15 * strength`; the single fbmMax division lives inside the
      // FBM sum (CPU: FBM.get3; GPU: the octave loop amp / fbmMax).
      noisePower: 0.15 * normalizedConfig.noise.strength,
      frequency: normalizedConfig.noise.frequency,
      positionAmount: normalizedConfig.noise.positionAmount,
      rotationAmount: normalizedConfig.noise.rotationAmount,
      sizeAmount: normalizedConfig.noise.sizeAmount,
      fbmMax: 2 - Math.pow(2, -normalizedConfig.noise.octaves),
    },
    isEnabled: true,
    burstStates: normalizedConfig.emission.bursts?.length
      ? normalizedConfig.emission.bursts.map(() => ({
          cyclesExecuted: 0,
          lastCycleTime: 0,
          probabilityPassed: false,
        }))
      : undefined,
  };

  // ?? 6. ParticleSystemInstance for the update loop ??
  const props: ParticleSystemInstance = {
    particleSystem,
    mappedAttributes: {
      position: buffers.position as unknown as THREE.BufferAttribute,
      isActive: buffers.orbitalIsActive as unknown as THREE.BufferAttribute,
      lifetime: buffers.particleState as unknown as THREE.BufferAttribute,
      startLifetime: buffers.startValues as unknown as THREE.BufferAttribute,
      startFrame: buffers.particleState as unknown as THREE.BufferAttribute,
      size: buffers.particleState as unknown as THREE.BufferAttribute,
      rotation: buffers.particleState as unknown as THREE.BufferAttribute,
      color: buffers.color as unknown as THREE.BufferAttribute,
    },
    // ?? Deprecated zero-size sentinels (GPU-only v4) ????
    // These legacy CPU particle-state fields are not authoritative anymore: the
    // compute kernels own the state in GPU storage. Only the TRAIL path (which
    // throws in v4) consumed them, so they are 0-length placeholders.
    scalarArray: new Float32Array(0),
    scalarInterleavedBuffer: new THREE.InterleavedBuffer(
      new Float32Array(0),
      SCALAR_STRIDE
    ),
    elapsedUniform,
    generalData,
    onUpdate: () => {},
    onComplete: () => {},
    creationTime: now + ((normalizedConfig.startDelay as number) || 0),
    lastEmissionTime: now,
    emissionAccumulator: 0,
    duration: normalizedConfig.duration,
    looping: normalizedConfig.looping,
    simulationSpace: normalizedConfig.simulationSpace,
    gravity: normalizedConfig.gravity,
    normalizedForceFields: forceFields,
    normalizedCollisionPlanes: collisionPlanes,
    emission: normalizedConfig.emission,
    normalizedConfig,
    iterationCount: 0,
    velocities: [],
    freeList: [],
    deactivateParticle: () => {},
    killParticle: () => {},
    activateParticle: () => {},
    computePipeline: pipeline,
    useGPUCompute: true,
    computeDispatchReady: false,
    maxParticles,
    material,
    geometry,
    rrType,
    requestedRendererType,
    effectiveRendererType: rrType,
    sharedUniforms,
    allComputeNodes: [
      ...((pipeline.computeNodes ?? []) as unknown[]),
      ...(ribbonPipeline ? [ribbonPipeline.ribbonNode] : []),
      ...subEntries.flatMap((e) => [
        e.init.commandBuildNode,
        e.init.childInitNode,
        ...(e.init.counterClearNode !== null &&
        e.init.counterClearNode !== undefined
          ? [e.init.counterClearNode]
          : []),
        ...((e.pipeline.computeNodes ?? []) as unknown[]),
      ]),
    ],
    passNames: [
      ...((pipeline.passNames ?? ['emit', 'simulate']) as string[]),
      ...(ribbonPipeline ? ['trail-ribbon'] : []),
      ...subEntries.flatMap((e, ei) => [
        `sub${ei}:command-build`,
        `sub${ei}:child-init`,
        `sub${ei}:counter-clear`,
        `sub${ei}:child-emit`,
        `sub${ei}:child-sim`,
      ]),
    ],
    fifoBaseStride,
    ribbonUniforms: ribbonPipeline
      ? (ribbonPipeline.uniforms as Record<string, { value: unknown }>)
      : undefined,
    ribbonBuffers: ribbonPipeline
      ? (ribbonPipeline.buffers as unknown as Record<
          string,
          THREE.BufferAttribute
        >)
      : undefined,
    frameParity: 0,
    subEntries: subEntries.map((e) => ({
      fifo: { capacity: e.fifo.capacity, windowSize: e.fifo.windowSize },
      requestedRendererType: e.requestedRendererType,
      effectiveRendererType: e.effectiveRendererType,
      pipeline: e.pipeline as unknown as Record<string, any>,
      init: e.init,
      gravity: e.gravity,
      noise: e.noise,
      rate: e.rate,
      acc: 0,
      isWorld: e.selfPose.isWorld,
      quat: [e.selfPose.qx, e.selfPose.qy, e.selfPose.qz, e.selfPose.qw] as [
        number,
        number,
        number,
        number,
      ],
      scale: [e.selfPose.sx, e.selfPose.sy, e.selfPose.sz] as [
        number,
        number,
        number,
      ],
      position: [
        _numOr(
          (
            e.cfg.transform as unknown as {
              position?: { x?: number; y?: number; z?: number };
            }
          )?.position?.x,
          0
        ),
        _numOr(
          (
            e.cfg.transform as unknown as {
              position?: { x?: number; y?: number; z?: number };
            }
          )?.position?.y,
          0
        ),
        _numOr(
          (
            e.cfg.transform as unknown as {
              position?: { x?: number; y?: number; z?: number };
            }
          )?.position?.z,
          0
        ),
      ] as [number, number, number],
    })),
  };
  // Self transform of each child object (LOCAL space) mirrors the main path.
  for (const e of subEntries) {
    if (!e.object) continue;
    const tf = e.cfg.transform as unknown as
      | {
          position?: Partial<THREE.Vector3>;
          rotation?: Partial<THREE.Vector3>;
          scale?: Partial<THREE.Vector3>;
        }
      | undefined;
    if (tf?.position) {
      e.object.position.set(
        _numOr(tf.position.x, 0),
        _numOr(tf.position.y, 0),
        _numOr(tf.position.z, 0)
      );
    }
    if (tf?.rotation) {
      e.object.rotation.set(
        THREE.MathUtils.degToRad(_numOr(tf.rotation.x, 0)),
        THREE.MathUtils.degToRad(_numOr(tf.rotation.y, 0)),
        THREE.MathUtils.degToRad(_numOr(tf.rotation.z, 0))
      );
    }
    if (tf?.scale) {
      e.object.scale.set(
        _numOr(tf.scale.x, 1),
        _numOr(tf.scale.y, 1),
        _numOr(tf.scale.z, 1)
      );
    }
    e.object.updateMatrix();
    const q = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        THREE.MathUtils.degToRad(
          _numOr((tf?.rotation as Partial<THREE.Vector3> | undefined)?.x, 0)
        ),
        THREE.MathUtils.degToRad(
          _numOr((tf?.rotation as Partial<THREE.Vector3> | undefined)?.y, 0)
        ),
        THREE.MathUtils.degToRad(
          _numOr((tf?.rotation as Partial<THREE.Vector3> | undefined)?.z, 0)
        ),
        'XYZ'
      )
    );
    const entry = props.subEntries?.[subEntries.indexOf(e)];
    if (entry) {
      entry.quat = [q.x, q.y, q.z, q.w];
      entry.scale = [
        _numOr(tf?.scale?.x, 1),
        _numOr(tf?.scale?.y, 1),
        _numOr(tf?.scale?.z, 1),
      ];
    }
  }
  createdParticleSystems.push(props);

  // Binding budgets per compute pass, derived from each pass's actual
  // resources (`passLayouts` of the owning pipelines; guaranteed WebGPU
  // per-stage storage limit = 8, never 16). The base emit/sim pool is
  // exactly the 8 per-particle buffers; trail + sub-emitter FIFOs live in
  // dedicated passes (trail-history, sub-birth/death-events, sub-command-
  // build, sub-child-init, sub-counter-clear); velocity-axis values are
  // seed-derived (no extra buffer).
  const _dbgPassCounts: Array<[string, number]> = [
    ...(
      (pipeline.passLayouts ?? []) as Array<{
        name: string;
        storageBindings: number;
      }>
    ).map((p) => [p.name, p.storageBindings] as [string, number]),
    ...(
      (ribbonPipeline?.passLayouts ?? []) as Array<{
        name: string;
        storageBindings: number;
      }>
    ).map((p) => [p.name, p.storageBindings] as [string, number]),
    ...subEntries.flatMap((e, ei): Array<[string, number]> => [
      ...(e.init.passLayouts ?? []).map(
        (p) => [`sub${ei}:${p.name}`, p.storageBindings] as [string, number]
      ),
      ...(e.pipeline.passLayouts ?? []).map(
        (p) => [`sub${ei}:${p.name}`, p.storageBindings] as [string, number]
      ),
    ]),
  ];
  const _dbgMaxPass = _dbgPassCounts.reduce((m, p) => Math.max(m, p[1]), 0);

  // ?? One-shot construction diagnostics (visible without DevTools commands) ??
  // `[PS:create]`, `[PS:config]` and `[PS:pipeline]` are logged exactly once
  // per system; the milestone probes themselves live in the examples harness.
  if (typeof console !== 'undefined' && console.log) {
    const logCfg = normalizedConfig as unknown as Record<string, any>;
    const shpU = pipeline.shapeUniforms as unknown as Record<
      string,
      { value: number } | undefined
    >;
    const sv = logCfg.startValues as Record<string, unknown> | undefined;
    const u = pipeline.uniforms as unknown as Record<
      string,
      { value: number } | undefined
    >;
    console.log(`[PS:create] system #${generalData.particleSystemId}`, {
      rendererType: rrType,
      requestedRendererType,
      effectiveRendererType: rrType,
      simulationSpace: normalizedConfig.simulationSpace,
      maxParticles,
      useInstancing,
    });
    console.log(`[PS:config] system #${generalData.particleSystemId}`, {
      shape: {
        publicKind: logCfg.shape?.shape ?? null,
        gpuShapeKind: shpU.shapeKind?.value ?? 0,
        radius: shpU.radius?.value ?? logCfg.shape?.radius ?? null,
        radiusThickness: shpU.radiusThickness?.value ?? null,
        arcDeg: shpU.arcDeg?.value ?? null,
        coneAngleDeg: shpU.coneAngleDeg?.value ?? null,
        rectScale: [shpU.rectScaleX?.value, shpU.rectScaleY?.value],
        rectRotationDeg: [shpU.rectRotXDeg?.value, shpU.rectRotYDeg?.value],
        boxScale: [shpU.boxSX?.value, shpU.boxSY?.value, shpU.boxSZ?.value],
        boxEmitFrom: shpU.boxEmitFrom?.value ?? null,
      },
      transform: {
        position: xform?.position ?? null,
        rotation: xform?.rotation ?? null,
        scale: xform?.scale ?? null,
      },
      emission: {
        rateOverTime: logCfg.emission?.rateOverTime ?? 0,
        rateOverDistance: logCfg.emission?.rateOverDistance ?? 0,
        bursts: logCfg.emission?.bursts?.length ?? 0,
      },
      startValues: {
        lifetime: sv?.startLifetime ?? null,
        speed: sv?.startSpeed ?? null,
        size: sv?.startSize ?? null,
        rotation: sv?.startRotation ?? null,
        color: sv?.startColor ?? null,
        opacity: sv?.startOpacity ?? null,
      },
      textureId:
        (config as { textureId?: string }).textureId ??
        (config as { _editorData?: { textureId?: string } })._editorData
          ?.textureId ??
        null,
      textureResolved: !!normalizedConfig.map,
      forceFieldCount: forceFields.length,
      collisionPlaneCount: collisionPlanes.length,
      subEmitterCount: (normalizedConfig.subEmitters ?? []).length,
      trailEnabled: !!trailDesc,
      modifiers: {
        linearVelocity:
          !!logCfg.velocityOverLifetime?.isActive &&
          (u.linearVelX !== undefined ||
            u.axisLinXMin !== undefined ||
            !!(
              logCfg.velocityOverLifetime?.linear &&
              Object.values(logCfg.velocityOverLifetime.linear).some(
                (value: any) => value !== undefined && value !== 0
              )
            )),
        orbitalVelocity:
          !!logCfg.velocityOverLifetime?.isActive &&
          !!(
            logCfg.velocityOverLifetime?.orbital &&
            Object.values(logCfg.velocityOverLifetime.orbital).some(
              (value: any) => value !== undefined && value !== 0
            )
          ),
        sizeOverLifetime: !!normalizedConfig.sizeOverLifetime?.isActive,
        opacityOverLifetime: !!normalizedConfig.opacityOverLifetime?.isActive,
        colorOverLifetime: !!normalizedConfig.colorOverLifetime?.isActive,
        rotationOverLifetime: !!normalizedConfig.rotationOverLifetime?.isActive,
        noise: !!normalizedConfig.noise?.isActive,
      },
    });
    console.log(
      `[PS:pipeline] system #${generalData.particleSystemId}: ` +
        `${((props.passNames ?? []) as string[]).join(' -> ') || 'emit -> simulate'}` +
        ` | storageBindings=${_dbgPassCounts.map((p) => `${p[0]}=${p[1]}≤8`).join(' ')}` +
        ` | packedFloats=${(pipeline.buffers.packedData as unknown as { length?: number })?.length ?? 0}`
    );
  }

  // Emission count scratch so bursts + rate-over-distance don't allocate
  // closures / objects per frame.
  let lastSeed = Math.random();

  const update = (cycleData: CycleData): void => {
    updateParticleSystemInstance(props, cycleData);
    lastSeed = Math.random();
  };

  const resumeEmitter = (): void => {
    generalData.isEnabled = true;
  };
  const pauseEmitter = (): void => {
    generalData.isEnabled = false;
  };
  const dispose = (): void => {
    destroyParticleSystem(particleSystem);
  };
  const updateConfig = (partial: Partial<ParticleSystemConfig>): void => {
    ObjectUtils.deepMerge(normalizedConfig, partial, {
      applyToFirstObject: true,
      skippedProperties: [],
    });
    // Scalars (gravity, duration, looping, noise params, emission rates, burst counts)
    // are read every frame straight from `normalizedConfig` in `updateParticleSystemInstance`
    // so no explicit copy is required below.
  };

  return {
    instance: particleSystem,
    resumeEmitter,
    pauseEmitter,
    dispose,
    update,
    updateConfig,
    /**
     * ?? Deprecated synchronous active count ????
     * Returns -1 (= unsupported) in the GPU-only engine: the authoritative count
     * is `maxParticles - allocator[0]` which lives in GPU storage and is only
     * available through an explicit (throttled) `getArrayBufferAsync` read-back.
     */
    getActiveParticleCount: () => -1,
    computeNode:
      props.allComputeNodes && props.allComputeNodes.length > 0
        ? props.allComputeNodes
        : (pipeline.computeNodes ?? pipeline.computeNode),
    /**
     * Binds the active perspective camera to every screen-space fluid pass
     * (depth / bilateral / thickness / blur). Call once after the demo
     * camera exists and again after any camera replacement (resize /
     * restart). No-op outside FLUID or when the camera is already bound.
     */
    bindCamera: (cam: unknown): void => {
      if (!cam) return;
      const mats = [material, trailMaterial] as Array<
        | (THREE.Material & { __fluidPassNodes?: Array<{ camera: unknown }> })
        | null
        | undefined
      >;
      for (const m of mats) {
        const passNodes = m?.__fluidPassNodes;
        if (!passNodes) continue;
        for (const node of passNodes) {
          if (node && node.camera == null) node.camera = cam;
        }
      }
    },
    /**
     * FLUID telemetry snapshot with no per-frame GPU read-back: solver id,
     * seeded particle count, ordered compute pass names, live box ratio and
     * the screen-space render pass count. `null` on non-solver systems.
     */
    getFluidTelemetry: (): FluidTelemetry | null =>
      fluidSolverId
        ? {
            solver: fluidSolverId,
            filledParticles: fluidHighWater,
            maxParticles,
            gridCount: fluidGridCount,
            passNames: [...fluidSolverPassNames],
            passCount: fluidSolverPassNames.length,
            boxWidthRatio: Number(
              (pipeline.uniforms as Record<string, { value: unknown }>)
                .fluidBoxWidthRatio?.value ?? fluidBoxWidthRatioValue
            ),
            screenSpacePasses: fluidScreenSpacePasses,
          }
        : null,
    /**
     * ?? Temporary one-shot GPU debug handle (deprecated, no per-frame cost) ????
     * getActiveParticleCount() stays -1; this object is the raw material for an
     * explicit 
enderer.getArrayBufferAsync(...) read-back (bytes, multiples of 4).
     * lastEmitCount() mirrors uEmitCount, the u32 count written per frame.
     */
    gpuDebug: {
      maxParticles,
      allocatorCount: pipeline.allocatorCount as number,
      /** Canonical requested vs effective GPU renderer classes (§2). */
      requestedRendererType,
      effectiveRendererType: rrType,
      /** u32 birth system seed for this pipeline (written ONCE at create). */
      systemSeed: (
        (pipeline.uniforms as Record<string, { value: unknown }>).seed as {
          value: unknown;
        }
      ).value as number,
      buffers: pipeline.buffers as unknown as Record<
        string,
        THREE.BufferAttribute
      >,
      emitNode: pipeline.emitNode,
      simNode: pipeline.simNode,
      passNames: (pipeline.passNames ?? ['emit', 'simulate']) as string[],
      allPassNames: (props.passNames ?? []) as string[],
      storageBindingCount: _dbgMaxPass,
      passBindingCounts: _dbgPassCounts,
      lastEmitCount: () =>
        (pipeline.uniforms.emitCount as { value: number }).value as number,
      /**
       * Per-sub-emitter-child canonical pairs (§2/§21): each child pool's own
       * requested vs effective renderer class + its events-per-frame.
       */
      subEmitters: (subEntries ?? []).map((e) => ({
        requestedRendererType: e.requestedRendererType ?? null,
        effectiveRendererType: e.effectiveRendererType,
        perEvent: e.perEvent,
      })),
      /** Decode summary for the `[PS:config]` / `[PS:pipeline]` logs. */
      snapshot: () => {
        const shp = normalizedConfig.shape as ShapeConfig & {
          sphere?: { radius?: number; radiusThickness?: number; arc?: number };
          cone?: {
            radius?: number;
            radiusThickness?: number;
            arc?: number;
            angle?: number;
          };
          circle?: { radius?: number; radiusThickness?: number; arc?: number };
          rectangle?: {
            scale?: { x?: number; y?: number };
            rotation?: { x?: number; y?: number };
          };
          box?: { scale?: unknown; emitFrom?: string };
        };
        const branch =
          shp.shape === 'CONE'
            ? shp.cone
            : shp.shape === 'CIRCLE'
              ? shp.circle
              : shp.sphere;
        const tex = normalizedConfig.map as unknown as
          { image?: { width?: number; height?: number } | null } | undefined;
        return {
          systemId: generalData.particleSystemId,
          // Canonical effective + original requested renderer classes (§2).
          effectiveRendererType: rrType,
          requestedRendererType,
          rendererType: rrType,
          simulationSpace: normalizedConfig.simulationSpace,
          maxParticles,
          shape: {
            publicShape: shp.shape,
            gpuShapeKind:
              (pipeline.shapeUniforms as Record<string, { value: number }>)
                ?.shapeKind?.value ?? 0,
            radius: branch?.radius ?? null,
            radiusThickness: branch?.radiusThickness ?? null,
            arcDeg: branch?.arc ?? null,
            coneAngleDeg:
              shp.shape === 'CONE' ? (shp.cone?.angle ?? null) : null,
            rectScale: shp.rectangle?.scale ?? null,
            rectRotation: shp.rectangle?.rotation ?? null,
            boxScale: shp.box?.scale ?? null,
            boxEmitFrom: shp.box?.emitFrom ?? null,
          },
          textureId:
            (config as { textureId?: string }).textureId ??
            (config as { _editorData?: { textureId?: string } })._editorData
              ?.textureId ??
            null,
          textureResolved: !!normalizedConfig.map,
          textureDimensions: tex?.image
            ? [tex.image.width ?? 0, tex.image.height ?? 0]
            : null,
          forceFieldCount: (normalizedConfig.forceFields ?? []).length,
          collisionPlaneCount: (normalizedConfig.collisionPlanes ?? []).length,
          subEmitterCount: (normalizedConfig.subEmitters ?? []).length,
          trailEnabled: !!(normalizedConfig.renderer as { trail?: unknown })
            .trail,
        };
      },
    },
  } as ParticleSystem;
};

// ?? GPU-only per-frame path: writes ~12 scalar uniforms + 2 dispatches ??
// No per-particle JS loop anywhere in this function.
const _lastUploadStampMap = new WeakMap<
  Record<string, THREE.BufferAttribute>,
  number
>();
/** One-shot first-frame upload marker for the sub-emitter command buffers. */
const _cmdUploadSeen = new WeakSet<object>();
const updateParticleSystemInstance = (
  props: ParticleSystemInstance,
  { now, delta, elapsed }: CycleData
): void => {
  const {
    generalData,
    normalizedConfig,
    particleSystem,
    elapsedUniform,
    creationTime,
    normalizedForceFields,
    normalizedCollisionPlanes,
    emission,
    computePipeline: pipeline,
    maxParticles = 0,
    allComputeNodes,
    subEntries,
    fifoBaseStride = 0,
    ribbonUniforms,
  } = props;
  if (!pipeline) return;
  const u = pipeline.uniforms as Record<string, { value: unknown }>;
  const dur = normalizedConfig.duration;
  const lifetime = now - creationTime;
  const loop = normalizedConfig.looping;
  const iterationTimeMs = loop ? lifetime % (dur * 1000) : lifetime;
  generalData.normalizedLifetimePercentage = Math.max(
    Math.min(iterationTimeMs / 1000 / dur, 1),
    0
  );
  (elapsedUniform as { value: number }).value = elapsed;

  // Emitter pose + gravity reference frame (oracle parity, scalar-only work).
  // WORLD: decompose the parent-composed source matrix; buffer is world space.
  // LOCAL: query matrixWorld; gravity is rotated by the inverse world
  // rotation and divided PER AXIS by the world scale.
  const gv = generalData.gravityVelocity;
  gv.set(0, normalizedConfig.gravity, 0);
  if (normalizedConfig.simulationSpace === SimulationSpace.WORLD) {
    particleSystem.updateMatrix();
    _tmpM1.copy(particleSystem.matrix);
    if (particleSystem.parent) {
      particleSystem.parent.updateMatrixWorld();
      _tmpM1.premultiply(particleSystem.parent.matrixWorld);
    }
    _tmpM1.decompose(
      generalData.currentWorldPosition,
      generalData.worldQuaternion,
      generalData.worldScale
    );
  } else {
    particleSystem.updateMatrixWorld();
    particleSystem.getWorldPosition(generalData.currentWorldPosition);
    particleSystem.getWorldQuaternion(generalData.worldQuaternion);
    particleSystem.getWorldScale(generalData.worldScale);
    _tmpQ1.copy(generalData.worldQuaternion).invert();
    gv.applyQuaternion(_tmpQ1);
    gv.x /= generalData.worldScale.x || 1;
    gv.y /= generalData.worldScale.y || 1;
    gv.z /= generalData.worldScale.z || 1;
  }

  // Rate-over-distance travel (same world-position delta the oracle uses).
  if (generalData.lastWorldPosition.x !== -99999) {
    _lastWorldPositionSnapshot.copy(generalData.lastWorldPosition);
    generalData.distanceFromLastEmitByDistance +=
      _lastWorldPositionSnapshot.distanceTo(generalData.currentWorldPosition);
  }
  generalData.lastWorldPosition.copy(generalData.currentWorldPosition);

  // Emission scalar count (rate over time + rate over distance + bursts).
  let emitCount = 0;
  if (generalData.isEnabled && (loop || iterationTimeMs < dur * 1000)) {
    const lastEmit = props.lastEmissionTime;
    const emissionDelta = now - lastEmit;
    if (emissionDelta > 0) {
      props.lastEmissionTime = now;
      if (emission.rateOverTime) {
        props.emissionAccumulator +=
          calculateValue(
            generalData.particleSystemId,
            emission.rateOverTime,
            generalData.normalizedLifetimePercentage
          ) *
          (emissionDelta / 1000);
      }
    }
    emitCount += Math.floor(props.emissionAccumulator);
    if (emitCount > 0) props.emissionAccumulator -= emitCount;
    if (
      emission.rateOverDistance &&
      generalData.distanceFromLastEmitByDistance > 0
    ) {
      const r = calculateValue(
        generalData.particleSystemId,
        emission.rateOverDistance,
        generalData.normalizedLifetimePercentage
      );
      if (r > 0) {
        const n = Math.floor(generalData.distanceFromLastEmitByDistance * r);
        emitCount += n;
        generalData.distanceFromLastEmitByDistance = Math.max(
          generalData.distanceFromLastEmitByDistance - n / r,
          0
        );
      }
    }
    // Bursts (advance internal cycle counters, add their counts).
    if (emission.bursts && generalData.burstStates) {
      const bursts = emission.bursts;
      const states = generalData.burstStates;
      const tSec = iterationTimeMs / 1000;
      for (let i = 0; i < bursts.length; i++) {
        const b = bursts[i];
        const s = states[i];
        const cyc = b.cycles ?? 1;
        const iv = b.interval ?? 0;
        const prob = b.probability ?? 1;
        if (loop && tSec < (b.time ?? 0) && s.cyclesExecuted > 0) {
          s.cyclesExecuted = 0;
          s.lastCycleTime = 0;
          s.probabilityPassed = false;
        }
        if (s.cyclesExecuted >= cyc) continue;
        const next = (b.time ?? 0) + s.cyclesExecuted * iv;
        if (tSec >= next) {
          if (s.cyclesExecuted === 0)
            s.probabilityPassed = Math.random() < prob;
          if (s.probabilityPassed) {
            emitCount += Math.floor(
              calculateValue(
                generalData.particleSystemId,
                b.count,
                generalData.normalizedLifetimePercentage
              )
            );
          }
          s.cyclesExecuted++;
          s.lastCycleTime = tSec;
        }
      }
    }
    if (emitCount > maxParticles) emitCount = maxParticles;
  }

  // Uniform writes (scalars only) ? no per-particle loops anywhere.
  (u.delta as { value: number }).value = delta;
  (u.deltaMs as { value: number }).value = delta * 1000;
  (u.gravityVelocity as { value: THREE.Vector3 }).value.copy(gv);
  (u.emitCount as { value: number }).value = emitCount;
  // Dynamic emit dispatch: the numeric `ComputeNode.count` is the real dispatch
  // size and also feeds the generated `instanceIndex >= count` bound guard.
  // Never dispatch 0 workgroups (r186 warns about it): the host always runs at
  // least one invocation and uEmitCount (u32) bounds the useful work inside
  // the kernel. See If(i.lessThan(uEmitCount), ...) in the emit kernel.
  (pipeline.emitNode as unknown as { count: number }).count = Math.max(
    1,
    emitCount
  );
  // The dedicated BIRTH event pass consumes exactly this frame's born slots.
  if (pipeline.subBirthEventsNode) {
    (pipeline.subBirthEventsNode as unknown as { count: number }).count =
      Math.max(1, emitCount);
  }
  // uSystemSeed is written ONCE at pipeline creation (§5): no per-frame seed.
  const n = generalData.noise;
  if (u.noiseStrength)
    (u.noiseStrength as { value: number }).value = n.strength;
  if (u.noisePower) (u.noisePower as { value: number }).value = n.noisePower;
  if (u.noiseFrequency)
    (u.noiseFrequency as { value: number }).value = n.frequency;
  if (u.noisePositionAmount)
    (u.noisePositionAmount as { value: number }).value = n.positionAmount;
  if (u.noiseRotationAmount)
    (u.noiseRotationAmount as { value: number }).value = n.rotationAmount;
  if (u.noiseSizeAmount)
    (u.noiseSizeAmount as { value: number }).value = n.sizeAmount;

  // ?? FLUID solver frame rules ??
  // The dambreak lattice is seeded once at creation, so the shape-emission pass
  // stays parked (`count = 1`, `uEmitCount = 0`) and the solver kernels own
  // `position` / `velocity` from the first frame on. The animated `z` squeeze of
  // the simulation box (`changeBoxSize` upstream) is refreshed here, like every
  // other scalar, so `updateConfig` picks it up without rebuilding the pool.
  if (generalData.fluidSolver) {
    (u.emitCount as { value: number }).value = 0;
    (pipeline.emitNode as unknown as { count: number }).count = 1;
    if (u.fluidBoxWidthRatio) {
      // Read the live `normalizedConfig` (patched by `updateConfig`) so the
      // `z` squeeze updates WITHOUT a pool rebuild; the creation-time
      // `generalData` snapshot is the fallback only.
      const solverName = String(
        (normalizedConfig.renderer.fluid as { solver?: string } | undefined)
          ?.solver ?? ''
      )
        .trim()
        .toUpperCase();
      const liveRatio =
        solverName === 'SPH'
          ? normalizedConfig.renderer.sph?.boxWidthRatio
          : normalizedConfig.renderer.mlsMpm?.boxWidthRatio;
      (u.fluidBoxWidthRatio as { value: number }).value =
        (typeof liveRatio === 'number' && Number.isFinite(liveRatio)
          ? liveRatio
          : generalData.fluidBoxWidthRatio) ?? 1;
    }
    // Spherical boundary + pointer force (WaterBall): scalar uniform writes
    // picked up live from `normalizedConfig.renderer.fluid` via
    // `updateConfig`, exactly like the `z` squeeze above.
    writeFluidDomainUniforms(
      u as Record<string, { value: unknown } | undefined>,
      normalizedConfig.renderer.fluid as
        | Parameters<typeof writeFluidDomainUniforms>[1]
        | undefined
    );
  }

  // ?? Emitter pose for the emit kernel (scalar/vector writes only) ??
  // LOCAL: identity quaternion, zero translation, unit scale ? the drawn object's
  //   matrixWorld applies the transform.
  // WORLD: matrixWorld is identity, so the full emitter pose is handed to the
  //   kernel directly (translation + world rotation + per-axis world scale).
  const pose = pipeline.emitterPose as
    | {
        positionW: { value: THREE.Vector4 };
        wrapperQuat: { value: THREE.Vector4 };
        worldScale: { value: THREE.Vector3 };
      }
    | undefined;
  if (pose) {
    if (normalizedConfig.simulationSpace === SimulationSpace.WORLD) {
      particleSystem.updateMatrix();
      _tmpM1.copy(particleSystem.matrix);
      if (particleSystem.parent) {
        particleSystem.parent.updateMatrixWorld();
        _tmpM1.premultiply(particleSystem.parent.matrixWorld);
      }
      _tmpM1.decompose(_tmpV1, _tmpQ1, _tmpV2);
      pose.positionW.value.set(_tmpV1.x, _tmpV1.y, _tmpV1.z, 1);
      pose.wrapperQuat.value.set(_tmpQ1.x, _tmpQ1.y, _tmpQ1.z, _tmpQ1.w);
      pose.worldScale.value.set(_tmpV2.x || 1, _tmpV2.y || 1, _tmpV2.z || 1);
    } else {
      pose.positionW.value.set(0, 0, 0, 0);
      pose.wrapperQuat.value.set(0, 0, 0, 1);
      pose.worldScale.value.set(1, 1, 1);
    }
  }

  // ?? FIFO ping-pong window index + trail clock ??
  // The integer counter attribute has exactly two slots (window 0/1); the
  // payload base is `index * channelWindowSize` computed inside the kernels.
  const parity = (props.frameParity ?? 0) % 2;
  const fifoBase = parity;
  if (u.fifoBase) (u.fifoBase as { value: number }).value = fifoBase;
  if (u.nowMs) (u.nowMs as { value: number }).value = now;
  if (ribbonUniforms?.nowMs) ribbonUniforms.nowMs.value = now;

  // ?? Sub-emitter children: scalar writes + their own continuous emission ??
  for (const e of subEntries ?? []) {
    const cp = e.pipeline as unknown as
      | {
          emitNode?: { count: number };
          uniforms: Record<string, { value: unknown }>;
          emitterPose?: {
            positionW: { value: THREE.Vector4 };
            wrapperQuat: { value: THREE.Vector4 };
            worldScale: { value: THREE.Vector3 };
          };
          allocatorCount?: number;
        }
      | undefined;
    if (!cp) continue;
    const cu = cp.uniforms;
    if (cu.delta) (cu.delta as { value: number }).value = delta;
    if (cu.deltaMs) (cu.deltaMs as { value: number }).value = delta * 1000;
    if (cu.nowMs) (cu.nowMs as { value: number }).value = now;
    // child uSystemSeed: written ONCE at pipeline creation (no per-frame seed).
    if (cu.gravityVelocity) {
      (cu.gravityVelocity as { value: THREE.Vector3 }).value.set(
        0,
        e.gravity,
        0
      );
    }
    if (e.noise) {
      if (cu.noiseStrength)
        (cu.noiseStrength as { value: number }).value = e.noise.strength;
      if (cu.noisePower)
        (cu.noisePower as { value: number }).value = e.noise.noisePower;
      if (cu.noiseFrequency)
        (cu.noiseFrequency as { value: number }).value = e.noise.frequency;
      if (cu.noisePositionAmount)
        (cu.noisePositionAmount as { value: number }).value =
          e.noise.positionAmount;
      if (cu.noiseRotationAmount)
        (cu.noiseRotationAmount as { value: number }).value =
          e.noise.rotationAmount;
      if (cu.noiseSizeAmount)
        (cu.noiseSizeAmount as { value: number }).value = e.noise.sizeAmount;
    }
    if (cu.fifoBase) (cu.fifoBase as { value: number }).value = fifoBase;
    // child-init pipeline system seed: written ONCE at creation.
    if (e.init.uniforms.fifoBase) e.init.uniforms.fifoBase.value = fifoBase;
    // Continuous rate-over-time of the child (bursts come from the events).
    let childEmit = 0;
    if (e.rate > 0) {
      e.acc += (e.rate * delta) / 1;
      childEmit = Math.floor(e.acc);
      if (childEmit > 0) e.acc -= childEmit;
    }
    const childCapacity = Math.max(2, (cp.allocatorCount ?? 2) - 1);
    if (childEmit > childCapacity) childEmit = childCapacity;
    if (cp.emitNode) cp.emitNode.count = Math.max(1, childEmit);
    if (cu.emitCount) (cu.emitCount as { value: number }).value = childEmit;
    // Child emitter pose: LOCAL relies on the object matrix; WORLD uses the
    // child's own transform on top of the (already world) event positions.
    const cpose = cp.emitterPose;
    if (cpose) {
      if (e.isWorld === 1) {
        cpose.positionW.value.set(
          e.position[0],
          e.position[1],
          e.position[2],
          1
        );
        cpose.wrapperQuat.value.set(e.quat[0], e.quat[1], e.quat[2], e.quat[3]);
        cpose.worldScale.value.set(e.scale[0], e.scale[1], e.scale[2]);
      } else {
        cpose.positionW.value.set(0, 0, 0, 0);
        cpose.wrapperQuat.value.set(0, 0, 0, 1);
        cpose.worldScale.value.set(1, 1, 1);
      }
    }
    const ip = e.init.uniforms as Record<string, { value: unknown }> & {
      positionW?: { value: THREE.Vector4 };
      wrapperQuat?: { value: THREE.Vector4 };
    };
    if (ip.positionW && ip.wrapperQuat) {
      if (e.isWorld === 1) {
        ip.positionW.value.set(e.position[0], e.position[1], e.position[2], 1);
        ip.wrapperQuat.value.set(e.quat[0], e.quat[1], e.quat[2], e.quat[3]);
      } else {
        ip.positionW.value.set(0, 0, 0, 0);
        ip.wrapperQuat.value.set(0, 0, 0, 1);
      }
    }
  }

  // Force-field / collision-plane records in the packed f32 uniform table.
  const ffInfo = pipeline.forceFieldInfo as {
    offset: number;
    countUniform: { value: number };
  } | null;
  const cInfo = (pipeline.collisionPlaneInfo ?? null) as {
    offset: number;
    countUniform: { value: number };
  } | null;
  if ((ffInfo || cInfo) && _tslMaterialFactory) {
    // Read-mostly uniform buffer (non-atomic f32): the CPU owns these record
    // regions, the kernels only load them.
    const cdArr = pipeline.buffers.packedData as Float32Array;
    const cdNode = pipeline.packedDataNode as unknown as {
      addUpdateRange(start: number, count: number): void;
      needsUpdate: boolean;
    };
    if (ffInfo && normalizedForceFields.length > 0) {
      const encFF = _tslMaterialFactory.encodeForceFieldsForGPU!(
        normalizedForceFields,
        generalData.particleSystemId,
        generalData.normalizedLifetimePercentage
      );
      let changedFF = false;
      for (let k = 0; k < encFF.length; k++)
        if (cdArr[ffInfo.offset + k] !== encFF[k]) {
          changedFF = true;
          break;
        }
      if (changedFF) {
        cdArr.set(encFF, ffInfo.offset);
        cdNode.addUpdateRange(ffInfo.offset, encFF.length);
        cdNode.needsUpdate = true;
      }
      ffInfo.countUniform.value = normalizedForceFields.length;
    }
    if (cInfo && normalizedCollisionPlanes.length > 0) {
      const encCP = _tslMaterialFactory.encodeCollisionPlanesForGPU!(
        normalizedCollisionPlanes
      );
      let changedCP = false;
      for (let k = 0; k < encCP.length; k++)
        if (cdArr[cInfo.offset + k] !== encCP[k]) {
          changedCP = true;
          break;
        }
      if (changedCP) {
        cdArr.set(encCP, cInfo.offset);
        cdNode.addUpdateRange(cInfo.offset, encCP.length);
        cdNode.needsUpdate = true;
      }
      cInfo.countUniform.value = normalizedCollisionPlanes.length;
    }
  }

  // First-frame buffer upload: storage attributes start with their CPU Float32Array
  // contents; after that the compute kernels own all changes on the GPU.
  const bufs = pipeline.buffers as unknown as Record<
    string,
    THREE.BufferAttribute
  >;
  let stamp = _lastUploadStampMap.get(bufs);
  if (stamp === undefined || stamp === 0) {
    for (const key of Object.keys(bufs)) {
      const a = bufs[key];
      if (a && 'needsUpdate' in a) a.needsUpdate = true;
    }
    _lastUploadStampMap.set(bufs, 1);
  } else {
    _lastUploadStampMap.set(bufs, stamp + 1);
  }

  // First-frame upload for the child pools, sub-emitter command buffers and
  // trail ribbon streams as well.
  for (const e of subEntries ?? []) {
    const cb = (
      e.pipeline as unknown as {
        buffers?: Record<string, THREE.BufferAttribute>;
      }
    )?.buffers;
    if (
      cb &&
      !_lastUploadStampMap.has(
        cb as unknown as Record<string, THREE.BufferAttribute>
      )
    ) {
      for (const key of Object.keys(cb)) {
        const a = cb[key];
        if (a && 'needsUpdate' in a) a.needsUpdate = true;
      }
      _lastUploadStampMap.set(
        cb as unknown as Record<string, THREE.BufferAttribute>,
        1
      );
    }
    const cmd = e.init.commandBuffer as THREE.BufferAttribute | undefined;
    if (cmd && 'needsUpdate' in cmd && !_cmdUploadSeen.has(cmd)) {
      cmd.needsUpdate = true;
      _cmdUploadSeen.add(cmd);
    }
  }
  const rb = (
    props as unknown as {
      ribbonBuffers?: Record<string, THREE.BufferAttribute>;
    }
  ).ribbonBuffers;
  if (rb && !_lastUploadStampMap.has(rb)) {
    for (const key of Object.keys(rb)) {
      const a = rb[key];
      if (a && 'needsUpdate' in a) a.needsUpdate = true;
    }
    _lastUploadStampMap.set(rb, 1);
  }

  // Emit + mark dispatch ready (editor does the actual `renderer.compute(...)`).
  props.computeDispatchReady = true;

  // Iteration counter (scalar) + FIFO ping-pong parity.
  props.iterationCount++;
  props.frameParity = (props.frameParity ?? 0) ^ 1;

  // Trail step (if the system uses TRAIL ? see `updateTrailGeometry` below).
  if (props.trailMesh) updateTrailGeometry(props, now);
};

// Tiny scratch for the world-space gravity transform above.
const _tmpQ1 = new THREE.Quaternion();
const _tmpV1 = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();
const _tmpM1 = new THREE.Matrix4();
/**
 * Evaluates a Catmull-Rom spline at parameter `t` (0..1) between points p1 and p2,
 * using p0 and p3 as control points. Writes result into `out`.
 */
const catmullRom = (
  out: Float32Array,
  outIdx: number,
  p0x: number,
  p0y: number,
  p0z: number,
  p1x: number,
  p1y: number,
  p1z: number,
  p2x: number,
  p2y: number,
  p2z: number,
  p3x: number,
  p3y: number,
  p3z: number,
  t: number
) => {
  const t2 = t * t;
  const t3 = t2 * t;
  out[outIdx] =
    0.5 *
    (2 * p1x +
      (-p0x + p2x) * t +
      (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 +
      (-p0x + 3 * p1x - 3 * p2x + p3x) * t3);
  out[outIdx + 1] =
    0.5 *
    (2 * p1y +
      (-p0y + p2y) * t +
      (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 +
      (-p0y + 3 * p1y - 3 * p2y + p3y) * t3);
  out[outIdx + 2] =
    0.5 *
    (2 * p1z +
      (-p0z + p2z) * t +
      (2 * p0z - 5 * p1z + 4 * p2z - p3z) * t2 +
      (-p0z + 3 * p1z - 3 * p2z + p3z) * t3);
};

/** Zeroes out a single trail vertex (both left+right sides). */
const clearTrailVertex = (
  vIdx: number,
  cIdx: number,
  aIdx: number,
  uvIdx: number,
  trailPosArr: Float32Array,
  trailNextArr: Float32Array,
  trailHalfWidthArr: Float32Array,
  trailUVArr: Float32Array,
  trailAlphaArr: Float32Array,
  trailColorArr: Float32Array,
  fallbackX: number,
  fallbackY: number,
  fallbackZ: number
) => {
  trailPosArr[vIdx] = fallbackX;
  trailPosArr[vIdx + 1] = fallbackY;
  trailPosArr[vIdx + 2] = fallbackZ;
  trailPosArr[vIdx + 3] = fallbackX;
  trailPosArr[vIdx + 4] = fallbackY;
  trailPosArr[vIdx + 5] = fallbackZ;
  trailNextArr[vIdx] = fallbackX;
  trailNextArr[vIdx + 1] = fallbackY;
  trailNextArr[vIdx + 2] = fallbackZ;
  trailNextArr[vIdx + 3] = fallbackX;
  trailNextArr[vIdx + 4] = fallbackY;
  trailNextArr[vIdx + 5] = fallbackZ;
  trailHalfWidthArr[aIdx] = 0;
  trailHalfWidthArr[aIdx + 1] = 0;
  trailUVArr[uvIdx] = 0;
  trailUVArr[uvIdx + 1] = 0;
  trailUVArr[uvIdx + 2] = 0;
  trailUVArr[uvIdx + 3] = 0;
  trailAlphaArr[aIdx] = 0;
  trailAlphaArr[aIdx + 1] = 0;
  trailColorArr[cIdx] = 0;
  trailColorArr[cIdx + 1] = 0;
  trailColorArr[cIdx + 2] = 0;
  trailColorArr[cIdx + 3] = 0;
  trailColorArr[cIdx + 4] = 0;
  trailColorArr[cIdx + 5] = 0;
  trailColorArr[cIdx + 6] = 0;
  trailColorArr[cIdx + 7] = 0;
};

/**
 * Writes a single trail ribbon vertex pair (left+right) into the typed arrays.
 */
const writeTrailVertex = (
  vIdx: number,
  cIdx: number,
  aIdx: number,
  uvIdx: number,
  hx: number,
  hy: number,
  hz: number,
  nx: number,
  ny: number,
  nz: number,
  halfWidth: number,
  t: number,
  alpha: number,
  fr: number,
  fg: number,
  fb: number,
  ca: number,
  trailPosArr: Float32Array,
  trailNextArr: Float32Array,
  trailHalfWidthArr: Float32Array,
  trailUVArr: Float32Array,
  trailAlphaArr: Float32Array,
  trailColorArr: Float32Array
) => {
  trailPosArr[vIdx] = hx;
  trailPosArr[vIdx + 1] = hy;
  trailPosArr[vIdx + 2] = hz;
  trailPosArr[vIdx + 3] = hx;
  trailPosArr[vIdx + 4] = hy;
  trailPosArr[vIdx + 5] = hz;
  trailNextArr[vIdx] = nx;
  trailNextArr[vIdx + 1] = ny;
  trailNextArr[vIdx + 2] = nz;
  trailNextArr[vIdx + 3] = nx;
  trailNextArr[vIdx + 4] = ny;
  trailNextArr[vIdx + 5] = nz;
  trailHalfWidthArr[aIdx] = halfWidth;
  trailHalfWidthArr[aIdx + 1] = halfWidth;
  trailUVArr[uvIdx] = 0;
  trailUVArr[uvIdx + 1] = t;
  trailUVArr[uvIdx + 2] = 1;
  trailUVArr[uvIdx + 3] = t;
  trailAlphaArr[aIdx] = alpha;
  trailAlphaArr[aIdx + 1] = alpha;
  trailColorArr[cIdx] = fr;
  trailColorArr[cIdx + 1] = fg;
  trailColorArr[cIdx + 2] = fb;
  trailColorArr[cIdx + 3] = ca;
  trailColorArr[cIdx + 4] = fr;
  trailColorArr[cIdx + 5] = fg;
  trailColorArr[cIdx + 6] = fb;
  trailColorArr[cIdx + 7] = ca;
};

// Scratch buffers reused each frame to avoid per-particle allocations
let _rawPoints: Float32Array | null = null;
let _rawPointsSize = 0;
let _smoothedPoints: Float32Array | null = null;
let _smoothedPointsSize = 0;
// Scratch buffer for connected ribbon particle indices (reused each frame).
// Uint32 so systems with more than 65535 particles don't silently wrap.
let _ribbonIndices: Uint32Array | null = null;
let _ribbonIndicesSize = 0;
let _ribbonCount = 0;

/**
 * Records current particle positions into the history ring buffer,
 * then rebuilds the triangle-strip ribbon geometry for all active particles.
 *
 * Supports:
 * - Adaptive sampling (minVertexDistance): frame-rate independent trail density
 * - Max time (maxTime): time-based trail expiry
 * - Catmull-Rom smoothing: eliminates sharp kinks between samples
 * - Twist prevention: consistent ribbon orientation during rapid direction changes
 * - Connected ribbons (ribbonId): multiple particles forming a single ribbon
 */
const updateTrailGeometry = (props: ParticleSystemInstance, now: number) => {
  const {
    generalData,
    trailPositionAttr,
    trailAlphaAttr,
    trailColorAttr,
    trailNextAttr: trailNextAttrCached,
    trailHalfWidthAttr: trailHalfWidthAttrCached,
    trailUVAttr: trailUVAttrCached,
    trailWidthCurveFn,
    trailOpacityCurveFn,
    trailColorOverTrailFns,
    trailConfig,
    mappedAttributes: ma,
  } = props;

  if (
    !trailPositionAttr ||
    !trailAlphaAttr ||
    !trailColorAttr ||
    !trailNextAttrCached ||
    !trailHalfWidthAttrCached ||
    !trailUVAttrCached ||
    !trailWidthCurveFn ||
    !trailOpacityCurveFn ||
    !trailConfig ||
    !generalData.positionHistory ||
    !generalData.positionHistoryIndex ||
    !generalData.positionHistoryCount
  )
    return;

  const trailLength = trailConfig.length;
  const positionHistory = generalData.positionHistory;
  const historyIndex = generalData.positionHistoryIndex;
  const historyCount = generalData.positionHistoryCount;
  const sampleTimes = generalData.trailSampleTimes;
  const lastSampledPos = generalData.trailLastSampledPosition;
  const prevNormal = generalData.trailPrevNormal;
  const minVertexDist = trailConfig.minVertexDistance;
  const minVertexDistSq = minVertexDist * minVertexDist;
  const maxTime = trailConfig.maxTime;
  const maxTimeMs = maxTime * 1000;
  const useSmoothing = trailConfig.smoothing;
  const subdivisions = trailConfig.smoothingSubdivisions;
  const useTwistPrevention = trailConfig.twistPrevention;
  const ribbonId = trailConfig.ribbonId;

  const trailScalarArr = props.scalarArray;
  const positionArr = ma.position.array;
  // Vertex-buffer fill counts from the previous frame ??? cleared slots stay
  // cleared (zero alpha/half-width), so re-clearing them every frame is
  // redundant work proportional to maxParticles ?? trailLength.
  const prevFilled = generalData.trailPrevFilledCount;

  const trailPosArr = trailPositionAttr.array as Float32Array;
  const trailAlphaArr = trailAlphaAttr.array as Float32Array;
  const trailColorArr = trailColorAttr.array as Float32Array;
  const trailNextArr = trailNextAttrCached.array as Float32Array;
  const trailUVArr = trailUVAttrCached.array as Float32Array;
  const trailHalfWidthArr = trailHalfWidthAttrCached.array as Float32Array;
  const verticesPerParticle = trailLength * 2;
  const hwm = generalData.highWaterIndex;
  const creationTimesLength = hwm > 0 ? hwm : generalData.creationTimes.length;
  let hasUpdates = false;

  // --- Connected Ribbons: collect particles sharing the same ribbonId ---
  const useRibbon = ribbonId !== undefined;
  let ribbonLeader = -1;
  if (useRibbon) {
    // Pre-allocate scratch buffer for ribbon indices
    if (!_ribbonIndices || _ribbonIndicesSize < creationTimesLength) {
      _ribbonIndices = new Uint32Array(creationTimesLength);
      _ribbonIndicesSize = creationTimesLength;
    }
    _ribbonCount = 0;
    for (let i = 0; i < creationTimesLength; i++) {
      if (trailScalarArr[i * SCALAR_STRIDE + S_IS_ACTIVE])
        _ribbonIndices[_ribbonCount++] = i;
    }
    // Insertion sort by creation time (typically nearly-sorted, O(n) best case)
    for (let i = 1; i < _ribbonCount; i++) {
      const key = _ribbonIndices[i];
      const keyTime = generalData.creationTimes[key];
      let j = i - 1;
      while (j >= 0 && generalData.creationTimes[_ribbonIndices[j]] > keyTime) {
        _ribbonIndices[j + 1] = _ribbonIndices[j];
        j--;
      }
      _ribbonIndices[j + 1] = key;
    }
    if (_ribbonCount > 0) ribbonLeader = _ribbonIndices[0];
  }

  for (let index = 0; index < creationTimesLength; index++) {
    const vertBase = index * verticesPerParticle;

    if (trailScalarArr[index * SCALAR_STRIDE + S_IS_ACTIVE]) {
      // Skip individual trail build for non-leader ribbon particles
      // (the leader's trail will be built by the connected ribbon section)
      if (useRibbon && _ribbonCount >= 2 && index !== ribbonLeader) {
        // Still record position history for this particle (needed for sampling)
        const posIdx = index * 3;
        const px = positionArr[posIdx];
        const py = positionArr[posIdx + 1];
        const pz = positionArr[posIdx + 2];
        const histBase = (index * trailLength + historyIndex[index]) * 3;
        positionHistory[histBase] = px;
        positionHistory[histBase + 1] = py;
        positionHistory[histBase + 2] = pz;
        if (sampleTimes) {
          sampleTimes[index * trailLength + historyIndex[index]] = now;
        }
        historyIndex[index] = (historyIndex[index] + 1) % trailLength;
        if (historyCount[index] < trailLength) historyCount[index]++;
        continue;
      }
      hasUpdates = true;
      const posIdx = index * 3;
      const px = positionArr[posIdx];
      const py = positionArr[posIdx + 1];
      const pz = positionArr[posIdx + 2];

      // --- Adaptive Sampling: only push a new sample if distance threshold met ---
      let shouldSample = true;
      if (minVertexDist > 0 && lastSampledPos && historyCount[index] > 0) {
        const lsIdx = index * 3;
        const dx = px - lastSampledPos[lsIdx];
        const dy = py - lastSampledPos[lsIdx + 1];
        const dz = pz - lastSampledPos[lsIdx + 2];
        if (dx * dx + dy * dy + dz * dz < minVertexDistSq) {
          shouldSample = false;
        }
      }

      if (shouldSample) {
        // Record the sample
        const histBase = (index * trailLength + historyIndex[index]) * 3;
        positionHistory[histBase] = px;
        positionHistory[histBase + 1] = py;
        positionHistory[histBase + 2] = pz;

        // Record timestamp for maxTime
        if (sampleTimes) {
          sampleTimes[index * trailLength + historyIndex[index]] = now;
        }

        historyIndex[index] = (historyIndex[index] + 1) % trailLength;
        if (historyCount[index] < trailLength) historyCount[index]++;

        // Update last sampled position
        if (lastSampledPos) {
          const lsIdx = index * 3;
          lastSampledPos[lsIdx] = px;
          lastSampledPos[lsIdx + 1] = py;
          lastSampledPos[lsIdx + 2] = pz;
        }
      }

      // --- MaxTime: determine effective count (expire old segments) ---
      let rawCount = historyCount[index];
      let effectiveCount = rawCount;
      if (maxTime > 0 && sampleTimes && rawCount > 0) {
        const sampleBase = index * trailLength;
        effectiveCount = 0;
        for (let s = 0; s < rawCount; s++) {
          const sampleSlot =
            (historyIndex[index] - 1 - s + trailLength * 2) % trailLength;
          const age = now - sampleTimes[sampleBase + sampleSlot];
          if (age <= maxTimeMs) {
            effectiveCount++;
          } else {
            break; // older samples are even older, stop
          }
        }
      }

      const count = effectiveCount;
      const ribbonWidth = trailConfig.width;

      // Get particle color from interleaved scalar buffer
      const trailBase = index * SCALAR_STRIDE;
      const cr = trailScalarArr[trailBase + S_COLOR_R];
      const cg = trailScalarArr[trailBase + S_COLOR_G];
      const cb = trailScalarArr[trailBase + S_COLOR_B];
      const ca = trailScalarArr[trailBase + S_COLOR_A];

      const ringOff = index * trailLength * 3;

      // --- Collect raw history points for this particle ---
      // We need them for both smoothing and the ribbon build.
      // rawPts: flat array of [x, y, z, x, y, z, ...] for count entries
      // rawPts[0..2] = head (most recent), rawPts[(count-1)*3..(count-1)*3+2] = tail
      const rawPtsSize = count * 3;
      // Reuse a scratch float array for raw points
      if (!_rawPoints || _rawPointsSize < rawPtsSize) {
        _rawPoints = new Float32Array(rawPtsSize);
        _rawPointsSize = rawPtsSize;
      }
      const rawPts = _rawPoints;
      for (let s = 0; s < count; s++) {
        const histSlot =
          ((historyIndex[index] - 1 - s + trailLength * 2) % trailLength) * 3 +
          ringOff;
        rawPts[s * 3] = positionHistory[histSlot];
        rawPts[s * 3 + 1] = positionHistory[histSlot + 1];
        rawPts[s * 3 + 2] = positionHistory[histSlot + 2];
      }

      // --- Catmull-Rom Smoothing ---
      let finalPts: Float32Array;
      let finalCount: number;

      if (useSmoothing && count >= 3) {
        // Interpolate between each pair of raw points with subdivisions
        const segmentCount = count - 1;
        finalCount = segmentCount * subdivisions + 1;
        const neededSize = finalCount * 3;

        // Resize global scratch buffer if needed
        if (!_smoothedPoints || _smoothedPointsSize < neededSize) {
          _smoothedPoints = new Float32Array(neededSize);
          _smoothedPointsSize = neededSize;
        }
        finalPts = _smoothedPoints;

        for (let seg = 0; seg < segmentCount; seg++) {
          // Control points: p0, p1, p2, p3
          const i0 = Math.max(0, seg - 1);
          const i1 = seg;
          const i2 = Math.min(count - 1, seg + 1);
          const i3 = Math.min(count - 1, seg + 2);

          const p0x = rawPts[i0 * 3],
            p0y = rawPts[i0 * 3 + 1],
            p0z = rawPts[i0 * 3 + 2];
          const p1x = rawPts[i1 * 3],
            p1y = rawPts[i1 * 3 + 1],
            p1z = rawPts[i1 * 3 + 2];
          const p2x = rawPts[i2 * 3],
            p2y = rawPts[i2 * 3 + 1],
            p2z = rawPts[i2 * 3 + 2];
          const p3x = rawPts[i3 * 3],
            p3y = rawPts[i3 * 3 + 1],
            p3z = rawPts[i3 * 3 + 2];

          for (let sub = 0; sub < subdivisions; sub++) {
            const t = sub / subdivisions;
            const outIdx = (seg * subdivisions + sub) * 3;
            catmullRom(
              finalPts,
              outIdx,
              p0x,
              p0y,
              p0z,
              p1x,
              p1y,
              p1z,
              p2x,
              p2y,
              p2z,
              p3x,
              p3y,
              p3z,
              t
            );
          }
        }
        // Last point = last raw point
        const lastOutIdx = (finalCount - 1) * 3;
        finalPts[lastOutIdx] = rawPts[(count - 1) * 3];
        finalPts[lastOutIdx + 1] = rawPts[(count - 1) * 3 + 1];
        finalPts[lastOutIdx + 2] = rawPts[(count - 1) * 3 + 2];
      } else {
        finalPts = rawPts;
        finalCount = count;
      }

      // Limit final count to the number of slots we can fill
      if (finalCount > trailLength) finalCount = trailLength;

      // Collapse degenerate segments: when two consecutive smoothed points are
      // nearly identical the shader tangent becomes zero, producing distorted
      // "squished" ribbon quads. Shift such points to the next distinct neighbor.
      if (useSmoothing && finalCount >= 2) {
        const MIN_SEG_DIST_SQ = 0.0001 * 0.0001;
        for (let d = 1; d < finalCount; d++) {
          const pi = (d - 1) * 3;
          const ci = d * 3;
          const dx = finalPts[ci] - finalPts[pi];
          const dy = finalPts[ci + 1] - finalPts[pi + 1];
          const dz = finalPts[ci + 2] - finalPts[pi + 2];
          if (dx * dx + dy * dy + dz * dz < MIN_SEG_DIST_SQ) {
            // Snap to previous point ??? the shader will get a near-zero tangent
            // but the vertex pair collapses to the same position, hiding it
            finalPts[ci] = finalPts[pi];
            finalPts[ci + 1] = finalPts[pi + 1];
            finalPts[ci + 2] = finalPts[pi + 2];
          }
        }
      }

      // --- Build ribbon vertices ---
      const prevFilledSlots = prevFilled ? prevFilled[index] : trailLength;
      if (prevFilled) prevFilled[index] = finalCount;
      for (let s = 0; s < trailLength; s++) {
        const vIdx = (vertBase + s * 2) * 3;
        const cIdx = (vertBase + s * 2) * 4;
        const aIdx = vertBase + s * 2;
        const uvIdxBase = (vertBase + s * 2) * 2;

        if (s >= finalCount) {
          // Slots at or beyond the previous fill count are already cleared.
          if (s >= prevFilledSlots) break;
          clearTrailVertex(
            vIdx,
            cIdx,
            aIdx,
            uvIdxBase,
            trailPosArr,
            trailNextArr,
            trailHalfWidthArr,
            trailUVArr,
            trailAlphaArr,
            trailColorArr,
            px,
            py,
            pz
          );
          continue;
        }

        const hx = finalPts[s * 3];
        const hy = finalPts[s * 3 + 1];
        const hz = finalPts[s * 3 + 2];

        // Compute an averaged tangent direction for the shader.
        // At interior points we average the forward and backward segment
        // directions so the billboard plane transitions smoothly through
        // bends instead of snapping per-segment.
        let nx: number, ny: number, nz: number;
        if (s > 0 && s < finalCount - 1) {
          // Interior: average of (prev???current) and (current???next)
          const px2 = finalPts[(s - 1) * 3];
          const py2 = finalPts[(s - 1) * 3 + 1];
          const pz2 = finalPts[(s - 1) * 3 + 2];
          const nx2 = finalPts[(s + 1) * 3];
          const ny2 = finalPts[(s + 1) * 3 + 1];
          const nz2 = finalPts[(s + 1) * 3 + 2];
          // Averaged tangent = (current - prev) + (next - current) = next - prev
          const atx = nx2 - px2;
          const aty = ny2 - py2;
          const atz = nz2 - pz2;
          const atLen = Math.sqrt(atx * atx + aty * aty + atz * atz);
          if (atLen > 0.0001) {
            // trailNext = current + normalized averaged tangent (shader computes tangent as trailNext - position)
            nx = hx + atx / atLen;
            ny = hy + aty / atLen;
            nz = hz + atz / atLen;
          } else {
            nx = finalPts[(s + 1) * 3];
            ny = finalPts[(s + 1) * 3 + 1];
            nz = finalPts[(s + 1) * 3 + 2];
          }
        } else if (s < finalCount - 1) {
          // Head: use forward direction
          nx = finalPts[(s + 1) * 3];
          ny = finalPts[(s + 1) * 3 + 1];
          nz = finalPts[(s + 1) * 3 + 2];
        } else if (finalCount >= 2) {
          // Tail: reuse the direction from the previous segment so the
          // ribbon end keeps the same orientation as the last real segment
          // instead of collapsing when the tangent aligns with the Y axis.
          const prevX = finalPts[(s - 1) * 3];
          const prevY = finalPts[(s - 1) * 3 + 1];
          const prevZ = finalPts[(s - 1) * 3 + 2];
          nx = hx + (hx - prevX);
          ny = hy + (hy - prevY);
          nz = hz + (hz - prevZ);
        } else {
          // Single point: nudge to avoid zero tangent
          nx = hx;
          ny = hy + 0.001;
          nz = hz;
        }

        // Trail percentage (0=head, 1=tail)
        const t = finalCount > 1 ? s / (finalCount - 1) : 0;

        // --- MaxTime: apply additional age-based fade ---
        let timeFade = 1.0;
        if (maxTime > 0 && sampleTimes && effectiveCount > 0) {
          // Map the current smoothed vertex back to the raw sample timeline.
          // When smoothing is active we interpolate between the two bracketing
          // raw samples' timestamps so the fade is smooth instead of stepping.
          const sampleBase = index * trailLength;
          if (useSmoothing && rawCount >= 2) {
            const rawF = (s / Math.max(finalCount - 1, 1)) * (rawCount - 1);
            const rawLo = Math.min(Math.floor(rawF), rawCount - 1);
            const rawHi = Math.min(rawLo + 1, rawCount - 1);
            const frac = rawF - rawLo;
            const slotLo =
              (historyIndex[index] - 1 - rawLo + trailLength * 2) % trailLength;
            const slotHi =
              (historyIndex[index] - 1 - rawHi + trailLength * 2) % trailLength;
            const ageLo = now - sampleTimes[sampleBase + slotLo];
            const ageHi = now - sampleTimes[sampleBase + slotHi];
            const age = ageLo + (ageHi - ageLo) * frac;
            timeFade = 1.0 - Math.min(age / maxTimeMs, 1.0);
          } else {
            const rawS = Math.min(s, rawCount - 1);
            const sampleSlot =
              (historyIndex[index] - 1 - rawS + trailLength * 2) % trailLength;
            const age = now - sampleTimes[sampleBase + sampleSlot];
            timeFade = 1.0 - Math.min(age / maxTimeMs, 1.0);
          }
        }

        const widthScale = trailWidthCurveFn(t);
        const opacityScale = trailOpacityCurveFn(t);
        const halfWidth = ribbonWidth * widthScale * 0.5;
        const alpha = ca * opacityScale * timeFade;

        const fr = trailColorOverTrailFns
          ? cr * trailColorOverTrailFns.r(t)
          : cr;
        const fg = trailColorOverTrailFns
          ? cg * trailColorOverTrailFns.g(t)
          : cg;
        const fb = trailColorOverTrailFns
          ? cb * trailColorOverTrailFns.b(t)
          : cb;

        writeTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          hx,
          hy,
          hz,
          nx,
          ny,
          nz,
          halfWidth,
          t,
          alpha,
          fr,
          fg,
          fb,
          ca,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr
        );
      }

      // --- Twist Prevention ---
      // After building the ribbon, ensure ribbon normals are consistent.
      // We compare the implied normal direction of consecutive segments and
      // flip if the dot product with the previous frame's normal is negative.
      if (useTwistPrevention && prevNormal && finalCount >= 2) {
        const nIdx = index * 3;
        // Compute current head tangent
        const tx = finalPts[3] - finalPts[0];
        const ty = finalPts[4] - finalPts[1];
        const tz = finalPts[5] - finalPts[2];
        const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (tLen > 0.0001) {
          const ntx = tx / tLen;
          const nty = ty / tLen;
          const ntz = tz / tLen;
          // Use a consistent up vector to compute a reference normal
          let upx = 0,
            upy = 1,
            upz = 0;
          const dot = ntx * upx + nty * upy + ntz * upz;
          if (Math.abs(dot) > 0.999) {
            upx = 1;
            upy = 0;
            upz = 0;
          }
          // cross(tangent, up) = normal
          let cnx = nty * upz - ntz * upy;
          let cny = ntz * upx - ntx * upz;
          let cnz = ntx * upy - nty * upx;
          const cnLen = Math.sqrt(cnx * cnx + cny * cny + cnz * cnz);
          if (cnLen > 0.0001) {
            cnx /= cnLen;
            cny /= cnLen;
            cnz /= cnLen;
          }

          // Check dot product with previous normal ??? if negative, flip
          const prevNx = prevNormal[nIdx];
          const prevNy = prevNormal[nIdx + 1];
          const prevNz = prevNormal[nIdx + 2];
          const hasPrev = prevNx !== 0 || prevNy !== 0 || prevNz !== 0;
          if (hasPrev) {
            const normalDot = cnx * prevNx + cny * prevNy + cnz * prevNz;
            if (normalDot < 0) {
              // Flip all ribbon offsets for this particle by swapping left/right half-widths
              for (let s = 0; s < Math.min(finalCount, trailLength); s++) {
                const aIdx = vertBase + s * 2;
                const hw = trailHalfWidthArr[aIdx];
                trailHalfWidthArr[aIdx] = -hw;
                trailHalfWidthArr[aIdx + 1] = -hw;
              }
              // Also flip the stored normal for next frame
              cnx = -cnx;
              cny = -cny;
              cnz = -cnz;
            }
          }

          // Store current normal for next frame
          prevNormal[nIdx] = cnx;
          prevNormal[nIdx + 1] = cny;
          prevNormal[nIdx + 2] = cnz;
        }
      }
    } else if (
      historyCount[index] > 0 ||
      (prevFilled && prevFilled[index] > 0)
    ) {
      // Particle just became inactive ??? collapse ribbon and clear history once
      hasUpdates = true;
      historyCount[index] = 0;
      historyIndex[index] = 0;
      const clearSlots = prevFilled ? prevFilled[index] : trailLength;
      if (prevFilled) prevFilled[index] = 0;
      for (let s = 0; s < clearSlots; s++) {
        const vIdx = (vertBase + s * 2) * 3;
        const cIdx = (vertBase + s * 2) * 4;
        const aIdx = vertBase + s * 2;
        const uvIdxBase = (vertBase + s * 2) * 2;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
      }
    }
  }

  // --- Connected Ribbons: chain particle positions with Catmull-Rom interpolation ---
  if (useRibbon && _ribbonCount >= 2 && _ribbonIndices) {
    hasUpdates = true;
    const leader = _ribbonIndices[0];
    const leaderVertBase = leader * verticesPerParticle;

    // The ribbon uses each particle's current position as a control point,
    // then fills `trailLength` vertices by Catmull-Rom interpolation between them.
    // This produces a smooth, continuous ribbon through all particle positions.
    const controlCount = _ribbonCount;
    const filledCount = Math.min(
      trailLength,
      Math.max(controlCount * 4, controlCount)
    );
    const chainSize = filledCount * 3;
    if (!_rawPoints || _rawPointsSize < chainSize) {
      _rawPoints = new Float32Array(chainSize);
      _rawPointsSize = chainSize;
    }

    if (controlCount === 2) {
      // Only 2 particles: linearly interpolate between them
      const p0Idx = _ribbonIndices[0] * 3;
      const p1Idx = _ribbonIndices[1] * 3;
      for (let i = 0; i < filledCount; i++) {
        const t = i / (filledCount - 1);
        _rawPoints[i * 3] =
          positionArr[p0Idx] + t * (positionArr[p1Idx] - positionArr[p0Idx]);
        _rawPoints[i * 3 + 1] =
          positionArr[p0Idx + 1] +
          t * (positionArr[p1Idx + 1] - positionArr[p0Idx + 1]);
        _rawPoints[i * 3 + 2] =
          positionArr[p0Idx + 2] +
          t * (positionArr[p1Idx + 2] - positionArr[p0Idx + 2]);
      }
    } else {
      // 3+ particles: Catmull-Rom interpolation through all control points
      const segments = controlCount - 1;
      const ptsPerSeg = Math.max(1, Math.floor((filledCount - 1) / segments));
      let wi = 0;
      for (let seg = 0; seg < segments && wi < filledCount; seg++) {
        const i0 = Math.max(0, seg - 1);
        const i1 = seg;
        const i2 = Math.min(controlCount - 1, seg + 1);
        const i3 = Math.min(controlCount - 1, seg + 2);
        const p0i = _ribbonIndices[i0] * 3;
        const p1i = _ribbonIndices[i1] * 3;
        const p2i = _ribbonIndices[i2] * 3;
        const p3i = _ribbonIndices[i3] * 3;
        const subCount = seg === segments - 1 ? filledCount - wi : ptsPerSeg;
        for (let sub = 0; sub < subCount && wi < filledCount; sub++) {
          const t = sub / subCount;
          catmullRom(
            _rawPoints,
            wi * 3,
            positionArr[p0i],
            positionArr[p0i + 1],
            positionArr[p0i + 2],
            positionArr[p1i],
            positionArr[p1i + 1],
            positionArr[p1i + 2],
            positionArr[p2i],
            positionArr[p2i + 1],
            positionArr[p2i + 2],
            positionArr[p3i],
            positionArr[p3i + 1],
            positionArr[p3i + 2],
            t
          );
          wi++;
        }
      }
      // Ensure last point is the last particle's position
      if (wi > 0) {
        const lastPIdx = _ribbonIndices[controlCount - 1] * 3;
        _rawPoints[(wi - 1) * 3] = positionArr[lastPIdx];
        _rawPoints[(wi - 1) * 3 + 1] = positionArr[lastPIdx + 1];
        _rawPoints[(wi - 1) * 3 + 2] = positionArr[lastPIdx + 2];
      }
    }

    const leaderBase = leader * SCALAR_STRIDE;
    const leaderCr = trailScalarArr[leaderBase + S_COLOR_R];
    const leaderCg = trailScalarArr[leaderBase + S_COLOR_G];
    const leaderCb = trailScalarArr[leaderBase + S_COLOR_B];
    const leaderCa = trailScalarArr[leaderBase + S_COLOR_A];

    const leaderPrevFilled = prevFilled ? prevFilled[leader] : trailLength;
    if (prevFilled) prevFilled[leader] = filledCount;
    for (let s = 0; s < trailLength; s++) {
      const vIdx = (leaderVertBase + s * 2) * 3;
      const cIdx = (leaderVertBase + s * 2) * 4;
      const aIdx = leaderVertBase + s * 2;
      const uvIdxBase = (leaderVertBase + s * 2) * 2;

      if (s >= filledCount) {
        // Slots at or beyond the previous fill count are already cleared.
        if (s >= leaderPrevFilled) break;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
        continue;
      }

      const ptIdx = s * 3;
      const ptx = _rawPoints[ptIdx];
      const pty = _rawPoints[ptIdx + 1];
      const ptz = _rawPoints[ptIdx + 2];

      // Averaged tangent for interior points
      let nx: number, ny: number, nz: number;
      if (s > 0 && s < filledCount - 1) {
        const px2 = _rawPoints[(s - 1) * 3];
        const py2 = _rawPoints[(s - 1) * 3 + 1];
        const pz2 = _rawPoints[(s - 1) * 3 + 2];
        const nx2 = _rawPoints[(s + 1) * 3];
        const ny2 = _rawPoints[(s + 1) * 3 + 1];
        const nz2 = _rawPoints[(s + 1) * 3 + 2];
        const atx = nx2 - px2;
        const aty = ny2 - py2;
        const atz = nz2 - pz2;
        const atLen = Math.sqrt(atx * atx + aty * aty + atz * atz);
        if (atLen > 0.0001) {
          nx = ptx + atx / atLen;
          ny = pty + aty / atLen;
          nz = ptz + atz / atLen;
        } else {
          nx = _rawPoints[(s + 1) * 3];
          ny = _rawPoints[(s + 1) * 3 + 1];
          nz = _rawPoints[(s + 1) * 3 + 2];
        }
      } else if (s < filledCount - 1) {
        nx = _rawPoints[(s + 1) * 3];
        ny = _rawPoints[(s + 1) * 3 + 1];
        nz = _rawPoints[(s + 1) * 3 + 2];
      } else if (filledCount >= 2) {
        // Tail: reuse previous segment direction
        const prevX = _rawPoints[(s - 1) * 3];
        const prevY = _rawPoints[(s - 1) * 3 + 1];
        const prevZ = _rawPoints[(s - 1) * 3 + 2];
        nx = ptx + (ptx - prevX);
        ny = pty + (pty - prevY);
        nz = ptz + (ptz - prevZ);
      } else {
        nx = ptx;
        ny = pty + 0.001;
        nz = ptz;
      }

      const t = filledCount > 1 ? s / (filledCount - 1) : 0;

      // --- MaxTime: apply age-based fade to connected ribbon ---
      let ribbonTimeFade = 1.0;
      if (maxTime > 0 && controlCount >= 2) {
        // Map the vertex to the nearest control point(s) and use
        // their creation times to compute an interpolated age.
        const ctrlF = t * (controlCount - 1);
        const ctrlLo = Math.min(Math.floor(ctrlF), controlCount - 1);
        const ctrlHi = Math.min(ctrlLo + 1, controlCount - 1);
        const frac = ctrlF - ctrlLo;
        const ageLo = now - generalData.creationTimes[_ribbonIndices[ctrlLo]];
        const ageHi = now - generalData.creationTimes[_ribbonIndices[ctrlHi]];
        const age = ageLo + (ageHi - ageLo) * frac;
        ribbonTimeFade = 1.0 - Math.min(age / maxTimeMs, 1.0);
      }

      const widthScale = trailWidthCurveFn(t);
      const opacityScale = trailOpacityCurveFn(t);
      const halfWidth = trailConfig.width * widthScale * 0.5;
      const alpha = leaderCa * opacityScale * ribbonTimeFade;
      const fr = trailColorOverTrailFns
        ? leaderCr * trailColorOverTrailFns.r(t)
        : leaderCr;
      const fg = trailColorOverTrailFns
        ? leaderCg * trailColorOverTrailFns.g(t)
        : leaderCg;
      const fb = trailColorOverTrailFns
        ? leaderCb * trailColorOverTrailFns.b(t)
        : leaderCb;

      writeTrailVertex(
        vIdx,
        cIdx,
        aIdx,
        uvIdxBase,
        ptx,
        pty,
        ptz,
        nx,
        ny,
        nz,
        halfWidth,
        t,
        alpha,
        fr,
        fg,
        fb,
        leaderCa,
        trailPosArr,
        trailNextArr,
        trailHalfWidthArr,
        trailUVArr,
        trailAlphaArr,
        trailColorArr
      );
    }

    // --- Twist Prevention for connected ribbon (applied to leader) ---
    if (useTwistPrevention && prevNormal && filledCount >= 2) {
      const nIdx = leader * 3;
      const tx = _rawPoints[3] - _rawPoints[0];
      const ty = _rawPoints[4] - _rawPoints[1];
      const tz = _rawPoints[5] - _rawPoints[2];
      const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
      if (tLen > 0.0001) {
        const ntx = tx / tLen;
        const nty = ty / tLen;
        const ntz = tz / tLen;
        let upx = 0,
          upy = 1,
          upz = 0;
        const dot = ntx * upx + nty * upy + ntz * upz;
        if (Math.abs(dot) > 0.999) {
          upx = 1;
          upy = 0;
          upz = 0;
        }
        let cnx = nty * upz - ntz * upy;
        let cny = ntz * upx - ntx * upz;
        let cnz = ntx * upy - nty * upx;
        const cnLen = Math.sqrt(cnx * cnx + cny * cny + cnz * cnz);
        if (cnLen > 0.0001) {
          cnx /= cnLen;
          cny /= cnLen;
          cnz /= cnLen;
        }
        const prevNx = prevNormal[nIdx];
        const prevNy = prevNormal[nIdx + 1];
        const prevNz = prevNormal[nIdx + 2];
        const hasPrev = prevNx !== 0 || prevNy !== 0 || prevNz !== 0;
        if (hasPrev) {
          const normalDot = cnx * prevNx + cny * prevNy + cnz * prevNz;
          if (normalDot < 0) {
            for (let s = 0; s < Math.min(filledCount, trailLength); s++) {
              const aIdx = leaderVertBase + s * 2;
              const hw = trailHalfWidthArr[aIdx];
              trailHalfWidthArr[aIdx] = -hw;
              trailHalfWidthArr[aIdx + 1] = -hw;
            }
            cnx = -cnx;
            cny = -cny;
            cnz = -cnz;
          }
        }
        prevNormal[nIdx] = cnx;
        prevNormal[nIdx + 1] = cny;
        prevNormal[nIdx + 2] = cnz;
      }
    }

    // Clear non-leader ribbon particles' trail vertices (only the slots that
    // were actually filled ??? already-cleared buffers are skipped entirely)
    for (let ri = 1; ri < _ribbonCount; ri++) {
      const pIdx = _ribbonIndices[ri];
      const pVertBase = pIdx * verticesPerParticle;
      const pClearSlots = prevFilled ? prevFilled[pIdx] : trailLength;
      if (prevFilled) prevFilled[pIdx] = 0;
      for (let s = 0; s < pClearSlots; s++) {
        const vIdx = (pVertBase + s * 2) * 3;
        const cIdx = (pVertBase + s * 2) * 4;
        const aIdx = pVertBase + s * 2;
        const uvIdxBase = (pVertBase + s * 2) * 2;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
      }
    }
  }

  if (hasUpdates) {
    trailPositionAttr.needsUpdate = true;
    trailAlphaAttr.needsUpdate = true;
    trailColorAttr.needsUpdate = true;
    trailNextAttrCached.needsUpdate = true;
    trailHalfWidthAttrCached.needsUpdate = true;
    trailUVAttrCached.needsUpdate = true;
  }
};

export const updateParticleSystems = (cycleData: CycleData) => {
  createdParticleSystems.forEach((props) =>
    updateParticleSystemInstance(props, cycleData)
  );
};
