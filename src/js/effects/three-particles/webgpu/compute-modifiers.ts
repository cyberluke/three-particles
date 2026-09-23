/**
 * GPU-owned WebGPU compute kernels for @cyberluke/three-particles 4.0.0.
 *
 * Per-particle state lives ONLY in GPU storage buffers. The CPU never
 * walks slots; it only writes 12 scalar uniforms per frame and issues
 * two `renderer.compute(...)` calls (emission, then simulation).
 *
 * Storage pool (8 storage bindings = the WebGPU compute guarantee
 * `minStorageBuffersPerStage = 8`; read-mostly tables live in uniform
 * bindings, which are counted separately, `minUniformBuffersPerStage = 4`):
 *   1. position          : vec4 * maxParticles   (xyz, w = padding)
 *   2. velocity          : vec4 * maxParticles   (xyz, w = padding)
 *   3. color             : vec4 * maxParticles   (r, g, b, a)
 *   4. particleState     : vec4 * maxParticles   (lifetime, size, rotation, startFrame)
 *   5. startValues       : vec4 * maxParticles   (startLife, startSize, startOpac, startColorR)
 *   6. startColorsExt    : vec4 * maxParticles   (startColorG, startColorB, rotSpeed, noiseOffset)
 *   7. orbitalIsActive   : vec4 * maxParticles   (orbitalOffset.xyz, isActive)
 *   8. allocator         : atomic<u32> * (maxParticles + 1)   (UINT, not f32)
 *        [0]                : freeCount (stack size)
 *        [1 .. maxParticles]: free slot ids
 *
 * One extra UNIFORM binding carries the read-mostly f32 tables (non-atomic,
 * so the curve / force-field / collision hot path stays a plain `f32` load):
 *        [0 .. curveLen-1]              baked curve samples
 *        [curveLen ...]                 force-field records      (optional)
 *        [...]                          collision-plane records (optional)
 *
 * The allocator is managed entirely on the GPU with integer `atomicAdd` /
 * `atomicSub` / `atomicLoad` / `atomicStore`; the CPU initialises it once at
 * pipeline creation and never touches it again.
 *
 * @module
 */
import { Vector3, Vector4 } from 'three';
import {
  Fn,
  float,
  vec3,
  vec4,
  uint as tuint,
  mix,
  floor,
  fract,
  sin,
  cos,
  sqrt,
  min as tslMin,
  max as tslMax,
  rand,
  storage,
  buffer,
  atomicAdd,
  atomicSub,
  atomicLoad,
  atomicStore,
  instanceIndex,
  uniform,
  If,
  compute,
  type ShaderNodeObject,
  type Node,
} from 'three/tsl';
import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import { sRGBToLinear } from '../color-utils.js';
import {
  COLLISION_PLANE_DATA_SIZE,
  createCollisionPlaneTSL,
} from './compute-collision-planes.js';
import {
  FORCE_FIELD_DATA_SIZE,
  createForceFieldTSL,
} from './compute-force-fields.js';
import {
  CURVE_RESOLUTION,
  bakeCurve,
  type BakedCurveMap,
} from './curve-bake.js';
import { snoise3D } from './tsl-noise.js';

// ??? Buffer layout constants ????????????????????????????????????????????????
// Each `vec4 * maxParticles` slot is stored as 4 floats in row order. Indices
// match the "1..8 = binding order" numbering above.
export const S_PSX = 0; // particleState.x: lifetime (ms)
export const S_PSY = 1; // particleState.y: size
export const S_PSZ = 2; // particleState.z: rotation (rad)
export const S_PSW = 3; // particleState.w: startFrame (0-based sheet index)
export const S_SVX = 0; // startValues.x: startLifetime (ms)
export const S_SVY = 1; // startValues.y: startSize
export const S_SVZ = 2; // startValues.z: startOpacity
export const S_SVW = 3; // startValues.w: startColorR
export const S_EXX = 0; // startColorsExt.x: startColorG
export const S_EXY = 1; // startColorsExt.y: startColorB
export const S_EXZ = 2; // startColorsExt.z: rotationSpeed (rad/s)
export const S_EXW = 3; // startColorsExt.w: noiseOffset
export const S_ORW = 3; // orbitalIsActive.w: isActive flag

/**
 * Floats per particle in the legacy CPU-side `curveData` init tail
 * (4 position+flag, 4 velocity+pad, 4 color, 4 state, 4 orbital, 8 ext).
 * Kept as an exported constant because the bounds-check parity tests
 * reproduce the kernel index arithmetic against it.
 */
export const INIT_STRIDE = 28;

/**
 * CPU reference for the GPU orbital-velocity rotation in the simulation
 * kernel: Euler(`speedX*dt`, `speedZ*dt`, `speedY*dt`) with order 'XYZ'
 * (intrinsic XYZ ⇒ matrix Rx·Ry·Rz, so the vector product applies Z first,
 * then Y, then X). Mutates `offset` in place, exactly mirroring the TSL
 * sequence `fx/fy/fz` in {@link createModifierComputeUpdate}.
 */
export function applyOrbitalRotation(
  offset: { x: number; y: number; z: number },
  speedX: number,
  speedY: number,
  speedZ: number,
  delta: number
): void {
  const angX = speedX * delta;
  const angY = speedZ * delta;
  const angZ = speedY * delta;
  // Z rotation (applied first).
  const c3 = Math.cos(angZ);
  const s3 = Math.sin(angZ);
  const zx = offset.x * c3 - offset.y * s3;
  const zy = offset.x * s3 + offset.y * c3;
  const zz = offset.z;
  // Y rotation.
  const c2 = Math.cos(angY);
  const s2 = Math.sin(angY);
  const yx = zx * c2 + zz * s2;
  const yz = -zx * s2 + zz * c2;
  const yy = zy;
  // X rotation (applied last).
  const c1 = Math.cos(angX);
  const s1 = Math.sin(angX);
  offset.x = yx;
  offset.y = yy * c1 - yz * s1;
  offset.z = yy * s1 + yz * c1;
}

// ??? Modifier flags ?????????????????????????????????????????????????????????
export type ModifierFlags = {
  sizeOverLifetime: boolean;
  opacityOverLifetime: boolean;
  colorOverLifetime: boolean;
  rotationOverLifetime: boolean;
  linearVelocity: boolean;
  orbitalVelocity: boolean;
  noise: boolean;
  forceFields: boolean;
  collisionPlanes: boolean;
};

/**
 * GPU shape enum. Matches the public {@link Shape} strings one-to-one:
 *   0 SPHERE | 1 CONE | 2 CIRCLE | 3 RECTANGLE | 4 BOX
 */
export const SHAPE_KIND = {
  SPHERE: 0,
  CONE: 1,
  CIRCLE: 2,
  RECTANGLE: 3,
  BOX: 4,
} as const;

/** {@link EmitFrom} encoded for the compute kernel: 0 VOLUME | 1 SHELL | 2 EDGE. */
export const BOX_EMIT_FROM = {
  VOLUME: 0,
  SHELL: 1,
  EDGE: 2,
} as const;

/**
 * Shape-emission parameters used by the emission compute kernel. One flat,
 * fully-resolved record: every field comes from the *nested* public
 * `ShapeConfig` branch (`shape.sphere` / `shape.cone` / …), never from
 * flattened `shapeType` / top-level `radius` keys.
 */
export type ShapeEmitParams = {
  /** 0 SPHERE | 1 CONE | 2 CIRCLE | 3 RECTANGLE | 4 BOX. */
  shapeKind: 0 | 1 | 2 | 3 | 4;
  /** sphere / cone / circle base radius. */
  radius: number;
  /** 1 = full volume, 0 = shell only (sphere, cone, circle). */
  radiusThickness: number;
  /** angular arc in **degrees** (360 = full). */
  arcDeg: number;
  /** cone opening angle in **degrees** (oracle default 90). */
  coneAngleDeg: number;
  /** rectangle local rotation (degrees, oracle x/y used). */
  rectangleRotXDeg: number;
  rectangleRotYDeg: number;
  /** rectangle scale. */
  rectangleScaleX: number;
  rectangleScaleY: number;
  /** box scale. */
  boxScaleX: number;
  boxScaleY: number;
  boxScaleZ: number;
  /** box emission mode: 0 VOLUME | 1 SHELL | 2 EDGE. */
  boxEmitFrom: 0 | 1 | 2;
  speedMin: number;
  speedMax: number;
  sizeMin: number;
  sizeMax: number;
  rotMin: number;
  rotMax: number;
  opacityMin: number;
  opacityMax: number;
  lifeMin: number; // startLifetime min (ms ? seconds to match GPU kernel)
  lifeMax: number;
  colorRMin: number;
  colorRMax: number;
  colorGMin: number;
  colorGMax: number;
  colorBMin: number;
  colorBMax: number;
  startFrameMin: number;
  startFrameMax: number;
  rotationCurveActive: boolean;
  rotationalXCurve: number; // index or -1
  rotationalYCurve: number;
  rotationalZCurve: number;
  linearXCurve: number;
  linearYCurve: number;
  linearZCurve: number;
  /** rotationOverLifetime speed range (SEPARATE from startRotation rotMin/rotMax). */
  rotOverLifeMin: number;
  rotOverLifeMax: number;
  /** noise.octaves (FBM octave count; 1 = single snoise sample). */
  noiseOctaves: number;
  /** noise.useRandomOffset — per-particle random offset in [0,100). */
  noiseUseRandomOffset: boolean;
};

/**
 * Default 4-arg shape for {@link createModifierComputeUpdate} so the pipeline
 * can be bootstrapped in 1-line tests (no-op emitter) and every field still
 * round-trips. The 5-arg / 10-arg calls used elsewhere all provide their own
 * concrete values, so this default only kicks in when the caller omits the
 * full parameter set.
 */
export const DEFAULT_SHAPE_EMIT_PARAMS: ShapeEmitParams = {
  shapeKind: 0,
  radius: 1,
  radiusThickness: 1,
  arcDeg: 360,
  coneAngleDeg: 90,
  rectangleRotXDeg: 0,
  rectangleRotYDeg: 0,
  rectangleScaleX: 1,
  rectangleScaleY: 1,
  boxScaleX: 1,
  boxScaleY: 1,
  boxScaleZ: 1,
  boxEmitFrom: 0,
  speedMin: 1,
  speedMax: 1,
  sizeMin: 1,
  sizeMax: 1,
  rotMin: 0,
  rotMax: 0,
  opacityMin: 1,
  opacityMax: 1,
  lifeMin: 1,
  lifeMax: 1,
  colorRMin: 1,
  colorRMax: 1,
  colorGMin: 1,
  colorGMax: 1,
  colorBMin: 1,
  colorBMax: 1,
  startFrameMin: 0,
  startFrameMax: 0,
  rotationCurveActive: false,
  rotationalXCurve: -1,
  rotationalYCurve: -1,
  rotationalZCurve: -1,
  linearXCurve: -1,
  linearYCurve: -1,
  linearZCurve: -1,
  rotOverLifeMin: 0,
  rotOverLifeMax: 0,
  noiseOctaves: 1,
  noiseUseRandomOffset: false,
};

/**
 * One GPU sub-emitter event channel, split into integer metadata + plain
 * float payload (WGSL integer atomics only):
 *   counter  : `array<atomic<u32>, 2>` — ping-pong window counters (event
 *              counts written this cycle; slot = window index 0/1).
 *   payload  : ordinary f32 storage; window w starts at `w * 6 * capacity`,
 *              event k lives at `windowBase + 6*k + 0..5 =
 *              (x, y, z, velX, velY, velZ)`.
 * The parent kernels append events; the child init kernel consumes them.
 */
export type SubEmitterFifo = {
  /** `atomic<u32>` ping-pong counters (itemSize 1, 2 elements). */
  counter: StorageBufferAttribute;
  /** 6 floats per event, 2 ping-pong windows, plain f32 storage. */
  payload: StorageBufferAttribute;
  /** 0 = BIRTH (written by the emission kernel), 1 = DEATH (sim kernel). */
  trigger: 0 | 1;
  /** Event capacity == `maxInstances` of the sub-emitter config. */
  capacity: number;
  /** Floats per window: 6 per event in the payload buffer. */
  windowSize: number;
};

/** Floats stored per sub-emitter event (position xyz + velocity xyz). */
export const SUB_EMITTER_EVENT_STRIDE = 6;

/** Floats in one FIFO payload window. */
export const subEmitterWindowSize = (capacity: number): number =>
  SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity);

// ─── u32 PCG birth hashing (§1: integer-first, uint-first) ────────────────
// Same arithmetic as Three.js r186 `nodes/math/Hash.js` (`hash()`), but
// implemented locally so every intermediate stays in u32; the final stage is
// the ONLY `uintBitsToFloat`-free f32 conversion. three's float `rand()` is
// intentionally not used for the primary birth entropy.

/** Raw uint PCG word (u32 chain: mul/add/shift/xor, no f32 intermediates). */
// r186 detail: `uint()` / `.toUint()` emit an empty snippet for atomic-result
// nodes (their nodeType is already 'uint'), so pass those through untouched.
const asU32 = (n: ShaderNodeObject<Node>): ShaderNodeObject<Node> =>
  n.nodeType === 'uint' ? n : n.toUint();

export const pcgRawU32 = (
  seedU: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => {
  const stateU = asU32(seedU).mul(tuint(747796405)).add(tuint(2891336453));
  const wordU = stateU
    .shiftRight(stateU.shiftRight(tuint(28)).add(tuint(4)))
    .bitXor(stateU)
    .mul(tuint(277803737));
  return wordU.shiftRight(tuint(22)).bitXor(wordU);
};

/** PCG hash in [0,1): integer mixing first, single f32 conversion at the end. */
export const pcg01 = (seedU: ShaderNodeObject<Node>): ShaderNodeObject<Node> =>
  pcgRawU32(seedU)
    .toFloat()
    .mul(float(1 / 4294967296));

/** Integer channel mix (all u32): birthNo*Knuth_odd ^ systemSeed ^ channel. */
export const mixBirthSeed = (
  birthNoU: ShaderNodeObject<Node>,
  systemSeedU: ShaderNodeObject<Node>,
  channelU: ShaderNodeObject<Node>
): ShaderNodeObject<Node> =>
  asU32(birthNoU)
    .mul(tuint(2654435761))
    .bitXor(asU32(systemSeedU))
    .bitXor(asU32(channelU));

/** Independent random channel from (birth number, system seed, channel id). */
export const randomChannel = (
  birthNoU: ShaderNodeObject<Node>,
  systemSeedU: ShaderNodeObject<Node>,
  channelU: ShaderNodeObject<Node>
): ShaderNodeObject<Node> =>
  pcg01(mixBirthSeed(birthNoU, systemSeedU, channelU));

/** 24-bit exact-in-f32 stable seed stored in `startColorsExt.w`. */
export const stableSeedU32 = (
  birthNoU: ShaderNodeObject<Node>,
  systemSeedU: ShaderNodeObject<Node>
): ShaderNodeObject<Node> =>
  pcgRawU32(mixBirthSeed(birthNoU, systemSeedU, CH.STABLE_SEED)).bitAnd(
    tuint(0x00ffffff)
  );

/** Stable seed recovered from `startColorsExt.w` (exact integer round-trip). */
export const stableSeedFromExt = (
  extW: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => extW.toUint();

/** One-time host seed for a pipeline (§5): u32 value written ONCE at creation. */
export const nextSystemSeed = (): number =>
  Math.floor(Math.random() * 0x100000000) >>> 0;

/** Dedicated uint channel constants (never reuse one id inside a pass). */
export const CH = {
  SHAPE_A: tuint(1),
  SHAPE_B: tuint(2),
  SHAPE_C: tuint(3),
  START_FRAME: tuint(4),
  SPEED: tuint(5),
  SIZE: tuint(6),
  ROTATION: tuint(7),
  OPACITY: tuint(8),
  LIFETIME: tuint(9),
  COLOR: tuint(10),
  ROTOL: tuint(11),
  STABLE_SEED: tuint(12),
  LIN_X: tuint(13),
  LIN_Y: tuint(14),
  LIN_Z: tuint(15),
  ORB_X: tuint(16),
  ORB_Y: tuint(17),
  ORB_Z: tuint(18),
  NOISE_PHASE: tuint(19),
} as const;

/**
 * Allocates the FIFO buffers for one sub-emitter channel: an integer
 * `atomic<u32>` counter pair (ping-pong) plus two ordinary f32 payload
 * windows. Float payload never goes through an atomic; counters are integer.
 */
export const createSubEmitterFifoAttribute = (
  capacity: number
): SubEmitterFifo =>
  ({
    counter: new StorageBufferAttribute(new Uint32Array(2), 1),
    payload: new StorageBufferAttribute(
      new Float32Array(2 * SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity)),
      1
    ),
    trigger: 1,
    capacity: Math.max(1, capacity),
    windowSize: subEmitterWindowSize(capacity),
  }) as SubEmitterFifo;

/** Trail-history descriptor consumed by the simulation kernel. */
export type TrailHistoryDesc = {
  /** Ordinary (non-atomic) vec4 f32 history ring: `(L+1)` rows per particle. */
  attribute: StorageBufferAttribute;
  /**
   * Integer ring metadata, `array<atomic<u32>>`, 2 words per particle:
   * word 2i = cursor, word 2i+1 = valid sample count (no float atomics).
   */
  meta: StorageBufferAttribute;
  /** Ring length `L` (`trail.length`). */
  length: number;
  /** Adaptive sampling threshold in world units (`trail.minVertexDistance`). */
  minVertexDistance: number;
  /** Sample age limit in ms (`trail.maxTime`, 0 = unlimited). */
  maxTime: number;
};

/** Ribbon build descriptor for {@link createTrailRibbonUpdate}. */
export type TrailRibbonDesc = {
  /** vec4 * (maxParticles*2L): (x, y, z, halfWidth). */
  position: StorageBufferAttribute;
  /** vec4 * (maxParticles*2L): (nextX, nextY, nextZ, alpha). */
  next: StorageBufferAttribute;
  /** vec4 * (maxParticles*2L): (uv.x = side 0/1, uv.y = t, colorR, colorG). */
  uvColorA: StorageBufferAttribute;
  /** vec4 * (maxParticles*2L): (colorB, colorA, 0, 0). */
  colorB: StorageBufferAttribute;
  /** Shared trail history buffer: ordinary vec4 f32 storage (same as {@link TrailHistoryDesc}). */
  history: StorageBufferAttribute;
  /** Shared integer ring metadata (atomic<u32>, 2 words per particle). */
  meta: StorageBufferAttribute;
  /** Particle RGBA pool of the owning system (read-mostly). */
  particleColor: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** Curve functions along the trail (t = 0 head \u2192 1 tail); omitted = identity. */
  curveFns: {
    width?: (t: number) => number;
    opacity?: (t: number) => number;
    colorR?: (t: number) => number;
    colorG?: (t: number) => number;
    colorB?: (t: number) => number;
  };
  /** Ribbon width in world units (`trail.width`). */
  width: number;
  /** Ring length `L` (`trail.length`). */
  length: number;
  /** Sample age limit in ms; 0 = unlimited. */
  maxTime: number;
  /** Number of particle slots. */
  maxParticles: number;
};

/** Per-frame compute-pipeline uniforms (CPU-side scalar writes only). */
export type ModifierUniforms = {
  delta: ShaderNodeObject<Node>;
  deltaMs: ShaderNodeObject<Node>;
  /** Current clock in ms, used for trail sample stamps. */
  nowMs: ShaderNodeObject<Node>;
  gravityVelocity: ShaderNodeObject<Node>;
  noiseStrength: ShaderNodeObject<Node>;
  noisePower: ShaderNodeObject<Node>;
  noiseFrequency: ShaderNodeObject<Node>;
  noisePositionAmount: ShaderNodeObject<Node>;
  noiseRotationAmount: ShaderNodeObject<Node>;
  noiseSizeAmount: ShaderNodeObject<Node>;
  /** Scalar count of new particles for this frame; also written to `emitNode.count`. */
  emitCount: ShaderNodeObject<Node>;
  /** Seed for GPU-side randomness (updated per frame by CPU). */
  seed: ShaderNodeObject<Node>;
};

/** GPU storage buffer references. */
export type ModifierStorageBuffers = {
  position: StorageBufferAttribute | StorageInstancedBufferAttribute;
  velocity: StorageBufferAttribute | StorageInstancedBufferAttribute;
  color: StorageBufferAttribute | StorageInstancedBufferAttribute;
  particleState: StorageBufferAttribute | StorageInstancedBufferAttribute;
  startValues: StorageBufferAttribute | StorageInstancedBufferAttribute;
  startColorsExt: StorageBufferAttribute | StorageInstancedBufferAttribute;
  orbitalIsActive: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** `array<atomic<u32>, 1>` ring-allocation counter (integer atomic only). */
  allocator: StorageBufferAttribute;
  /** Trail ring integer metadata (`atomic<u32>`, 2 words per particle), or null. */
  trailMeta: StorageBufferAttribute | null;
  /** Packed read-mostly f32 tables (curves + force fields + collision planes), uniform-backed. */
  packedData: Float32Array;
};

/** Actual per-pass resource budget derived from that pass's bound nodes. */
export type PassLayout = {
  name: string;
  storageBindings: number;
  uniformBindings: number;
};

/** The complete GPU-owned pipeline handle returned by `createComputePipeline`. */
export type ModifierComputePipeline = {
  /** Emission compute node (dispatched before simulation each frame). */
  emitNode: ReturnType<typeof compute>;
  /** Per-frame simulation compute node. */
  simNode: ReturnType<typeof compute>;
  /** Dedicated trail-history pass node (null when no TRAIL renderer). */
  trailHistoryNode: ReturnType<typeof compute> | null;
  /** Dedicated sub-emitter BIRTH event pass node (null without BIRTH FIFOs). */
  subBirthEventsNode: ReturnType<typeof compute> | null;
  /** Dedicated sub-emitter DEATH event pass node (null without DEATH FIFOs). */
  subDeathEventsNode: ReturnType<typeof compute> | null;
  /** Convenience: every node of this pipeline in dispatch order. */
  computeNodes: ReturnType<typeof compute>[];
  /** Real per-pass storage/uniform budgets (no synthetic totals). */
  passLayouts: PassLayout[];
  /** Uniforms the CPU writes before each pair of compute dispatches. */
  uniforms: ModifierUniforms;
  /** Shape / curve parameters (scalar float uniforms) baked at creation. */
  shapeUniforms: Record<string, ShaderNodeObject<Node>>;
  /** Storage buffers (shared by both kernels). */
  buffers: ModifierStorageBuffers;
  /** Element count of the uint atomic allocator stack (`maxParticles + 1`). */
  allocatorCount: number;
  /** Non-atomic f32 uniform-table node (baked curves + force fields + collision planes). */
  packedDataNode: ShaderNodeObject<Node>;
  /** Semantic compute-pass names in dispatch order (for `[PS:pipeline]` logging). */
  passNames: string[];
  /** Trail ring integer meta attribute (null when the system has no trail). */
  trailMeta: StorageBufferAttribute | null;

  /** Emitter-pose uniform values for sub-emitter init kernel. */
  emitterPose: {
    /** (x, y, z, isWorldFlag) ? translation only in WORLD simulation. */
    positionW: ShaderNodeObject<Node> | null;
    wrapperQuat: ShaderNodeObject<Node> | null;
    worldScale: ShaderNodeObject<Node> | null;
  };
  /** Force field offset + count uniform (null when disabled). */
  forceFieldInfo: {
    offset: number;
    countUniform: ShaderNodeObject<Node>;
  } | null;
  /** Collision plane offset + count uniform (null when disabled). */
  collisionPlaneInfo: {
    offset: number;
    countUniform: ShaderNodeObject<Node>;
  } | null;
};

// ??? Storage pool creation ??????????????????????????????????????????????????
/**
 * Creates GPU buffers for one particle system.
 *
 * 8 storage bindings: the 7 per-particle `vec4` state buffers plus one
 * `uint`-typed atomic allocator; the read-mostly f32 tables (baked curves,
 * force fields, collision planes) live in a plain `Float32Array` that is bound
 * as a uniform buffer, so they do not consume a storage binding and stay
 * non-atomic `f32` loads on the hot path.
 *
 * The CPU only fills the two CPU-owned buffers once; after the first upload
 * the compute kernels own all changes.
 */
export function createModifierStorageBuffers(
  maxParticles: number,
  instanced: boolean,
  curveData: Float32Array,
  hasForceFields = false,
  hasCollisionPlanes = false,
  trailLength = 0
): { buffers: ModifierStorageBuffers; allocatorCount: number } {
  const Cls = instanced
    ? StorageInstancedBufferAttribute
    : StorageBufferAttribute;

  const curveLen = Math.max(curveData.length, 1);
  const ffSize = hasForceFields ? FORCE_FIELD_DATA_SIZE : 0;
  const cpSize = hasCollisionPlanes ? COLLISION_PLANE_DATA_SIZE : 0;
  // Read-mostly f32 table layout (uniform-backed, non-atomic):
  //   [0 .. curveLen-1]                        baked curve samples
  //   [curveLen .. curveLen + ffSize - 1]      force-field records   (if any)
  //   [... .. + cpSize - 1]                    collision-plane records (if any)
  const packedData = new Float32Array(curveLen + ffSize + cpSize);
  packedData.set(curveData, 0);

  // Ring allocator: ONE integer `atomic<u32>` counter (index 0 of the binding,
  // the tail padding keeps the binding size identical to legacy). Allocation
  // is `slot = fetchAdd(counter) % (maxParticles + 1)`, so the counter can
  // never underflow and every slot index is always in range (a plain
  // atomicSub stack wraps 0 -> 0xffffffff and indexes out of bounds).
  const allocatorCount = maxParticles + 1;
  const allocatorData = new Uint32Array(allocatorCount);
  allocatorData[0] = 0;
  for (let i = 0; i < maxParticles; i++) allocatorData[i + 1] = i;

  // Trail ring metadata: integer atomic words only (cursor, count per slot).
  const trailMeta =
    trailLength > 0
      ? new StorageBufferAttribute(
          new Uint32Array(Math.max(1, maxParticles) * 2),
          1
        )
      : null;

  return {
    buffers: {
      position: new Cls(new Float32Array(maxParticles * 4), 4),
      velocity: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      color: new Cls(new Float32Array(maxParticles * 4), 4),
      particleState: new Cls(new Float32Array(maxParticles * 4), 4),
      startValues: new Cls(new Float32Array(maxParticles * 4), 4),
      startColorsExt: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      orbitalIsActive: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      allocator: new StorageBufferAttribute(allocatorData, 1),
      trailMeta,
      packedData,
    },
    allocatorCount,
  };
}

// ??? Curve lookup helper ????????????????????????????????????????????????????
function createCurveLookup(sCurveData: ShaderNodeObject<Node>) {
  return Fn(
    ({
      curveIndex,
      t,
    }: {
      curveIndex: ShaderNodeObject<Node>;
      t: ShaderNodeObject<Node>;
    }) => {
      const clamped = tslMin(t, float(1.0));
      const pos = clamped.mul(CURVE_RESOLUTION - 1);
      const idx0 = floor(pos);
      const f = fract(pos);
      const base = curveIndex.mul(CURVE_RESOLUTION);
      const v0 = sCurveData.element(base.add(idx0));
      const v1 = sCurveData.element(
        base.add(tslMin(idx0.add(1.0), float(CURVE_RESOLUTION - 1)))
      );
      return mix(v0, v1, f);
    }
  );
}

// ??? Kernel builder ????????????????????????????????????????????????????????
/**
 * Builds two compute kernels sharing the same buffers: a small "emission" pass
 * (count = per-frame `emitCount`) and a full "simulation" pass (count =
 * `maxParticles`). The emission pass is 1-based sized at creation and its
 * `ComputeNode.count` is refreshed from `emitCount` every frame, which also
 * regenerates the shader-side `instanceIndex >= count` bound guard.
 *
 * Both kernels address the same `uint` atomic allocator stack with
 * `atomicSub` / `atomicAdd` / `atomicLoad` / `atomicStore`, so a slot returned
 * by the sim kernel's death handling is immediately available to the emission
 * kernel on the next frame.
 */
/** Fully-resolved scalar node set consumed by {@link shapeEmitNodes}. */
type ShapeNodeUniforms = {
  kind: ShaderNodeObject<Node>;
  radius: ShaderNodeObject<Node>;
  thickness: ShaderNodeObject<Node>;
  arcDeg: ShaderNodeObject<Node>;
  coneAngleDeg: ShaderNodeObject<Node>;
  rectRX: ShaderNodeObject<Node>;
  rectRY: ShaderNodeObject<Node>;
  rectSX: ShaderNodeObject<Node>;
  rectSY: ShaderNodeObject<Node>;
  boxSX: ShaderNodeObject<Node>;
  boxSY: ShaderNodeObject<Node>;
  boxSZ: ShaderNodeObject<Node>;
  boxFrom: ShaderNodeObject<Node>;
  speedMin: ShaderNodeObject<Node>;
  speedMax: ShaderNodeObject<Node>;
};

/**
 * Oracle-faithful shape emission in the emitter's LOCAL frame for all 5 kinds
 * (0 SPHERE | 1 CONE | 2 CIRCLE | 3 RECTANGLE | 4 BOX), ported formula-by-
 * formula from the CPU helpers in `three-particles-utils.ts`.
 *
 * `rA` / `rB` / `rC` are three independent 0..1 randoms and `rSpeed` the speed
 * random; the returned components are still unrotated (the caller applies the
 * wrapper quaternion). Shared by the emission kernel and the sub-emitter init
 * kernel so both use identical shape math.
 */
function shapeEmitNodes(
  u: ShapeNodeUniforms,
  rA: ShaderNodeObject<Node>,
  rB: ShaderNodeObject<Node>,
  rC: ShaderNodeObject<Node>,
  rSpeed: ShaderNodeObject<Node>
): {
  px: ShaderNodeObject<Node>;
  py: ShaderNodeObject<Node>;
  pz: ShaderNodeObject<Node>;
  vx: ShaderNodeObject<Node>;
  vy: ShaderNodeObject<Node>;
  vz: ShaderNodeObject<Node>;
} {
  // Arc / cone angle are degrees; PI/180 = 0.01745329.
  const DEG = float(0.01745329);
  const uRadius = u.radius;
  const uRadiusThickness = u.thickness;

  // SPHERE: theta = 2*PI*u, phi = acos(2v-1), dir*sphereDistance.
  const thetaS = rA.mul(u.arcDeg).mul(DEG);
  const cosPhi = rB.mul(float(2.0)).sub(float(1.0));
  const sinPhi = sqrt(float(1.0).sub(cosPhi.mul(cosPhi)));
  const dirSx = sinPhi.mul(cos(thetaS));
  const dirSy = sinPhi.mul(sin(thetaS));
  const dirSz = cosPhi;
  const distS = uRadius
    .mul(float(1.0).sub(uRadiusThickness))
    .add(uRadius.mul(uRadiusThickness).mul(rC));
  const pSx = dirSx.mul(distS);
  const pSy = dirSy.mul(distS);
  const pSz = dirSz.mul(distS);
  // vel = position * (1/|position|) * speed = dir * speed.
  const spS = mix(u.speedMin, u.speedMax, rSpeed);
  const vSx = dirSx.mul(spS);
  const vSy = dirSy.mul(spS);
  const vSz = dirSz.mul(spS);

  // CONE + CIRCLE share the base-disc position (same oracle math).
  const thetaB = rA.mul(u.arcDeg).mul(DEG);
  const dirBx = cos(thetaB);
  const dirBy = sin(thetaB);
  const distB = uRadius
    .mul(float(1.0).sub(uRadiusThickness))
    .add(uRadius.mul(uRadiusThickness).mul(rB));
  const pBx = dirBx.mul(distB);
  const pBy = dirBy.mul(distB);
  const pBz = float(0.0);
  // CONE velocity: spread by the opening angle, |dist|/radius * angle(rad).
  const nAngle = distB
    .div(uRadius.max(float(1e-6)))
    .mul(u.coneAngleDeg.mul(DEG));
  const spB = mix(u.speedMin, u.speedMax, rSpeed);
  const sinNA = sin(nAngle);
  const vCx = dirBx.mul(sinNA).mul(spB);
  const vCy = dirBy.mul(sinNA).mul(spB);
  const vCz = cos(nAngle).mul(spB);
  // CIRCLE velocity: radial in XY, no Z.
  const vIx = dirBx.mul(spB);
  const vIy = dirBy.mul(spB);
  const vIz = float(0.0);

  // RECTANGLE: uniform XY inside scale, tilted by the local rotation.
  const rxOff = rA.mul(u.rectSX).sub(u.rectSX.mul(float(0.5)));
  const ryOff = rB.mul(u.rectSY).sub(u.rectSY.mul(float(0.5)));
  const rotXr = u.rectRX.mul(DEG);
  const rotYr = u.rectRY.mul(DEG);
  const pRx = rxOff.mul(cos(rotYr));
  const pRy = ryOff.mul(cos(rotXr));
  const pRz = rxOff.mul(sin(rotYr)).sub(ryOff.mul(sin(rotXr)));

  // BOX: VOLUME / SHELL / EDGE on the +-scale/2 lattice.
  const halfX = u.boxSX.mul(float(0.5));
  const halfY = u.boxSY.mul(float(0.5));
  const halfZ = u.boxSZ.mul(float(0.5));
  const pVx = rA.mul(u.boxSX).sub(halfX);
  const pVy = rB.mul(u.boxSY).sub(halfY);
  const pVz = rC.mul(u.boxSZ).sub(halfZ);
  const side = floor(rA.mul(float(6.0))).min(float(5.0));
  const pa = side.sub(floor(side.div(float(3.0))).mul(float(3.0)));
  const a0 = side.greaterThan(float(2.0)).select(float(1.0), float(0.0));
  const isPa0 = pa.equal(float(0.0));
  const isPa1 = pa.equal(float(1.0));
  // SHELL lattice coordinates (a0 on `pa`, rB / rC on the other two axes).
  const shX = isPa0.select(a0, isPa1.select(rC, rB));
  const shY = isPa0.select(rB, isPa1.select(a0, rC));
  const shZ = isPa0.select(rC, isPa1.select(rB, a0));
  const pShx = shX.mul(u.boxSX).sub(halfX);
  const pShy = shY.mul(u.boxSY).sub(halfY);
  const pShz = shZ.mul(u.boxSZ).sub(halfZ);
  // EDGE lattice: one axis pinned, one of four edges, one free coordinate.
  const edge = floor(rB.mul(float(4.0))).min(float(3.0));
  const lowEdge = edge.lessThan(float(2.0));
  const e1 = lowEdge.select(rC, edge.sub(float(2.0)));
  const e2 = lowEdge.select(edge, rC);
  const edX = isPa0.select(a0, isPa1.select(e2, e1));
  const edY = isPa0.select(e1, isPa1.select(a0, e2));
  const edZ = isPa0.select(e2, isPa1.select(e1, a0));
  const pEdx = edX.mul(u.boxSX).sub(halfX);
  const pEdy = edY.mul(u.boxSY).sub(halfY);
  const pEdz = edZ.mul(u.boxSZ).sub(halfZ);
  const boxFrom = u.boxFrom;
  const pBX = boxFrom
    .equal(float(0.0))
    .select(pVx, boxFrom.equal(float(1.0)).select(pShx, pEdx));
  const pBY = boxFrom
    .equal(float(0.0))
    .select(pVy, boxFrom.equal(float(1.0)).select(pShy, pEdy));
  const pBZ = boxFrom
    .equal(float(0.0))
    .select(pVz, boxFrom.equal(float(1.0)).select(pShz, pEdz));
  // RECTANGLE / BOX velocities are straight local +Z at `speed`.
  const vPlaneZ = mix(u.speedMin, u.speedMax, rSpeed);

  // Kind dispatch: 0 SPHERE | 1 CONE | 2 CIRCLE | 3 RECTANGLE | 4 BOX.
  const kind = u.kind;
  const isSphereKind = kind.equal(float(0.0));
  const isConeKind = kind.equal(float(1.0));
  const isCircleKind = kind.equal(float(2.0));
  const isRectKind = kind.equal(float(3.0));
  const planeX = isRectKind.select(pRx, pBX);
  const planeY = isRectKind.select(pRy, pBY);
  const planeZ = isRectKind.select(pRz, pBZ);
  // CONE and CIRCLE both use the base-disc position; only RECT/BOX take the
  // planar branch (CIRCLE no longer "falls through" to the BOX lattice).
  const isDiscKind = isConeKind.or(isCircleKind);
  const discX = isDiscKind.select(pBx, planeX);
  const discY = isDiscKind.select(pBy, planeY);
  const discZ = isDiscKind.select(pBz, planeZ);
  const px = isSphereKind.select(pSx, discX);
  const py = isSphereKind.select(pSy, discY);
  const pz = isSphereKind.select(pSz, discZ);
  const cOrI_X = isConeKind.select(vCx, vIx);
  const cOrI_Y = isConeKind.select(vCy, vIy);
  const cOrI_Z = isConeKind.select(vCz, vIz);
  const isDisc = isConeKind.or(isCircleKind);
  const nonSphereVX = isDisc.select(cOrI_X, float(0.0));
  const nonSphereVY = isDisc.select(cOrI_Y, float(0.0));
  const nonSphereVZ = isDisc.select(cOrI_Z, vPlaneZ);
  const vx = isSphereKind.select(vSx, nonSphereVX);
  const vy = isSphereKind.select(vSy, nonSphereVY);
  const vz = isSphereKind.select(vSz, nonSphereVZ);

  return { px, py, pz, vx, vy, vz };
}

/**
 * Rotates a local-frame vector by a unit quaternion stored as (x, y, z, w).
 * v' = v*(2w^2-1) + 2*(q.xyz . v)*q.xyz + 2w*(q.xyz x v)
 */
function quatRotateNodes(
  x: ShaderNodeObject<Node>,
  y: ShaderNodeObject<Node>,
  z: ShaderNodeObject<Node>,
  q: ShaderNodeObject<Node>
): [ShaderNodeObject<Node>, ShaderNodeObject<Node>, ShaderNodeObject<Node>] {
  const qx = q.x;
  const qy = q.y;
  const qz = q.z;
  const qw = q.w;
  const projD = qx.mul(x).add(qy.mul(y)).add(qz.mul(z)).mul(float(2.0));
  const scaleV = qw.mul(qw).mul(float(2.0)).sub(float(1.0));
  const twoW = qw.mul(float(2.0));
  return [
    x
      .mul(scaleV)
      .add(qx.mul(projD))
      .add(qy.mul(z).sub(qz.mul(y)).mul(twoW)),
    y
      .mul(scaleV)
      .add(qy.mul(projD))
      .add(qz.mul(x).sub(qx.mul(z)).mul(twoW)),
    z
      .mul(scaleV)
      .add(qz.mul(projD))
      .add(qx.mul(y).sub(qy.mul(x)).mul(twoW)),
  ];
}

/** Raw velocity-over-lifetime axis values (constant / range / curve). */
export type VelocityAxisValues = {
  linear: Array<
    | import('../types.js').Constant
    | import('../types.js').RandomBetweenTwoConstants
    | import('../types.js').LifetimeCurve
    | undefined
  >;
  orbital: Array<
    | import('../types.js').Constant
    | import('../types.js').RandomBetweenTwoConstants
    | import('../types.js').LifetimeCurve
    | undefined
  >;
};

export function createModifierComputeUpdate(
  buffers: ModifierStorageBuffers,
  maxParticles: number,
  curveMap: BakedCurveMap,
  flags: ModifierFlags,
  shapeParams: ShapeEmitParams = DEFAULT_SHAPE_EMIT_PARAMS,
  forceFieldCount = 0,
  collisionPlaneCount = 0,
  subFifos: SubEmitterFifo[] = [],
  trailDesc?: TrailHistoryDesc,
  velocityValues?: VelocityAxisValues
): ModifierComputePipeline {
  // ?? Per-frame uniforms ??
  const uDelta = uniform(float(0));
  const uDeltaMs = uniform(float(0));
  const uNowMs = uniform(float(0));
  const uGravityVelocity = uniform(new Vector3(0, 0, 0));
  // One-time system seed (§5): u32, written ONCE at pipeline creation, never
  // per frame. All birth entropy is derived from it by integer channel mixes.
  const uSystemSeed = uniform(nextSystemSeed(), 'uint');
  const uSeed = uSystemSeed;
  // Declared u32 so it matches instanceIndex exactly in the WGSL guard and in
  // If(i.lessThan(uEmitCount), ...); the CPU writes the integer count per frame.
  const uEmitCount = uniform(0, 'uint');
  const uNoiseStrength = uniform(float(0));
  const uNoisePower = uniform(float(0));
  const uNoiseFrequency = uniform(float(1));
  const uNoisePosAmount = uniform(float(0));
  const uNoiseRotAmount = uniform(float(0));
  const uNoiseSizeAmount = uniform(float(0));

  // ?? Shape / config scalar uniforms ??
  const shapeUniforms: Record<string, ShaderNodeObject<Node>> = {};
  const sh = (name: string, v: number) => {
    const u = uniform(float(v));
    shapeUniforms[name] = u;
    return u;
  };
  const uShape = sh('shapeKind', shapeParams.shapeKind);
  const uRadius = sh('radius', shapeParams.radius);
  const uRadiusThickness = sh('radiusThickness', shapeParams.radiusThickness);
  const uArcDeg = sh('arcDeg', shapeParams.arcDeg);
  const uConeAngleDeg = sh('coneAngleDeg', shapeParams.coneAngleDeg);
  const uRectRX = sh('rectangleRotXDeg', shapeParams.rectangleRotXDeg);
  const uRectRY = sh('rectangleRotYDeg', shapeParams.rectangleRotYDeg);
  const uRectSX = sh('rectangleScaleX', shapeParams.rectangleScaleX);
  const uRectSY = sh('rectangleScaleY', shapeParams.rectangleScaleY);
  const uBoxSX = sh('boxScaleX', shapeParams.boxScaleX);
  const uBoxSY = sh('boxScaleY', shapeParams.boxScaleY);
  const uBoxSZ = sh('boxScaleZ', shapeParams.boxScaleZ);
  const uBoxEmitFrom = sh('boxEmitFrom', shapeParams.boxEmitFrom);
  // ?? Emitter pose (written once per frame by the CPU, scalar only) ??
  // w = 1 for WORLD simulation (add translation + axis scale), 0 for LOCAL.
  const uEmitterPos = uniform(new Vector4(0, 0, 0, 0));
  // Wrapper quaternion (x, y, z, w). Identity in LOCAL space, emitter world
  // rotation in WORLD space ? mirrors generalData.wrapperQuaternion.
  const uWrapperQuat = uniform(new Vector4(0, 0, 0, 1));
  // Emitter world scale, applied to the rotated shape offset in WORLD space.
  const uWorldScale = uniform(new Vector3(1, 1, 1));

  const uSpeedMin = sh('speedMin', shapeParams.speedMin);
  const uSpeedMax = sh('speedMax', shapeParams.speedMax);
  const uSizeMin = sh('sizeMin', shapeParams.sizeMin);
  const uSizeMax = sh('sizeMax', shapeParams.sizeMax);
  const uRotMin = sh('rotMin', shapeParams.rotMin);
  const uRotMax = sh('rotMax', shapeParams.rotMax);
  // rotationOverLifetime has its OWN range (oracle: separate
  // lifetimeValues.rotationOverLifetime, never startRotation's min/max).
  const uRotOLMin = sh('rotOverLifeMin', shapeParams.rotOverLifeMin);
  const uRotOLMax = sh('rotOverLifeMax', shapeParams.rotOverLifeMax);
  const uOpMin = sh('opacityMin', shapeParams.opacityMin);
  const uOpMax = sh('opacityMax', shapeParams.opacityMax);
  const uLifeMin = sh('lifeMin', shapeParams.lifeMin);
  const uLifeMax = sh('lifeMax', shapeParams.lifeMax);
  const uCRR = sh('colorRMin', sRGBToLinear(shapeParams.colorRMin));
  const uCRX = sh('colorRMax', sRGBToLinear(shapeParams.colorRMax));
  const uCGR = sh('colorGMin', sRGBToLinear(shapeParams.colorGMin));
  const uCGX = sh('colorGMax', sRGBToLinear(shapeParams.colorGMax));
  const uCBR = sh('colorBMin', sRGBToLinear(shapeParams.colorBMin));
  const uCBX = sh('colorBMax', sRGBToLinear(shapeParams.colorBMax));
  const uFrMin = sh('startFrameMin', shapeParams.startFrameMin);
  const uFrMax = sh('startFrameMax', shapeParams.startFrameMax);

  // ?? Particle-state storage nodes (bindings 1-7) ??
  const sPos = storage(buffers.position, 'vec4', maxParticles);
  const sVel = storage(buffers.velocity, 'vec4', maxParticles);
  const sCol = storage(buffers.color, 'vec4', maxParticles);
  const sPS = storage(buffers.particleState, 'vec4', maxParticles);
  const sSV = storage(buffers.startValues, 'vec4', maxParticles);
  const sEx = storage(buffers.startColorsExt, 'vec4', maxParticles);
  const sOIA = storage(buffers.orbitalIsActive, 'vec4', maxParticles);

  // ?? Binding 8: integer ring-allocator counter ??
  // `array<atomic<u32>, 1>` (only index 0 is live): a monotonic birth counter,
  // slot = counter mod maxParticles, so allocation can never underflow.
  // With bindings 1-7 above this completes the guaranteed WebGPU budget of
  // exactly 8 storage buffers for the base pipeline (no optional binding 9).
  const allocatorCount = maxParticles + 1;
  const ringMod = float(maxParticles);
  // u32 ring modulus (§4): birthNo and slot modulo stay native integers.
  const ringModU = tuint(maxParticles);
  const sAllocator = storage(
    buffers.allocator,
    'uint',
    Math.max(1, allocatorCount)
  ).toAtomic();

  // ?? Read-mostly f32 tables: uniform buffer binding, non-atomic ??
  // Baked curve samples plus optional force-field / collision-plane records.
  // A plain uniform binding: it neither consumes a storage slot nor becomes
  // `atomic<f32>`, so curve sampling stays a straight `f32` load.
  const sCD = buffer(buffers.packedData, 'float', buffers.packedData.length);
  // Curve table sampler — must be in lexical scope BEFORE `simAxis`, the
  // emission/simulation kernels and every size/opacity/color/velocity
  // lifetime-curve lookup below.
  const lookupCurve = createCurveLookup(sCD);

  // ?? Sub-emitter FIFO channels: integer `atomic<u32>` counters + f32 payloads ??
  // The CPU writes the window index (0/1 frame parity) into uFifoBase; each
  // channel's payload window starts at `window * 6 * capacity`.
  const uFifoBase = uniform(float(0), 'uint');
  const fifoNodes = subFifos.map((f) => ({
    trigger: f.trigger,
    capacity: Math.max(1, f.capacity),
    windowSize: subEmitterWindowSize(Math.max(1, f.capacity)),
    count: storage(
      f.counter,
      'uint',
      Math.max(1, (f.counter.array as Uint32Array).length)
    ).toAtomic(),
    payload: storage(
      f.payload,
      'float',
      Math.max(1, (f.payload.array as Float32Array).length)
    ),
  }));
  // Per-trigger channel views: each event pass only binds its own channels,
  // which keeps both passes at <= 8 storage bindings (see pass builders below).
  const birthFifos = fifoNodes.filter((f) => f.trigger === 0);
  const deathFifos = fifoNodes.filter((f) => f.trigger === 1);
  const hasDeathFifo = deathFifos.length > 0;

  const writeFifoEvent = (
    f: {
      capacity: number;
      windowSize: number;
      count: ShaderNodeObject<Node>;
      payload: ShaderNodeObject<Node>;
    },
    x: ShaderNodeObject<Node>,
    y: ShaderNodeObject<Node>,
    z: ShaderNodeObject<Node>,
    vx: ShaderNodeObject<Node>,
    vy: ShaderNodeObject<Node>,
    vz: ShaderNodeObject<Node>
  ): void => {
    // `atomicAdd` on the integer counter gives the pre-increment event index;
    // indices beyond `capacity` are the explicitly dropped events.
    const winBase = uFifoBase.mul(float(f.windowSize)).toVar();
    const oldCount = float(
      atomicAdd(f.count.element(uFifoBase), tuint(1))
    ).toVar();
    If(oldCount.lessThan(float(f.capacity)), () => {
      const b = winBase.add(oldCount.mul(float(SUB_EMITTER_EVENT_STRIDE)));
      f.payload.element(b).assign(x);
      f.payload.element(b.add(float(1.0))).assign(y);
      f.payload.element(b.add(float(2.0))).assign(z);
      f.payload.element(b.add(float(3.0))).assign(vx);
      f.payload.element(b.add(float(4.0))).assign(vy);
      f.payload.element(b.add(float(5.0))).assign(vz);
    });
  };

  // ?? Trail history ring: plain vec4 f32 samples + integer atomic metadata ??
  const trailRows = trailDesc ? trailDesc.length + 1 : 0;
  const sTrail: ShaderNodeObject<Node> | null = trailDesc
    ? storage(trailDesc.attribute, 'vec4', trailRows * maxParticles)
    : null;
  const sTrailMeta: ShaderNodeObject<Node> | null =
    trailDesc && buffers.trailMeta
      ? storage(buffers.trailMeta, 'uint', maxParticles * 2).toAtomic()
      : null;

  const curveLen = Math.max(curveMap.data.length, 1);

  // ?? Force-field + collision-plane TSL readers ??
  const forceFieldOffset = curveLen;
  const collisionOffset =
    forceFieldOffset + (flags.forceFields ? FORCE_FIELD_DATA_SIZE : 0);

  const ffNodes = flags.forceFields
    ? createForceFieldTSL(sCD, forceFieldOffset, forceFieldCount)
    : null;
  const cpNodes = flags.collisionPlanes
    ? createCollisionPlaneTSL(sCD, collisionOffset, collisionPlaneCount)
    : null;

  // Per-axis raw velocity-over-lifetime values (constant / random-range /
  // curve) — oracle parity: random ranges are sampled PER PARTICLE from the
  // stable birth seed stored in `startColorsExt.w` (no extra storage buffer).
  // Destination contract:
  //   startColorsExt.x = startColorG   startColorsExt.y = startColorB
  //   startColorsExt.z = rotationOverLifetime speed   .w = particleSeed
  type AxisSpec = {
    /** -1 = not a lifetime curve; >= 0 = baked curve table index. */
    ci: number;
    min: number;
    max: number;
    /** true when the config value is a random-between-two range. */
    isRange: boolean;
  };
  const parseAxis = (
    rawAxis:
      | import('../types.js').Constant
      | import('../types.js').RandomBetweenTwoConstants
      | import('../types.js').LifetimeCurve
      | undefined,
    curveIdx: number
  ): AxisSpec => {
    if (curveIdx >= 0) {
      return { ci: curveIdx, min: 0, max: 0, isRange: false };
    }
    if (
      rawAxis &&
      typeof rawAxis === 'object' &&
      'min' in rawAxis &&
      'max' in rawAxis
    ) {
      const mn = Number(rawAxis.min) || 0;
      const mx = Number(rawAxis.max) || 0;
      return { ci: -1, min: mn, max: mx, isRange: mn !== mx };
    }
    const c = typeof rawAxis === 'number' ? rawAxis : 0;
    return { ci: -1, min: c, max: c, isRange: false };
  };
  const vv = velocityValues ?? {
    linear: [undefined, undefined, undefined],
    orbital: [undefined, undefined, undefined],
  };
  const linAxes: AxisSpec[] = [
    parseAxis(vv.linear[0], curveMap.linearVelX),
    parseAxis(vv.linear[1], curveMap.linearVelY),
    parseAxis(vv.linear[2], curveMap.linearVelZ),
  ];
  const orbAxes: AxisSpec[] = [
    parseAxis(vv.orbital[0], curveMap.orbitalVelX),
    parseAxis(vv.orbital[1], curveMap.orbitalVelY),
    parseAxis(vv.orbital[2], curveMap.orbitalVelZ),
  ];
  const axisUniforms = new Map<
    AxisSpec,
    [ShaderNodeObject<Node>, ShaderNodeObject<Node>]
  >();
  for (const a of [...linAxes, ...orbAxes]) {
    if (a.isRange) {
      axisUniforms.set(a, [uniform(float(a.min)), uniform(float(a.max))]);
    }
  }
  // Per-particle axis value for `startColorsExt.w = particleSeed`:
  //   - lifetime curve       -> `lookupCurve` on the baked table
  //   - random range         -> `mix(min, max, rand(seed + salt))` (stable per
  //                              particle; only the immutable birth seed is
  //                              used, NEVER the changing frame seed)
  //   - constant             -> direct passthrough
  // Six distinct fixed salts keep the six axis streams decorrelated.
  const simAxis = (
    a: AxisSpec,
    lifePct: ShaderNodeObject<Node>,
    particleSeed: ShaderNodeObject<Node>,
    salt: ShaderNodeObject<Node>
  ): ShaderNodeObject<Node> => {
    if (a.ci >= 0) {
      return lookupCurve({
        curveIndex: float(a.ci),
        t: lifePct,
      });
    }

    if (a.isRange) {
      const [mn, mx] = axisUniforms.get(a)!;
      return mix(
        mn,
        mx,
        pcg01(particleSeed.toUint().mul(tuint(2654435761)).bitXor(salt))
      );
    }

    return float(a.min);
  };

  //
  // `i` = invocation index (0 .. emitCount-1). Each invocation:
  //   1) atomically pops one uint slot id off the allocator stack,
  //   2) draws 14 independent randoms from `uSeed + 16*i + k`,
  //   3) builds the shape position + velocity (5 shape kinds, oracle math),
  //   4) rotates them by the emitter wrapper quaternion,
  //   5) writes vec4 slots on pos/vel/color/particleState/startValues/ext/orbital.
  // The random noise phase offset is derived from the stable birth seed by
  // the simulation pass (`useRandomOffset` honored there, §6 of the rescue).
  const emitKernel = Fn(() => {
    const i = instanceIndex;
    // Explicit count guard: the host dispatches max(1, emitCount) invocations, so the
    // kernel itself must skip the extra one when emitCount === 0.
    If(i.lessThan(uEmitCount), () => {
      // Race-safe ring allocation (never underflows), all in u32 (§4):
      // birthNo and the slot modulo use the native integer `%`.
      const birthNo = atomicAdd(sAllocator.element(0), tuint(1)).toVar();
      const slotIdx = birthNo.mod(ringModU).toVar();
      // Independent random channels: mix(birthNo, systemSeed, channelId) in
      // u32, PCG once per channel, single f32 conversion at the end.
      const rcA = randomChannel(birthNo, uSystemSeed, CH.SHAPE_A);
      const rcB = randomChannel(birthNo, uSystemSeed, CH.SHAPE_B);
      const rcC = randomChannel(birthNo, uSystemSeed, CH.SHAPE_C);
      const rcSpeed = randomChannel(birthNo, uSystemSeed, CH.SPEED);
      const rcSize = randomChannel(birthNo, uSystemSeed, CH.SIZE);
      const rcRot = randomChannel(birthNo, uSystemSeed, CH.ROTATION);
      const rcOpacity = randomChannel(birthNo, uSystemSeed, CH.OPACITY);
      // (shape kind A/B/C + speed, then sheet/size/rot/opacity/life/color/rotol)
      const rcSheet = randomChannel(birthNo, uSystemSeed, CH.START_FRAME);
      const rcLife = randomChannel(birthNo, uSystemSeed, CH.LIFETIME);
      const rcColor = randomChannel(birthNo, uSystemSeed, CH.COLOR);
      const rcRotOl = randomChannel(birthNo, uSystemSeed, CH.ROTOL);

      // ?? Shape emission (all 5 kinds, oracle math) in the emitter's local frame.
      const shE = shapeEmitNodes(
        {
          kind: uShape,
          radius: uRadius,
          thickness: uRadiusThickness,
          arcDeg: uArcDeg,
          coneAngleDeg: uConeAngleDeg,
          rectRX: uRectRX,
          rectRY: uRectRY,
          rectSX: uRectSX,
          rectSY: uRectSY,
          boxSX: uBoxSX,
          boxSY: uBoxSY,
          boxSZ: uBoxSZ,
          boxFrom: uBoxEmitFrom,
          speedMin: uSpeedMin,
          speedMax: uSpeedMax,
        },
        rcA,
        rcB,
        rcC,
        rcSpeed
      );
      const pxL = shE.px;
      const pyL = shE.py;
      const pzL = shE.pz;
      const vxL = shE.vx;
      const vyL = shE.vy;
      const vzL = shE.vz;

      // ?? Emitter pose: rotate by the wrapper quaternion, then (WORLD only)
      // apply the per-axis world scale and add the emitter translation.
      const [rotPX, rotPY, rotPZ] = quatRotateNodes(
        pxL,
        pyL,
        pzL,
        uWrapperQuat
      );
      const [rotVX, rotVY, rotVZ] = quatRotateNodes(
        vxL,
        vyL,
        vzL,
        uWrapperQuat
      );

      // LOCAL (w = 0): identity quaternion, no translation, no extra scale
      // (three applies matrixWorld at draw time). WORLD (w = 1): per-axis world
      // scale on the rotated offset, plus the emitter translation.
      const isWorld = uEmitterPos.w.greaterThan(0.5);
      const sxf = isWorld.select(uWorldScale.x, float(1.0));
      const syf = isWorld.select(uWorldScale.y, float(1.0));
      const szf = isWorld.select(uWorldScale.z, float(1.0));
      const ox = rotPX.mul(sxf).add(uEmitterPos.x);
      const oy = rotPY.mul(syf).add(uEmitterPos.y);
      const oz = rotPZ.mul(szf).add(uEmitterPos.z);

      // Start values (dedicated channels so no field aliases another).
      const opac = mix(uOpMin, uOpMax, rcOpacity);
      const clR = mix(uCRR, uCRX, rcColor);
      const clG = mix(uCGR, uCGX, rcColor);
      const clB = mix(uCBR, uCBX, rcColor);
      const slife = mix(uLifeMin, uLifeMax, rcLife).mul(float(1000.0));
      const ssize = mix(uSizeMin, uSizeMax, rcSize);
      const srot = mix(uRotMin, uRotMax, rcRot);
      const startFrame = floor(mix(uFrMin, uFrMax, rcSheet)).toVar();
      // Separate per-particle rotationOverLifetime speed (oracle keeps its own
      // min/max, distinct from startRotation's rotMin/rotMax).
      const rotSpeed = mix(uRotOLMin, uRotOLMax, rcRotOl);
      // 24-bit stable per-particle seed (§8): created ONCE at birth via the
      // RAW PCG mix (& 0xFFFFFF is exact in f32) and immutable for the whole
      // lifetime. sim derives every non-curve axis + noise phase from it.
      const stableSeedU = stableSeedU32(birthNo, uSystemSeed);

      sPos.element(slotIdx).assign(vec4(ox, oy, oz, float(0.0)));
      sVel.element(slotIdx).assign(vec4(rotVX, rotVY, rotVZ, float(0.0)));
      sCol.element(slotIdx).assign(vec4(clR, clG, clB, opac));
      // lifetime=0, size, rotation, startFrame
      sPS.element(slotIdx).assign(vec4(float(0.0), ssize, srot, startFrame));
      // startValues = (startLife, size, opacity, colorR)
      sSV.element(slotIdx).assign(vec4(slife, ssize, opac, clR));
      // ext = (colorG, colorB, rotSpeed, stableSeed 0..0xFFFFFF)
      sEx
        .element(slotIdx)
        .assign(vec4(clG, clB, rotSpeed, stableSeedU.toFloat()));
      // Orbital pivot = rotated shape offset (oracle positionOffset), w=1.
      sOIA.element(slotIdx).assign(vec4(rotPX, rotPY, rotPZ, float(1.0)));

      // BIRTH events: written by the dedicated `sub-birth-events` pass below
      // (base emit keeps exactly its 8 storage bindings — no FIFO nodes here).
    });
  });

  const emitNode = compute(emitKernel(), maxParticles);

  // Simulation kernel — physics + modifier order mirrors the oracle CPU
  // `updateParticleSystems` inner loop exactly:
  //   gravity -> force fields -> integrate -> collisions -> modifiers ->
  //   lifetime += dt -> trail sample -> write back -> death.
  const noiseOctavesCount = Math.max(
    1,
    Math.round(shapeParams.noiseOctaves || 1)
  );
  // three-noise FBm divides the accumulated octaves by
  // `max = 1 + 0.5 + ... + 0.5^octaves = 2 - 2^-octaves` (the sum includes
  // the trailing 0.5^n exactly like the oracle's `fbmMax` constant).
  const noiseFbmMax = 2 - Math.pow(2, -noiseOctavesCount);
  if (!Number.isFinite(noiseFbmMax) || noiseFbmMax <= 0) {
    throw new Error(
      `three-particles: invalid FBM normalization ${noiseFbmMax}`
    );
  }
  const simKernel = Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(maxParticles)), () => {
      const oiaVec = sOIA.element(i).toVar();
      If(oiaVec.w.greaterThanEqual(float(0.5)), () => {
        const pos = sPos.element(i).xyz.toVar();
        const vel = sVel.element(i).xyz.toVar();
        const ps = sPS.element(i).toVar();
        const sv = sSV.element(i);
        const ex = sEx.element(i);
        const startLife = sv.x;
        const life = ps.x;
        const lifePct = tslMin(life.div(startLife), float(1.0));

        // 1. Gravity
        vel.assign(vel.sub(vec3(uGravityVelocity).mul(uDelta)));

        // 2. Force fields
        if (ffNodes) ffNodes.apply({ pos, vel, delta: uDelta });

        // 3. Integrate
        pos.assign(pos.add(vel.mul(uDelta)));

        // 4. Collisions
        if (cpNodes)
          cpNodes.apply({
            pos,
            vel,
            oiaVec,
            sColorNode: sCol,
            ps,
            startLife,
            particleIdx: i,
            sOrbitalIsActiveNode: sOIA,
          });

        // 5. Modifiers — oracle semantics.
        // 5a. Linear velocity over lifetime: per-axis value is either the
        // baked curve table (`lookupCurve`) or derived from the immutable
        // birth seed in `startColorsExt.w` (constant direct, range via a
        // fixed salt). position.w / velocity.w are plain padding 0.
        if (flags.linearVelocity) {
          const lvx = simAxis(
            linAxes[0],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_X
          );
          const lvy = simAxis(
            linAxes[1],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_Y
          );
          const lvz = simAxis(
            linAxes[2],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_Z
          );
          pos.assign(pos.add(vec3(lvx, lvy, lvz).mul(uDelta)));
        }
        if (flags.orbitalVelocity) {
          // Pivot + offset mirror the oracle `positionOffset`: subtract it,
          // rotate the offset by the Euler, add it back (offset mutation is
          // stored in oia.xyz). Axis speeds derive from the birth seed.
          const offset = vec3(oiaVec.x, oiaVec.y, oiaVec.z).toVar();
          pos.assign(pos.sub(offset));
          const oX = simAxis(
            orbAxes[0],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_X
          );
          const oY = simAxis(
            orbAxes[1],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_Y
          );
          const oZ = simAxis(
            orbAxes[2],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_Z
          );
          // Oracle: Euler(speedX*dt, speedZ*dt, speedY*dt) with order 'XYZ' —
          // intrinsic XYZ. Its matrix is Rx·Ry·Rz, so the vector product
          // applies Z FIRST, then Y, then X (extrinsic Z→Y→X). Keep the
          // per-axis mapping: Euler.x = oX, Euler.y = oZ, Euler.z = oY.
          const angX = oX.mul(uDelta);
          const angY = oZ.mul(uDelta);
          const angZ = oY.mul(uDelta);
          const c3 = cos(angZ),
            s3 = sin(angZ);
          const zx = offset.x.mul(c3).sub(offset.y.mul(s3));
          const zy = offset.x.mul(s3).add(offset.y.mul(c3));
          const zz = offset.z;
          const c2 = cos(angY),
            s2 = sin(angY);
          const yx = zx.mul(c2).add(zz.mul(s2));
          const yz = zx.mul(s2).negate().add(zz.mul(c2));
          const yy = zy;
          const c1 = cos(angX),
            s1 = sin(angX);
          const fx = yx;
          const fy = yy.mul(c1).sub(yz.mul(s1));
          const fz = yy.mul(s1).add(yz.mul(c1));
          pos.assign(pos.add(vec3(fx, fy, fz)));
          oiaVec.assign(vec4(fx, fy, fz, oiaVec.w));
        }
        // 5b. Size / opacity / color over lifetime — start value * multiplier.
        if (flags.sizeOverLifetime) {
          const s = lookupCurve({
            curveIndex: float(curveMap.sizeOverLifetime),
            t: lifePct,
          });
          ps.y.assign(s.mul(sv.y));
        }
        if (flags.opacityOverLifetime) {
          const op = lookupCurve({
            curveIndex: float(curveMap.opacityOverLifetime),
            t: lifePct,
          });
          const col = sCol.element(i).toVar();
          col.w.assign(op.mul(sv.z));
          sCol.element(i).assign(col);
        }
        if (flags.colorOverLifetime) {
          // Oracle: final = startChannel * curveMultiplier (NO mix
          // interpolation against the current color).
          const col = sCol.element(i).toVar();
          const sce = sEx.element(i);
          if (curveMap.colorR >= 0) {
            col.x.assign(
              sv.w.mul(
                lookupCurve({ curveIndex: float(curveMap.colorR), t: lifePct })
              )
            );
          }
          if (curveMap.colorG >= 0) {
            col.y.assign(
              sce.x.mul(
                lookupCurve({ curveIndex: float(curveMap.colorG), t: lifePct })
              )
            );
          }
          if (curveMap.colorB >= 0) {
            col.z.assign(
              sce.y.mul(
                lookupCurve({ curveIndex: float(curveMap.colorB), t: lifePct })
              )
            );
          }
          sCol.element(i).assign(col);
        }
        // 5c. Rotation over lifetime: speed * delta * 0.02 (oracle factor).
        if (flags.rotationOverLifetime) {
          ps.z.assign(ps.z.add(ex.z.mul(uDelta).mul(float(0.02))));
        }
        // 5d. Noise — oracle FBM on a scalar input: per-axis coordinates
        // (t,0,0) / (t,t,0) / (t,t,t); per-octave input scaling
        // t -> t*frequency*2^k with amplitude 0.5^k, divided by
        // fbmMax = 2 - 2^-octaves. uNoisePower carries the single
        // 0.15*strength (per-channel amounts applied below). Random offset
        // comes from the STABLE birth seed (never the frame seed).
        if (flags.noise) {
          const noiseOffset = shapeParams.noiseUseRandomOffset
            ? pcg01(
                stableSeedFromExt(ex.w)
                  .toUint()
                  .mul(tuint(2654435761))
                  .bitXor(CH.NOISE_PHASE)
              ).mul(float(100.0))
            : float(0.0);
          const np = lifePct
            .add(noiseOffset)
            .mul(float(10.0))
            .mul(uNoiseStrength)
            .mul(uNoiseFrequency);
          let noiseX = float(0.0).toVar();
          let noiseY = float(0.0).toVar();
          let noiseZ = float(0.0).toVar();
          let amp = 1.0;
          let lac = 1.0;
          for (let o = 0; o < noiseOctavesCount; o++) {
            const t = np.mul(float(lac));
            const sc = float(amp / noiseFbmMax);
            noiseX.assign(
              noiseX.add(snoise3D({ v: vec3(t, float(0), float(0)) }).mul(sc))
            );
            noiseY.assign(
              noiseY.add(snoise3D({ v: vec3(t, t, float(0)) }).mul(sc))
            );
            noiseZ.assign(noiseZ.add(snoise3D({ v: vec3(t, t, t) }).mul(sc)));
            amp *= 0.5;
            lac *= 2.0;
          }
          If(uNoisePosAmount.greaterThan(float(0.001)), () => {
            pos.assign(
              pos.add(
                vec3(noiseX, noiseY, noiseZ)
                  .mul(uNoisePower)
                  .mul(uNoisePosAmount)
              )
            );
          });
          If(uNoiseRotAmount.greaterThan(float(0.001)), () => {
            ps.z.assign(ps.z.add(noiseX.mul(uNoisePower).mul(uNoiseRotAmount)));
          });
          If(uNoiseSizeAmount.greaterThan(float(0.001)), () => {
            ps.y.assign(
              ps.y.add(noiseX.mul(uNoisePower).mul(uNoiseSizeAmount))
            );
          });
        }

        ps.x.assign(ps.x.add(uDeltaMs));

        // Trail history is sampled by the dedicated `trail-history` pass
        // (see `trailHistoryNode` below), keeping this kernel at 8 bindings.

        // Position/velocity vec4 padding stays plain 0 (axes are seed-derived).
        sPos.element(i).assign(vec4(pos, float(0.0)));
        sVel.element(i).assign(vec4(vel, float(0.0)));
        sPS.element(i).assign(ps);
        sOIA.element(i).assign(oiaVec);

        // Death: ring allocator needs no push (the monotonically increasing
        // birth counter owns recycling); mark the slot inactive + zero color.
        // With DEATH-trigger sub-emitters the transient marker is -1; the
        // dedicated `sub-death-events` pass writes the FIFO then clears to 0.
        If(ps.x.greaterThan(startLife), () => {
          const inactive = sOIA.element(i).toVar();
          sOIA
            .element(i)
            .assign(
              vec4(
                inactive.x,
                inactive.y,
                inactive.z,
                hasDeathFifo ? float(-1.0) : float(0.0)
              )
            );
          sCol
            .element(i)
            .assign(vec4(float(0.0), float(0.0), float(0.0), float(0.0)));
        });
      });
    });
  });

  const simNode = compute(simKernel(), maxParticles);

  // ?? Dedicated trail-history pass (runs AFTER `simulate`) ??
  // Bindings: position, orbitalIsActive, trailHistory, trailMeta (= 4 <= 8).
  // Same adaptive ring sampling the inlined sim block used; reads the
  // post-simulation position straight back from the position pool.
  let trailHistoryNode: ReturnType<typeof compute> | null = null;
  if (sTrail && sTrailMeta && trailDesc) {
    const trailHistoryKernel = Fn(() => {
      const i = instanceIndex;
      If(float(i).lessThan(float(maxParticles)), () => {
        const oiaVec = sOIA.element(i).toVar();
        // active (1) or death-pending (-1, event pass not yet run) => sample.
        const activeNow = oiaVec.w.greaterThanEqual(float(0.5));
        const pendingDeath = oiaVec.w.lessThan(float(-0.5));
        If(activeNow.or(pendingDeath), () => {
          const pos = sPos.element(i).toVar();
          const L = float(trailDesc.length);
          const tr = float(trailRows);
          const curIdx = i.mul(tuint(2));
          const cursor = float(atomicLoad(sTrailMeta.element(curIdx))).toVar();
          const count = float(
            atomicLoad(sTrailMeta.element(curIdx.add(tuint(1))))
          ).toVar();
          const baseI = i.mul(float(trailRows));
          const prev = sTrail.element(baseI.add(cursor)).toVar();
          const ddx = pos.x.sub(prev.x);
          const ddy = pos.y.sub(prev.y);
          const ddz = pos.z.sub(prev.z);
          const dist = sqrt(ddx.mul(ddx).add(ddy.mul(ddy)).add(ddz.mul(ddz)));
          const firstSample = count.lessThan(float(0.5));
          const farEnough = dist.greaterThanEqual(
            float(Math.max(1e-6, trailDesc.minVertexDistance))
          );
          If(firstSample.or(farEnough), () => {
            // cursor indexes the LAST written sample; 0..L-1, wrapping.
            const newCursor = cursor
              .greaterThanEqual(L.sub(float(1)))
              .select(float(0), cursor.add(float(1)));
            sTrail
              .element(baseI.add(newCursor))
              .assign(vec4(pos.x, pos.y, pos.z, uNowMs));
            atomicStore(sTrailMeta.element(curIdx), newCursor.toUint());
            atomicStore(
              sTrailMeta.element(curIdx.add(tuint(1))),
              tslMin(count.add(float(1)), float(trailDesc.length)).toUint()
            );
          });
        });
      });
    });
    trailHistoryNode = compute(trailHistoryKernel(), maxParticles);
  }

  // ?? Dedicated sub-emitter event passes (never bound into base emit/sim) ??
  // BIRTH: born slots are recovered from the monotonic allocator counter
  // (counterAfter - emitCount + i, mod maxParticles); position/velocity are
  // read back from the parent pools the emit kernel just wrote.
  let subBirthEventsNode: ReturnType<typeof compute> | null = null;
  if (birthFifos.length > 0) {
    const subBirthKernel = Fn(() => {
      const i = instanceIndex;
      If(i.lessThan(uEmitCount), () => {
        const counterAfter = atomicLoad(sAllocator.element(0)).toVar();
        const birthNo = counterAfter.sub(uEmitCount).add(i).toVar();
        const slot = birthNo.mod(ringModU).toVar();
        const p = sPos.element(slot).toVar();
        const v = sVel.element(slot).toVar();
        for (const f of birthFifos) {
          writeFifoEvent(f, p.x, p.y, p.z, v.x, v.y, v.z);
        }
      });
    });
    subBirthEventsNode = compute(subBirthKernel(), maxParticles);
  }

  // DEATH: the sim kernel marks `orbitalIsActive.w = -1` for slots that died
  // this step; this pass appends the events, then clears the marker to 0.
  let subDeathEventsNode: ReturnType<typeof compute> | null = null;
  if (deathFifos.length > 0) {
    const subDeathKernel = Fn(() => {
      const i = instanceIndex;
      If(float(i).lessThan(float(maxParticles)), () => {
        const oiaVec = sOIA.element(i).toVar();
        If(oiaVec.w.equal(float(-1.0)), () => {
          const p = sPos.element(i).toVar();
          const v = sVel.element(i).toVar();
          for (const f of deathFifos) {
            writeFifoEvent(f, p.x, p.y, p.z, v.x, v.y, v.z);
          }
          sOIA
            .element(i)
            .assign(vec4(oiaVec.x, oiaVec.y, oiaVec.z, float(0.0)));
        });
      });
    });
    subDeathEventsNode = compute(subDeathKernel(), maxParticles);
  }

  // ?? Real per-pass layouts (derived from each pass's actual resources) ??
  const layout = (
    name: string,
    storageNodes: Array<ShaderNodeObject<Node> | null>,
    uniformNodes: Array<ShaderNodeObject<Node> | null>
  ): PassLayout => ({
    name,
    storageBindings: new Set(
      storageNodes.filter((n) => n !== null && n !== undefined)
    ).size,
    uniformBindings: new Set(
      uniformNodes.filter((n) => n !== null && n !== undefined)
    ).size,
  });
  const basePool = [sPos, sVel, sCol, sPS, sSV, sEx, sOIA, sAllocator];
  const emitUniforms: Array<ShaderNodeObject<Node>> = [
    uEmitCount,
    uSeed,
    uShape,
    uRadius,
    uRadiusThickness,
    uArcDeg,
    uConeAngleDeg,
    uRectRX,
    uRectRY,
    uRectSX,
    uRectSY,
    uBoxSX,
    uBoxSY,
    uBoxSZ,
    uBoxEmitFrom,
    uSpeedMin,
    uSpeedMax,
    uSizeMin,
    uSizeMax,
    uRotMin,
    uRotMax,
    uOpMin,
    uOpMax,
    uLifeMin,
    uLifeMax,
    uCRR,
    uCRX,
    uCGR,
    uCGX,
    uCBR,
    uCBX,
    uFrMin,
    uFrMax,
    uRotOLMin,
    uRotOLMax,
    uWrapperQuat,
    uEmitterPos,
    uWorldScale,
  ];
  const simUniforms: Array<ShaderNodeObject<Node>> = [
    uDelta,
    uDeltaMs,
    uGravityVelocity,
    uNoiseStrength,
    uNoisePower,
    uNoiseFrequency,
    uNoisePosAmount,
    uNoiseRotAmount,
    uNoiseSizeAmount,
    sCD as ShaderNodeObject<Node>,
    ...Array.from(axisUniforms.values()).flat(),
  ];
  const passLayouts: PassLayout[] = [
    layout('emit', basePool, emitUniforms),
    layout('simulate', basePool, simUniforms),
  ];
  if (trailHistoryNode) {
    passLayouts.push(
      layout('trail-history', [sPos, sOIA, sTrail, sTrailMeta], [uNowMs])
    );
  }
  if (subBirthEventsNode) {
    passLayouts.push(
      layout(
        'sub-birth-events',
        [
          sAllocator,
          sPos,
          sVel,
          ...birthFifos.flatMap((f) => [f.count, f.payload]),
        ],
        [uEmitCount, uFifoBase]
      )
    );
  }
  if (subDeathEventsNode) {
    passLayouts.push(
      layout(
        'sub-death-events',
        [sPos, sVel, sOIA, ...deathFifos.flatMap((f) => [f.count, f.payload])],
        [uFifoBase]
      )
    );
  }

  return {
    emitNode,
    simNode,
    trailHistoryNode,
    subBirthEventsNode,
    subDeathEventsNode,
    computeNodes: [
      emitNode,
      ...(subBirthEventsNode ? [subBirthEventsNode] : []),
      simNode,
      ...(subDeathEventsNode ? [subDeathEventsNode] : []),
      ...(trailHistoryNode ? [trailHistoryNode] : []),
    ],
    passLayouts,
    uniforms: {
      delta: uDelta,
      deltaMs: uDeltaMs,
      nowMs: uNowMs,
      gravityVelocity: uGravityVelocity,
      noiseStrength: uNoiseStrength,
      noisePower: uNoisePower,
      noiseFrequency: uNoiseFrequency,
      noisePositionAmount: uNoisePosAmount,
      noiseRotationAmount: uNoiseRotAmount,
      noiseSizeAmount: uNoiseSizeAmount,
      emitCount: uEmitCount,
      seed: uSeed,
      /** Ping-pong FIFO window index for this frame (0 or 1). */
      fifoBase: uFifoBase,
    },
    shapeUniforms,
    buffers,
    allocatorCount,
    packedDataNode: sCD as ShaderNodeObject<Node>,
    passNames: [
      'emit',
      ...(subBirthEventsNode ? ['sub-birth-events'] : []),
      'simulate',
      ...(subDeathEventsNode ? ['sub-death-events'] : []),
      ...(trailHistoryNode ? ['trail-history'] : []),
    ],
    trailMeta: buffers.trailMeta,
    // Emitter-pose uniforms, refreshed once per frame by the CPU (scalar only).
    emitterPose: {
      positionW: uEmitterPos,
      wrapperQuat: uWrapperQuat,
      worldScale: uWorldScale,
    },
    forceFieldInfo: ffNodes
      ? { offset: forceFieldOffset, countUniform: ffNodes.countUniform }
      : null,
    collisionPlaneInfo: cpNodes
      ? { offset: collisionOffset, countUniform: cpNodes.countUniform }
      : null,
  };
}

// ─── Sub-emitter init kernel ─────────────────────────────────────────────────

/** Compute pipeline for one sub-emitter child pool. */
export type SubEmitterInitPipeline = {
  /** Pass A: FIFO -> spawn commands (+ ring slot allocation). */
  commandBuildNode: unknown;
  /** Pass B: spawn commands -> child state (8 bindings, no allocator). */
  childInitNode: unknown;
  /** 1-invocation pass that zeroes the *other* ping-pong counter. */
  counterClearNode: unknown;
  /** CPU-owned vec4 command buffer (uploaded once, kernels own it after). */
  commandBuffer: StorageBufferAttribute;
  /** Real budgets for every pass this pipeline owns. */
  passLayouts: PassLayout[];
  passName: string;
  counterClearPassName: string;
  /** Per-frame scalar uniforms. */
  uniforms: {
    seed: ShaderNodeObject<Node>;
    fifoBase: ShaderNodeObject<Node>;
    inherit: ShaderNodeObject<Node>;
    /** Emitter pose, written like the main pipeline's `emitterPose`. */
    positionW: ShaderNodeObject<Node>;
    wrapperQuat: ShaderNodeObject<Node>;
  };
  buffers: ModifierStorageBuffers;
};

/**
 * Builds the sub-emitter init kernel: reads the parent event FIFO and, per
 * event, initializes `particlesPerEvent` slots in the child pool using the
 * child's own shape math (+ the inherited speed of the parent particle).
 *
 * Slot state layout is identical to {@link createModifierComputeUpdate}; the
 * child's *simulation* pass is a normal modifier pipeline created on the same
 * buffers (its emission node is unused because every child particle is created
 * by an event).
 */
export function createSubEmitterInitUpdate(
  child: ModifierStorageBuffers,
  childMax: number,
  childParams: ShapeEmitParams,
  parent: ModifierStorageBuffers,
  parentMax: number,
  fifo: SubEmitterFifo,
  inheritVelocity: number,
  particlesPerEvent: number,
  childVelValues?: VelocityAxisValues
): SubEmitterInitPipeline {
  const capacity = Math.max(1, fifo.capacity);
  const perEvent = Math.max(1, particlesPerEvent);
  const windowSize = subEmitterWindowSize(capacity);

  // One-time system seed for this child init pipeline (§5): u32, written ONCE.
  const uSystemSeed = uniform(nextSystemSeed(), 'uint');
  const uSeed = uSystemSeed;
  const uInherit = uniform(float(Math.max(0, inheritVelocity)));
  const uFifoBase = uniform(float(0), 'uint');
  const uWrapperQuat = uniform(new Vector4(0, 0, 0, 1));
  const uEmitterPos = uniform(new Vector4(0, 0, 0, 0));
  const uWorldScale = uniform(new Vector3(1, 1, 1));

  // Child shape scalars (same flat table as the main pipeline).
  const cRadius = uniform(float(childParams.radius));
  const cThickness = uniform(float(childParams.radiusThickness));
  const cArc = uniform(float(childParams.arcDeg));
  const cAngle = uniform(float(childParams.coneAngleDeg));
  const cRRX = uniform(float(childParams.rectangleRotXDeg));
  const cRRY = uniform(float(childParams.rectangleRotYDeg));
  const cRSX = uniform(float(childParams.rectangleScaleX));
  const cRSY = uniform(float(childParams.rectangleScaleY));
  const cBSX = uniform(float(childParams.boxScaleX));
  const cBSY = uniform(float(childParams.boxScaleY));
  const cBSZ = uniform(float(childParams.boxScaleZ));
  const cBF = uniform(float(childParams.boxEmitFrom));
  const cKind = uniform(float(childParams.shapeKind));
  const cSpeedMin = uniform(float(childParams.speedMin));
  const cSpeedMax = uniform(float(childParams.speedMax));
  const cSizeMin = uniform(float(childParams.sizeMin));
  const cSizeMax = uniform(float(childParams.sizeMax));
  const cRotMin = uniform(float(childParams.rotMin));
  const cRotMax = uniform(float(childParams.rotMax));
  const cOpMin = uniform(float(childParams.opacityMin));
  const cOpMax = uniform(float(childParams.opacityMax));
  const cLifeMin = uniform(float(childParams.lifeMin));
  const cLifeMax = uniform(float(childParams.lifeMax));
  const cCRR = uniform(float(sRGBToLinear(childParams.colorRMin)));
  const cCRX = uniform(float(sRGBToLinear(childParams.colorRMax)));
  const cCGR = uniform(float(sRGBToLinear(childParams.colorGMin)));
  const cCGX = uniform(float(sRGBToLinear(childParams.colorGMax)));
  const cCBR = uniform(float(sRGBToLinear(childParams.colorBMin)));
  const cCBX = uniform(float(sRGBToLinear(childParams.colorBMax)));
  const cFrMin = uniform(float(childParams.startFrameMin));
  const cFrMax = uniform(float(childParams.startFrameMax));

  // Child pool (bindings 1..7) + its own ring allocator (binding 8). The
  // per-particle velocity axes are seed-derived (no per-axis table buffer).
  const cPos = storage(child.position, 'vec4', childMax);
  const cVel = storage(child.velocity, 'vec4', childMax);
  const cCol = storage(child.color, 'vec4', childMax);
  const cPS = storage(child.particleState, 'vec4', childMax);
  const cSV = storage(child.startValues, 'vec4', childMax);
  const cEx = storage(child.startColorsExt, 'vec4', childMax);
  const cOIA = storage(child.orbitalIsActive, 'vec4', childMax);
  const cAlloc = storage(
    child.allocator,
    'uint',
    Math.max(1, childMax + 1)
  ).toAtomic();
  const cRingMod = float(childMax);
  // Uint ring modulus for the integer slot modulo in Pass A (§4).
  const cRingModU = tuint(childMax);
  void parent;
  void parentMax; // read through the FIFO payload only
  // Compact spawn-command buffer (vec4 stream). Command `m` = the child
  // particle with flat index `m` (event i, slot jj -> m = i*perEvent + jj):
  //   vec4 1+2m : (ringSlot, eventX, eventY, eventZ)
  //   vec4 2+2m : (velX, velY, velZ, 0)
  // vec4 slot 0 is the header (x = live event count written by Pass A).
  const commandBuffer = new StorageBufferAttribute(
    new Float32Array(4 * (1 + capacity * perEvent)),
    4
  );
  const fifoCounter = storage(
    fifo.counter,
    'uint',
    Math.max(1, (fifo.counter.array as Uint32Array).length)
  ).toAtomic();
  const fifoPayload = storage(
    fifo.payload,
    'float',
    Math.max(1, (fifo.payload.array as Float32Array).length)
  );
  const sCmd = storage(commandBuffer, 'vec4', 1 + capacity * perEvent);

  // Child velocity-over-lifetime axes (parity with the main emit kernel).
  const cParseAxis = (
    rawAxis:
      | import('../types.js').Constant
      | import('../types.js').RandomBetweenTwoConstants
      | undefined,
    ci: number
  ): { min: number; max: number; isRange: boolean } => {
    if (rawAxis && typeof rawAxis === 'object' && 'min' in rawAxis) {
      const mn = Number(rawAxis.min) || 0;
      const mx = Number(rawAxis.max) || 0;
      return { min: mn, max: mx, isRange: mn !== mx };
    }
    const c = typeof rawAxis === 'number' ? rawAxis : 0;
    return { min: c, max: c, isRange: false };
  };
  const cVv = childVelValues ?? {
    linear: [undefined, undefined, undefined],
    orbital: [undefined, undefined, undefined],
  };
  const cLin = [0, 1, 2].map((k) => cParseAxis(cVv.linear[k], -1));
  const cOrb = [0, 1, 2].map((k) => cParseAxis(cVv.orbital[k], -1));
  void cLin;
  void cOrb; // seed-derived in the child sim kernel (no axis buffer)

  // The *other* window's counter is cleared by a dedicated 1-invocation
  // pass so the next frame's writers start from 0 (no race with this frame).
  const otherIdx = uFifoBase.equal(float(0)).select(tuint(1), tuint(0));
  const counterClearKernel = Fn(() => {
    atomicStore(fifoCounter.element(otherIdx), tuint(0));
  });
  const counterClearNode = compute(counterClearKernel(), 1);

  // ?? Pass A: events -> compact spawn commands (4 storage bindings) ??
  // fifoCounter + fifoPayload + child allocator + command buffer. Allocates
  // the ring slots here so Pass B needs no allocator (8-binding budget).
  const commandBuildKernel = Fn(() => {
    const i = instanceIndex;
    const winBase = uFifoBase.mul(float(windowSize)).toVar();
    const count = float(atomicLoad(fifoCounter.element(uFifoBase))).toVar();
    If(float(i).equal(float(0)), () => {
      // Header (vec4 slot 0): x = this frame's live event count.
      sCmd.element(float(0)).assign(vec4(count, float(0), float(0), float(0)));
    });
    If(float(i).lessThan(count), () => {
      const eb = winBase.add(float(i).mul(float(SUB_EMITTER_EVENT_STRIDE)));
      const eX = fifoPayload.element(eb).toVar();
      const eY = fifoPayload.element(eb.add(float(1))).toVar();
      const eZ = fifoPayload.element(eb.add(float(2))).toVar();
      const vX = fifoPayload.element(eb.add(float(3))).toVar();
      const vY = fifoPayload.element(eb.add(float(4))).toVar();
      const vZ = fifoPayload.element(eb.add(float(5))).toVar();
      // Unrolled per-event particle loop (particlesPerEvent is a host constant).
      for (let jj = 0; jj < perEvent; jj++) {
        // Ring slot: uint modulo with the native integer `%` (§4).
        const birthNo = atomicAdd(cAlloc.element(0), tuint(1)).toVar();
        const slot = birthNo.mod(cRingModU).toVar();
        const m = float(i.mul(float(perEvent)).add(float(jj)));
        sCmd
          .element(float(1).add(m.mul(float(2))))
          .assign(vec4(slot.toFloat(), eX, eY, eZ));
        sCmd
          .element(float(2).add(m.mul(float(2))))
          .assign(vec4(vX, vY, vZ, float(0)));
      }
    });
  });
  const commandBuildNode = compute(commandBuildKernel(), capacity);

  // ?? Pass B: spawn commands -> child state (exactly 8 storage bindings) ??
  // command buffer + the 7 child pools; no allocator, no FIFO nodes.
  const childInitKernel = Fn(() => {
    const i = instanceIndex;
    const header = sCmd.element(float(0)).toVar();
    If(float(i).lessThan(header.x.mul(float(perEvent))), () => {
      const m = float(i);
      // uint index for the integer channel mixes (§8).
      // `i` (instanceIndex) is already 'uint' in r186 — no extra conversion.
      const mU = i;
      const c0 = sCmd.element(float(1).add(m.mul(float(2)))).toVar();
      const c1 = sCmd.element(float(2).add(m.mul(float(2)))).toVar();
      const slot = c0.x;
      const eX = c0.y;
      const eY = c0.z;
      const eZ = c0.w;
      const vX = c1.x;
      const vY = c1.y;
      const vZ = c1.z;

      // Oracle: startSpeed += |parentVelocity| * inheritVelocity
      const parentSpeed = sqrt(
        vX.mul(vX).add(vY.mul(vY)).add(vZ.mul(vZ))
      ).toVar();
      const spAdd = parentSpeed.mul(uInherit);

      {
        // Integer channel mixes per child particle `m` (flat command index):
        // one dedicated channel id + one PCG draw each (u32, uint-first).
        const rcA = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_A)
        );
        const rcB = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_B)
        );
        const rcC = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_C)
        );
        const rcSpeed = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SPEED)
        );
        const rcSize = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SIZE)
        );
        const rcRot = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.ROTATION)
        );
        const rcOpacity = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.OPACITY)
        );
        const rcSheet = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.START_FRAME)
        );
        const rcLife = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.LIFETIME)
        );
        const rcColor = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.COLOR)
        );
        const rcRotOl = pcg01(
          mU.mul(tuint(2654435761)).bitXor(uSystemSeed).bitXor(CH.ROTOL)
        );

        const shE = shapeEmitNodes(
          {
            kind: cKind,
            radius: cRadius,
            thickness: cThickness,
            arcDeg: cArc,
            coneAngleDeg: cAngle,
            rectRX: cRRX,
            rectRY: cRRY,
            rectSX: cRSX,
            rectSY: cRSY,
            boxSX: cBSX,
            boxSY: cBSY,
            boxSZ: cBSZ,
            boxFrom: cBF,
            speedMin: cSpeedMin.add(spAdd),
            speedMax: cSpeedMax.add(spAdd),
          },
          rcA,
          rcB,
          rcC,
          rcSpeed
        );

        // Child shape offset in the child's own frame, then the same
        // emitter-pose rules as the main kernel.
        const [rx, ry, rz] = quatRotateNodes(
          shE.px,
          shE.py,
          shE.pz,
          uWrapperQuat
        );
        const [rvx, rvy, rvz] = quatRotateNodes(
          shE.vx,
          shE.vy,
          shE.vz,
          uWrapperQuat
        );
        const isWorld = uEmitterPos.w.greaterThan(0.5);
        const sxf = isWorld.select(uWorldScale.x, float(1.0));
        const syf = isWorld.select(uWorldScale.y, float(1.0));
        const szf = isWorld.select(uWorldScale.z, float(1.0));
        const px = rx.mul(sxf).add(eX);
        const py = ry.mul(syf).add(eY);
        const pz = rz.mul(szf).add(eZ);

        const opac = mix(cOpMin, cOpMax, rcOpacity);
        const clR = mix(cCRR, cCRX, rcColor);
        const clG = mix(cCGR, cCGX, rcColor);
        const clB = mix(cCBR, cCBX, rcColor);
        const slife = mix(cLifeMin, cLifeMax, rcLife).mul(float(1000.0));
        const ssize = mix(cSizeMin, cSizeMax, rcSize);
        const srot = mix(cRotMin, cRotMax, rcRot);
        const startFrame = floor(mix(cFrMin, cFrMax, rcSheet)).toVar();
        // Separate rotationOverLifetime speed (own min/max, oracle parity).
        const rotSpeed = mix(
          float(childParams.rotOverLifeMin),
          float(childParams.rotOverLifeMax),
          rcRotOl
        );

        // 24-bit stable per-child-particle seed (§8), exact in f32, stored
        // in startColorsExt.w; the child sim recovers it with `toUint()`.
        const stableSeedU = pcgRawU32(
          mixBirthSeed(mU, uSystemSeed, CH.STABLE_SEED)
        )
          .bitAnd(tuint(0x00ffffff))
          .toVar();

        cPos.element(slot).assign(vec4(px, py, pz, float(0.0)));
        cVel.element(slot).assign(vec4(rvx, rvy, rvz, float(0.0)));
        cCol.element(slot).assign(vec4(clR, clG, clB, opac));
        cPS.element(slot).assign(vec4(float(0.0), ssize, srot, startFrame));
        cSV.element(slot).assign(vec4(slife, ssize, opac, clR));
        // ext = (colorG, colorB, rotSpeed, stableSeed 0..0xFFFFFF)
        cEx
          .element(slot)
          .assign(vec4(clG, clB, rotSpeed, stableSeedU.toFloat()));
        // Orbital pivot = rotated child shape offset (oracle parity).
        cOIA.element(slot).assign(vec4(rx, ry, rz, float(1.0)));
      }
    });
  });

  const childInitNode = compute(
    childInitKernel(),
    Math.max(1, capacity * perEvent)
  );

  const initPassLayouts: PassLayout[] = [
    {
      name: 'sub-command-build',
      storageBindings: 4, // fifoCounter, fifoPayload, allocator, commands
      uniformBindings: 1, // uFifoBase
    },
    {
      // command buffer + the 7 child pools; NO allocator here.
      name: 'sub-child-init',
      storageBindings: 8,
      uniformBindings: 35, // seed/inherit/pose + 30 child shape scalars
    },
    {
      name: 'sub-counter-clear',
      storageBindings: 1,
      uniformBindings: 1,
    },
  ];

  return {
    commandBuildNode,
    childInitNode,
    counterClearNode,
    commandBuffer,
    passLayouts: initPassLayouts,
    passName: fifo.trigger === 0 ? 'sub-birth' : 'sub-death',
    counterClearPassName: 'fifo-counter-clear',
    uniforms: {
      seed: uSeed,
      fifoBase: uFifoBase,
      inherit: uInherit,
      positionW: uEmitterPos,
      wrapperQuat: uWrapperQuat,
    },
    buffers: child,
  };
}

// ─── Trail ribbon build kernel ───────────────────────────────────────────────

/** Compute pipeline that expands the trail history into ribbon vertices. */
export type TrailRibbonPipeline = {
  ribbonNode: unknown;
  /** Real budget for the single ribbon expansion pass. */
  passLayouts: PassLayout[];
  uniforms: { nowMs: ShaderNodeObject<Node> };
  buffers: Record<string, StorageBufferAttribute>;
};

/**
 * Expands the per-particle trail history ring into the ribbon vertex streams
 * consumed by `tsl-trail-ribbon-material` (position+halfWidth, next+alpha,
 * uv+color). Slot `s` maps to the history sample `cursor - s`, so the newest
 * sample is the head (t = 0) and older samples run toward the tail (t = 1),
 * exactly like the CPU oracle. Unused slots get half-width 0.
 */
export function createTrailRibbonUpdate(
  desc: TrailRibbonDesc
): TrailRibbonPipeline {
  const L = desc.length;
  const rows = L + 1;
  const vertexCount = Math.max(1, desc.maxParticles) * Math.max(1, L);

  const uNowMs = uniform(float(0));

  // Bake the along-trail curves into one read-mostly f32 table; each curve
  // occupies exactly CURVE_RESOLUTION floats, inactive curves get identity.
  const activeFns = [
    desc.curveFns.width ?? ((t: number) => t),
    desc.curveFns.opacity ?? ((t: number) => t),
    desc.curveFns.colorR ?? ((t: number) => t),
    desc.curveFns.colorG ?? ((t: number) => t),
    desc.curveFns.colorB ?? ((t: number) => t),
  ];
  const curveData = new Float32Array(activeFns.length * CURVE_RESOLUTION);
  activeFns.forEach((fn, k) => {
    curveData.set(bakeCurve(fn, CURVE_RESOLUTION), k * CURVE_RESOLUTION);
  });
  const IDX_WIDTH = 0;
  const IDX_OPACITY = 1;
  const IDX_CR = 2;
  const IDX_CG = 3;
  const IDX_CB = 4;

  const aPos = storage(desc.position, 'vec4', vertexCount * 2);
  const aNext = storage(desc.next, 'vec4', vertexCount * 2);
  const aUVA = storage(desc.uvColorA, 'vec4', vertexCount * 2);
  const aColB = storage(desc.colorB, 'vec4', vertexCount * 2);
  const hist = storage(desc.history, 'vec4', rows * desc.maxParticles);
  const sMeta = storage(
    desc.meta,
    'uint',
    Math.max(1, (desc.meta.array as Uint32Array).length)
  ).toAtomic();
  const pColor = storage(desc.particleColor, 'vec4', desc.maxParticles);
  const sCD = buffer(curveData, 'float', curveData.length);
  const lookupCurve = createCurveLookup(sCD);

  const halfWidthBase = float(desc.width * 0.5);

  const kernel = Fn(() => {
    const idx = instanceIndex;
    const lf = float(L);
    const rowF = float(rows);
    const i = floor(float(idx).div(lf));
    const s = float(idx).sub(i.mul(lf));

    If(i.lessThan(float(desc.maxParticles)), () => {
      const base = i.mul(rowF);
      const cursor = float(atomicLoad(sMeta.element(i.mul(float(2))))).toVar();
      const count = float(
        atomicLoad(sMeta.element(i.mul(float(2)).add(float(1))))
      ).toVar();

      If(count.greaterThan(float(0.5)), () => {
        // Ring sample of slot s: (cursor - s) mod L, oldest -> newest order.
        const raw = cursor.sub(s).add(lf);
        const si = raw.sub(floor(raw.div(lf)).mul(lf));
        const sample = hist.element(base.add(si)).toVar();
        const nextRaw = si.add(float(1));
        const ni = nextRaw.sub(floor(nextRaw.div(lf)).mul(lf));
        const nextSample = hist.element(base.add(ni)).toVar();

        // Along-trail parameter: head (newest) = 0, tail = 1.
        const t = count
          .greaterThan(float(1.5))
          .select(s.div(tslMax(count.sub(float(1)), float(1))), float(0));

        // Curves along the trail (baked, identity when not supplied).
        const wScale = lookupCurve({
          curveIndex: float(IDX_WIDTH),
          t,
        });
        const oScale = lookupCurve({
          curveIndex: float(IDX_OPACITY),
          t,
        });
        const cr = lookupCurve({ curveIndex: float(IDX_CR), t });
        const cg = lookupCurve({ curveIndex: float(IDX_CG), t });
        const cb = lookupCurve({ curveIndex: float(IDX_CB), t });

        // Particle colour (RGBA, already modifier-shaped) modulated by the
        // along-trail colour curves; the current particle is slot `i`.
        const pcol = pColor.element(i).toVar();

        // Age expiry (trail.maxTime) and unused-slot masking.
        const inRange = s.lessThan(count);
        const ageOk =
          desc.maxTime > 0
            ? uNowMs.sub(sample.w).lessThanEqual(float(desc.maxTime))
            : inRange;
        const alive = inRange.and(ageOk);
        const hw = alive.select(halfWidthBase.mul(wScale), float(0));
        const alpha = alive.select(oScale.mul(pcol.w), float(0));

        const nPos = count
          .greaterThan(float(1.5))
          .select(nextSample.xyz, sample.xyz);

        // Two vertices per slot: side 0 (uv.x = 0) and side 1 (uv.x = 1).
        for (let side = 0; side < 2; side++) {
          const vi = float(idx).mul(float(2)).add(float(side));
          aPos.element(vi).assign(vec4(sample.x, sample.y, sample.z, hw));
          aNext.element(vi).assign(vec4(nPos.x, nPos.y, nPos.z, alpha));
          aUVA
            .element(vi)
            .assign(vec4(float(side), t, cr.mul(pcol.x), cg.mul(pcol.y)));
          aColB
            .element(vi)
            .assign(vec4(cb.mul(pcol.z), alpha, float(0), float(0)));
        }
      });
    });
  });

  const ribbonNode = compute(kernel(), vertexCount);

  return {
    ribbonNode,
    // 4 ribbon streams + history + meta + particle color = 7 storage bindings.
    passLayouts: [
      { name: 'trail-ribbon', storageBindings: 7, uniformBindings: 2 },
    ],
    uniforms: { nowMs: uNowMs },
    buffers: {
      position: desc.position,
      next: desc.next,
      uvColorA: desc.uvColorA,
      colorB: desc.colorB,
      history: desc.history,
    },
  };
}
