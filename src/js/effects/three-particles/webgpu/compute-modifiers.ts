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
import {
  CURVE_RESOLUTION,
  bakeCurve,
  type BakedCurveMap,
} from './curve-bake.js';
import { snoise3D } from './tsl-noise.js';
import {
  FORCE_FIELD_DATA_SIZE,
  createForceFieldTSL,
} from './compute-force-fields.js';
import {
  COLLISION_PLANE_DATA_SIZE,
  createCollisionPlaneTSL,
} from './compute-collision-planes.js';
import { sRGBToLinear } from '../color-utils.js';

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

/**
 * Allocates the FIFO buffers for one sub-emitter channel: an integer
 * `atomic<u32>` counter pair (ping-pong) plus two ordinary f32 payload
 * windows. Float payload never goes through an atomic; counters are integer.
 */
export const createSubEmitterFifoAttribute = (capacity: number): SubEmitterFifo =>
  ({
    counter: new StorageBufferAttribute(new Uint32Array(2), 1),
    payload: new StorageBufferAttribute(
      new Float32Array(
        2 * SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity)
      ),
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
  /**
   * Per-particle velocity-axis values (linear xyz + orbital.x). Present only
   * when velocityOverLifetime is active (`axes == null` otherwise, so the base
   * pool stays at the 8 guaranteed storage slots). orbital.y lives in
   * `position.w`, orbital.z in `velocity.w`.
   */
  axes: StorageBufferAttribute | null;
  /** Trail ring integer metadata (`atomic<u32>`, 2 words per particle), or null. */
  trailMeta: StorageBufferAttribute | null;
  /** Packed read-mostly f32 tables (curves + force fields + collision planes), uniform-backed. */
  packedData: Float32Array;
};

/** The complete GPU-owned pipeline handle returned by `createComputePipeline`. */
export type ModifierComputePipeline = {
  /** Emission compute node (dispatched before simulation each frame). */
  emitNode: ReturnType<typeof compute>;
  /** Per-frame simulation compute node. */
  simNode: ReturnType<typeof compute>;
  /** Convenience: both nodes in dispatch order. */
  computeNodes: ReturnType<typeof compute>[];
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
  hasVelocityAxes = false,
  trailLength = 0
): { buffers: ModifierStorageBuffers; allocatorCount: number } {
  const Cls = instanced
    ? StorageInstancedBufferAttribute
    : StorageBufferAttribute;

  const curveLen = Math.max(curveData.length, 1);
  const ffSize  = hasForceFields     ? FORCE_FIELD_DATA_SIZE     : 0;
  const cpSize  = hasCollisionPlanes ? COLLISION_PLANE_DATA_SIZE : 0;
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
  const allocatorData  = new Uint32Array(allocatorCount);
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
      position:        new Cls(new Float32Array(maxParticles * 4), 4),
      velocity:        new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
      color:           new Cls(new Float32Array(maxParticles * 4), 4),
      particleState:   new Cls(new Float32Array(maxParticles * 4), 4),
      startValues:     new Cls(new Float32Array(maxParticles * 4), 4),
      startColorsExt:  new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
      orbitalIsActive: new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
      allocator:       new StorageBufferAttribute(allocatorData, 1),
      axes: hasVelocityAxes
        ? new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4)
        : null,
      trailMeta,
      packedData,
    },
    allocatorCount,
  };
}

// ??? Curve lookup helper ????????????????????????????????????????????????????
function createCurveLookup(sCurveData: ShaderNodeObject<Node>) {
  return Fn(({ curveIndex, t }: { curveIndex: ShaderNodeObject<Node>; t: ShaderNodeObject<Node> }) => {
    const clamped = tslMin(t, float(1.0));
    const pos = clamped.mul(CURVE_RESOLUTION - 1);
    const idx0 = floor(pos);
    const f = fract(pos);
    const base = curveIndex.mul(CURVE_RESOLUTION);
    const v0 = sCurveData.element(base.add(idx0));
    const v1 = sCurveData.element(base.add(tslMin(idx0.add(1.0), float(CURVE_RESOLUTION - 1))));
    return mix(v0, v1, f);
  });
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
    x.mul(scaleV).add(qx.mul(projD)).add(qy.mul(z).sub(qz.mul(y)).mul(twoW)),
    y.mul(scaleV).add(qy.mul(projD)).add(qz.mul(x).sub(qx.mul(z)).mul(twoW)),
    z.mul(scaleV).add(qz.mul(projD)).add(qx.mul(y).sub(qy.mul(x)).mul(twoW)),
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
  shapeParams: ShapeEmitParams,
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
  const uSeed = uniform(float(0));
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
  const sPos  = storage(buffers.position, 'vec4', maxParticles);
  const sVel  = storage(buffers.velocity, 'vec4', maxParticles);
  const sCol  = storage(buffers.color, 'vec4', maxParticles);
  const sPS   = storage(buffers.particleState, 'vec4', maxParticles);
  const sSV   = storage(buffers.startValues, 'vec4', maxParticles);
  const sEx   = storage(buffers.startColorsExt, 'vec4', maxParticles);
  const sOIA  = storage(buffers.orbitalIsActive, 'vec4', maxParticles);

  // ?? Binding 8 + optional 9: integer ring-allocator counter + axes ?
  // `array<atomic<u32>, 1>` (only index 0 is live): a monotonic birth counter,
  // slot = counter mod maxParticles, so allocation can never underflow.
  const allocatorCount = maxParticles + 1;
  const ringMod = float(maxParticles);
  const sAllocator     = storage(buffers.allocator, 'uint', Math.max(1, allocatorCount)).toAtomic();
  // Per-particle velocity-axis values (linear.xyz + orbital.x); exists only
  // when velocityOverLifetime is active; orbital.y / orbital.z ride in the
  // unused vec4 padding of `position.w` / `velocity.w`.
  const hasAxes = buffers.axes !== null;
  const sAxes = hasAxes
    ? storage(buffers.axes as StorageBufferAttribute, 'vec4', maxParticles)
    : null;

  // ?? Read-mostly f32 tables: uniform buffer binding, non-atomic ??
  // Baked curve samples plus optional force-field / collision-plane records.
  // A plain uniform binding: it neither consumes a storage slot nor becomes
  // `atomic<f32>`, so curve sampling stays a straight `f32` load.
  const sCD = buffer(buffers.packedData, 'float', buffers.packedData.length);

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
  const forceFieldOffset  = curveLen;
  const collisionOffset   = forceFieldOffset + (flags.forceFields ? FORCE_FIELD_DATA_SIZE : 0);

  const ffNodes = flags.forceFields
    ? createForceFieldTSL(sCD, forceFieldOffset, forceFieldCount) : null;
  const cpNodes = flags.collisionPlanes
    ? createCollisionPlaneTSL(sCD, collisionOffset, collisionPlaneCount) : null;

  // Per-axis raw velocity-over-lifetime values (constant / random-range /
  // curve) — oracle parity: random ranges are sampled PER PARTICLE at birth.
  // The per-particle destination slots are:
  //   linear  -> `axes.xyz`        orbital.x -> `axes.w`
  //   orbital.y -> `position.w`    orbital.z -> `velocity.w`
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
    if (rawAxis && typeof rawAxis === 'object' && 'min' in rawAxis && 'max' in rawAxis) {
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
  const sampleAxis = (
    a: AxisSpec,
    r: ShaderNodeObject<Node>
  ): ShaderNodeObject<Node> => {
    if (a.isRange) {
      const [mn, mx] = axisUniforms.get(a)!;
      return mix(mn, mx, r);
    }
    return float(a.min);
  };

  // Per-particle axis value used by the simulation pass: curve axes look up
  // the baked table at the lifetime percentage; the others read the slot
  // written by the emission kernel (per-particle constant or fresh random).
  const simAxis = (
    a: AxisSpec,
    stored: ShaderNodeObject<Node>,
    lifePct: ShaderNodeObject<Node>
  ): ShaderNodeObject<Node> =>
    a.ci >= 0
      ? lookupCurve({ curveIndex: float(a.ci), t: lifePct })
      : stored;

  //
  // `i` = invocation index (0 .. emitCount-1). Each invocation:
  //   1) atomically pops one uint slot id off the allocator stack,
  //   2) draws 14 independent randoms from `uSeed + 16*i + k`,
  //   3) builds the shape position + velocity (5 shape kinds, oracle math),
  //   4) rotates them by the emitter wrapper quaternion,
  //   5) writes vec4 slots on pos/vel/color/particleState/startValues/ext/orbital.
  // Random noise phase offset scale: `Math.random() * 100` only with
  // `useRandomOffset`, else the deterministic 0 (oracle parity).
  const noiseOffsetScale = float(shapeParams.noiseUseRandomOffset ? 100.0 : 0.0);
  const emitKernel = Fn(() => {
    const i = instanceIndex;
    // Explicit count guard: the host dispatches max(1, emitCount) invocations, so the
    // kernel itself must skip the extra one when emitCount === 0.
    If(i.lessThan(uEmitCount), () => {
      // Race-safe ring allocation (never underflows): slot = birthNo mod N.
      const birthNo = float(atomicAdd(sAllocator.element(0), tuint(1))).toVar();
      const slotIdx = birthNo
        .sub(floor(birthNo.div(ringMod)).mul(ringMod))
        .toVar();
      // 14 independent randoms per particle (stride 16 keeps them unique).
      const base2 = float(i).mul(float(16.0));
      const rnd = (k: number) => rand(uSeed.add(base2.add(float(k + 0.13))));
      const rNoise = rnd(0);
      const rA = rnd(1); // 1st angular / x / axis-1
      const rB = rnd(2); // 2nd angular / y / ratio / axis-2
      const rC = rnd(3); // 3rd ratio / z / axis-3
      const rSheet = rnd(5);
      const rSpeed = rnd(6);
      const rSize = rnd(7);
      const rRot = rnd(8);
      const rOp = rnd(9);
      const rLife = rnd(10);
      const rColor = rnd(11);
      const rRotSpeed = rnd(12);

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
        rA,
        rB,
        rC,
        rSpeed
      );
      const pxL = shE.px;
      const pyL = shE.py;
      const pzL = shE.pz;
      const vxL = shE.vx;
      const vyL = shE.vy;
      const vzL = shE.vz;

      // ?? Emitter pose: rotate by the wrapper quaternion, then (WORLD only)
      // apply the per-axis world scale and add the emitter translation.
      const [rotPX, rotPY, rotPZ] = quatRotateNodes(pxL, pyL, pzL, uWrapperQuat);
      const [rotVX, rotVY, rotVZ] = quatRotateNodes(vxL, vyL, vzL, uWrapperQuat);

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

      // Start values (fresh randoms so no field aliases another).
      const opac = mix(uOpMin, uOpMax, rOp);
      const clR = mix(uCRR, uCRX, rColor);
      const clG = mix(uCGR, uCGX, rColor);
      const clB = mix(uCBR, uCBX, rColor);
      const slife = mix(uLifeMin, uLifeMax, rLife).mul(float(1000.0));
      const ssize = mix(uSizeMin, uSizeMax, rSize);
      const srot = mix(uRotMin, uRotMax, rRot);
      const startFrame = floor(mix(uFrMin, uFrMax, rSheet)).toVar();
      // Separate per-particle rotationOverLifetime speed (oracle keeps its own
      // min/max, distinct from startRotation's rotMin/rotMax).
      const rotSpeed = mix(uRotOLMin, uRotOLMax, rRotSpeed);
      // Random noise phase offset only with `useRandomOffset` (oracle
      // `Math.random() * 100`, else 0).
      const noiseOff = rNoise.mul(noiseOffsetScale);

      sPos.element(slotIdx).assign(
        vec4(
          ox,
          oy,
          oz,
          flags.orbitalVelocity ? sampleAxis(orbAxes[1], rC) : float(0.0)
        )
      );
      sVel.element(slotIdx).assign(
        vec4(
          rotVX,
          rotVY,
          rotVZ,
          flags.orbitalVelocity ? sampleAxis(orbAxes[2], rA) : float(0.0)
        )
      );
      sCol.element(slotIdx).assign(vec4(clR, clG, clB, opac));
      // lifetime=0, size, rotation, startFrame
      sPS.element(slotIdx).assign(vec4(float(0.0), ssize, srot, startFrame));
      // startValues = (startLife, size, opacity, colorR)
      sSV.element(slotIdx).assign(vec4(slife, ssize, opac, clR));
      // ext = (colorG, colorB, rotSpeed, noiseOffset)
      sEx.element(slotIdx).assign(vec4(clG, clB, rotSpeed, noiseOff));
      // Orbital pivot = rotated shape offset (oracle positionOffset), w=1.
      sOIA.element(slotIdx).assign(vec4(rotPX, rotPY, rotPZ, float(1.0)));

      // Per-particle velocity-axis values: (linear.x, linear.y, linear.z,
      // orbital.x); orbital.y/.z ride `position.w` / `velocity.w` (written
      // above). Curve axes keep their baked table lookup.
      if (hasAxes && sAxes) {
        sAxes.element(slotIdx).assign(
          vec4(
            sampleAxis(linAxes[0], rA),
            sampleAxis(linAxes[1], rB),
            sampleAxis(linAxes[2], rC),
            flags.orbitalVelocity ? sampleAxis(orbAxes[0], rB) : float(0.0)
          )
        );
      }

      // BIRTH events for the configured sub-emitter channels.
      for (const f of fifoNodes) {
        if (f.trigger === 0) {
          writeFifoEvent(f, ox, oy, oz, rotVX, rotVY, rotVZ);
        }
      }
    });
  });

  const emitNode = compute(emitKernel(), maxParticles);

  // Simulation kernel — physics + modifier order mirrors the oracle CPU
  // `updateParticleSystems` inner loop exactly:
  //   gravity -> force fields -> integrate -> collisions -> modifiers ->
  //   lifetime += dt -> trail sample -> write back -> death.
  const noiseOctavesCount = Math.max(1, Math.round(shapeParams.noiseOctaves || 1));
  // three-noise FBm divides the accumulated octaves by
  // `max = 1 + 0.5 + ... + 0.5^octaves = 2 - 2^-octaves` (the sum includes
  // the trailing 0.5^n exactly like the oracle's `fbmMax` constant).
  const noiseFbmMax = 2 - Math.pow(2, -noiseOctavesCount);
  const noiseOctDiv = Array.from(
    { length: noiseOctavesCount },
    () => float(noiseFbmMax)
  );
  const simKernel = Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(maxParticles)), () => {
      const oiaVec = sOIA.element(i).toVar();
      If(oiaVec.w.greaterThanEqual(float(0.5)), () => {
        const pos = sPos.element(i).xyz.toVar();
        const posW = sPos.element(i).w.toVar();
        const vel = sVel.element(i).xyz.toVar();
        const velW = sVel.element(i).w.toVar();
        const ps  = sPS.element(i).toVar();
        const sv  = sSV.element(i);
        const ex  = sEx.element(i);
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
        if (cpNodes) cpNodes.apply({
          pos, vel, oiaVec,
          sColorNode: sCol, ps, startLife,
          particleIdx: i, sOrbitalIsActiveNode: sOIA,
        });

        // 5. Modifiers — oracle semantics.
        // 5a. Linear velocity over lifetime: per-axis source is either the
        // baked curve table or the per-particle value written at birth
        // (constant or fresh random sample kept for this particle).
        if (flags.linearVelocity && hasAxes && sAxes) {
          const ax = sAxes.element(i);
          const lvx = simAxis(linAxes[0], ax.x, lifePct);
          const lvy = simAxis(linAxes[1], ax.y, lifePct);
          const lvz = simAxis(linAxes[2], ax.z, lifePct);
          pos.assign(pos.add(vec3(lvx, lvy, lvz).mul(uDelta)));
        }
        if (flags.orbitalVelocity && hasAxes && sAxes) {
          // Pivot + offset mirror the oracle `positionOffset`: subtract it,
          // rotate the offset by the Euler, add it back (offset mutation is
          // stored in oia.xyz).
          const offset = vec3(oiaVec.x, oiaVec.y, oiaVec.z).toVar();
          pos.assign(pos.sub(offset));
          const ax = sAxes.element(i);
          const oX = hasAxes ? simAxis(orbAxes[0], ax.w, lifePct) : float(0.0);
          const oY = hasAxes ? simAxis(orbAxes[1], posW, lifePct) : float(0.0);
          const oZ = hasAxes ? simAxis(orbAxes[2], velW, lifePct) : float(0.0);
          // Oracle: Euler(speedX*dt, speedZ*dt, speedY*dt) with order 'XYZ' —
          // intrinsic XYZ. Its matrix is Rx·Ry·Rz, so the vector product
          // applies Z FIRST, then Y, then X (extrinsic Z→Y→X). Keep the
          // per-axis mapping: Euler.x = oX, Euler.y = oZ, Euler.z = oY.
          const angX = oX.mul(uDelta);
          const angY = oZ.mul(uDelta);
          const angZ = oY.mul(uDelta);
          const c3 = cos(angZ), s3 = sin(angZ);
          const zx = offset.x.mul(c3).sub(offset.y.mul(s3));
          const zy = offset.x.mul(s3).add(offset.y.mul(c3));
          const zz = offset.z;
          const c2 = cos(angY), s2 = sin(angY);
          const yx = zx.mul(c2).add(zz.mul(s2));
          const yz = zx.mul(s2).negate().add(zz.mul(c2));
          const yy = zy;
          const c1 = cos(angX), s1 = sin(angX);
          const fx = yx;
          const fy = yy.mul(c1).sub(yz.mul(s1));
          const fz = yy.mul(s1).add(yz.mul(c1));
          pos.assign(pos.add(vec3(fx, fy, fz)));
          oiaVec.assign(vec4(fx, fy, fz, oiaVec.w));
        }
        // 5b. Size / opacity / color over lifetime — start value * multiplier.
        if (flags.sizeOverLifetime) {
          const s = lookupCurve({ curveIndex: float(curveMap.sizeOverLifetime), t: lifePct });
          ps.y.assign(s.mul(sv.y));
        }
        if (flags.opacityOverLifetime) {
          const op = lookupCurve({ curveIndex: float(curveMap.opacityOverLifetime), t: lifePct });
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
            col.x.assign(sv.w.mul(lookupCurve({ curveIndex: float(curveMap.colorR), t: lifePct })));
          }
          if (curveMap.colorG >= 0) {
            col.y.assign(sce.x.mul(lookupCurve({ curveIndex: float(curveMap.colorG), t: lifePct })));
          }
          if (curveMap.colorB >= 0) {
            col.z.assign(sce.y.mul(lookupCurve({ curveIndex: float(curveMap.colorB), t: lifePct })));
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
        // 0.15*strength (per-channel amounts applied below).
        if (flags.noise) {
          const np = lifePct
            .add(ex.w)
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
            const sc = float(amp / noiseOctDiv[0]);
            noiseX.assign(noiseX.add(snoise3D({ v: vec3(t, float(0), float(0)) }).mul(sc)));
            noiseY.assign(noiseY.add(snoise3D({ v: vec3(t, t, float(0)) }).mul(sc)));
            noiseZ.assign(noiseZ.add(snoise3D({ v: vec3(t, t, t) }).mul(sc)));
            amp *= 0.5;
            lac *= 2.0;
          }
          If(uNoisePosAmount.greaterThan(float(0.001)), () => {
            pos.assign(
              pos.add(vec3(noiseX, noiseY, noiseZ).mul(uNoisePower).mul(uNoisePosAmount))
            );
          });
          If(uNoiseRotAmount.greaterThan(float(0.001)), () => {
            ps.z.assign(ps.z.add(noiseX.mul(uNoisePower).mul(uNoiseRotAmount)));
          });
          If(uNoiseSizeAmount.greaterThan(float(0.001)), () => {
            ps.y.assign(ps.y.add(noiseX.mul(uNoisePower).mul(uNoiseSizeAmount)));
          });
        }

        ps.x.assign(ps.x.add(uDeltaMs));

        // Trail history: adaptive ring sample (oracle minVertexDistance).
        // Metadata = integer atomics (cursor,count); samples = plain f32.
        if (sTrail && sTrailMeta && trailDesc) {
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
          const dist = sqrt(
            ddx.mul(ddx).add(ddy.mul(ddy)).add(ddz.mul(ddz))
          );
          const firstSample = count.lessThan(float(0.5));
          const farEnough = dist.greaterThanEqual(
            float(Math.max(1e-6, trailDesc.minVertexDistance))
          );
          If(firstSample.or(farEnough), () => {
            // cursor indexes the LAST written sample; 0..L-1, wrapping.
            const newCursor = cursor
              .greaterThanEqual(L.sub(float(1)))
              .select(float(0), cursor.add(float(1)));
            sTrail.element(baseI.add(newCursor)).assign(
              vec4(pos.x, pos.y, pos.z, uNowMs)
            );
            atomicStore(
              sTrailMeta.element(curIdx),
              newCursor.toUint()
            );
            atomicStore(
              sTrailMeta.element(curIdx.add(tuint(1))),
              tslMin(count.add(float(1)), float(trailDesc.length)).toUint()
            );
          });
        }

        sPos.element(i).assign(vec4(pos, posW));
        sVel.element(i).assign(vec4(vel, velW));
        sPS.element(i).assign(ps);
        sOIA.element(i).assign(oiaVec);

        // Death: ring allocator needs no push (the monotonically increasing
        // birth counter owns recycling); mark the slot inactive + zero color.
        If(ps.x.greaterThan(startLife), () => {
          const inactive = sOIA.element(i).toVar();
          sOIA.element(i).assign(vec4(inactive.x, inactive.y, inactive.z, float(0.0)));
          sCol.element(i).assign(vec4(float(0.0), float(0.0), float(0.0), float(0.0)));

          // DEATH events for the configured sub-emitter channels.
          for (const f of fifoNodes) {
            if (f.trigger === 1) {
              writeFifoEvent(
                f,
                pos.x,
                pos.y,
                pos.z,
                vel.x,
                vel.y,
                vel.z
              );
            }
          }
        });
      });
    });
  });

  const simNode = compute(simKernel(), maxParticles);

  return {
    emitNode,
    simNode,
    computeNodes: [emitNode, simNode],
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
      'simulate',
      ...(sTrail ? ['trail-history(sample)'] : []),
      ...fifoNodes.map((f) => (f.trigger === 0 ? 'sub-birth(events)' : 'sub-death(events)')),
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
  /** Consumer node; dispatch it directly after the parent kernels. */
  initNode: unknown;
  /** 1-invocation pass that zeroes the *other* ping-pong counter. */
  counterClearNode: unknown;
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

  const uSeed = uniform(float(0));
  const uInherit = uniform(float(Math.max(0, inheritVelocity)));
  const uFifoBase = uniform(float(0));
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

  // Child pool (bindings 1..7) + its own ring allocator (binding 8) + the
  // optional per-axis table.
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
  const cHasAxes = child.axes !== null;
  const cAxes = cHasAxes
    ? storage(child.axes as StorageBufferAttribute, 'vec4', childMax)
    : null;
  // Read-mostly parent pool — only bound when the child needs the live
  // parent arrays (currently neither: positions/velocities travel inside
  // the FIFO payload records).
  const cOrbOn =
    childVelValues &&
    (childVelValues.orbital[0] !== undefined ||
      childVelValues.orbital[1] !== undefined ||
      childVelValues.orbital[2] !== undefined);
  const cLinOn =
    childVelValues &&
    (childVelValues.linear[0] !== undefined ||
      childVelValues.linear[1] !== undefined ||
      childVelValues.linear[2] !== undefined);
  const cCurve = cHasAxes && (cLinOn || cOrbOn);
  // Read-mostly parent pool + the event FIFO (u32 counters + plain f32
  // payload — no atomic floats).
  const pPos = storage(parent.position, 'vec4', parentMax);
  const pVel = storage(parent.velocity, 'vec4', parentMax);
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
  const cNoiseOffsetScale = float(
    childParams.noiseUseRandomOffset ? 100.0 : 0.0
  );

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
  const cSample = (
    a: { min: number; max: number; isRange: boolean },
    r: ShaderNodeObject<Node>
  ): ShaderNodeObject<Node> =>
    a.isRange ? mix(float(a.min), float(a.max), r) : float(a.min);

  // The *other* window's counter is cleared by a dedicated 1-invocation
  // pass so the next frame's writers start from 0 (no race with this frame).
  const otherIdx = uFifoBase.equal(float(0)).select(tuint(1), tuint(0));
  const counterClearKernel = Fn(() => {
    atomicStore(fifoCounter.element(otherIdx), tuint(0));
  });
  const counterClearNode = compute(counterClearKernel(), 1);

  const kernel = Fn(() => {
    const i = instanceIndex;
    const winBase = uFifoBase.mul(float(windowSize)).toVar();
    const count = float(atomicLoad(fifoCounter.element(uFifoBase))).toVar();

    If(float(i).lessThan(count), () => {
      const eb = winBase.add(float(i).mul(float(SUB_EMITTER_EVENT_STRIDE)));
      const eX = fifoPayload.element(eb).toVar();
      const eY = fifoPayload.element(eb.add(float(1))).toVar();
      const eZ = fifoPayload.element(eb.add(float(2))).toVar();
      const vX = fifoPayload.element(eb.add(float(3))).toVar();
      const vY = fifoPayload.element(eb.add(float(4))).toVar();
      const vZ = fifoPayload.element(eb.add(float(5))).toVar();

      // Oracle: startSpeed += |parentVelocity| * inheritVelocity
      const parentSpeed = sqrt(
        vX.mul(vX).add(vY.mul(vY)).add(vZ.mul(vZ))
      ).toVar();
      const spAdd = parentSpeed.mul(uInherit);

      // Unrolled per-event particle loop (particlesPerEvent is a host constant).
      for (let jj = 0; jj < perEvent; jj++) {
        // Ring slot (race-safe, never underflows).
        const birthNo = float(atomicAdd(cAlloc.element(0), tuint(1))).toVar();
        const slot = birthNo
          .sub(floor(birthNo.div(cRingMod)).mul(cRingMod))
          .toVar();
        {
          const rBase = float(i).mul(float(16 * perEvent)).add(float(jj * 16));
          const rnd = (k: number) => rand(uSeed.add(rBase.add(float(k + 0.13))));
          const rNoise = rnd(0);
          const rA = rnd(1);
          const rB = rnd(2);
          const rC = rnd(3);
          const rSheet = rnd(5);
          const rSpeed = rnd(6);
          const rSize = rnd(7);
          const rRot = rnd(8);
          const rOp = rnd(9);
          const rLife = rnd(10);
          const rColor = rnd(11);
          const rRotSpeed = rnd(12);

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
            rA,
            rB,
            rC,
            rSpeed
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

          const opac = mix(cOpMin, cOpMax, rOp);
          const clR = mix(cCRR, cCRX, rColor);
          const clG = mix(cCGR, cCGX, rColor);
          const clB = mix(cCBR, cCBX, rColor);
          const slife = mix(cLifeMin, cLifeMax, rLife).mul(float(1000.0));
          const ssize = mix(cSizeMin, cSizeMax, rSize);
          const srot = mix(cRotMin, cRotMax, rRot);
          const startFrame = floor(mix(cFrMin, cFrMax, rSheet)).toVar();
          // Separate rotationOverLifetime speed (own min/max, oracle parity).
          const rotSpeed = mix(
            float(childParams.rotOverLifeMin),
            float(childParams.rotOverLifeMax),
            rRotSpeed
          );

          cPos.element(slot).assign(
            vec4(
              px,
              py,
              pz,
              cOrbOn ? cSample(cOrb[1], rC) : float(0.0)
            )
          );
          cVel.element(slot).assign(
            vec4(
              rvx,
              rvy,
              rvz,
              cOrbOn ? cSample(cOrb[2], rA) : float(0.0)
            )
          );
          cCol.element(slot).assign(vec4(clR, clG, clB, opac));
          cPS.element(slot).assign(
            vec4(float(0.0), ssize, srot, startFrame)
          );
          cSV.element(slot).assign(vec4(slife, ssize, opac, clR));
          cEx.element(slot).assign(
            vec4(clG, clB, rotSpeed, rNoise.mul(cNoiseOffsetScale))
          );
          // Orbital pivot = rotated child shape offset (oracle parity).
          cOIA.element(slot).assign(vec4(rx, ry, rz, float(1.0)));
          if (cHasAxes && cAxes) {
            cAxes.element(slot).assign(
              vec4(
                cSample(cLin[0], rA),
                cSample(cLin[1], rB),
                cSample(cLin[2], rC),
                cOrbOn ? cSample(cOrb[0], rB) : float(0.0)
              )
            );
          }
        }
      }
    });
  });

  const initNode = compute(kernel(), capacity);

  return {
    initNode,
    counterClearNode,
    passName: fifo.trigger === 0 ? 'sub-birth-init' : 'sub-death-init',
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
      const cursor = float(
        atomicLoad(sMeta.element(i.mul(float(2))))
      ).toVar();
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
        const ageOk = desc.maxTime > 0
          ? uNowMs.sub(sample.w).lessOrEqual(float(desc.maxTime))
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
          aPos.element(vi).assign(
            vec4(sample.x, sample.y, sample.z, hw)
          );
          aNext.element(vi).assign(
            vec4(nPos.x, nPos.y, nPos.z, alpha)
          );
          aUVA.element(vi).assign(
            vec4(float(side), t, cr.mul(pcol.x), cg.mul(pcol.y))
          );
          aColB.element(vi).assign(
            vec4(cb.mul(pcol.z), alpha, float(0), float(0))
          );
        }
      });
    });
  });

  const ribbonNode = compute(kernel(), vertexCount);

  return {
    ribbonNode,
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


