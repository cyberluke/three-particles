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
};

/**
 * One GPU sub-emitter event channel. Storage layout (atomic f32, itemSize 1):
 *   [0]                  event count written this cycle
 *   [1 + 6*k + 0..5]     event k = (x, y, z, velX, velY, velZ)
 * The parent kernels append events; the child init kernel consumes them.
 */
export type SubEmitterFifo = {
  attribute: StorageBufferAttribute;
  /** 0 = BIRTH (written by the emission kernel), 1 = DEATH (sim kernel). */
  trigger: 0 | 1;
  /** Event capacity == `maxInstances` of the sub-emitter config. */
  capacity: number;
  /** Floats per window: 1 counter + 6 per event. */
  windowSize: number;
};

/** Floats stored per sub-emitter event (position xyz + velocity xyz). */
export const SUB_EMITTER_EVENT_STRIDE = 6;

/** Floats in one FIFO window (1 counter + 6 per event). */
export const subEmitterWindowSize = (capacity: number): number =>
  1 + SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity);

/**
 * Allocates the atomic f32 FIFO for one sub-emitter channel. Two ping-pong
 * windows so the reader (init kernel) can zero the *other* window in the same
 * dispatch without racing the writers.
 */
export const createSubEmitterFifoAttribute = (
  capacity: number
): StorageBufferAttribute =>
  new StorageBufferAttribute(
    new Float32Array(
      2 * (1 + SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity))
    ),
    1
  );

/** Trail-history descriptor consumed by the simulation kernel. */
export type TrailHistoryDesc = {
  /** atomic f32, itemSize 4: `(L+1)` rows per particle; row L = (cursor, count). */
  attribute: StorageBufferAttribute;
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
  /** Shared trail history buffer (same attribute as {@link TrailHistoryDesc}). */
  history: StorageBufferAttribute;
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
  /** 8th storage binding: `array<atomic<u32>, maxParticles + 1>` slot allocator. */
  allocator: StorageBufferAttribute;
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
  /** Emitter pose uniforms, written once per frame by the CPU. */
  emitterPose: {
    /** (x, y, z, isWorldFlag) ? translation only in WORLD simulation. */
    positionW: ShaderNodeObject<Node>;
    /** Wrapper quaternion (x, y, z, w): identity LOCAL, world rot WORLD. */
    wrapperQuat: ShaderNodeObject<Node>;
    /** Emitter world scale, applied per axis to the rotated offset in WORLD. */
    worldScale: ShaderNodeObject<Node>;
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
  hasCollisionPlanes = false
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

  // Allocator stack (uint atomics): [0] = freeCount, [1..maxParticles] = slot ids.
  // TODO(r186 follow-up): `atomicSub` wraps at 0 when the stack is exhausted
  // (freeCount 0 -> 4294967295). Needs a race-safe clamp (e.g. a second
  // counter or a modulo-`maxParticles+1` fold) - deliberately not changed here.
  const allocatorCount = maxParticles + 1;
  const allocatorData  = new Uint32Array(allocatorCount);
  allocatorData[0] = maxParticles;
  for (let i = 0; i < maxParticles; i++) allocatorData[i + 1] = i;

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
  const discX = isConeKind.select(pBx, planeX); // CIRCLE falls through
  const discY = isConeKind.select(pBy, planeY);
  const discZ = isConeKind.select(pBz, planeZ);
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

export function createModifierComputeUpdate(
  buffers: ModifierStorageBuffers,
  maxParticles: number,
  curveMap: BakedCurveMap,
  flags: ModifierFlags,
  shapeParams: ShapeEmitParams,
  forceFieldCount = 0,
  collisionPlaneCount = 0,
  subFifos: SubEmitterFifo[] = [],
  trailDesc?: TrailHistoryDesc
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

  // ?? Binding 8: the uint atomic allocator stack ??
  // `array<atomic<u32>, maxParticles + 1>` with [0] = freeCount and
  // [1 .. maxParticles] = free slot ids, so addressing is 0-based and needs no
  // float base offset any more.
  const allocatorCount = maxParticles + 1;
  const sAllocator     = storage(buffers.allocator, 'uint', allocatorCount).toAtomic();

  // ?? Read-mostly f32 tables: uniform buffer binding, non-atomic ??
  // Baked curve samples plus optional force-field / collision-plane records.
  // A plain uniform binding: it neither consumes a storage slot nor becomes
  // `atomic<f32>`, so curve sampling stays a straight `f32` load.
  const sCD = buffer(buffers.packedData, 'float', buffers.packedData.length);

  // ?? Sub-emitter FIFO channels (one atomic f32 storage binding each) ??
  const uFifoBase = uniform(float(0));
  const fifoNodes = subFifos.map((f) => ({
    trigger: f.trigger,
    capacity: Math.max(1, f.capacity),
    node: storage(
      f.attribute,
      'float',
      (f.attribute.array as Float32Array).length
    ).toAtomic(),
  }));

  const writeFifoEvent = (
    n: ShaderNodeObject<Node>,
    capacity: number,
    x: ShaderNodeObject<Node>,
    y: ShaderNodeObject<Node>,
    z: ShaderNodeObject<Node>,
    vx: ShaderNodeObject<Node>,
    vy: ShaderNodeObject<Node>,
    vz: ShaderNodeObject<Node>
  ): void => {
    // Ping-pong window base comes from the CPU (frame parity). `atomicAdd` on
    // the counter slot returns the pre-increment count, so event k lives at
    // base + 1 + 6*k.
    const base = uFifoBase.toVar();
    const oldCount = atomicAdd(n.element(base), float(1.0)).toVar();
    If(oldCount.lessThan(float(capacity)), () => {
      const b = base.add(float(1.0)).add(oldCount.mul(float(SUB_EMITTER_EVENT_STRIDE)));
      n.element(b).assign(x);
      n.element(b.add(float(1.0))).assign(y);
      n.element(b.add(float(2.0))).assign(z);
      n.element(b.add(float(3.0))).assign(vx);
      n.element(b.add(float(4.0))).assign(vy);
      n.element(b.add(float(5.0))).assign(vz);
    });
  };

  // ?? Trail history ring (vec4 rows: L samples + 1 meta row per particle) ??
  const trailRows = trailDesc ? trailDesc.length + 1 : 0;
  const sTrail: ShaderNodeObject<Node> | null = trailDesc
    ? storage(trailDesc.attribute, 'vec4', trailRows * maxParticles).toAtomic()
    : null;

  const curveLen = Math.max(curveMap.data.length, 1);

  // ?? Force-field + collision-plane TSL readers ??
  const forceFieldOffset  = curveLen;
  const collisionOffset   = forceFieldOffset + (flags.forceFields ? FORCE_FIELD_DATA_SIZE : 0);

  const ffNodes = flags.forceFields
    ? createForceFieldTSL(sCD, forceFieldOffset, forceFieldCount) : null;
  const cpNodes = flags.collisionPlanes
    ? createCollisionPlaneTSL(sCD, collisionOffset, collisionPlaneCount) : null;

  const lookupCurve = createCurveLookup(sCD);

  // ????? Emission kernel ?????
  //
  // `i` = invocation index (0 .. emitCount-1). Each invocation:
  //   1) atomically pops one uint slot id off the allocator stack,
  //   2) draws 14 independent randoms from `uSeed + 16*i + k`,
  //   3) builds the shape position + velocity (5 shape kinds, oracle math),
  //   4) rotates them by the emitter wrapper quaternion,
  //   5) writes vec4 slots on pos/vel/color/particleState/startValues/ext/orbital.
  const emitKernel = Fn(() => {
    const i = instanceIndex;
    // Explicit count guard: the host dispatches max(1, emitCount) invocations, so the
    // kernel itself must skip the extra one when emitCount === 0.
    If(i.lessThan(uEmitCount), () => {
    // Pop: `oldTop` = freeCount *before* the decrement (integer atomic).
    const oldTop = atomicSub(sAllocator.element(0), tuint(1)).toVar();
    If(oldTop.greaterThan(tuint(0)), () => {
      // The wanted slot id sits at index 1 + (oldTop - 1) === oldTop.
      const slotIdx = atomicLoad(sAllocator.element(oldTop)).toVar();
      // 14 independent randoms per particle (stride 16 keeps them unique).
      const base2 = i.mul(float(16.0));
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
      const rotSpeed = mix(uRotMin, uRotMax, rRotSpeed);

      sPos.element(slotIdx).assign(vec4(ox, oy, oz, float(0.0)));
      sVel.element(slotIdx).assign(vec4(rotVX, rotVY, rotVZ, float(0.0)));
      sCol.element(slotIdx).assign(vec4(clR, clG, clB, opac));
      // lifetime=0, size, rotation, startFrame
      sPS.element(slotIdx).assign(vec4(float(0.0), ssize, srot, startFrame));
      // startValues = (startLife, size, opacity, colorR)
      sSV.element(slotIdx).assign(vec4(slife, ssize, opac, clR));
      // ext = (colorG, colorB, rotSpeed, noiseOffset)
      sEx.element(slotIdx).assign(
        vec4(clG, clB, rotSpeed, rNoise.mul(float(100.0)))
      );
      // Orbital offset = birth position in pool space (rotation pivot), w=1.
      sOIA.element(slotIdx).assign(vec4(ox, oy, oz, float(1.0)));

      // BIRTH events for the configured sub-emitter channels.
      for (const f of fifoNodes) {
        if (f.trigger === 0) {
          writeFifoEvent(f.node, f.capacity, ox, oy, oz, rotVX, rotVY, rotVZ);
        }
      }
    });
    });
  });

  const emitNode = compute(emitKernel(), maxParticles);

  // ????? Simulation kernel (identical physics to the previous engine) ?????
  const simKernel = Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(maxParticles)), () => {
      const oiaVec = sOIA.element(i).toVar();
      If(oiaVec.w.greaterThanEqual(float(0.5)), () => {
        const pos = sPos.element(i).xyz.toVar();
        const vel = sVel.element(i).xyz.toVar();
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

        // 5. Modifiers (curves + noise) ? mirror of previous engine.
        if (flags.linearVelocity) {
          const lvx = curveMap.linearVelX >= 0 ? lookupCurve({ curveIndex: float(curveMap.linearVelX), t: lifePct }) : float(0.0);
          const lvy = curveMap.linearVelY >= 0 ? lookupCurve({ curveIndex: float(curveMap.linearVelY), t: lifePct }) : float(0.0);
          const lvz = curveMap.linearVelZ >= 0 ? lookupCurve({ curveIndex: float(curveMap.linearVelZ), t: lifePct }) : float(0.0);
          pos.assign(pos.add(vec3(lvx, lvy, lvz).mul(uDelta)));
        }
        if (flags.orbitalVelocity && (curveMap.orbitalVelX >= 0 || curveMap.orbitalVelY >= 0 || curveMap.orbitalVelZ >= 0)) {
          const offset = vec3(oiaVec.x, oiaVec.y, oiaVec.z).toVar();
          pos.assign(pos.sub(offset));
          const ovx = curveMap.orbitalVelX >= 0 ? lookupCurve({ curveIndex: float(curveMap.orbitalVelX), t: lifePct }) : float(0.0);
          const ovy = curveMap.orbitalVelY >= 0 ? lookupCurve({ curveIndex: float(curveMap.orbitalVelY), t: lifePct }) : float(0.0);
          const ovz = curveMap.orbitalVelZ >= 0 ? lookupCurve({ curveIndex: float(curveMap.orbitalVelZ), t: lifePct }) : float(0.0);
          // Mirror the CPU 'XYZ Euler (x, z, y)' intrinsic rotation. Apply
          // z-rot = ovy*dt, y-rot = ovz*dt, x-rot = ovx*dt around the offset.
          const angX = ovx.mul(uDelta);
          const angY = ovz.mul(uDelta);
          const angZ = ovy.mul(uDelta);
          const c1 = cos(angX), s1 = sin(angX);
          const c2 = cos(angY), s2 = sin(angY);
          const c3 = cos(angZ), s3 = sin(angZ);
          const ny = offset.y.mul(c1).sub(offset.z.mul(s1));
          const nz = offset.y.mul(s1).add(offset.z.mul(c1));
          const nx1 = offset.x.mul(c2).add(nz.mul(s2));
          const nz1 = offset.x.mul(s2).negate().add(nz.mul(c2));
          const fx = nx1.mul(c3).sub(ny.mul(s3));
          const fy = nx1.mul(s3).add(ny.mul(c3));
          const fz = nz1;
          pos.assign(pos.add(vec3(fx, fy, fz)));
          oiaVec.assign(vec4(fx, fy, fz, oiaVec.w));
        }
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
          const col = sCol.element(i).toVar();
          const cr = lookupCurve({ curveIndex: float(curveMap.colorR), t: lifePct }).mix(col.x, lifePct);
          const cg = lookupCurve({ curveIndex: float(curveMap.colorG), t: lifePct }).mix(col.y, lifePct);
          const cb = lookupCurve({ curveIndex: float(curveMap.colorB), t: lifePct }).mix(col.z, lifePct);
          col.assign(vec4(cr, cg, cb, col.w));
          sCol.element(i).assign(col);
        }
        if (flags.rotationOverLifetime) {
          ps.z.assign(ps.z.add(ex.z.mul(uDelta)));
        }
        if (flags.noise) {
          const freq = uNoiseFrequency;
          const p3 = pos.mul(freq);
          const seed = ex.w;
          // Single shared simplex implementation (`tsl-noise.ts` `snoise3D`).
          // Each axis is sampled at a decorrelated offset so the three
          // components differ; the per-particle `noiseOffset` (startColorsExt.w)
          // keeps the deterministic per-slot seed semantics.
          const noiseSample = (
            x: ShaderNodeObject<Node>,
            y: ShaderNodeObject<Node>,
            z: ShaderNodeObject<Node>,
            offset: ShaderNodeObject<Node>
          ) => snoise3D({ v: vec3(x.add(offset), y.add(offset), z.add(offset)) });

          const nx = noiseSample(p3.x, p3.y, p3.z, seed);
          const ny = noiseSample(
            p3.y.add(float(31.41)),
            p3.z.sub(float(17.53)),
            p3.x.add(float(23.07)),
            seed
          );
          const nz = noiseSample(
            p3.z.sub(float(51.07)),
            p3.x.add(float(13.11)),
            p3.y.add(float(41.79)),
            seed
          );
          const noiseVec = vec3(nx, ny, nz).mul(uNoisePower);
          If(uNoisePosAmount.greaterThan(float(0.001)), () => {
            pos.assign(pos.add(noiseVec.mul(uNoisePosAmount)));
          });
          If(uNoiseRotAmount.greaterThan(float(0.001)), () => {
            ps.z.assign(ps.z.add(nx.mul(uNoisePower).mul(uNoiseRotAmount)));
          });
          If(uNoiseSizeAmount.greaterThan(float(0.001)), () => {
            ps.y.assign(ps.y.add(nx.mul(uNoisePower).mul(uNoiseSizeAmount)));
          });
        }

        ps.x.assign(ps.x.add(uDeltaMs));

        // ?? Trail history: adaptive ring sample (oracle minVertexDistance) ??
        if (sTrail && trailDesc) {
          const L = float(trailDesc.length);
          const tr = float(trailRows);
          const metaIdx = i.mul(tr).add(tr.sub(float(1)));
          const meta = atomicLoad(sTrail.element(metaIdx)).toVar();
          const cursor = meta.x;
          const count = meta.y;
          const base = i.mul(tr);
          const prev = atomicLoad(sTrail.element(base.add(cursor))).toVar();
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
            sTrail.element(base.add(newCursor)).assign(
              vec4(pos.x, pos.y, pos.z, uNowMs)
            );
            sTrail.element(metaIdx).assign(
              vec4(
                newCursor,
                tslMin(count.add(float(1)), float(trailDesc.length)),
                float(0),
                float(0)
              )
            );
          });
        }

        sPos.element(i).assign(vec4(pos, float(0.0)));
        sVel.element(i).assign(vec4(vel, float(0.0)));
        sPS.element(i).assign(ps);
        sOIA.element(i).assign(oiaVec);

        // ??? Death: push the uint slot id back onto the atomic allocator stack ???
        If(ps.x.greaterThan(startLife), () => {
          const inactive = sOIA.element(i).toVar();
          sOIA.element(i).assign(vec4(inactive.x, inactive.y, inactive.z, float(0.0)));
          sCol.element(i).assign(vec4(float(0.0), float(0.0), float(0.0), float(0.0)));
          // Only pushed once: `atomicAdd` hands back the pre-increment count, and
          // inactive slots are skipped by the w >= 0.5 branch, so the sim pass
          // cannot double-push. Stack item index = 1 + oldCount.
          const oldTop = atomicAdd(sAllocator.element(0), tuint(1)).toVar();
          atomicStore(sAllocator.element(oldTop.add(tuint(1))), tuint(i));

          // DEATH events for the configured sub-emitter channels.
          for (const f of fifoNodes) {
            if (f.trigger === 1) {
              writeFifoEvent(
                f.node,
                f.capacity,
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
      /** Ping-pong FIFO window base for this frame (0 or `windowSize`). */
      fifoBase: uFifoBase,
    },
    shapeUniforms,
    buffers,
    allocatorCount,
    packedDataNode: sCD as ShaderNodeObject<Node>,
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
  particlesPerEvent: number
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

  // Child pool (bindings 1..7) + its own allocator (binding 8).
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
    childMax + 1
  ).toAtomic();

  // Read-mostly parent pool + the event FIFO.
  const pPos = storage(parent.position, 'vec4', parentMax);
  const pVel = storage(parent.velocity, 'vec4', parentMax);
  const fifoNode = storage(
    fifo.attribute,
    'float',
    (fifo.attribute.array as Float32Array).length
  ).toAtomic();

  const kernel = Fn(() => {
    const i = instanceIndex;
    const base = uFifoBase.toVar();
    const count = atomicLoad(fifoNode.element(base)).toVar();

    If(float(i).lessThan(count), () => {
      const eb = base.add(float(1)).add(float(i).mul(float(SUB_EMITTER_EVENT_STRIDE)));
      const eX = atomicLoad(fifoNode.element(eb)).toVar();
      const eY = atomicLoad(fifoNode.element(eb.add(float(1)))).toVar();
      const eZ = atomicLoad(fifoNode.element(eb.add(float(2)))).toVar();
      const vX = atomicLoad(fifoNode.element(eb.add(float(3)))).toVar();
      const vY = atomicLoad(fifoNode.element(eb.add(float(4)))).toVar();
      const vZ = atomicLoad(fifoNode.element(eb.add(float(5)))).toVar();

      // Oracle: startSpeed += |parentVelocity| * inheritVelocity
      const parentSpeed = sqrt(
        vX.mul(vX).add(vY.mul(vY)).add(vZ.mul(vZ))
      ).toVar();
      const spAdd = parentSpeed.mul(uInherit);

      // Unrolled per-event particle loop (particlesPerEvent is a host constant).
      for (let jj = 0; jj < perEvent; jj++) {
        const oldTop = atomicSub(cAlloc.element(0), tuint(1)).toVar();
        If(oldTop.greaterThan(tuint(0)), () => {
          const slot = atomicLoad(cAlloc.element(oldTop)).toVar();
          const rBase = i.mul(float(16 * perEvent)).add(float(jj * 16));
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
          const rotSpeed = mix(cRotMin, cRotMax, rRotSpeed);

          cPos.element(slot).assign(vec4(px, py, pz, float(0.0)));
          cVel.element(slot).assign(vec4(rvx, rvy, rvz, float(0.0)));
          cCol.element(slot).assign(vec4(clR, clG, clB, opac));
          cPS.element(slot).assign(
            vec4(float(0.0), ssize, srot, startFrame)
          );
          cSV.element(slot).assign(vec4(slife, ssize, opac, clR));
          cEx.element(slot).assign(
            vec4(clG, clB, rotSpeed, rNoise.mul(float(100.0)))
          );
          cOIA.element(slot).assign(vec4(px, py, pz, float(1.0)));
        });
      }

      // The first invocation clears the *other* ping-pong window counter, so
      // the next frame's writers start from 0 without a separate dispatch.
      If(i.equal(float(0)), () => {
        const otherBase = uFifoBase.equal(float(0)).select(
          float(windowSize),
          float(0)
        );
        atomicStore(fifoNode.element(otherBase), float(0));
      });
    });
  });

  const initNode = compute(kernel(), capacity);

  return {
    initNode,
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
      const meta = atomicLoad(hist.element(base.add(lf))).toVar();
      const cursor = meta.x;
      const count = meta.y;

      If(count.greaterThan(float(0.5)), () => {
        // Ring sample of slot s: (cursor - s) mod L, oldest -> newest order.
        const raw = cursor.sub(s).add(lf);
        const si = raw.sub(floor(raw.div(lf)).mul(lf));
        const sample = atomicLoad(hist.element(base.add(si))).toVar();
        const nextRaw = si.add(float(1));
        const ni = nextRaw.sub(floor(nextRaw.div(lf)).mul(lf));
        const nextSample = atomicLoad(hist.element(base.add(ni))).toVar();

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
        const pcol = atomicLoad(pColor.element(i)).toVar();

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


