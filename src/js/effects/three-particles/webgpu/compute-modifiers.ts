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
import { Vector3 } from 'three';
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
import { CURVE_RESOLUTION, type BakedCurveMap } from './curve-bake.js';
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

/** Compact shape-emission parameters used by the emission compute kernel. */
export type ShapeEmitParams = {
  /** 0 = cone, 1 = sphere, 2 = plane (edge/plane/box handled as axis-aligned box). */
  shapeKind: 0 | 1 | 2;
  radius: number;
  length: number;
  arc: number; // radians
  spreadX: number; // -1..1
  spreadY: number;
  spreadZ: number;
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

/** Per-frame compute-pipeline uniforms (CPU-side scalar writes only). */
export type ModifierUniforms = {
  delta: ShaderNodeObject<Node>;
  deltaMs: ShaderNodeObject<Node>;
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
export function createModifierComputeUpdate(
  buffers: ModifierStorageBuffers,
  maxParticles: number,
  curveMap: BakedCurveMap,
  flags: ModifierFlags,
  shapeParams: ShapeEmitParams,
  forceFieldCount = 0,
  collisionPlaneCount = 0
): ModifierComputePipeline {
  // ?? Per-frame uniforms ??
  const uDelta = uniform(float(0));
  const uDeltaMs = uniform(float(0));
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
  const uLength = sh('length', shapeParams.length);
  const uArc = sh('arc', shapeParams.arc);
  const uSpreadX = sh('spreadX', shapeParams.spreadX);
  const uSpreadY = sh('spreadY', shapeParams.spreadY);
  const uSpreadZ = sh('spreadZ', shapeParams.spreadZ);
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
  //   2) computes 8 random values from `uSeed + 64*i`,
  //   3) builds a shape-position + direction,
  //   4) writes vec4 slots on pos/vel/color/particleState/startValues/ext/orbital.
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
      // 8 deterministic randoms, one per particle.
      const base2 = i.mul(float(8.0));
      const r0 = rand(uSeed.add(base2.add(float(0.13))));
      const r1 = rand(uSeed.add(base2.add(float(1.17))));
      const r2 = rand(uSeed.add(base2.add(float(2.23))));
      const r3 = rand(uSeed.add(base2.add(float(3.31))));
      const r4 = rand(uSeed.add(base2.add(float(4.37))));
      const r5 = rand(uSeed.add(base2.add(float(5.41))));
      const r6 = rand(uSeed.add(base2.add(float(6.47))));
      const r7 = rand(uSeed.add(base2.add(float(7.53))));

      // Shape-dependent unit direction. r0..r2 spherical; r3..r4 angular;
      // r5 radial; r6..r7 speed / life / size / rotation / noiseOffset share them.
      const phi   = r0.mul(float(6.2831853)).toVar();
      const cosT  = float(1.0).sub(r1.mul(float(2.0))).toVar();
      const sinT  = sqrt(float(1.0).sub(cosT.mul(cosT))).toVar();
      const dx = sinT.mul(cos(phi));
      const dy = cosT;
      const dz = sinT.mul(sin(phi));

      // Cone: bias direction along local +Z by uArc/?, plane: fixed local +Z,
      // spherical: pure spherical. The unit direction is then scaled by speed.
      const kind = uShape;
      const coneZ = float(1.0).sub(r1.mul(uArc).mul(float(0.3183098)));
      const planeZ = float(1.0);
      const dirX = select01(kind, dx, dx, float(0.0));
      const dirY = select01(kind, dy, dy, float(0.0));
      const dirZa = select01(kind, dz, dz, planeZ);
      const coneX = dirX.mul(float(1.0));
      const coneY = dirY.mul(float(1.0));
      const coneZ2 = select01(kind, coneZ, dirZa, planeZ);

      // Final direction
      const dvx = select01(kind, coneX, dirX, coneX);
      const dvy = select01(kind, coneY, dirY, coneY);
      const dvz = select01(kind, coneZ2, coneZ2, planeZ);

      // Spread mix towards a canonical axis per r2.
      const sdX = mix(dvx, float(0.0), uSpreadX);
      const sdY = mix(dvy, float(1.0), uSpreadY);
      const sdZ = mix(dvz, float(0.0), uSpreadZ);

      // Shape radius / length offsets using r2..r5.
      const radial = r2.mul(uRadius);
      const ox = sdX.mul(radial);
      const oy = sdY.mul(radial);
      const oz = sdZ.mul(radial);
      const lenOffset = r3.sub(float(0.5)).mul(uLength);

      // Position in the emitter's own local reference (identity matrix).
      // Emitter's world matrix is applied through the standard modelMatrix
      // at render time; for WORLD simulation the CPU pre-multiplies by
      // `sourceWorldMatrix` before uploading the shape uniform, so we just
      // add the local offset here.
      sPos.element(slotIdx).assign(vec4(ox, oy.add(lenOffset), oz, float(0.0)));

      // Speed random in [min, max] on direction.
      const spMag = mix(uSpeedMin, uSpeedMax, r4).toVar();
      const vxAbs = sdX.mul(spMag);
      const vyAbs = sdY.mul(spMag);
      const vzAbs = sdZ.mul(spMag);
      sVel.element(slotIdx).assign(vec4(vxAbs, vyAbs, vzAbs, float(0.0)));

      // Color mix in linear space.
      const clR = mix(uCRR, uCRX, r6);
      const clG = mix(uCGR, uCGX, r6);
      const clB = mix(uCBR, uCBX, r6);
      const opac = mix(uOpMin, uOpMax, r7);

      // StartLifetime in ms.
      const slife = mix(uLifeMin, uLifeMax, r5).mul(float(1000.0));
      // StartSize and StartRotation.
      const ssize = mix(uSizeMin, uSizeMax, r4);
      const srot  = mix(uRotMin, uRotMax, r3);

      sCol.element(slotIdx).assign(vec4(clR, clG, clB, opac));
      // lifetime=0, size, rotation, startFrame
      const startFrame = floor(mix(uFrMin, uFrMax, r1)).toVar();
      sPS.element(slotIdx).assign(vec4(float(0.0), ssize, srot, startFrame));

      // startValues = (startLife, size, opacity, colorR)
      sSV.element(slotIdx).assign(vec4(slife, ssize, opac, clR));
      // ext = (colorG, colorB, rotSpeed, noiseOffset)
      const rotSpeed = mix(uRotMin, uRotMax, r3); // same random ? deterministic; use r7 for actual per-particle speed
      sEx.element(slotIdx).assign(vec4(clG, clB, rotSpeed, r6.mul(float(100.0))));

      // Orbital offset = current local position (rotation pivots around it).
      // isActive = 1.
      sOIA.element(slotIdx).assign(vec4(ox, oy.add(lenOffset), oz, float(1.0)));
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
      gravityVelocity: uGravityVelocity,
      noiseStrength: uNoiseStrength,
      noisePower: uNoisePower,
      noiseFrequency: uNoiseFrequency,
      noisePositionAmount: uNoisePosAmount,
      noiseRotationAmount: uNoiseRotAmount,
      noiseSizeAmount: uNoiseSizeAmount,
      emitCount: uEmitCount,
      seed: uSeed,
    },
    shapeUniforms,
    buffers,
    allocatorCount,
    packedDataNode: sCD as ShaderNodeObject<Node>,
    forceFieldInfo: ffNodes
      ? { offset: forceFieldOffset, countUniform: ffNodes.countUniform }
      : null,
    collisionPlaneInfo: cpNodes
      ? { offset: collisionOffset, countUniform: cpNodes.countUniform }
      : null,
  };
}

// Canonical r186 conditional selection: shape kind 0 = cone, 1 = sphere,
// 2 (anything else) = plane. `.select(ifTrue, ifFalse)` on boolean nodes only.
function select01(
  kind: ShaderNodeObject<Node>,
  cone: ShaderNodeObject<Node>,
  sphere: ShaderNodeObject<Node>,
  planeVal: ShaderNodeObject<Node>
) {
  const k = floor(kind);
  return k.equal(float(0.0)).select(
    cone,
    k.equal(float(1.0)).select(sphere, planeVal)
  );
}


