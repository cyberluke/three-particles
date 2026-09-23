import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import {
  createModifierStorageBuffers,
  createModifierComputeUpdate,
  INIT_STRIDE,
  type ModifierFlags,
} from '../js/effects/three-particles/webgpu/compute-modifiers.js';
import type { BakedCurveMap } from '../js/effects/three-particles/webgpu/curve-bake.js';

// ─── Helpers ─────────────────────────────────────────────────────────────────

const NO_FLAGS: ModifierFlags = {
  sizeOverLifetime: false,
  opacityOverLifetime: false,
  colorOverLifetime: false,
  rotationOverLifetime: false,
  linearVelocity: false,
  orbitalVelocity: false,
  noise: false,
  forceFields: false,
  collisionPlanes: false,
};

const EMPTY_CURVE_MAP: BakedCurveMap = {
  data: new Float32Array(0),
  curveCount: 0,
  sizeOverLifetime: -1,
  opacityOverLifetime: -1,
  colorR: -1,
  colorG: -1,
  colorB: -1,
  linearVelX: -1,
  linearVelY: -1,
  linearVelZ: -1,
  orbitalVelX: -1,
  orbitalVelY: -1,
  orbitalVelZ: -1,
};

// ─── createModifierStorageBuffers ────────────────────────────────────────────

describe('createModifierStorageBuffers', () => {
  it('creates the 8 storage bindings + packed f32 table (curves only)', () => {
    const curveData = new Float32Array(256);
    const { buffers, allocatorCount } = createModifierStorageBuffers(
      50,
      false,
      curveData
    );

    expect(buffers.position).toBeInstanceOf(StorageBufferAttribute);
    expect(buffers.position.array).toHaveLength(200); // 50 * 4 (vec4)
    expect(buffers.velocity.array).toHaveLength(200);
    expect(buffers.color.array).toHaveLength(200);
    expect(buffers.particleState.array).toHaveLength(200);
    expect(buffers.startValues.array).toHaveLength(200);
    expect(buffers.startColorsExt.array).toHaveLength(200);
    expect(buffers.orbitalIsActive.array).toHaveLength(200);
    // allocator: uint stack of maxParticles + 1 elements, pre-filled 0..n-1
    expect(allocatorCount).toBe(51);
    expect(buffers.allocator.array).toHaveLength(51);
    expect((buffers.allocator.array as Uint32Array)[0]).toBe(0);
    expect((buffers.allocator.array as Uint32Array)[50]).toBe(49);
    // packedData = curve samples only (no force fields / collision planes)
    expect(buffers.packedData).toHaveLength(256);
    expect(buffers.trailMeta).toBeNull();
    // No separate emitQueue buffer
    expect((buffers as Record<string, unknown>).emitQueue).toBeUndefined();
  });

  it('clamps the packed table to 1 element when curve data is empty', () => {
    const { buffers } = createModifierStorageBuffers(
      10,
      false,
      new Float32Array(0)
    );
    expect(buffers.packedData).toHaveLength(1);
  });

  it('appends force-field and collision-plane records to packedData', () => {
    const { buffers } = createModifierStorageBuffers(
      10,
      false,
      new Float32Array(64),
      true,
      true
    );
    // 64 curves + 16*12 force-field words + 16*12 collision-plane words
    expect(buffers.packedData).toHaveLength(64 + 192 + 192);
  });

  it('uses instanced storage attributes when requested', () => {
    const { buffers } = createModifierStorageBuffers(
      25,
      true,
      new Float32Array(0)
    );
    expect(buffers.position).toBeInstanceOf(StorageInstancedBufferAttribute);
    expect(buffers.particleState).toBeInstanceOf(
      StorageInstancedBufferAttribute
    );
  });

  it('creates integer trail meta only when trailLength > 0', () => {
    const noTrail = createModifierStorageBuffers(8, false, new Float32Array(0));
    expect(noTrail.buffers.trailMeta).toBeNull();

    const withTrail = createModifierStorageBuffers(
      8,
      false,
      new Float32Array(0),
      false,
      false,
      4
    );
    expect(withTrail.buffers.trailMeta).not.toBeNull();
    expect(withTrail.buffers.trailMeta!.array).toHaveLength(16); // 2 words/particle
  });
});

// ─── createModifierComputeUpdate ─────────────────────────────────────────────

describe('createModifierComputeUpdate', () => {
  it('creates the two-pass pipeline with no modifiers active', () => {
    const { buffers } = createModifierStorageBuffers(
      100,
      false,
      new Float32Array(0)
    );
    const pipeline = createModifierComputeUpdate(
      buffers,
      100,
      EMPTY_CURVE_MAP,
      NO_FLAGS
    );

    expect(pipeline.emitNode).toBeDefined();
    expect(pipeline.simNode).toBeDefined();
    expect(pipeline.computeNodes).toEqual([
      pipeline.emitNode,
      pipeline.simNode,
    ]);
    expect(pipeline.passNames).toEqual(['emit', 'simulate']);
    expect(pipeline.passLayouts).toHaveLength(2);
    // Base pool stays within the 8-storage-binding WebGPU guarantee.
    for (const pass of pipeline.passLayouts) {
      expect(pass.storageBindings).toBe(8);
    }
    expect(pipeline.uniforms).toBeDefined();
    expect(pipeline.uniforms.delta).toBeDefined();
    expect(pipeline.uniforms.deltaMs).toBeDefined();
    expect(pipeline.uniforms.gravityVelocity).toBeDefined();
    expect(pipeline.uniforms.noiseStrength).toBeDefined();
    expect(pipeline.uniforms.emitCount).toBeDefined();
    expect(pipeline.uniforms.seed).toBeDefined();
    expect(pipeline.buffers).toBe(buffers);
    expect(pipeline.allocatorCount).toBe(101);
    expect(pipeline.packedDataNode).toBeDefined();
    expect(pipeline.trailHistoryNode).toBeNull();
    expect(pipeline.subBirthEventsNode).toBeNull();
    expect(pipeline.subDeathEventsNode).toBeNull();
    expect(pipeline.forceFieldInfo).toBeNull();
    expect(pipeline.collisionPlaneInfo).toBeNull();
  });

  it('creates pipeline with all lifetime modifiers active', () => {
    const curveData = new Float32Array(256 * 8); // 8 curves
    const { buffers } = createModifierStorageBuffers(50, false, curveData);
    const allFlags: ModifierFlags = {
      sizeOverLifetime: true,
      opacityOverLifetime: true,
      colorOverLifetime: true,
      rotationOverLifetime: true,
      linearVelocity: true,
      orbitalVelocity: true,
      noise: true,
      forceFields: false,
      collisionPlanes: false,
    };
    const curveMap: BakedCurveMap = {
      data: curveData,
      curveCount: 8,
      sizeOverLifetime: 0,
      opacityOverLifetime: 1,
      colorR: 2,
      colorG: 3,
      colorB: 4,
      linearVelX: 5,
      linearVelY: 6,
      linearVelZ: 7,
      orbitalVelX: -1,
      orbitalVelY: -1,
      orbitalVelZ: -1,
    };

    const pipeline = createModifierComputeUpdate(
      buffers,
      50,
      curveMap,
      allFlags
    );
    expect(pipeline.simNode).toBeDefined();
    expect(pipeline.buffers.packedData).toHaveLength(256 * 8);
  });

  it('exposes force-field / collision-plane offsets after the curve table', () => {
    const { buffers } = createModifierStorageBuffers(
      20,
      false,
      new Float32Array(32),
      true,
      true
    );
    const flags: ModifierFlags = {
      ...NO_FLAGS,
      forceFields: true,
      collisionPlanes: true,
    };
    const curveMap: BakedCurveMap = { ...EMPTY_CURVE_MAP, data: new Float32Array(32) };
    const pipeline = createModifierComputeUpdate(
      buffers,
      20,
      curveMap,
      flags,
      undefined,
      3,
      2
    );
    expect(pipeline.forceFieldInfo).not.toBeNull();
    expect(pipeline.forceFieldInfo!.offset).toBe(32);
    expect(pipeline.forceFieldInfo!.countUniform).toBeDefined();
    expect(pipeline.collisionPlaneInfo).not.toBeNull();
    expect(pipeline.collisionPlaneInfo!.offset).toBe(32 + 192);
    expect(pipeline.collisionPlaneInfo!.countUniform).toBeDefined();
  });

  it('creates pipeline with instanced buffers', () => {
    const { buffers } = createModifierStorageBuffers(
      25,
      true,
      new Float32Array(0)
    );
    const pipeline = createModifierComputeUpdate(
      buffers,
      25,
      EMPTY_CURVE_MAP,
      NO_FLAGS
    );
    expect(pipeline.simNode).toBeDefined();
  });

  it('adds a trail-history pass when a trail descriptor is provided', () => {
    const { buffers } = createModifierStorageBuffers(
      10,
      false,
      new Float32Array(0),
      false,
      false,
      4
    );
    const pipeline = createModifierComputeUpdate(
      buffers,
      10,
      EMPTY_CURVE_MAP,
      NO_FLAGS,
      undefined,
      0,
      0,
      [],
      {
        attribute: new StorageBufferAttribute(
          new Float32Array(10 * (4 + 1) * 4),
          4
        ),
        meta: buffers.trailMeta!,
        length: 4,
        minVertexDistance: 0.01,
        maxTime: 0,
      }
    );
    expect(pipeline.trailHistoryNode).not.toBeNull();
    expect(pipeline.passNames).toEqual(['emit', 'simulate', 'trail-history']);
    expect(pipeline.computeNodes).toContain(pipeline.trailHistoryNode);
    expect(pipeline.trailMeta).toBe(buffers.trailMeta);
  });
});

// ─── Layout constant ─────────────────────────────────────────────────────────

describe('INIT_STRIDE', () => {
  it('is the 28-float per-particle init block used by the kernel index math', () => {
    expect(INIT_STRIDE).toBe(28);
  });
});
