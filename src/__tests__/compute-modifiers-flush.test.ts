/**
 * Tests for the GPU-owned compute pipeline structure (4.x architecture).
 *
 * The CPU no longer walks particle slots: `createModifierStorageBuffers`
 * allocates the 8 storage bindings + the packed f32 uniform table once, and
 * `createModifierComputeUpdate` returns the compute nodes in dispatch order.
 * These tests pin the dispatch structure and the ring-allocator init.
 */

import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import {
  createModifierStorageBuffers,
  createModifierComputeUpdate,
  createSubEmitterFifoAttribute,
  INIT_STRIDE,
  type ModifierFlags,
} from '../js/effects/three-particles/webgpu/compute-modifiers.js';
import type { BakedCurveMap } from '../js/effects/three-particles/webgpu/curve-bake.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const MAX_PARTICLES = 10;

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

function makeBuffers(curveData = new Float32Array(256)) {
  return createModifierStorageBuffers(MAX_PARTICLES, true, curveData).buffers;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('compute pipeline dispatch structure', () => {
  it('base pipeline dispatches exactly emit → simulate', () => {
    const buffers = makeBuffers();
    const pipeline = createModifierComputeUpdate(
      buffers,
      MAX_PARTICLES,
      EMPTY_CURVE_MAP,
      NO_FLAGS
    );
    expect(pipeline.passNames).toEqual(['emit', 'simulate']);
    expect(pipeline.computeNodes).toHaveLength(2);
    expect(pipeline.computeNodes[0]).toBe(pipeline.emitNode);
    expect(pipeline.computeNodes[1]).toBe(pipeline.simNode);
  });

  it('ring allocator buffer is pre-filled 0..maxParticles-1 after the counter', () => {
    const buffers = makeBuffers();
    const arr = buffers.allocator.array as Uint32Array;
    expect(arr).toHaveLength(MAX_PARTICLES + 1);
    expect(arr[0]).toBe(0); // freeCount / birth counter
    for (let i = 0; i < MAX_PARTICLES; i++) {
      expect(arr[i + 1]).toBe(i); // slot ids
    }
  });

  it('BIRTH FIFOs insert the sub-birth-events pass between emit and simulate', () => {
    const buffers = makeBuffers();
    const fifo = createSubEmitterFifoAttribute(4);
    fifo.trigger = 0;
    const pipeline = createModifierComputeUpdate(
      buffers,
      MAX_PARTICLES,
      EMPTY_CURVE_MAP,
      NO_FLAGS,
      undefined,
      0,
      0,
      [fifo]
    );
    expect(pipeline.subBirthEventsNode).not.toBeNull();
    expect(pipeline.subDeathEventsNode).toBeNull();
    expect(pipeline.passNames).toEqual([
      'emit',
      'sub-birth-events',
      'simulate',
    ]);
  });

  it('DEATH FIFOs append the sub-death-events pass after simulate', () => {
    const buffers = makeBuffers();
    const fifo = createSubEmitterFifoAttribute(4);
    fifo.trigger = 1;
    const pipeline = createModifierComputeUpdate(
      buffers,
      MAX_PARTICLES,
      EMPTY_CURVE_MAP,
      NO_FLAGS,
      undefined,
      0,
      0,
      [fifo]
    );
    expect(pipeline.subDeathEventsNode).not.toBeNull();
    expect(pipeline.passNames).toEqual([
      'emit',
      'simulate',
      'sub-death-events',
    ]);
  });

  it('emitCount and fifoBase uniforms are exposed for per-frame CPU writes', () => {
    const buffers = makeBuffers();
    const pipeline = createModifierComputeUpdate(
      buffers,
      MAX_PARTICLES,
      EMPTY_CURVE_MAP,
      NO_FLAGS
    );
    expect(pipeline.uniforms.emitCount).toBeDefined();
    expect(pipeline.uniforms.fifoBase).toBeDefined();
    expect(pipeline.uniforms.seed).toBeDefined();
  });

  it('packed table stays a plain f32 array (uniform binding, not storage)', () => {
    const buffers = makeBuffers(new Float32Array(64));
    expect(buffers.packedData).toBeInstanceOf(Float32Array);
    expect(buffers.packedData).toHaveLength(64);
    // The 8 storage bindings are the 7 vec4 pools + the uint allocator only
    // (instanced mode swaps the vec4 pools for their instanced subclass).
    const storageAttrs = [
      buffers.position,
      buffers.velocity,
      buffers.color,
      buffers.particleState,
      buffers.startValues,
      buffers.startColorsExt,
      buffers.orbitalIsActive,
      buffers.allocator,
    ];
    expect(
      storageAttrs.every(
        (a) =>
          a instanceof StorageBufferAttribute ||
          a instanceof StorageInstancedBufferAttribute
      )
    ).toBe(true);
  });

  it('INIT_STRIDE keeps the 28-float per-particle init block size', () => {
    expect(INIT_STRIDE).toBe(28);
  });
});
