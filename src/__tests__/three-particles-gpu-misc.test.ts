/**
 * Miscellaneous GPU contract tests for three-particles.ts (4.x GPU-only).
 *
 * The mock factory mirrors the real `ModifierComputePipeline` shape:
 * emit/sim nodes, pass layouts, allocator capacity, shape uniforms and the
 * packed f32 table.
 */

import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import { SimulationBackend } from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';
import { enableWebGPU } from '../webgpu.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

afterEach(() => {
  enableWebGPU();
});

function createMockGPUBuffers(maxParticles: number) {
  return {
    position: new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
    velocity: new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
    color: new StorageBufferAttribute(new Float32Array(maxParticles * 4), 4),
    particleState: new StorageBufferAttribute(
      new Float32Array(maxParticles * 4),
      4
    ),
    startValues: new StorageBufferAttribute(
      new Float32Array(maxParticles * 4),
      4
    ),
    startColorsExt: new StorageBufferAttribute(
      new Float32Array(maxParticles * 4),
      4
    ),
    orbitalIsActive: new StorageBufferAttribute(
      new Float32Array(maxParticles * 4),
      4
    ),
    allocator: new StorageBufferAttribute(new Uint32Array(maxParticles + 1), 1),
    trailMeta: null,
    packedData: new Float32Array(1),
  };
}

function createMockComputePipeline(maxParticles: number) {
  const buffers = createMockGPUBuffers(maxParticles);
  const emitNode = { isNode: true, count: 1 };
  const simNode = { isNode: true, count: maxParticles };
  return {
    emitNode,
    simNode,
    computeNodes: [emitNode, simNode],
    passLayouts: [
      { name: 'emit', storageBindings: 8, uniformBindings: 1 },
      { name: 'simulate', storageBindings: 8, uniformBindings: 1 },
    ],
    passNames: ['emit', 'simulate'],
    allocatorCount: maxParticles + 1,
    shapeUniforms: { shapeKind: { value: 0 } },
    uniforms: {
      delta: { value: 0 },
      deltaMs: { value: 0 },
      nowMs: { value: 0 },
      gravityVelocity: { value: new THREE.Vector3() },
      emitCount: { value: 0 },
      seed: { value: 11 },
      fifoBase: { value: 0 },
      noiseStrength: { value: 0 },
      noisePower: { value: 0 },
      noiseFrequency: { value: 1 },
      noisePositionAmount: { value: 0 },
      noiseRotationAmount: { value: 0 },
      noiseSizeAmount: { value: 0 },
    },
    buffers,
    packedDataNode: {
      addUpdateRange: (_start: number, _count: number) => {},
      needsUpdate: false,
    },
    trailMeta: null,
    forceFieldInfo: null,
    collisionPlaneInfo: null,
  };
}

function createFullGPUFactory(maxParticles = 10) {
  const pipeline = createMockComputePipeline(maxParticles);
  return {
    factory: {
      createTSLParticleMaterial: jest.fn(
        () => new THREE.ShaderMaterial() as unknown as THREE.Material
      ),
      createTSLTrailMaterial: jest.fn(
        () => new THREE.ShaderMaterial() as unknown as THREE.Material
      ),
      createComputePipeline: jest.fn(() => pipeline),
      createSubEmitterFifoAttribute: jest.fn((capacity: number) => ({
        counter: new StorageBufferAttribute(new Uint32Array(2), 1),
        payload: new StorageBufferAttribute(
          new Float32Array(2 * 6 * Math.max(1, capacity)),
          1
        ),
        trigger: 1 as const,
        capacity: Math.max(1, capacity),
        windowSize: 6 * Math.max(1, capacity),
      })),
      createTrailRibbonUpdate: jest.fn(() => ({
        ribbonNode: { isNode: true },
        passLayouts: [
          { name: 'trail-ribbon', storageBindings: 4, uniformBindings: 1 },
        ],
        uniforms: { nowMs: { value: 0 } },
        buffers: {
          position: pipeline.buffers.position,
          next: pipeline.buffers.velocity,
          uvColorA: pipeline.buffers.color,
          colorB: pipeline.buffers.particleState,
        },
      })),
      encodeShapeEmitParams: jest.fn(() => ({ shapeKind: 0 })),
      encodeForceFieldsForGPU: jest.fn(() => new Float32Array(0)),
      encodeCollisionPlanesForGPU: jest.fn(() => new Float32Array(0)),
    },
    pipeline,
  };
}

// ─── Noise initialization ────────────────────────────────────────────────────

describe('noise offset initialization', () => {
  it('initializes without error when useRandomOffset is true', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 20,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        noise: {
          isActive: true,
          useRandomOffset: true,
          strength: 1,
          frequency: 1,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
        },
      },
      1000
    );

    expect(ps.instance).toBeDefined();
    expect(() =>
      ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 })
    ).not.toThrow();
    ps.dispose();
  });

  it('works without noise active', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        noise: { isActive: false },
      },
      1000
    );

    expect(() =>
      ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 })
    ).not.toThrow();
    ps.dispose();
  });
});

// ─── GPU noise uniform propagation ──────────────────────────────────────────

describe('GPU noise uniforms', () => {
  it('propagates noise config to GPU uniforms', () => {
    const { factory, pipeline } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        noise: {
          isActive: true,
          useRandomOffset: false,
          strength: 2.5,
          frequency: 3.0,
          power: 1.5,
          positionAmount: 0.8,
          rotationAmount: 0.3,
          sizeAmount: 0.1,
        },
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });

    expect(pipeline.uniforms.noiseStrength.value).toBeCloseTo(2.5);
    expect(pipeline.uniforms.noiseFrequency.value).toBeCloseTo(3.0);
    expect(pipeline.uniforms.noisePositionAmount.value).toBeCloseTo(0.8);
    expect(pipeline.uniforms.noiseRotationAmount.value).toBeCloseTo(0.3);
    expect(pipeline.uniforms.noiseSizeAmount.value).toBeCloseTo(0.1);
    // noisePower is normalized by the FBM max accumulator (1.5 / 4).
    expect(pipeline.uniforms.noisePower.value).toBeCloseTo(0.375);
    ps.dispose();
  });
});

// ─── Noise via updateConfig ──────────────────────────────────────────────────

describe('noise offset initialization via updateConfig', () => {
  it('enables noise with random offsets via updateConfig', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 15,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        noise: {
          isActive: false,
          useRandomOffset: false,
          strength: 0,
          frequency: 0.5,
          power: 1,
          positionAmount: 0,
          rotationAmount: 0,
          sizeAmount: 0,
        },
      },
      1000
    );

    ps.updateConfig({
      noise: {
        isActive: true,
        useRandomOffset: true,
        strength: 2,
        frequency: 1,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0,
        sizeAmount: 0,
      },
    });

    expect(() =>
      ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 })
    ).not.toThrow();
    ps.dispose();
  });

  it('reuses the pipeline when updateConfig updates noise twice', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        noise: {
          isActive: true,
          useRandomOffset: true,
          strength: 1,
          frequency: 1,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
        },
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });

    ps.updateConfig({
      noise: {
        isActive: true,
        useRandomOffset: true,
        strength: 3,
        frequency: 2,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0,
        sizeAmount: 0,
      },
    });

    expect(() =>
      ps.update({ now: 1032, delta: 0.016, elapsed: 0.032 })
    ).not.toThrow();
    ps.dispose();
  });
});

// ─── Material disposal ───────────────────────────────────────────────────────

describe('material disposal', () => {
  it('disposes trail system without error', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 3,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        renderer: {
          rendererType: 'TRAIL',
          trail: { length: 4, minVertexDistance: 0.1 },
        },
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    expect(() => ps.dispose()).not.toThrow();
  });

  it('disposes non-trail system without error', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 5,
        duration: 5,
        looping: true,
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    expect(() => ps.dispose()).not.toThrow();
  });

  it('disposes instanced renderer without error', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 5,
        duration: 5,
        looping: true,
        renderer: { rendererType: 'INSTANCED' },
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    expect(() => ps.dispose()).not.toThrow();
  });

  it('disposes mesh renderer without error', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 5,
        duration: 5,
        looping: true,
        renderer: {
          rendererType: 'MESH',
          mesh: { geometry: new THREE.BoxGeometry(0.1, 0.1, 0.1) },
        },
      },
      1000
    );

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    expect(() => ps.dispose()).not.toThrow();
  });
});
