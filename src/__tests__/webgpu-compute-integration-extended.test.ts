/**
 * Extended WebGPU compute integration tests (4.x GPU-only contract).
 *
 * Uses a fully-mocked TSL factory whose compute pipeline mirrors the real
 * `ModifierComputePipeline` shape: emit/sim nodes, real pass layouts, the
 * 8-binding base pool, the ring allocator and the packed f32 table.
 */

import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import {
  SimulationBackend,
  SimulationSpace,
} from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';

// ─── Mock GPU pipeline ──────────────────────────────────────────────────────

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
    allocator: new StorageBufferAttribute(
      new Uint32Array(maxParticles + 1),
      1
    ),
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
      seed: { value: 7 },
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
    emitterPose: {
      positionW: { value: new THREE.Vector4(0, 0, 0, 0) },
      wrapperQuat: { value: new THREE.Vector4(0, 0, 0, 1) },
      worldScale: { value: new THREE.Vector3(1, 1, 1) },
    },
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

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('GPU compute integration', () => {
  afterEach(() => {
    registerTSLMaterialFactory(
      null as unknown as Parameters<typeof registerTSLMaterialFactory>[0]
    );
  });

  it('creates GPU pipeline when factory has all compute methods', () => {
    const { factory } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    expect(factory.createComputePipeline).toHaveBeenCalledTimes(1);
    expect(factory.createComputePipeline).toHaveBeenCalledWith(
      10,
      expect.any(Boolean),
      expect.any(Object),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Array),
      undefined
    );
    ps.dispose();
  });

  it('uses GPU buffer attributes for geometry (not CPU interleaved)', () => {
    const { factory, pipeline } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    // The material factory was called with gpuCompute=true
    const callArgs = factory.createTSLParticleMaterial.mock.calls[0];
    expect(callArgs[3]).toBe(true); // gpuCompute flag

    // The geometry attributes ARE the compute-owned storage buffers.
    const geom = (ps.instance as THREE.Points).geometry;
    expect(geom.getAttribute('position')).toBe(pipeline.buffers.position);
    expect(geom.getAttribute('color')).toBe(pipeline.buffers.color);
    expect(geom.getAttribute('particleState')).toBe(
      pipeline.buffers.particleState
    );
    expect(geom.getAttribute('startValues')).toBe(
      pipeline.buffers.startValues
    );
    ps.dispose();
  });

  it('GPU pipeline has expected uniform structure', () => {
    const { factory, pipeline } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    expect(pipeline.uniforms.delta).toBeDefined();
    expect(pipeline.uniforms.gravityVelocity).toBeDefined();
    expect(pipeline.uniforms.noiseStrength).toBeDefined();
    expect(pipeline.uniforms.emitCount).toBeDefined();
    expect(pipeline.buffers).toBeDefined();
    expect(pipeline.buffers.allocator).toBeDefined();
    expect(pipeline.buffers.packedData).toBeInstanceOf(Float32Array);
    ps.dispose();
  });

  it('uses GPU compute + ribbon pipeline for the trail renderer', () => {
    const { factory } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        renderer: { rendererType: 'TRAIL', trail: { length: 4 } },
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    // Trail systems run the base kernels AND the ribbon expansion pass.
    expect(factory.createComputePipeline).toHaveBeenCalled();
    expect(factory.createTrailRibbonUpdate).toHaveBeenCalledTimes(1);
    expect(ps.instance).toBeInstanceOf(THREE.Mesh);
    ps.dispose();
  });

  it('handles world space simulation with GPU compute', () => {
    const { factory } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        simulationSpace: SimulationSpace.WORLD,
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    // WORLD-space GPU systems hold matrixWorld at identity — the buffer
    // stores world coordinates directly, so no per-frame compensation
    // uniform is needed.
    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    expect(ps.instance.matrixWorld.elements).toEqual(
      new THREE.Matrix4().elements
    );
    ps.dispose();
  });

  it('GPU WORLD: instance is not wrapped and matrixWorld stays identity under parent motion', () => {
    const { factory } = createFullGPUFactory(10);
    registerTSLMaterialFactory(factory);

    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 10 },
        simulationSpace: SimulationSpace.WORLD,
        simulationBackend: SimulationBackend.GPU,
      },
      1000
    );

    // No Gyroscope / wrapper — instance is the direct renderable.
    expect(ps.instance).toBeInstanceOf(THREE.Object3D);
    expect(ps.instance.children).toHaveLength(0);

    const parent = new THREE.Group();
    parent.add(ps.instance);
    parent.position.set(50, 0, 0);
    parent.quaternion.setFromEuler(new THREE.Euler(0, Math.PI / 2, 0));
    parent.updateMatrixWorld(true);

    ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    parent.position.set(-30, 12, 7);
    parent.updateMatrixWorld(true);
    ps.update({ now: 1032, delta: 0.016, elapsed: 0.032 });

    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(ps.instance.matrixWorld.elements).toEqual(
      new THREE.Matrix4().elements
    );
    ps.dispose();
  });
});
