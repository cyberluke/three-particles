import * as THREE from 'three';
import {
  PointsNodeMaterial,
  StorageBufferAttribute,
} from 'three/webgpu';
import { SimulationBackend } from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';
import { enableWebGPU } from '../webgpu.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

// ─── Helpers ─────────────────────────────────────────────────────────────────

const createTestSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
): ParticleSystem =>
  createParticleSystem(
    {
      maxParticles: 10,
      duration: 5,
      looping: true,
      ...config,
    },
    startTime
  );

function createMockFactory() {
  const pipeline = {
    emitNode: { isNode: true, count: 1 },
    simNode: { isNode: true },
    computeNodes: [],
    passLayouts: [
      { name: 'emit', storageBindings: 8, uniformBindings: 1 },
      { name: 'simulate', storageBindings: 8, uniformBindings: 1 },
    ],
    passNames: ['emit', 'simulate'],
    allocatorCount: 11,
    shapeUniforms: { shapeKind: { value: 0 } },
    uniforms: {
      delta: { value: 0 },
      deltaMs: { value: 0 },
      gravityVelocity: { value: new THREE.Vector3() },
      emitCount: { value: 0 },
      seed: { value: 1 },
    },
    buffers: {} as Record<string, unknown>,
    packedDataNode: {
      addUpdateRange: (_s: number, _c: number) => {},
      needsUpdate: false,
    },
    trailMeta: null,
    forceFieldInfo: null,
    collisionPlaneInfo: null,
  };
  // The geometry contract reads the SAME storage attributes the pipeline owns.
  const mk = () => new StorageBufferAttribute(new Float32Array(40), 4);
  pipeline.buffers.position = mk();
  pipeline.buffers.velocity = mk();
  pipeline.buffers.color = mk();
  pipeline.buffers.particleState = mk();
  pipeline.buffers.startValues = mk();
  pipeline.buffers.startColorsExt = mk();
  pipeline.buffers.orbitalIsActive = mk();
  pipeline.buffers.allocator = new StorageBufferAttribute(
    new Uint32Array(11),
    1
  );
  pipeline.buffers.packedData = new Float32Array(1);

  return {
    mockFactory: {
      createTSLParticleMaterial: jest.fn(
        () => new THREE.ShaderMaterial() as unknown as THREE.Material
      ),
      createTSLTrailMaterial: jest.fn(
        () => new THREE.ShaderMaterial() as unknown as THREE.Material
      ),
      createComputePipeline: jest.fn(() => pipeline),
      createSubEmitterFifoAttribute: jest.fn(),
      createSubEmitterInitUpdate: jest.fn(),
      createTrailRibbonUpdate: jest.fn(),
      encodeShapeEmitParams: jest.fn(),
      encodeForceFieldsForGPU: jest.fn(),
      encodeCollisionPlanesForGPU: jest.fn(),
    },
    pipeline,
  };
}

// ─── registerTSLMaterialFactory ──────────────────────────────────────────────

describe('registerTSLMaterialFactory', () => {
  it('is exported and callable', () => {
    expect(typeof registerTSLMaterialFactory).toBe('function');
  });

  it('does not throw when registering a factory', () => {
    const { mockFactory } = createMockFactory();
    expect(() => registerTSLMaterialFactory(mockFactory)).not.toThrow();
    enableWebGPU();
  });
});

// ─── TSL material branching ──────────────────────────────────────────────────

describe('TSL material branching', () => {
  afterEach(() => {
    // Restore the real factory so later suites keep the GPU-only pipeline.
    enableWebGPU();
  });

  it('uses the TSL node material from the real factory registered at setup', () => {
    const ps = createTestSystem();
    const points = ps.instance as THREE.Points;
    expect(points.material).toBeInstanceOf(PointsNodeMaterial);
    ps.dispose();
  });

  it('calls the mock TSL factory when registered (GPU backend)', () => {
    const { mockFactory } = createMockFactory();
    registerTSLMaterialFactory(mockFactory);

    const ps = createTestSystem({
      simulationBackend: SimulationBackend.GPU,
    });
    expect(mockFactory.createTSLParticleMaterial).toHaveBeenCalled();
    expect(mockFactory.createComputePipeline).toHaveBeenCalled();
    ps.dispose();
  });

  it('passes renderer type and uniforms to TSL factory', () => {
    const { mockFactory } = createMockFactory();
    registerTSLMaterialFactory(mockFactory);

    const ps = createTestSystem({
      simulationBackend: SimulationBackend.GPU,
    });

    const [rendererType, uniforms, config] =
      mockFactory.createTSLParticleMaterial.mock.calls[0];

    // Renderer type should be POINTS (default)
    expect(rendererType).toBe('POINTS');

    // Uniforms should include standard particle system uniforms
    expect(uniforms).toHaveProperty('elapsed');
    expect(uniforms).toHaveProperty('map');
    expect(uniforms).toHaveProperty('tiles');
    expect(uniforms).toHaveProperty('fps');
    expect(uniforms).toHaveProperty('softParticlesEnabled');

    // Renderer config should include material properties
    expect(config).toHaveProperty('transparent');
    expect(config).toHaveProperty('blending');
    expect(config).toHaveProperty('depthTest');
    expect(config).toHaveProperty('depthWrite');

    ps.dispose();
  });
});

// ─── SimulationBackend config field ──────────────────────────────────────────

describe('simulationBackend config field', () => {
  it('defaults to AUTO in created systems', () => {
    const ps = createTestSystem();
    // System should create successfully with default AUTO backend
    expect(ps.instance).toBeDefined();
    ps.dispose();
  });

  it('accepts CPU backend without error (maps to the GPU kernel)', () => {
    const ps = createTestSystem({
      simulationBackend: SimulationBackend.CPU,
    });
    expect(ps.instance).toBeDefined();
    ps.dispose();
  });

  it('accepts GPU backend without error', () => {
    const ps = createTestSystem({
      simulationBackend: SimulationBackend.GPU,
    });
    expect(ps.instance).toBeDefined();
    ps.dispose();
  });
});
