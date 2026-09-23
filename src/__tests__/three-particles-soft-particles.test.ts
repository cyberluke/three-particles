/**
 * Soft-particles uniform tests for the GPU-only (4.x) engine.
 *
 * The TSL factory receives the normalized soft-particle uniforms through the
 * shared-uniforms object, so the mock factory records them directly.
 */

import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';
import { enableWebGPU } from '../webgpu.js';

// ─── Mock factory that records the shared uniforms ──────────────────────────

function mockFactoryWithCapture() {
  const captured: {
    uniforms?: Record<string, { value: unknown }>;
  } = {};
  const mk = (n: number, itemSize: number) =>
    new StorageBufferAttribute(new Float32Array(n * itemSize), itemSize);
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
    buffers: {
      position: mk(10, 4),
      velocity: mk(10, 4),
      color: mk(10, 4),
      particleState: mk(10, 4),
      startValues: mk(10, 4),
      startColorsExt: mk(10, 4),
      orbitalIsActive: mk(10, 4),
      allocator: new StorageBufferAttribute(new Uint32Array(11), 1),
      packedData: new Float32Array(1),
    },
    packedDataNode: {
      addUpdateRange: (_s: number, _c: number) => {},
      needsUpdate: false,
    },
    trailMeta: null,
    forceFieldInfo: null,
    collisionPlaneInfo: null,
  };
  const factory = {
    createTSLParticleMaterial: jest.fn(
      (_type: unknown, uniforms: Record<string, { value: unknown }>) => {
        captured.uniforms = uniforms;
        return new THREE.ShaderMaterial() as unknown as THREE.Material;
      }
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
  };
  return { factory, captured };
}

const createWithFactory = (
  config: Record<string, unknown>,
  startTime = 1000
) => {
  const { factory, captured } = mockFactoryWithCapture();
  registerTSLMaterialFactory(factory);
  const ps = createParticleSystem(
    { maxParticles: 10, duration: 5, looping: true, ...config },
    startTime
  );
  return { ps, captured };
};

afterEach(() => {
  enableWebGPU();
});

// ─── Defaults ───────────────────────────────────────────────────────────────

describe('soft particle uniforms — defaults', () => {
  it('should default softParticlesEnabled to false', () => {
    const { ps, captured } = createWithFactory({});
    expect(captured.uniforms!.softParticlesEnabled.value).toBe(false);
    ps.dispose();
  });

  it('should default softParticlesIntensity to 1.0', () => {
    const { ps, captured } = createWithFactory({});
    expect(captured.uniforms!.softParticlesIntensity.value).toBe(1);
    ps.dispose();
  });

  it('should default sceneDepthTexture to null', () => {
    const { ps, captured } = createWithFactory({});
    expect(captured.uniforms!.sceneDepthTexture.value).toBeNull();
    ps.dispose();
  });

  it('should default cameraNearFar to (0.1, 1000)', () => {
    const { ps, captured } = createWithFactory({});
    const cnf = captured.uniforms!.cameraNearFar.value as THREE.Vector2;
    expect(cnf.x).toBeCloseTo(0.1);
    expect(cnf.y).toBeCloseTo(1000);
    ps.dispose();
  });
});

// ─── Enabled flag resolution ────────────────────────────────────────────────

describe('soft particle uniforms — enabled flag', () => {
  it('should set softParticlesEnabled to true when enabled', () => {
    const depth = new THREE.DataTexture(new Float32Array(4), 2, 2);
    const { ps, captured } = createWithFactory({
      renderer: {
        softParticles: { enabled: true, depthTexture: depth },
      },
    });
    expect(captured.uniforms!.softParticlesEnabled.value).toBe(true);
    expect(captured.uniforms!.sceneDepthTexture.value).toBe(depth);
    ps.dispose();
  });

  it('keeps enabled true when no depthTexture is provided', () => {
    const { ps, captured } = createWithFactory({
      renderer: { softParticles: { enabled: true } },
    });
    expect(captured.uniforms!.softParticlesEnabled.value).toBe(true);
    expect(captured.uniforms!.sceneDepthTexture.value).toBeNull();
    ps.dispose();
  });

  it('should disable when enabled is false even with depthTexture', () => {
    const depth = new THREE.DataTexture(new Float32Array(4), 2, 2);
    const { ps, captured } = createWithFactory({
      renderer: {
        softParticles: { enabled: false, depthTexture: depth },
      },
    });
    expect(captured.uniforms!.softParticlesEnabled.value).toBe(false);
    ps.dispose();
  });
});

// ─── Intensity clamping ─────────────────────────────────────────────────────

describe('soft particle uniforms — intensity clamp', () => {
  it('should use default intensity when not specified', () => {
    const { ps, captured } = createWithFactory({
      renderer: { softParticles: { enabled: true } },
    });
    expect(captured.uniforms!.softParticlesIntensity.value).toBe(1);
    ps.dispose();
  });

  it('should clamp intensity to minimum 0.001 when set to zero', () => {
    const { ps, captured } = createWithFactory({
      renderer: { softParticles: { enabled: true, intensity: 0 } },
    });
    expect(captured.uniforms!.softParticlesIntensity.value).toBeCloseTo(0.001);
    ps.dispose();
  });

  it('should clamp negative intensity to minimum 0.001', () => {
    const { ps, captured } = createWithFactory({
      renderer: { softParticles: { enabled: true, intensity: -5 } },
    });
    expect(captured.uniforms!.softParticlesIntensity.value).toBeCloseTo(0.001);
    ps.dispose();
  });
});

// ─── Per-renderer presence ──────────────────────────────────────────────────

describe('soft particle uniforms — all renderer classes', () => {
  it.each(['POINTS', 'INSTANCED', 'MESH', 'TRAIL'])(
    'should include soft particle uniforms for %s renderer',
    (type) => {
      const { ps, captured } = createWithFactory({
        renderer: {
          rendererType: type,
          mesh: { geometry: new THREE.BoxGeometry(1, 1, 1) },
          softParticles: { enabled: true },
        },
      });
      expect(captured.uniforms).toHaveProperty('softParticlesEnabled');
      expect(captured.uniforms).toHaveProperty('softParticlesIntensity');
      expect(captured.uniforms).toHaveProperty('sceneDepthTexture');
      expect(captured.uniforms).toHaveProperty('cameraNearFar');
      ps.dispose();
    }
  );
});
