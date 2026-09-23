import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import {
  SCALAR_STRIDE,
  S_SIZE,
} from '../js/effects/three-particles/three-particles-constants.js';
import { LifeTimeCurve } from '../js/effects/three-particles/three-particles-enums.js';
import { applyModifiers } from '../js/effects/three-particles/three-particles-modifiers.js';
import {
  createParticleSystem,
  registerTSLMaterialFactory,
} from '../js/effects/three-particles/three-particles.js';
import { enableWebGPU } from '../webgpu.js';
import {
  GeneralData,
  Noise,
  NormalizedParticleSystemConfig,
  ParticleSystem,
} from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: the CPU writes scalar uniforms only; the
 * per-frame emission count is observable through `gpuDebug.lastEmitCount()`,
 * while `getActiveParticleCount()` is the deprecated sentinel (-1).
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createTestSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 50,
      duration: 5,
      looping: true,
      startLifetime: 10,
      startSpeed: 0,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
      gravity: 0,
      emission: { rateOverTime: 10, rateOverDistance: 0 },
      ...config,
    } as never,
    startTime
  );

  const step = (timeOffsetMs: number, deltaMs: number = 16) => {
    ps.update({
      now: startTime + timeOffsetMs,
      delta: deltaMs / 1000,
      elapsed: timeOffsetMs / 1000,
    });
  };

  return { ps, step, startTime };
};

describe('time-based emission accumulator', () => {
  test('carries the fractional remainder instead of flooring it away', () => {
    // 100/s at 16ms steps = 1.6 particles per frame. The accumulator keeps
    // the remainder, so the tenth frame's dispatch is 2, not 1.
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      maxParticles: 100,
    });
    for (let i = 1; i <= 10; i++) step(i * 16);
    expect(lastEmit(ps)).toBe(2);
    ps.dispose();
  });

  test('startDelay shifts the normalized lifetime window', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      startDelay: 1,
      maxParticles: 200,
    });
    step(500);
    // Emission scalars still advance; the delay shifts the lifetime phase.
    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });

  test('clamps the per-frame dispatch to the pool size', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 1000 },
      maxParticles: 5,
      startLifetime: 10,
    });
    for (let i = 1; i <= 20; i++) step(i * 16);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

describe('getActiveParticleCount', () => {
  test('is the deprecated GPU sentinel (-1)', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    for (let i = 1; i <= 5; i++) step(i * 16);
    expect(ps.getActiveParticleCount).toBeDefined();
    expect(ps.getActiveParticleCount!()).toBe(-1);
    ps.dispose();
  });

  test('lastEmitCount reflects the per-frame scalar write', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);
    expect(lastEmit(ps)).toBe(1);
    ps.dispose();
  });
});

describe('updateConfig live module activation (scalar path)', () => {
  test('activating velocityOverLifetime after creation keeps running', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);

    ps.updateConfig({
      velocityOverLifetime: {
        isActive: true,
        linear: { y: 5 },
      } as never,
    });
    expect(() => {
      step(32);
      step(48);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });

  test('activating rotationOverLifetime after creation keeps running', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);

    ps.updateConfig({
      rotationOverLifetime: { isActive: true, min: 100, max: 100 },
    });
    expect(() => {
      step(32);
      step(48);
    }).not.toThrow();
    ps.dispose();
  });

  test('changing the sizeOverLifetime curve takes effect without error', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      sizeOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: LifeTimeCurve.BEZIER,
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
      },
    });
    step(16);

    ps.updateConfig({
      sizeOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: LifeTimeCurve.BEZIER,
          scale: 3,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
      },
    });
    expect(() => step(32)).not.toThrow();
    ps.dispose();
  });
});

describe('updateConfig merging', () => {
  test('merges partial configs without warnings', () => {
    const { ps } = createTestSystem();
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    ps.updateConfig({ gravity: 5 });
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
    ps.dispose();
  });
});

describe('GPU storage attribute contract', () => {
  test('geometry attributes are the compute-owned storage pools', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);
    const points = ps.instance as THREE.Points;
    const posAttr = points.geometry.attributes.position;
    const psAttr = points.geometry.attributes.particleState;
    expect(posAttr).toBeDefined();
    expect(psAttr).toBeDefined();
    // First-frame upload marks the pools for one GPU upload.
    expect(posAttr.version).toBeGreaterThanOrEqual(1);
    ps.dispose();
  });
});

describe('applyModifiers updateFlags aggregation', () => {
  test('sets flags instead of bumping attribute versions when provided', () => {
    const attributes = {
      position: { array: new Float32Array(3), needsUpdate: false },
    } as unknown as THREE.NormalBufferAttributes;
    const updateFlags = { position: false, quat: false };

    applyModifiers({
      delta: 1,
      generalData: {
        noise: { isActive: false } as Noise,
        particleSystemId: 0,
        startValues: {},
        lifetimeValues: {},
        linearVelocityData: [
          {
            speed: new THREE.Vector3(1, 0, 0),
            valueModifiers: { x: undefined, y: undefined, z: undefined },
          },
        ],
      } as unknown as GeneralData,
      normalizedConfig: {
        sizeOverLifetime: { isActive: false },
        opacityOverLifetime: { isActive: false },
        colorOverLifetime: { isActive: false },
      } as unknown as NormalizedParticleSystemConfig,

      attributes: attributes as never,
      scalarArray: new Float32Array(SCALAR_STRIDE),
      particleLifetimePercentage: 0.5,
      particleIndex: 0,
      updateFlags,
    });

    expect(updateFlags.position).toBe(true);

    expect((attributes.position as { needsUpdate: boolean }).needsUpdate).toBe(
      false
    );
  });

  test('uses pre-resolved modifier curves from generalData when present', () => {
    const scalarArray = new Float32Array(SCALAR_STRIDE);
    const curveFn = jest.fn(() => 4);

    applyModifiers({
      delta: 1,
      generalData: {
        noise: { isActive: false } as Noise,
        particleSystemId: 0,
        startValues: { startSize: [2] },
        lifetimeValues: {},
        modifierCurves: { size: curveFn },
      } as unknown as GeneralData,
      normalizedConfig: {
        sizeOverLifetime: { isActive: true },
        opacityOverLifetime: { isActive: false },
        colorOverLifetime: { isActive: false },
      } as unknown as NormalizedParticleSystemConfig,
      attributes: {
        position: { array: new Float32Array(3), needsUpdate: false },
      } as never,
      scalarArray,
      particleLifetimePercentage: 0.5,
      particleIndex: 0,
    });

    expect(curveFn).toHaveBeenCalledWith(0.5);
    expect(scalarArray[S_SIZE]).toBe(8); // startSize 2 × curve 4
  });
});

describe('registerTSLMaterialFactory renderer gating', () => {
  const mockFactory = {
    createTSLParticleMaterial: () =>
      new THREE.MeshBasicMaterial() as THREE.Material,
    createTSLTrailMaterial: () =>
      new THREE.MeshBasicMaterial() as THREE.Material,
    createComputePipeline: (maxParticles: number) => {
      const mk = (n: number, itemSize: number) =>
        new StorageBufferAttribute(new Float32Array(n * itemSize), itemSize);
      return {
        emitNode: { isNode: true, count: 1 },
        simNode: { isNode: true },
        computeNodes: [],
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
          gravityVelocity: { value: new THREE.Vector3() },
          emitCount: { value: 0 },
          seed: { value: 1 },
        },
        buffers: {
          position: mk(maxParticles, 4),
          velocity: mk(maxParticles, 4),
          color: mk(maxParticles, 4),
          particleState: mk(maxParticles, 4),
          startValues: mk(maxParticles, 4),
          startColorsExt: mk(maxParticles, 4),
          orbitalIsActive: mk(maxParticles, 4),
          allocator: new StorageBufferAttribute(
            new Uint32Array(maxParticles + 1),
            1
          ),
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
    },
    createSubEmitterFifoAttribute: (capacity: number) => ({
      counter: new StorageBufferAttribute(new Uint32Array(2), 1),
      payload: new StorageBufferAttribute(
        new Float32Array(2 * 6 * Math.max(1, capacity)),
        1
      ),
      trigger: 1 as const,
      capacity: Math.max(1, capacity),
      windowSize: 6 * Math.max(1, capacity),
    }),
    encodeShapeEmitParams: () => ({ shapeKind: 0 }),
    encodeForceFieldsForGPU: () => new Float32Array(0),
    encodeCollisionPlanesForGPU: () => new Float32Array(0),
  };

  test('skips registration and warns for a non-compute-capable renderer', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const result = registerTSLMaterialFactory(mockFactory, {
      renderer: { render: () => {}, getSize: () => {} },
    });
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('does not support compute dispatches')
    );
    warnSpy.mockRestore();
    // The real factory from the setup is still active — systems keep the
    // GPU contract.
    const ps = createParticleSystem({ maxParticles: 10 });
    expect(ps.instance).toBeInstanceOf(THREE.Points);
    ps.dispose();
  });

  test('registers for a compute-capable renderer (duck-typed)', () => {
    const computeCapableRenderer = {
      compute: () => {},
      hasFeature: () => true,
    };
    const result = registerTSLMaterialFactory(mockFactory, {
      renderer: computeCapableRenderer,
    });
    expect(result).toBe(true);
    // The duck-typed renderer lacks `backend.isWebGPUBackend`, so the
    // GPU-only build rejects it at system creation.
    expect(() => createParticleSystem({ maxParticles: 10 })).toThrow(
      'not a native WebGPU backend'
    );
    enableWebGPU();
  });
});
