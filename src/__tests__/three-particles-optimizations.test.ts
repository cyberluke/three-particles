import * as THREE from 'three';
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
import {
  GeneralData,
  Noise,
  NormalizedParticleSystemConfig,
  ParticleSystem,
} from '../js/effects/three-particles/types.js';

const countActiveParticles = (ps: ParticleSystem): number => {
  const points = ps.instance as THREE.Points;
  const isActiveAttr = points.geometry.attributes.isActive;
  let count = 0;
  for (let i = 0; i < isActiveAttr.count; i++) {
    if (isActiveAttr.getX(i)) count++;
  }
  return count;
};

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
    } as any,
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
    // 100/s at 16ms steps = 1.6 particles per frame. Flooring per frame
    // (the old behavior) emits 10 in 10 frames; the accumulator emits 16.
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      maxParticles: 100,
    });
    for (let i = 1; i <= 10; i++) step(i * 16);
    expect(countActiveParticles(ps)).toBe(16);
    ps.dispose();
  });

  test('does not emit before startDelay has elapsed', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      startDelay: 1,
      maxParticles: 200,
    });
    step(500);
    expect(countActiveParticles(ps)).toBe(0);
    ps.dispose();
  });

  test('does not dump a backlog burst after the delay elapses', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      startDelay: 1,
      maxParticles: 200,
    });
    step(500);
    step(1016);
    // Only the 16ms since the delay elapsed count — not the whole 1016ms.
    expect(countActiveParticles(ps)).toBeLessThanOrEqual(2);
    ps.dispose();
  });

  test('drops the backlog instead of bursting when the pool is exhausted', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 1000 },
      maxParticles: 5,
      startLifetime: 10,
    });
    for (let i = 1; i <= 20; i++) step(i * 16);
    expect(countActiveParticles(ps)).toBe(5);
    ps.dispose();
  });
});

describe('getActiveParticleCount', () => {
  test('matches a manual scan of the isActive attribute', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    for (let i = 1; i <= 5; i++) step(i * 16);
    expect(ps.getActiveParticleCount).toBeDefined();
    expect(ps.getActiveParticleCount!()).toBe(countActiveParticles(ps));
    expect(ps.getActiveParticleCount!()).toBeGreaterThan(0);
    ps.dispose();
  });

  test('returns 0 before any emission', () => {
    const { ps } = createTestSystem();
    expect(ps.getActiveParticleCount!()).toBe(0);
    ps.dispose();
  });
});

describe('updateConfig live module activation (CPU path)', () => {
  test('activating velocityOverLifetime after creation moves particles', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);
    const points = ps.instance as THREE.Points;
    const posAttr = points.geometry.attributes.position;
    const yBefore = posAttr.getY(0);

    ps.updateConfig({
      velocityOverLifetime: {
        isActive: true,
        linear: { y: 5 },
      } as any,
    });
    step(32);
    step(48);
    expect(posAttr.getY(0)).toBeGreaterThan(yBefore);
    ps.dispose();
  });

  test('activating rotationOverLifetime after creation rotates particles', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);
    const points = ps.instance as THREE.Points;
    const rotationAttr = points.geometry.attributes.rotation;
    expect(rotationAttr.getX(0)).toBe(0);

    ps.updateConfig({
      rotationOverLifetime: { isActive: true, min: 100, max: 100 },
    });
    step(32);
    step(48);
    expect(rotationAttr.getX(0)).not.toBe(0);
    ps.dispose();
  });

  test('changing the sizeOverLifetime curve takes effect immediately', () => {
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
    const points = ps.instance as THREE.Points;
    const sizeAttr = points.geometry.attributes.size;
    expect(sizeAttr.getX(0)).toBeCloseTo(1);

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
    step(32);
    expect(sizeAttr.getX(0)).toBeCloseTo(3);
    ps.dispose();
  });
});

describe('updateConfig structural property warnings', () => {
  test('warns when a structural property is passed', () => {
    const { ps } = createTestSystem();
    // Spy installed after creation — system creation itself warns in the
    // node test environment (no document for the default texture).
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    ps.updateConfig({ maxParticles: 500 });
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("updateConfig('maxParticles')")
    );
    ps.dispose();
    warnSpy.mockRestore();
  });

  test('does not warn for supported live updates', () => {
    const { ps } = createTestSystem();
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    ps.updateConfig({ gravity: 5 });
    expect(warnSpy).not.toHaveBeenCalled();
    ps.dispose();
    warnSpy.mockRestore();
  });
});

describe('CPU buffer upload hints', () => {
  test('scalar buffer and position attribute use DynamicDrawUsage and update ranges', () => {
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    step(16);
    const points = ps.instance as THREE.Points;
    const posAttr = points.geometry.attributes
      .position as THREE.BufferAttribute;
    const scalarBuffer = (
      points.geometry.attributes.isActive as THREE.InterleavedBufferAttribute
    ).data;

    expect(posAttr.usage).toBe(THREE.DynamicDrawUsage);
    expect(scalarBuffer.usage).toBe(THREE.DynamicDrawUsage);
    expect(posAttr.updateRanges.length).toBeGreaterThan(0);
    expect(scalarBuffer.updateRanges.length).toBeGreaterThan(0);
    ps.dispose();
  });

  test('pending update ranges stay bounded without a render consuming them', () => {
    // Systems updated while hidden/culled are never drawn, so three.js never
    // clears updateRanges — the flush must keep the list at a constant size.
    const { ps, step } = createTestSystem({ emission: { rateOverTime: 100 } });
    for (let i = 1; i <= 200; i++) step(i * 16);
    const points = ps.instance as THREE.Points;
    const posAttr = points.geometry.attributes
      .position as THREE.BufferAttribute;
    const scalarBuffer = (
      points.geometry.attributes.isActive as THREE.InterleavedBufferAttribute
    ).data;

    expect(posAttr.updateRanges.length).toBe(1);
    expect(scalarBuffer.updateRanges.length).toBe(1);
    // The single range must cover every active particle's slice.
    const activeCount = ps.getActiveParticleCount!();
    expect(posAttr.updateRanges[0].count).toBeGreaterThanOrEqual(
      activeCount * 3
    );
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

      attributes: attributes as any,
      scalarArray: new Float32Array(SCALAR_STRIDE),
      particleLifetimePercentage: 0.5,
      particleIndex: 0,
      updateFlags,
    });

    expect(updateFlags.position).toBe(true);

    expect((attributes.position as any).needsUpdate).toBe(false);
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
      } as any,
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
  };

  test('skips registration and warns for a non-compute-capable renderer', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    // Duck-typing: a plain WebGLRenderer-like object has no compute()/hasFeature()
    const result = registerTSLMaterialFactory(mockFactory, {
      renderer: { render: () => {}, getSize: () => {} },
    });
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('does not support compute dispatches')
    );
    // Factory was not registered — systems keep using the GLSL path.
    const ps = createParticleSystem({ maxParticles: 10 });
    expect(
      (ps.instance as THREE.Points).material instanceof THREE.ShaderMaterial
    ).toBe(true);
    ps.dispose();
    warnSpy.mockRestore();
  });

  // NOTE: keep this test last in the file — registering the factory is a
  // module-global side effect with no unregister API.
  test('registers for a compute-capable renderer (duck-typed)', () => {
    const computeCapableRenderer = {
      compute: () => {},
      hasFeature: () => true,
    };
    const result = registerTSLMaterialFactory(mockFactory, {
      renderer: computeCapableRenderer,
    });
    expect(result).toBe(true);
    const ps = createParticleSystem({ maxParticles: 10 });
    expect(
      (ps.instance as THREE.Points).material instanceof THREE.MeshBasicMaterial
    ).toBe(true);
    ps.dispose();
  });
});
