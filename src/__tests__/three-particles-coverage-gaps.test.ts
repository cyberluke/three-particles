import * as THREE from 'three';
import { SubEmitterTrigger } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) coverage gaps: the scalar-only update path, the FIFO event
 * graph and the child render-object lifecycle.
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
      maxParticles: 10,
      duration: 5,
      looping: true,
      startLifetime: 0.2,
      startSpeed: 1,
      emission: { rateOverTime: 50, rateOverDistance: 0 },
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

// ─── Sub-emitter child graph ─────────────────────────────────────────────────

describe('sub-emitter cleanup — child render objects', () => {
  it('should dispose sub-emitter instance when parent is disposed', () => {
    const scene = new THREE.Group();
    const { ps, step } = createTestSystem({
      subEmitters: [
        {
          trigger: SubEmitterTrigger.BIRTH,
          config: {
            maxParticles: 3,
            duration: 1,
            looping: false,
            startLifetime: 0.3,
            startSpeed: 0,
            emission: { rateOverTime: 5 },
          },
        },
      ],
    });

    scene.add(ps.instance);
    step(16);
    expect(ps.instance.children.length).toBe(1);

    ps.dispose();
    expect(scene.children.length).toBe(0);
  });

  it('should correctly read the child pool contract attributes', () => {
    const { ps, step } = createTestSystem({
      subEmitters: [
        {
          trigger: SubEmitterTrigger.DEATH,
          config: {
            maxParticles: 3,
            duration: 1,
            looping: false,
            startLifetime: 0.3,
            startSpeed: 0,
            emission: { rateOverTime: 5 },
          },
        },
      ],
    });

    for (let i = 1; i <= 10; i++) step(i * 16);

    const child = ps.instance.children[0] as THREE.Points;
    expect(child).toBeInstanceOf(THREE.Points);
    expect(child.geometry.attributes.particleState).toBeDefined();
    expect(child.geometry.attributes.startValues).toBeDefined();

    ps.dispose();
  });

  it('should not throw when parent particle system is not added to a scene', () => {
    const { ps, step } = createTestSystem({});
    expect(() => {
      step(16);
      ps.dispose();
    }).not.toThrow();
  });
});

// ─── Scalar uniform writes ───────────────────────────────────────────────────

describe('scalar-only per-frame path', () => {
  it('advances the emission accumulator across frames', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      maxParticles: 100,
    });

    step(16);
    expect(lastEmit(ps)).toBe(1);
    step(32, 16);
    expect(lastEmit(ps)).toBe(2); // 0.6 + 1.6 = 2.2 → 2
    step(48, 16);
    expect(lastEmit(ps)).toBe(1); // 0.2 + 1.6 = 1.8 → 1

    ps.dispose();
  });

  it('keeps the per-frame dispatch within maxParticles', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 1000 },
      maxParticles: 5,
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

// ─── Start values decoding ───────────────────────────────────────────────────

describe('start values decoding', () => {
  it('accepts min/max pairs for startSpeed', () => {
    const { ps, step } = createTestSystem({
      startSpeed: { min: 1, max: 3 },
    });
    expect(() => step(16)).not.toThrow();
    ps.dispose();
  });

  it('accepts undefined startSpeed fallback', () => {
    const { ps, step } = createTestSystem({ startSpeed: undefined });
    expect(() => step(16)).not.toThrow();
    ps.dispose();
  });

  it('accepts inheritVelocity with min/max startSpeed', () => {
    const { ps, step } = createTestSystem({
      startSpeed: { min: 1, max: 2 },
      subEmitters: [
        {
          trigger: SubEmitterTrigger.DEATH,
          config: {
            maxParticles: 3,
            startSpeed: { min: 0.5, max: 1.5 },
            emission: { rateOverTime: 2 },
          },
          inheritVelocity: 0.5,
        },
      ],
    });

    expect(() => {
      for (let i = 1; i <= 10; i++) step(i * 16);
    }).not.toThrow();
    expect(ps.instance.children.length).toBe(1);
    ps.dispose();
  });
});
