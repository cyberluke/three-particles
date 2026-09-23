import * as THREE from 'three';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: rate-over-distance emission is a CPU scalar
 * computed from the world-position delta and published through
 * `gpuDebug.lastEmitCount()`.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createDistanceTestSystem = (
  rateOverDistance: number,
  options: { maxParticles?: number; duration?: number; looping?: boolean } = {}
) => {
  const startTime = 1000;
  const ps = createParticleSystem(
    {
      emission: {
        rateOverTime: 0,
        rateOverDistance,
      },
      maxParticles: options.maxParticles ?? 200,
      duration: options.duration ?? 10,
      looping: options.looping ?? true,
      startLifetime: 10,
      startSpeed: 0,
    },
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

describe('Rate Over Distance - Pause/Resume', () => {
  it('should emit particles when moving with rateOverDistance', () => {
    const { ps, step } = createDistanceTestSystem(1);
    const instance = ps.instance as THREE.Points;

    // First frame initializes lastWorldPosition (no accumulation).
    step(16);
    expect(lastEmit(ps)).toBe(0);

    // Move 10 units → floor(10 * 1) = 10 particles requested.
    instance.position.x = 10;
    step(32);
    expect(lastEmit(ps)).toBe(10);

    ps.dispose();
  });

  it('accumulates world distance while paused and emits it on resume', () => {
    const { ps, step } = createDistanceTestSystem(1);
    const instance = ps.instance as THREE.Points;

    step(16);

    instance.position.x = 2;
    step(32);
    expect(lastEmit(ps)).toBe(2);

    // Pause: no per-frame emit, but the world-position delta keeps
    // accumulating into the distance budget.
    ps.pauseEmitter();
    instance.position.x = 1000;
    step(48);
    expect(lastEmit(ps)).toBe(0);

    // Resume: the full accumulated distance (998 + 1) is consumed at once,
    // clamped to the pool size.
    ps.resumeEmitter();
    instance.position.x = 1001;
    step(64);
    expect(lastEmit(ps)).toBe(200);

    ps.dispose();
  });

  it('clamps the resumed distance dispatch to maxParticles', () => {
    const { ps, step } = createDistanceTestSystem(10);
    const instance = ps.instance as THREE.Points;

    step(16);

    ps.pauseEmitter();
    instance.position.x = 100;
    step(32);
    instance.position.x = 200;
    step(48);
    expect(lastEmit(ps)).toBe(0);

    // 201 units * rate 10 = 2010 → clamped to the 200-slot pool.
    ps.resumeEmitter();
    instance.position.x = 201;
    step(64);
    expect(lastEmit(ps)).toBe(200);

    ps.dispose();
  });

  it('resumes normal distance emission after unpause', () => {
    const { ps, step } = createDistanceTestSystem(1, { maxParticles: 600 });
    const instance = ps.instance as THREE.Points;

    step(16);

    ps.pauseEmitter();
    instance.position.x = 500;
    step(32);

    // Resume and move 5 units: total 505 units → 505 particles.
    ps.resumeEmitter();
    instance.position.x = 505;
    step(48);
    expect(lastEmit(ps)).toBe(505);

    ps.dispose();
  });
});
