import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: burst timing is evaluated on the CPU into the
 * per-frame scalar `emitCount` (clamped to `maxParticles`), observable via
 * `gpuDebug.lastEmitCount()`.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createBurstSystem = (
  bursts: Array<Record<string, unknown>>,
  config: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 50,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 0,
      emission: { rateOverTime: 0, rateOverDistance: 0, bursts },
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

describe('Burst Emission', () => {
  it('should emit particles at burst time', () => {
    const { ps, step } = createBurstSystem([{ time: 0, count: 5 }]);
    step(16);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });

  it('should emit at time 0 on the first frame', () => {
    const { ps, step } = createBurstSystem([{ time: 0, count: 3 }]);
    step(0);
    expect(lastEmit(ps)).toBe(3);
    ps.dispose();
  });

  it('should handle multiple bursts at different times', () => {
    const { ps, step } = createBurstSystem([
      { time: 0, count: 2 },
      { time: 0.1, count: 3 },
    ]);

    step(16); // t = 16ms: first burst only
    expect(lastEmit(ps)).toBe(2);

    step(116, 100); // t = 116ms: second burst fires
    expect(lastEmit(ps)).toBe(3);

    ps.dispose();
  });

  it('should support multi-cycle bursts with interval', () => {
    const { ps, step } = createBurstSystem([
      { time: 0, count: 2, cycles: 3, interval: 0.1 },
    ]);

    step(16);
    expect(lastEmit(ps)).toBe(2);
    step(116, 100);
    expect(lastEmit(ps)).toBe(2);
    step(216, 100);
    expect(lastEmit(ps)).toBe(2);
    // cycles exhausted (3 of 3 fired) — nothing left to emit.
    step(316, 100);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });

  it('should not exceed maxParticles', () => {
    const { ps, step } = createBurstSystem([{ time: 0, count: 100 }], {
      maxParticles: 10,
    });
    step(16);
    expect(lastEmit(ps)).toBe(10);
    ps.dispose();
  });

  it('should not emit when probability is 0', () => {
    const { ps, step } = createBurstSystem([
      { time: 0, count: 5, probability: 0 },
    ]);
    step(16);
    expect(lastEmit(ps)).toBe(0);
    ps.dispose();
  });

  it('should always emit when probability is 1', () => {
    const { ps, step } = createBurstSystem([
      { time: 0, count: 5, probability: 1 },
    ]);
    step(16);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });

  it('should handle random count range', () => {
    const { ps, step } = createBurstSystem([
      { time: 0, count: { min: 4, max: 4 } },
    ]);
    step(16);
    expect(lastEmit(ps)).toBe(4);
    ps.dispose();
  });

  it('should not emit burst before its time', () => {
    const { ps, step } = createBurstSystem([{ time: 1, count: 5 }]);
    step(16);
    expect(lastEmit(ps)).toBe(0);
    step(1016, 1000);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });

  it('should reset burst states on loop', () => {
    const { ps, step } = createBurstSystem([{ time: 0.5, count: 2 }], {
      looping: true,
      duration: 1,
    });

    step(516); // first cycle: burst fires
    expect(lastEmit(ps)).toBe(2);

    step(1016, 500); // new loop, t = 16ms < 0.5s: reset, no fire
    expect(lastEmit(ps)).toBe(0);

    step(1516, 500); // second cycle: burst fires again
    expect(lastEmit(ps)).toBe(2);

    ps.dispose();
  });

  it('should not emit bursts after duration in non-looping mode', () => {
    const { ps, step } = createBurstSystem([{ time: 0, count: 2 }], {
      looping: false,
      duration: 1,
    });

    step(16);
    expect(lastEmit(ps)).toBe(2);

    step(1500, 1484);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });

  it('should handle burst with cycles=1 (default behavior)', () => {
    const { ps, step } = createBurstSystem([{ time: 0, count: 2 }]);

    step(16);
    expect(lastEmit(ps)).toBe(2);

    step(116, 100);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });
});
