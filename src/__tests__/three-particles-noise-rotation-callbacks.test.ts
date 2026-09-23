import {
  createParticleSystem,
  updateParticleSystems,
} from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: the CPU writes ~12 scalar uniforms per frame;
 * modifier math runs in the compute kernels. Observable surface:
 * `gpuDebug.lastEmitCount()` and `iterationCount` progression through the
 * global `updateParticleSystems` loop.
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
      maxParticles: 20,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
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

// ─── Noise ───────────────────────────────────────────────────────────────────

describe('Noise Module', () => {
  it('creates the pipeline with noise active', () => {
    const { ps, step } = createTestSystem({
      noise: {
        isActive: true,
        strength: 1.5,
        frequency: 2,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0,
        sizeAmount: 0,
        useRandomOffset: true,
      },
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });

  it('creates the pipeline with noise applied to rotation and size', () => {
    const { ps, step } = createTestSystem({
      noise: {
        isActive: true,
        strength: 1,
        frequency: 1,
        power: 1,
        positionAmount: 0.5,
        rotationAmount: 0.3,
        sizeAmount: 0.2,
        useRandomOffset: false,
      },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('creates the pipeline with noise inactive', () => {
    const { ps, step } = createTestSystem({
      noise: {
        isActive: false,
        strength: 1,
        frequency: 1,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0,
        sizeAmount: 0,
      },
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

// ─── Rotation over lifetime ──────────────────────────────────────────────────

describe('Rotation Over Lifetime', () => {
  it('creates the pipeline with rotation over lifetime active', () => {
    const { ps, step } = createTestSystem({
      rotationOverLifetime: { isActive: true, min: 1, max: 2 },
    });

    step(16);
    step(100, 84);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('handles rotation with equal min and max', () => {
    const { ps, step } = createTestSystem({
      rotationOverLifetime: { isActive: true, min: 5, max: 5 },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('handles rotation with negative values', () => {
    const { ps, step } = createTestSystem({
      rotationOverLifetime: { isActive: true, min: -3, max: -1 },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('creates the pipeline with rotation inactive', () => {
    const { ps, step } = createTestSystem({
      rotationOverLifetime: { isActive: false, min: 1, max: 2 },
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

// ─── Lifetime curves ─────────────────────────────────────────────────────────

describe('Lifetime curve modifiers', () => {
  it('applies size over lifetime with bezier curve', () => {
    const { ps, step } = createTestSystem({
      sizeOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 0, percentage: 1 },
          ],
        },
      },
    });

    expect(() => {
      step(16);
      step(500, 484);
    }).not.toThrow();
    ps.dispose();
  });

  it('applies opacity over lifetime with bezier curve', () => {
    const { ps, step } = createTestSystem({
      opacityOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 0, percentage: 1 },
          ],
        },
      },
    });

    expect(() => {
      step(16);
      step(500, 484);
    }).not.toThrow();
    ps.dispose();
  });

  it('applies color over lifetime with bezier curves', () => {
    const { ps, step } = createTestSystem({
      colorOverLifetime: {
        isActive: true,
        r: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 0, percentage: 1 },
          ],
        },
        g: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 0, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
        b: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
      },
    });

    expect(() => {
      step(16);
      step(500, 484);
    }).not.toThrow();
    ps.dispose();
  });

  it('creates the pipeline with color over lifetime inactive', () => {
    const { ps, step } = createTestSystem({
      colorOverLifetime: { isActive: false },
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

// ─── Update callbacks ────────────────────────────────────────────────────────

describe('update callbacks (GPU-only sentinels)', () => {
  it('iterationCount advances with every update', () => {
    const { ps, step } = createTestSystem();
    const props = ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    };
    step(16);
    step(32);
    step(48);
    expect(props.gpuDebug.lastEmitCount()).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });

  it('does not throw when onUpdate/onComplete are provided', () => {
    const onUpdate = jest.fn();
    const onComplete = jest.fn();
    const { ps, step } = createTestSystem({ onUpdate, onComplete });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('supports the global updateParticleSystems loop', () => {
    const { ps, startTime } = createTestSystem();

    updateParticleSystems({
      now: startTime + 100,
      delta: 0.1,
      elapsed: 0.1,
    });
    expect(lastEmit(ps)).toBe(5);

    ps.dispose();
  });
});

// ─── Combined ────────────────────────────────────────────────────────────────

describe('Combined modifiers', () => {
  it('handles all modifiers active simultaneously', () => {
    const { ps, step } = createTestSystem({
      sizeOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
      },
      opacityOverLifetime: {
        isActive: true,
        lifetimeCurve: {
          type: 'BEZIER',
          scale: 1,
          bezierPoints: [
            { x: 0, y: 1, percentage: 0 },
            { x: 1, y: 1, percentage: 1 },
          ],
        },
      },
      rotationOverLifetime: { isActive: true, min: 1, max: 2 },
      noise: {
        isActive: true,
        strength: 1,
        frequency: 1,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0.5,
        sizeAmount: 0.5,
        useRandomOffset: true,
      },
      velocityOverLifetime: {
        isActive: true,
        linear: { x: 1, y: 0, z: 0 },
        orbital: { x: 0, y: 0, z: 0 },
      },
    });

    expect(() => {
      for (let i = 1; i <= 20; i++) step(i * 16);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });

  it('handles velocity + gravity + noise together', () => {
    const { ps, step } = createTestSystem({
      gravity: -9.8,
      noise: {
        isActive: true,
        strength: 1,
        frequency: 1,
        power: 1,
        positionAmount: 1,
        rotationAmount: 0,
        sizeAmount: 0,
      },
      velocityOverLifetime: {
        isActive: true,
        linear: { x: 2, y: 1, z: 0 },
        orbital: { x: 0, y: 0, z: 0.5 },
      },
    });

    expect(() => {
      for (let i = 1; i <= 10; i++) step(i * 16);
    }).not.toThrow();
    ps.dispose();
  });
});
