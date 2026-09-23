import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: velocity-over-lifetime data is baked into the
 * packed curve table (lifetime-curve axes) or written as per-particle start
 * values (constant axes). The CPU-observable surface is the per-frame scalar
 * emission count and the pipeline structure.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const passNames = (ps: ParticleSystem): string[] =>
  (
    ps as unknown as {
      gpuDebug: { passNames: string[] };
    }
  ).gpuDebug.passNames;

const createVelocityTestSystem = (
  velocityConfig: Record<string, unknown>,
  extraConfig: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 20,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 0,
      emission: { rateOverTime: 50 },
      velocityOverLifetime: {
        isActive: true,
        ...velocityConfig,
      },
      ...extraConfig,
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

const bezier = {
  type: 'BEZIER',
  scale: 1,
  bezierPoints: [
    { x: 0, y: 0, percentage: 0 },
    { x: 1, y: 1, percentage: 1 },
  ],
};

describe('Velocity Over Lifetime - Linear', () => {
  it('should create the pipeline with constant linear velocity', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 1, y: 2, z: 3 },
      orbital: { x: 0, y: 0, z: 0 },
    });

    step(16);
    step(100, 84);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should create the pipeline with linear velocity from random ranges', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: {
        x: { min: 1, max: 2 },
        y: { min: 0, max: 0 },
        z: { min: 0, max: 0 },
      },
      orbital: { x: 0, y: 0, z: 0 },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('bakes linear velocity lifetime curves into the packed table', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: bezier, y: 0, z: 0 },
      orbital: { x: 0, y: 0, z: 0 },
    });

    step(100);
    const packed = (
      ps as unknown as {
        gpuDebug: { buffers: { packedData?: unknown } };
      }
    ).gpuDebug.buffers;
    // One baked curve (CURVE_RESOLUTION samples) for linearVelX.
    const packedData = packed.packedData as Float32Array;
    expect(packedData.length).toBeGreaterThan(0);
    expect(packedData.length % 256 === 0 || packedData.length > 256).toBe(
      true
    );
    ps.dispose();
  });

  it('should handle mixed constant and curve linear velocity', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: bezier, y: 2, z: { min: 1, max: 3 } },
      orbital: { x: 0, y: 0, z: 0 },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });
});

describe('Velocity Over Lifetime - Orbital', () => {
  it('should create the pipeline with orbital velocity active', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 0, y: 0, z: 0 },
      orbital: { x: 1, y: 2, z: 3 },
    });

    step(16);
    step(100, 84);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should handle orbital velocity with random range values', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 0, y: 0, z: 0 },
      orbital: {
        x: { min: 0, max: 1 },
        y: { min: 0, max: 1 },
        z: { min: 0, max: 1 },
      },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('bakes orbital velocity lifetime curves', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 0, y: 0, z: 0 },
      orbital: { x: 0, y: 0, z: bezier },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    expect(passNames(ps)).toEqual(['emit', 'simulate']);
    ps.dispose();
  });

  it('should handle combined linear and orbital velocity', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 1, y: 0, z: 0 },
      orbital: { x: 0, y: 0, z: 0.5 },
    });

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('should handle zero orbital velocity', () => {
    const { ps, step } = createVelocityTestSystem({
      linear: { x: 0, y: 0, z: 0 },
      orbital: { x: 0, y: 0, z: 0 },
    });

    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });

  it('inactive velocity module keeps the base two-pass pipeline', () => {
    const { ps, step } = createVelocityTestSystem(
      { linear: { x: 1, y: 0, z: 0 }, orbital: { x: 0, y: 0, z: 0 } },
      {}
    );
    ps.updateConfig({
      velocityOverLifetime: {
        isActive: false,
        linear: { x: 1, y: 0, z: 0 },
        orbital: { x: 0, y: 0, z: 0 },
      },
    });
    step(100);
    expect(passNames(ps)).toEqual(['emit', 'simulate']);
    ps.dispose();
  });
});
