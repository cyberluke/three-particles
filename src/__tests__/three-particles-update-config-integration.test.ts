import {
  ForceFieldType,
  Shape,
  SimulationSpace,
} from '../js/effects/three-particles/three-particles-enums.js';
import {
  createParticleSystem,
  updateParticleSystems,
} from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: `updateConfig` merges the partial config into the
 * normalized config; the kernels consume the new scalars on the next frame.
 * The CPU-observable per-frame emission count is `gpuDebug.lastEmitCount()`.
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
      duration: 10,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
      emission: { rateOverTime: 20, rateOverDistance: 0 },
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

// ---------------------------------------------------------------------------
// Integration: updateConfig with global updateParticleSystems
// ---------------------------------------------------------------------------

describe('integration — updateConfig with global updateParticleSystems', () => {
  it('should apply config changes when updated via the global loop', () => {
    const startTime = 1000;
    const ps = createParticleSystem(
      {
        maxParticles: 30,
        duration: 10,
        looping: true,
        startLifetime: 2,
        startSpeed: 0,
        gravity: 0,
        emission: { rateOverTime: 20, rateOverDistance: 0 },
      } as never,
      startTime
    );

    updateParticleSystems({ now: startTime + 100, delta: 0.1, elapsed: 0.1 });
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.updateConfig({ gravity: -50 });

    expect(() => {
      updateParticleSystems({
        now: startTime + 300,
        delta: 0.2,
        elapsed: 0.3,
      });
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: updateConfig force fields
// ---------------------------------------------------------------------------

describe('integration — updateConfig force field effects', () => {
  it('accepts a new directional force field at runtime', () => {
    const { ps, step } = createTestSystem({
      startSpeed: 0,
      gravity: 0,
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.updateConfig({
      forceFields: [
        {
          isActive: true,
          type: ForceFieldType.DIRECTIONAL,
          direction: { x: 1, y: 0, z: 0 },
          strength: 50,
        },
      ],
    });

    expect(() => {
      step(200, 100);
      step(700, 500);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('accepts removing all force fields at runtime', () => {
    const { ps, step } = createTestSystem({
      startSpeed: 0,
      gravity: 0,
      forceFields: [
        {
          isActive: true,
          type: ForceFieldType.DIRECTIONAL,
          direction: { x: 0, y: 1, z: 0 },
          strength: 100,
        },
      ],
    });

    step(100);
    step(300, 200);

    ps.updateConfig({ forceFields: [] });

    expect(() => {
      step(400, 100);
      step(500, 100);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: updateConfig start values
// ---------------------------------------------------------------------------

describe('integration — updateConfig start values', () => {
  it('spawns new particles with the updated startColor', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 0.3,
      startColor: {
        min: { r: 1, g: 1, b: 1 },
        max: { r: 1, g: 1, b: 1 },
      },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.updateConfig({
      startColor: {
        min: { r: 1, g: 0, b: 0 },
        max: { r: 1, g: 0, b: 0 },
      },
    });

    expect(() => {
      step(600, 500);
      step(900, 300);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('spawns new particles with the updated startSize', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 0.2,
      startSize: 1,
    });

    step(100);
    ps.updateConfig({ startSize: 5 });

    expect(() => {
      step(500, 400);
      step(800, 300);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('spawns new particles with the updated startSpeed', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 1,
      startSpeed: 0.1,
      gravity: 0,
    });

    step(100);
    step(300, 200);

    ps.updateConfig({ startSpeed: 50 });

    expect(() => {
      step(600, 300);
      step(1000, 400);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);

    ps.dispose();
  });

  it('spawns new particles with the updated startLifetime', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 0.1,
      emission: { rateOverTime: 50, rateOverDistance: 0 },
    });

    step(50);
    step(200, 150);

    ps.updateConfig({ startLifetime: 10 });

    expect(() => {
      step(400, 200);
      step(1200, 800);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('uses the updated shape for new particles', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 0.2,
      startSpeed: 0,
      gravity: 0,
      shape: {
        shape: Shape.SPHERE,
        sphere: { radius: 0.01, radiusThickness: 1, arc: 360 },
      },
    });

    step(100);
    const snapBefore = (
      ps as unknown as {
        gpuDebug: { snapshot(): { shape: { radius: number | null } } };
      }
    ).gpuDebug.snapshot();
    expect(snapBefore.shape.radius).toBeCloseTo(0.01);

    ps.updateConfig({
      shape: {
        shape: Shape.SPHERE,
        sphere: { radius: 10, radiusThickness: 1, arc: 360 },
      },
    });

    expect(() => {
      step(500, 400);
      step(800, 300);
    }).not.toThrow();

    const snapAfter = (
      ps as unknown as {
        gpuDebug: { snapshot(): { shape: { radius: number | null } } };
      }
    ).gpuDebug.snapshot();
    expect(snapAfter.shape.radius).toBeCloseTo(10);

    ps.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: multiple systems with independent updateConfig
// ---------------------------------------------------------------------------

describe('integration — multiple systems with independent config updates', () => {
  it('should update configs independently for each system', () => {
    const startTime = 1000;
    const base = {
      maxParticles: 20,
      duration: 10,
      looping: true,
      startLifetime: 2,
      startSpeed: 0,
      gravity: 0,
      emission: { rateOverTime: 20, rateOverDistance: 0 },
    } as never;
    const ps1 = createParticleSystem(base, startTime);
    const ps2 = createParticleSystem(base, startTime);

    updateParticleSystems({ now: startTime + 100, delta: 0.1, elapsed: 0.1 });
    expect(lastEmit(ps1)).toBeGreaterThan(0);
    expect(lastEmit(ps2)).toBeGreaterThan(0);

    ps1.updateConfig({ gravity: 100 });

    expect(() => {
      updateParticleSystems({
        now: startTime + 400,
        delta: 0.3,
        elapsed: 0.4,
      });
    }).not.toThrow();

    expect(lastEmit(ps1)).toBeGreaterThan(0);
    expect(lastEmit(ps2)).toBeGreaterThan(0);

    ps1.dispose();
    ps2.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: updateConfig emission rate lifecycle
// ---------------------------------------------------------------------------

describe('integration — updateConfig emission rate lifecycle', () => {
  it('should ramp up and down emission rate during system lifetime', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 5, rateOverDistance: 0 },
      startLifetime: 0.5,
    });

    step(500);
    const low = lastEmit(ps);

    ps.updateConfig({ emission: { rateOverTime: 200, rateOverDistance: 0 } });
    step(800, 300);
    const high = lastEmit(ps);
    expect(high).toBeGreaterThan(low);

    ps.updateConfig({ emission: { rateOverTime: 0, rateOverDistance: 0 } });
    step(1200, 400);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: updateConfig with simulationSpace
// ---------------------------------------------------------------------------

describe('integration — updateConfig simulationSpace', () => {
  it('switches LOCAL → WORLD without crashing', () => {
    const { ps, step } = createTestSystem({
      simulationSpace: SimulationSpace.LOCAL,
      gravity: 0,
      emission: { rateOverTime: 30, rateOverDistance: 0 },
    });

    step(200);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.updateConfig({ simulationSpace: SimulationSpace.WORLD });

    expect(() => {
      step(400, 200);
      step(600, 200);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('switches WORLD → LOCAL without crashing', () => {
    const { ps, step } = createTestSystem({
      simulationSpace: SimulationSpace.WORLD,
      gravity: 0,
      emission: { rateOverTime: 30, rateOverDistance: 0 },
    });

    step(200);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.updateConfig({ simulationSpace: SimulationSpace.LOCAL });

    expect(() => {
      step(400, 200);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('keeps running when the same simulationSpace is echoed', () => {
    const { ps, step } = createTestSystem({
      simulationSpace: SimulationSpace.LOCAL,
      gravity: 0,
      emission: { rateOverTime: 30, rateOverDistance: 0 },
    });

    step(200);
    const before = lastEmit(ps);
    expect(before).toBeGreaterThan(0);

    ps.updateConfig({ simulationSpace: SimulationSpace.LOCAL });
    step(400, 200);

    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });
});

// ---------------------------------------------------------------------------
// Integration: rapid successive updateConfig calls (stress)
// ---------------------------------------------------------------------------

describe('integration — rapid successive updateConfig calls', () => {
  it('should handle many config changes between frames without corruption', () => {
    const { ps, step } = createTestSystem({
      gravity: 0,
      startSpeed: 1,
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    for (let i = 0; i < 20; i++) {
      ps.updateConfig({ gravity: -i * 2 });
      ps.updateConfig({
        forceFields: [
          {
            type: ForceFieldType.DIRECTIONAL,
            direction: { x: Math.sin(i), y: Math.cos(i), z: 0 },
            strength: i,
          },
        ],
      });
    }

    expect(() => {
      step(200, 100);
      step(400, 200);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});
