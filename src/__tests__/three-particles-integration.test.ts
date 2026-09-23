import * as THREE from 'three';
import {
  Shape,
  SimulationSpace,
} from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import {
  serializeParticleSystem,
  deserializeParticleSystem,
} from '../js/effects/three-particles/three-particles-serialization.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) integration tests: creation, per-frame scalar emission,
 * shape decoding, serialization round-trip and disposal.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const snapshotOf = (ps: ParticleSystem) =>
  (
    ps as unknown as {
      gpuDebug: {
        snapshot(): {
          shape: { gpuShapeKind: number; radius: number | null };
          simulationSpace: string;
          maxParticles: number;
          forceFieldCount: number;
          subEmitterCount: number;
        };
      };
    }
  ).gpuDebug.snapshot();

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

describe('integration — emission lifecycle', () => {
  it('should emit across multiple frames with the accumulator', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 100 },
      maxParticles: 100,
    });

    step(16);
    expect(lastEmit(ps)).toBe(1);
    step(100, 84);
    expect(lastEmit(ps)).toBe(9); // 0.6 + 8.4 = 9.0

    ps.dispose();
  });

  it('should respect maxParticles limit during continuous emission', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 1000 },
      maxParticles: 10,
    });

    step(100);
    expect(lastEmit(ps)).toBe(10);

    ps.dispose();
  });

  it('should apply gravity, force fields and size modifier simultaneously', () => {
    const { ps, step } = createTestSystem({
      gravity: -9.8,
      forceFields: [
        {
          type: 'DIRECTIONAL',
          direction: { x: 1, y: 0, z: 0 },
          strength: 2,
        },
      ],
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
    });

    expect(() => {
      step(100);
      step(500, 400);
    }).not.toThrow();
    expect(snapshotOf(ps).forceFieldCount).toBe(1);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

describe('integration — serialization', () => {
  it('should preserve a complex config through serialize → deserialize', () => {
    const config = {
      maxParticles: 30,
      duration: 3,
      looping: false,
      startLifetime: { min: 0.5, max: 1.5 },
      startSpeed: 2,
      startSize: { min: 0.2, max: 0.8 },
      gravity: -4,
      simulationSpace: SimulationSpace.WORLD,
      shape: {
        shape: Shape.CONE,
        cone: { radius: 1.5, angle: 30, radiusThickness: 0.5, arc: 180 },
      },
      emission: { rateOverTime: 25, bursts: [{ time: 0, count: 4 }] },
    };

    const json = serializeParticleSystem(config as never);
    const restored = deserializeParticleSystem(json);

    expect(restored.maxParticles).toBe(30);
    expect(restored.duration).toBe(3);
    expect(restored.looping).toBe(false);
    expect(restored.gravity).toBe(-4);
    expect(restored.simulationSpace).toBe(SimulationSpace.WORLD);
    expect(
      (restored.shape as { cone: { angle: number } }).cone.angle
    ).toBe(30);

    const { ps } = createTestSystem(restored as Record<string, unknown>);
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(1);
    expect(snapshotOf(ps).simulationSpace).toBe(SimulationSpace.WORLD);
    ps.dispose();
  });

  it('should produce a working particle system from deserialized config', () => {
    const json = serializeParticleSystem({
      maxParticles: 10,
      duration: 2,
      looping: true,
      emission: { rateOverTime: 100 },
    } as never);
    const restored = deserializeParticleSystem(json);

    const { ps, step } = createTestSystem(
      restored as Record<string, unknown>,
      1000
    );
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });
});

describe('integration — shapes', () => {
  const cases: Array<[Shape, number]> = [
    [Shape.SPHERE, 0],
    [Shape.CONE, 1],
    [Shape.CIRCLE, 2],
    [Shape.RECTANGLE, 3],
    [Shape.BOX, 4],
  ];

  it.each(cases)('should emit from %s shape', (shape, kind) => {
    const { ps, step } = createTestSystem({ shape: { shape } });
    step(100);
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(kind);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });
});

describe('integration — lifecycle and space', () => {
  it('should stop emitting after duration ends in non-looping mode', () => {
    const { ps, step } = createTestSystem({
      looping: false,
      duration: 1,
      emission: { rateOverTime: 100 },
    });

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    step(1500, 1400);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });

  it('should stop emitting when paused and resume when unpaused', () => {
    const { ps, step } = createTestSystem();

    ps.pauseEmitter();
    step(100);
    expect(lastEmit(ps)).toBe(0);

    ps.resumeEmitter();
    step(200, 100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should use different instance types for WORLD vs LOCAL space', () => {
    const local = createTestSystem({});
    const world = createTestSystem({
      simulationSpace: SimulationSpace.WORLD,
    });

    expect(local.ps.instance).toBeInstanceOf(THREE.Points);
    expect(world.ps.instance).toBeInstanceOf(THREE.Points);
    expect(world.ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(local.ps.instance.matrixWorldAutoUpdate).not.toBe(false);

    local.ps.dispose();
    world.ps.dispose();
  });
});
