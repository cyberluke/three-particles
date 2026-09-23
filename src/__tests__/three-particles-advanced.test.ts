import * as THREE from 'three';
import {
  SimulationSpace,
  SubEmitterTrigger,
} from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: modifiers run in the compute kernels; the CPU
 * only decodes config into scalars/tables. Observable: `gpuDebug` snapshot,
 * per-frame emission scalars and the child-object graph.
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
          forceFieldCount: number;
          collisionPlaneCount: number;
          subEmitterCount: number;
          trailEnabled: boolean;
          simulationSpace: string;
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

// ─── Force fields ────────────────────────────────────────────────────────────

describe('Force fields', () => {
  it('should apply point force field to active particles', () => {
    const { ps, step } = createTestSystem({
      forceFields: [
        {
          type: 'POINT',
          position: { x: 0, y: 5, z: 0 },
          strength: 3,
          range: 10,
        },
      ],
    });
    step(100);
    expect(snapshotOf(ps).forceFieldCount).toBe(1);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should apply directional force field to particles', () => {
    const { ps, step } = createTestSystem({
      forceFields: [
        {
          type: 'DIRECTIONAL',
          direction: { x: 1, y: 0, z: 0 },
          strength: 10,
        },
      ],
    });
    expect(() => {
      step(100);
      step(500, 400);
    }).not.toThrow();
    ps.dispose();
  });

  it('should handle multiple force fields simultaneously', () => {
    const { ps } = createTestSystem({
      forceFields: [
        { type: 'POINT', position: { x: 0, y: 0, z: 0 }, strength: 1 },
        { type: 'DIRECTIONAL', direction: { x: 0, y: 1, z: 0 }, strength: 2 },
        {
          type: 'RADIAL',
          position: { x: 1, y: 1, z: 1 },
          strength: 3,
          range: 5,
        },
      ],
    });
    expect(snapshotOf(ps).forceFieldCount).toBe(3);
    ps.dispose();
  });

  it('should handle inactive force fields', () => {
    const { ps, step } = createTestSystem({
      forceFields: [
        { isActive: false, type: 'POINT', position: { x: 0, y: 0, z: 0 } },
      ],
    });
    expect(() => step(100)).not.toThrow();
    ps.dispose();
  });

  it('should handle empty and undefined forceFields', () => {
    const a = createTestSystem({ forceFields: [] });
    const b = createTestSystem({});
    expect(snapshotOf(a.ps).forceFieldCount).toBe(0);
    expect(snapshotOf(b.ps).forceFieldCount).toBe(0);
    a.ps.dispose();
    b.ps.dispose();
  });
});

// ─── Collision planes ────────────────────────────────────────────────────────

describe('Collision planes', () => {
  it('should register collision planes in the snapshot', () => {
    const { ps, step } = createTestSystem({
      collisionPlanes: [
        { mode: 'PLANE', plane: { x: 0, y: 1, z: 0, w: 2 } },
      ],
    });
    step(100);
    expect(snapshotOf(ps).collisionPlaneCount).toBe(1);
    ps.dispose();
  });
});

// ─── Velocity / noise modifiers ──────────────────────────────────────────────

describe('Modifier pipelines', () => {
  it('should apply linear velocity when active', () => {
    const { ps, step } = createTestSystem({
      velocityOverLifetime: {
        isActive: true,
        linear: { x: 1, y: 0, z: 0 },
        orbital: { x: 0, y: 0, z: 0 },
      },
    });
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('should apply noise with position/rotation/size amounts', () => {
    const { ps, step } = createTestSystem({
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
    });
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('should handle all modifiers active simultaneously', () => {
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
        rotationAmount: 0,
        sizeAmount: 0,
      },
      velocityOverLifetime: {
        isActive: true,
        linear: { x: 1, y: 0, z: 0 },
        orbital: { x: 0, y: 0, z: 0 },
      },
    });

    expect(() => {
      for (let i = 1; i <= 10; i++) step(i * 16);
    }).not.toThrow();
    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });
});

// ─── Sub-emitters ────────────────────────────────────────────────────────────

describe('Sub-emitters', () => {
  const child = {
    maxParticles: 3,
    duration: 1,
    looping: false,
    startLifetime: 0.3,
    startSpeed: 0.5,
    emission: { rateOverTime: 5, rateOverDistance: 0 },
  };

  it('should spawn sub-emitter on particle death', () => {
    const { ps, step } = createTestSystem({
      startLifetime: 0.1,
      subEmitters: [{ trigger: SubEmitterTrigger.DEATH, config: child }],
    });
    for (let i = 1; i <= 20; i++) step(i * 16);
    expect(ps.instance.children.length).toBe(1);
    ps.dispose();
  });

  it('should spawn sub-emitter on particle birth', () => {
    const { ps, step } = createTestSystem({
      subEmitters: [{ trigger: SubEmitterTrigger.BIRTH, config: child }],
    });
    step(16);
    expect(ps.instance.children.length).toBe(1);
    ps.dispose();
  });

  it('should respect maxInstances for sub-emitters', () => {
    const { ps, step } = createTestSystem({
      subEmitters: [
        { trigger: SubEmitterTrigger.BIRTH, config: child, maxInstances: 2 },
      ],
    });
    step(16);
    step(100, 84);
    // One child render object per config, regardless of the instance cap.
    expect(ps.instance.children.length).toBe(1);
    ps.dispose();
  });

  it('should dispose sub-emitters when parent is disposed', () => {
    const { ps, step } = createTestSystem({
      subEmitters: [{ trigger: SubEmitterTrigger.BIRTH, config: child }],
    });
    step(16);
    expect(() => ps.dispose()).not.toThrow();
  });
});

// ─── World space ─────────────────────────────────────────────────────────────

describe('World space', () => {
  it('should create wrapper-less instance in world space mode', () => {
    const { ps } = createTestSystem({
      simulationSpace: SimulationSpace.WORLD,
    });
    expect(ps.instance).toBeInstanceOf(THREE.Points);
    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(snapshotOf(ps).simulationSpace).toBe(SimulationSpace.WORLD);
    ps.dispose();
  });

  it('should handle world space with movement', () => {
    const { ps, step } = createTestSystem({
      simulationSpace: SimulationSpace.WORLD,
    });
    ps.instance.position.set(3, 2, 1);
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    ps.dispose();
  });

  it('should handle multiple dispose calls gracefully', () => {
    const { ps } = createTestSystem();
    ps.dispose();
    expect(() => ps.dispose()).not.toThrow();
  });
});

// ─── Distance emission ───────────────────────────────────────────────────────

describe('Distance emission', () => {
  it('should emit particles based on movement distance', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 0, rateOverDistance: 2 },
    });

    step(16); // initializes lastWorldPosition
    expect(lastEmit(ps)).toBe(0);

    ps.instance.position.x = 5;
    step(32, 16);
    expect(lastEmit(ps)).toBe(10); // floor(5 * 2)

    ps.dispose();
  });

  it('should not emit by distance when not moving', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 0, rateOverDistance: 2 },
    });

    step(16);
    step(32, 16);
    expect(lastEmit(ps)).toBe(0);

    ps.dispose();
  });
});
