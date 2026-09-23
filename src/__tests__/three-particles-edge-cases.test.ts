import * as THREE from 'three';
import { Shape } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract edge cases: creation, scalar emission counts,
 * allocator sizing and shape decoding.
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
          shape: {
            gpuShapeKind: number;
            radius: number | null;
            radiusThickness: number | null;
            arcDeg: number | null;
            coneAngleDeg: number | null;
            rectScale: unknown;
            rectRotation: unknown;
          };
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
      maxParticles: 50,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
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

describe('Edge cases - maxParticles', () => {
  it('should handle maxParticles of 1', () => {
    const { ps } = createTestSystem({ maxParticles: 1 });
    const dbg = (
      ps as unknown as { gpuDebug: { allocatorCount: number } }
    ).gpuDebug;
    expect(dbg.allocatorCount).toBe(2);
    ps.dispose();
  });

  it('should handle large maxParticles', () => {
    const { ps } = createTestSystem({ maxParticles: 10000 });
    const geom = (ps.instance as THREE.Points).geometry;
    expect(geom.getAttribute('position').count).toBe(10000);
    ps.dispose();
  });

  it('should not emit more particles than maxParticles', () => {
    const { ps, step } = createTestSystem({
      maxParticles: 5,
      emission: { rateOverTime: 1000 },
    });
    step(100);
    expect(lastEmit(ps)).toBe(5);
    ps.dispose();
  });
});

describe('Edge cases - zero and extreme values', () => {
  it('should handle zero emission rate', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 0, rateOverDistance: 0 },
    });
    step(100);
    expect(lastEmit(ps)).toBe(0);
    ps.dispose();
  });

  it('should handle very small duration', () => {
    const { ps, step } = createTestSystem({ duration: 0.001 });
    expect(() => step(100)).not.toThrow();
    ps.dispose();
  });

  it('should handle very large duration', () => {
    const { ps, step } = createTestSystem({ duration: 100000 });
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should handle zero startSpeed', () => {
    const { ps, step } = createTestSystem({ startSpeed: 0 });
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should handle zero startSize', () => {
    const { ps, step } = createTestSystem({ startSize: 0 });
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });

  it('should handle zero startOpacity', () => {
    const { ps, step } = createTestSystem({ startOpacity: 0 });
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);
    ps.dispose();
  });
});

describe('Edge cases - LifetimeCurve start values', () => {
  const bezier = {
    type: 'BEZIER',
    scale: 1,
    bezierPoints: [
      { x: 0, y: 0, percentage: 0 },
      { x: 1, y: 1, percentage: 1 },
    ],
  };

  it('should handle bezier curve for startSpeed', () => {
    const { ps, step } = createTestSystem({ startSpeed: bezier });
    expect(() => step(100)).not.toThrow();
    ps.dispose();
  });

  it('should handle bezier curve for startLifetime', () => {
    const { ps, step } = createTestSystem({ startLifetime: bezier });
    expect(() => step(100)).not.toThrow();
    ps.dispose();
  });

  it('should handle bezier curve for startSize', () => {
    const { ps, step } = createTestSystem({ startSize: bezier });
    expect(() => step(100)).not.toThrow();
    ps.dispose();
  });
});

describe('Edge cases - emission', () => {
  it('should handle rateOverTime as random range', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: { min: 100, max: 100 } },
    });
    step(100);
    expect(lastEmit(ps)).toBe(10);
    ps.dispose();
  });

  it('should emit by both time and distance simultaneously', () => {
    const { ps, step } = createTestSystem({
      emission: { rateOverTime: 10, rateOverDistance: 1 },
    });
    step(100); // time: floor(10*0.1) = 1
    expect(lastEmit(ps)).toBe(1);

    (ps.instance as THREE.Points).position.x = 2;
    step(200, 100); // time 1 + distance 2
    expect(lastEmit(ps)).toBe(3);

    ps.dispose();
  });
});

describe('Edge cases - renderer config', () => {
  it('should handle all blending modes', () => {
    for (const blending of [0, 1, 2, 3, 4]) {
      const { ps } = createTestSystem({ renderer: { blending } });
      const material = (ps.instance as THREE.Points).material as THREE.Material;
      expect(material.blending).toBe(blending);
      ps.dispose();
    }
  });

  it('should handle depthTest false', () => {
    const { ps } = createTestSystem({ renderer: { depthTest: false } });
    const material = (ps.instance as THREE.Points).material as THREE.Material;
    expect(material.depthTest).toBe(false);
    ps.dispose();
  });

  it('should handle transparent false', () => {
    const { ps } = createTestSystem({ renderer: { transparent: false } });
    const material = (ps.instance as THREE.Points).material as THREE.Material;
    expect(material.transparent).toBe(false);
    ps.dispose();
  });
});

describe('Edge cases - shape variations', () => {
  it('should handle cone with 90 degree angle', () => {
    const { ps } = createTestSystem({
      shape: { shape: Shape.CONE, cone: { angle: 90, radius: 1 } },
    });
    expect(snapshotOf(ps).shape.coneAngleDeg).toBe(90);
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(1);
    ps.dispose();
  });

  it('should handle sphere with small arc', () => {
    const { ps } = createTestSystem({
      shape: { shape: Shape.SPHERE, sphere: { arc: 30 } },
    });
    expect(snapshotOf(ps).shape.arcDeg).toBe(30);
    ps.dispose();
  });

  it('should handle sphere with zero radiusThickness (shell only)', () => {
    const { ps } = createTestSystem({
      shape: { shape: Shape.SPHERE, sphere: { radiusThickness: 0 } },
    });
    expect(snapshotOf(ps).shape.radiusThickness).toBe(0);
    ps.dispose();
  });

  it('should handle circle with small arc', () => {
    const { ps } = createTestSystem({
      shape: { shape: Shape.CIRCLE, circle: { arc: 45 } },
    });
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(2);
    expect(snapshotOf(ps).shape.arcDeg).toBe(45);
    ps.dispose();
  });

  it('should handle rectangle with rotation', () => {
    const { ps } = createTestSystem({
      shape: {
        shape: Shape.RECTANGLE,
        rectangle: { rotation: { x: 10, y: 20 } },
      },
    });
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(3);
    ps.dispose();
  });

  it('should handle rectangle with zero scale', () => {
    const { ps } = createTestSystem({
      shape: {
        shape: Shape.RECTANGLE,
        rectangle: { scale: { x: 0, y: 0 } },
      },
    });
    expect(snapshotOf(ps).shape.gpuShapeKind).toBe(3);
    ps.dispose();
  });
});

describe('Edge cases - lifecycle', () => {
  it('should handle creating and disposing multiple systems', () => {
    const systems = [1, 2, 3].map((n) =>
      createParticleSystem({ maxParticles: n * 10 }, 1000)
    );
    for (const ps of systems) {
      expect(() => ps.dispose()).not.toThrow();
    }
  });

  it('should handle disposing middle system from a list', () => {
    const a = createParticleSystem({ maxParticles: 5 }, 1000);
    const b = createParticleSystem({ maxParticles: 5 }, 1000);
    const c = createParticleSystem({ maxParticles: 5 }, 1000);
    b.dispose();
    expect(() => {
      a.update({ now: 1100, delta: 0.1, elapsed: 0.1 });
      c.update({ now: 1100, delta: 0.1, elapsed: 0.1 });
    }).not.toThrow();
    a.dispose();
    c.dispose();
  });

  it('should handle update with very large delta', () => {
    const { ps, step } = createTestSystem();
    expect(() => step(10000, 9840)).not.toThrow();
    ps.dispose();
  });

  it('should handle update with very small delta', () => {
    const { ps, step } = createTestSystem();
    expect(() => step(1, 1)).not.toThrow();
    ps.dispose();
  });

  it('should handle rapid successive updates', () => {
    const { ps, step } = createTestSystem();
    expect(() => {
      for (let i = 1; i <= 200; i++) step(i * 1);
    }).not.toThrow();
    ps.dispose();
  });
});
