import * as THREE from 'three';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) contract: `updateConfig` merges the partial config into the
 * normalized config that the kernels read every frame. The CPU-observable
 * effect is the per-frame scalar emission count.
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

describe('ParticleSystem.updateConfig', () => {
  it('should be a function on the returned particle system', () => {
    const { ps } = createTestSystem();
    expect(typeof ps.updateConfig).toBe('function');
    ps.dispose();
  });

  describe('gravity updates', () => {
    it('should update gravity in real time', () => {
      const { ps, step } = createTestSystem({ gravity: 0 });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      ps.updateConfig({ gravity: -9.8 });
      expect(() => {
        step(200, 100);
        step(500, 300);
      }).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });

  describe('emission rate updates', () => {
    it('should change emission rate at runtime', () => {
      const { ps, step } = createTestSystem({
        emission: { rateOverTime: 5, rateOverDistance: 0 },
      });

      step(500);
      const low = lastEmit(ps);

      ps.updateConfig({ emission: { rateOverTime: 100, rateOverDistance: 0 } });
      step(1000, 500);
      const high = lastEmit(ps);

      expect(high).toBeGreaterThan(low);
      ps.dispose();
    });
  });

  describe('noise updates', () => {
    it('should enable noise at runtime', () => {
      const { ps, step } = createTestSystem({
        noise: {
          isActive: false,
          strength: 1,
          frequency: 1,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
          useRandomOffset: false,
        },
      });

      step(100);
      ps.updateConfig({
        noise: {
          isActive: true,
          strength: 2,
          frequency: 0.5,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
          useRandomOffset: false,
        },
      });

      expect(() => step(200, 100)).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should disable noise at runtime', () => {
      const { ps, step } = createTestSystem({
        noise: {
          isActive: true,
          strength: 1,
          frequency: 1,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
          useRandomOffset: false,
        },
      });

      step(100);
      ps.updateConfig({
        noise: {
          isActive: false,
          strength: 0,
          frequency: 1,
          power: 1,
          positionAmount: 0,
          rotationAmount: 0,
          sizeAmount: 0,
          useRandomOffset: false,
        },
      });

      expect(() => step(200, 100)).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });

  describe('duration and looping updates', () => {
    it('should update duration', () => {
      const { ps } = createTestSystem({ duration: 5 });
      ps.updateConfig({ duration: 10 });
      expect(() =>
        ps.update({ now: 6000, delta: 0.016, elapsed: 5 })
      ).not.toThrow();
      ps.dispose();
    });

    it('should update looping', () => {
      const { ps } = createTestSystem({ looping: true });
      ps.updateConfig({ looping: false });
      expect(() =>
        ps.update({ now: 6000, delta: 0.016, elapsed: 5 })
      ).not.toThrow();
      ps.dispose();
    });
  });

  describe('color and size config updates', () => {
    it('should update startColor for new particles', () => {
      const { ps, step } = createTestSystem({
        startColor: {
          min: { r: 1, g: 1, b: 1 },
          max: { r: 1, g: 1, b: 1 },
        },
      });

      step(100);
      ps.updateConfig({
        startColor: {
          min: { r: 1, g: 0, b: 0 },
          max: { r: 1, g: 0, b: 0 },
        },
      });

      expect(() => step(500, 400)).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should update startSize for new particles', () => {
      const { ps, step } = createTestSystem({ startSize: 1 });
      step(100);
      ps.updateConfig({ startSize: 5 });
      expect(() => step(500, 400)).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });

  describe('multiple sequential updates', () => {
    it('should handle multiple updateConfig calls', () => {
      const { ps, step } = createTestSystem();

      step(100);
      ps.updateConfig({ gravity: -5 });
      step(200, 100);
      ps.updateConfig({
        gravity: -10,
        forceFields: [
          {
            type: 'POINT',
            position: { x: 0, y: 5, z: 0 },
            strength: 3,
            range: 10,
          },
        ],
      });
      step(300, 100);
      ps.updateConfig({
        forceFields: [],
        emission: { rateOverTime: 50, rateOverDistance: 0 },
      });
      step(400, 100);

      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });

  describe('partial config merging', () => {
    it('should only update specified properties', () => {
      const { ps, step } = createTestSystem({
        gravity: -5,
        duration: 10,
      });

      step(100);
      ps.updateConfig({ gravity: -20 });

      expect(() => step(200, 100)).not.toThrow();
      const snap = (
        ps as unknown as {
          gpuDebug: { snapshot(): { maxParticles: number } };
        }
      ).gpuDebug.snapshot();
      expect(snap.maxParticles).toBe(50);
      ps.dispose();
    });
  });
});
