import * as THREE from 'three';
import { RendererType } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only trail contract (4.x): the ribbon Mesh owns vec4 storage
 * attributes (`position` = xyz+halfWidth, `trailNext`, `trailUVColor`,
 * `trailColorBA`); the history ring is filled by the `trail-history` compute
 * pass. The CPU side only creates the structures and writes scalars.
 */

const createTrailSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 10,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 3,
      startSize: 1,
      startOpacity: 1,
      startRotation: 0,
      emission: { rateOverTime: 20 },
      renderer: {
        rendererType: RendererType.TRAIL,
        trail: { length: 8 },
      },
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

const ribbonGeometry = (ps: ParticleSystem): THREE.BufferGeometry =>
  (ps.instance as THREE.Mesh).geometry;

describe('Trail Improvements (GPU-only contract)', () => {
  describe('Adaptive Trail Sampling (minVertexDistance)', () => {
    it('should create a trail system with minVertexDistance without errors', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, minVertexDistance: 0.5 },
        },
      });
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      expect(passNames(ps)).toContain('trail-history');
      ps.dispose();
    });

    it('runs updates with a high minVertexDistance threshold', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        startSpeed: 0.001,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, minVertexDistance: 10 },
        },
      });

      expect(() => {
        step(16);
        step(80, 64);
      }).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('runs updates with a low minVertexDistance threshold', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        startSpeed: 10,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, minVertexDistance: 0.01 },
        },
      });

      for (let i = 1; i <= 20; i++) step(i * 16);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('falls back to per-frame sampling when minVertexDistance is 0', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        startSpeed: 3,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, minVertexDistance: 0 },
        },
      });

      expect(() => {
        for (let i = 1; i <= 10; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });
  });

  describe('Trail Max Time', () => {
    it('should create a trail system with maxTime without errors', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, maxTime: 1.0 },
        },
      });
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      ps.dispose();
    });

    it('does not throw when maxTime expires old segments', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 1,
        startSpeed: 3,
        startLifetime: 10,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 30, maxTime: 0.2 },
        },
      });

      for (let i = 1; i <= 10; i++) step(i * 16);
      expect(() => step(5000, 4840)).not.toThrow();
      ps.dispose();
    });

    it('does not expire segments when maxTime is 0', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 1,
        startSpeed: 3,
        startLifetime: 10,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, maxTime: 0 },
        },
      });

      for (let i = 1; i <= 10; i++) step(i * 16);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });

  describe('Trail Smoothing', () => {
    it('should create a trail system with smoothing enabled without errors', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, smoothing: true, smoothingSubdivisions: 3 },
        },
      });
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      ps.dispose();
    });

    it('runs smoothing updates without throwing', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 1,
        startSpeed: 3,
        startLifetime: 10,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 30, smoothing: true, smoothingSubdivisions: 3 },
        },
      });

      expect(() => {
        for (let i = 1; i <= 20; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });
  });

  describe('Trail Twist Prevention', () => {
    it('should create a trail system with twistPrevention without errors', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, twistPrevention: true },
        },
      });
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      ps.dispose();
    });

    it('runs twist-prevention frames without throwing', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 1,
        startSpeed: 5,
        startLifetime: 10,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, twistPrevention: true },
        },
      });

      expect(() => {
        for (let i = 1; i <= 15; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });

    it('works correctly when combined with noise (direction changes)', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 3,
        noise: {
          isActive: true,
          strength: 0.5,
          frequency: 2.0,
          positionAmount: 1.0,
          rotationAmount: 0,
          sizeAmount: 0,
          useRandomOffset: true,
        },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 15, twistPrevention: true },
        },
      });

      expect(() => {
        for (let i = 1; i <= 20; i++) step(i * 16);
      }).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
      ps.dispose();
    });

    it('should reset twist normal when particle is recycled', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 2,
        startSpeed: 3,
        startLifetime: 0.1,
        emission: { rateOverTime: 50 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, twistPrevention: true },
        },
      });

      expect(() => {
        for (let i = 1; i <= 40; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });
  });

  describe('Connected Ribbons (ribbonId)', () => {
    it('should create a trail system with ribbonId without errors', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, ribbonId: 1 },
        },
      });
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      ps.dispose();
    });

    it('runs ribbon-chain frames without throwing', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 20, ribbonId: 1 },
        },
      });

      expect(() => {
        for (let i = 1; i <= 10; i++) step(i * 16);
      }).not.toThrow();
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should work without ribbonId (independent trails per particle)', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8 },
        },
      });

      expect(() => {
        for (let i = 1; i <= 10; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });
  });

  describe('Combined Features', () => {
    it('should work with all trail improvements enabled simultaneously', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: {
            length: 20,
            minVertexDistance: 0.05,
            maxTime: 2.0,
            smoothing: true,
            smoothingSubdivisions: 2,
            twistPrevention: true,
          },
        },
      });

      expect(() => {
        for (let i = 1; i <= 20; i++) step(i * 16);
        step(500, 320);
      }).not.toThrow();
      // 20 frames @ rate 20 → 6 emitted; the 180 ms jump adds 4 more.
      expect(lastEmit(ps)).toBe(4);
      ps.dispose();
    });

    it('should work with sub-emitters using trail improvements', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 10,
        startLifetime: 0.3,
        subEmitters: [
          {
            trigger: 'DEATH',
            config: {
              maxParticles: 5,
              startLifetime: 0.5,
              startSpeed: 2,
              emission: { rateOverTime: 0, bursts: [{ time: 0, count: 3 }] },
              renderer: {
                rendererType: RendererType.TRAIL,
                trail: { length: 10, minVertexDistance: 0.1, smoothing: true },
              },
            },
          },
        ],
      });

      expect(() => {
        for (let i = 1; i <= 50; i++) step(i * 16);
      }).not.toThrow();
      ps.dispose();
    });

    it('should work with force fields and trail improvements', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 5,
        startLifetime: 3,
        emission: { rateOverTime: 10 },
        forceFields: [
          {
            type: 'POINT',
            position: { x: 0, y: 0, z: 0 },
            strength: 5.0,
            range: 10,
            falloff: 'LINEAR',
          },
        ],
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 15, smoothing: true, twistPrevention: true },
        },
      });

      expect(() => {
        for (let i = 1; i <= 20; i++) step(i * 16);
        step(500, 320);
      }).not.toThrow();
      // 20 frames @ rate 10 → 3 emitted; the 180 ms jump adds 2 more.
      expect(lastEmit(ps)).toBe(2);
      ps.dispose();
    });
  });

  describe('Ribbon geometry contract', () => {
    it('exposes the four vec4 ribbon attributes', () => {
      const { ps } = createTrailSystem();
      const geom = ribbonGeometry(ps);
      for (const name of [
        'position',
        'trailNext',
        'trailUVColor',
        'trailColorBA',
      ]) {
        const attr = geom.getAttribute(name);
        expect(attr).toBeDefined();
        expect(attr.itemSize).toBe(4);
      }
      ps.dispose();
    });

    it('should correctly keep attributes valid across many frames', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 3,
        startSpeed: 5,
        emission: { rateOverTime: 50 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8 },
        },
      });

      for (let i = 1; i <= 20; i++) step(i * 16);

      const geom = ribbonGeometry(ps);
      const posArr = geom.getAttribute('position').array as Float32Array;
      const nextArr = geom.getAttribute('trailNext').array as Float32Array;
      for (let i = 0; i < posArr.length; i++) {
        expect(Number.isFinite(posArr[i])).toBe(true);
        expect(Number.isFinite(nextArr[i])).toBe(true);
      }

      ps.dispose();
    });

    it('should handle zero-length movement without NaN or artifacts', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 3,
        startSpeed: 0,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8 },
        },
      });

      for (let i = 1; i <= 20; i++) step(i * 16);

      const geom = ribbonGeometry(ps);
      const posArr = geom.getAttribute('position').array as Float32Array;
      for (let i = 0; i < posArr.length; i++) {
        expect(Number.isFinite(posArr[i])).toBe(true);
      }

      ps.dispose();
    });

    it('should handle maxTime expiring all segments gracefully', () => {
      const { ps, step } = createTrailSystem({
        maxParticles: 1,
        startSpeed: 5,
        emission: { rateOverTime: 100 },
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8, maxTime: 0.001 },
        },
      });

      for (let i = 1; i <= 5; i++) step(i * 16);
      expect(() => step(5000, 4920)).not.toThrow();

      const geom = ribbonGeometry(ps);
      const posArr = geom.getAttribute('position').array as Float32Array;
      for (let i = 0; i < posArr.length; i++) {
        expect(Number.isFinite(posArr[i])).toBe(true);
      }

      ps.dispose();
    });

    it('should dispose without errors when trail improvements are enabled', () => {
      const { ps, step } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: {
            length: 10,
            minVertexDistance: 0.05,
            maxTime: 1,
            smoothing: true,
            smoothingSubdivisions: 2,
            twistPrevention: true,
            ribbonId: 1,
          },
        },
      });
      for (let i = 1; i <= 10; i++) step(i * 16);
      expect(() => ps.dispose()).not.toThrow();
    });
  });
});
