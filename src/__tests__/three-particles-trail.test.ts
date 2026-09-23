import * as THREE from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { RendererType } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only trail contract (4.x): the visible object is the ribbon Mesh; the
 * history ring is filled by the `trail-history` compute pass. CPU-observable:
 * `gpuDebug.lastEmitCount()`, `passNames`, ribbon geometry sizes.
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

const createTrailSystem = (
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
      renderer: {
        rendererType: RendererType.TRAIL,
        trail: { length: 4, dieWithParticle: true },
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

describe('Trail / Ribbon Renderer (RendererType.TRAIL)', () => {
  describe('creation', () => {
    it('should create a THREE.Mesh with the ribbon geometry', () => {
      const { ps } = createTrailSystem();
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      expect((ps.instance as THREE.Mesh).frustumCulled).toBe(false);
      ps.dispose();
    });

    it('should have ribbon attributes on the geometry', () => {
      const { ps } = createTrailSystem();
      const geom = (ps.instance as THREE.Mesh).geometry;
      for (const name of [
        'position',
        'trailNext',
        'trailUVColor',
        'trailColorBA',
      ]) {
        expect(geom.getAttribute(name)).toBeDefined();
      }
      ps.dispose();
    });

    it('should have correct vertex count based on maxParticles and trail length', () => {
      const { ps } = createTrailSystem();
      const geom = (ps.instance as THREE.Mesh).geometry;
      // 2 vertices per history sample: 20 particles * 4 samples * 2
      expect(geom.getAttribute('position').count).toBe(20 * 4 * 2);
      ps.dispose();
    });

    it('should have correct index count based on maxParticles and trail length', () => {
      const { ps } = createTrailSystem();
      const geom = (ps.instance as THREE.Mesh).geometry;
      // 6 indices per quad: 20 * (4 - 1) quads * 6
      expect(geom.getIndex()!.count).toBe(20 * (4 - 1) * 6);
      ps.dispose();
    });

    it('should use DoubleSide rendering', () => {
      const { ps } = createTrailSystem();
      const material = (ps.instance as THREE.Mesh)
        .material as MeshBasicNodeMaterial;
      expect(material).toBeInstanceOf(MeshBasicNodeMaterial);
      expect(material.side).toBe(THREE.DoubleSide);
      ps.dispose();
    });

    it('should accept widthOverTrail/opacityOverTrail without type field (legacy format)', () => {
      expect(() =>
        createTrailSystem({
          renderer: {
            rendererType: RendererType.TRAIL,
            trail: {
              length: 3,
              widthOverTrail: {
                scale: 1,
                bezierPoints: [
                  { x: 0, y: 1, percentage: 0 },
                  { x: 1, y: 0, percentage: 1 },
                ],
              },
              opacityOverTrail: {
                scale: 1,
                bezierPoints: [
                  { x: 0, y: 1, percentage: 0 },
                  { x: 1, y: 0, percentage: 1 },
                ],
              },
            },
          },
        })
      ).not.toThrow();
    });

    it('should use the default trail length of 20 when not specified', () => {
      const ps = createParticleSystem(
        {
          maxParticles: 5,
          duration: 5,
          looping: true,
          renderer: { rendererType: RendererType.TRAIL, trail: {} },
        } as never,
        1000
      );
      const geom = (ps.instance as THREE.Mesh).geometry;
      expect(geom.getAttribute('position').count).toBe(5 * 20 * 2);
      ps.dispose();
    });
  });

  describe('compute passes', () => {
    it('adds the trail-history pass to the dispatch list', () => {
      const { ps } = createTrailSystem();
      expect(passNames(ps)).toContain('trail-history');
      ps.dispose();
    });

    it('keeps every pass within the 8-storage-binding guarantee', () => {
      const { ps } = createTrailSystem();
      const counts = (
        ps as unknown as {
          gpuDebug: { passBindingCounts: Array<[string, number]> };
        }
      ).gpuDebug;
      for (const [, storageBindings] of counts.passBindingCounts) {
        expect(storageBindings).toBeLessThanOrEqual(8);
      }
      ps.dispose();
    });
  });

  describe('particle simulation', () => {
    it('should emit particles over time', () => {
      const { ps, step } = createTrailSystem();

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      ps.dispose();
    });

    it('should stop emitting after the non-looping duration', () => {
      const { ps, step } = createTrailSystem({
        looping: false,
        duration: 1,
        startLifetime: 0.5,
      });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      step(1500, 1400);
      expect(lastEmit(ps)).toBe(0);

      ps.dispose();
    });
  });

  describe('trail configuration', () => {
    it('should support custom trail length', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 8 },
        },
      });
      const geom = (ps.instance as THREE.Mesh).geometry;
      expect(geom.getAttribute('position').count).toBe(20 * 8 * 2);
      ps.dispose();
    });

    it('should support blending mode configuration', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 4 },
          blending: 2,
          transparent: true,
        },
      });
      const material = (ps.instance as THREE.Mesh).material as THREE.Material;
      expect(material.blending).toBe(2);
      ps.dispose();
    });

    it('should support transparency configuration', () => {
      const { ps } = createTrailSystem({
        renderer: {
          rendererType: RendererType.TRAIL,
          trail: { length: 4 },
          transparent: false,
        },
      });
      const material = (ps.instance as THREE.Mesh).material as THREE.Material;
      expect(material.transparent).toBe(false);
      ps.dispose();
    });
  });

  describe('integration with other features', () => {
    it('should work with gravity', () => {
      const { ps, step } = createTrailSystem({ gravity: -9.8 });
      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should work with opacity over lifetime', () => {
      const { ps, step } = createTrailSystem({
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
      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should work with noise', () => {
      const { ps, step } = createTrailSystem({
        noise: {
          isActive: true,
          strength: 1,
          frequency: 1,
          power: 1,
          positionAmount: 1,
          rotationAmount: 0,
          sizeAmount: 0,
        },
      });
      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('should work with burst emission', () => {
      const { ps, step } = createTrailSystem({
        emission: {
          rateOverTime: 0,
          rateOverDistance: 0,
          bursts: [{ time: 0, count: 3 }],
        },
      });
      step(16);
      expect(lastEmit(ps)).toBe(3);
      ps.dispose();
    });
  });

  describe('dispose', () => {
    it('should clean up the ribbon without error', () => {
      const { ps, step } = createTrailSystem();
      step(100);
      expect(() => ps.dispose()).not.toThrow();
    });
  });
});
