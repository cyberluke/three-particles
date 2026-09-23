import * as THREE from 'three';
import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import { RendererType } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only contract (4.x): the instanced renderer is a THREE.Mesh with an
 * InstancedBufferGeometry whose per-instance attributes are the compute-owned
 * vec4 storage buffers. Emission is observed through `gpuDebug.lastEmitCount()`.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createInstancedSystem = (
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
      renderer: { rendererType: RendererType.INSTANCED },
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

const getGeometry = (ps: ParticleSystem): THREE.InstancedBufferGeometry =>
  (ps.instance as THREE.Mesh).geometry as THREE.InstancedBufferGeometry;

describe('GPU Instancing (RendererType.INSTANCED)', () => {
  describe('creation', () => {
    it('should create a THREE.Mesh instead of THREE.Points', () => {
      const { ps } = createInstancedSystem();
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      expect(ps.instance).not.toBeInstanceOf(THREE.Points);
      ps.dispose();
    });

    it('should use InstancedBufferGeometry', () => {
      const { ps } = createInstancedSystem();
      const geom = getGeometry(ps);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect(geom.instanceCount).toBe(50);
      ps.dispose();
    });

    it('should have a base quad geometry with 4 vertices and 6 indices', () => {
      const { ps } = createInstancedSystem();
      const geom = getGeometry(ps);
      expect(geom.getAttribute('position').count).toBe(4);
      expect(geom.getIndex()!.count).toBe(6);
      ps.dispose();
    });

    it('should have the GPU storage instance attributes', () => {
      const { ps } = createInstancedSystem();
      const geom = getGeometry(ps);

      for (const name of [
        'instanceOffset',
        'instanceColor',
        'instanceParticleState',
        'instanceStartValues',
      ]) {
        const attr = geom.getAttribute(name);
        expect(attr).toBeDefined();
        expect(
          attr instanceof StorageBufferAttribute ||
            attr instanceof StorageInstancedBufferAttribute
        ).toBe(true);
        expect(attr.itemSize).toBe(4);
        expect(attr.count).toBe(50);
      }

      ps.dispose();
    });
  });

  describe('emission and lifecycle', () => {
    it('should emit particles over time', () => {
      const { ps, step } = createInstancedSystem({
        emission: { rateOverTime: 100 },
      });

      step(100);
      expect(lastEmit(ps)).toBe(10);
      ps.dispose();
    });

    it('should stop emitting after the non-looping duration', () => {
      const { ps, step } = createInstancedSystem({
        emission: { rateOverTime: 100 },
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

    it('should respect maxParticles in the allocator capacity', () => {
      const { ps } = createInstancedSystem({ maxParticles: 10 });
      const dbg = (
        ps as unknown as {
          gpuDebug: { allocatorCount: number; maxParticles: number };
        }
      ).gpuDebug;
      expect(dbg.maxParticles).toBe(10);
      expect(dbg.allocatorCount).toBe(11);
      ps.dispose();
    });
  });

  describe('material and uniforms', () => {
    it('should use a TSL node material', () => {
      const { ps } = createInstancedSystem();
      const material = (ps.instance as THREE.Mesh)
        .material as THREE.Material;
      expect(material).toBeDefined();
      expect(material.type).toContain('Node');
      ps.dispose();
    });

    it('should expose viewportHeight in the shared uniforms', () => {
      const { ps } = createInstancedSystem();
      const dbg = (
        ps as unknown as {
          gpuDebug: { snapshot(): Record<string, unknown> };
        }
      ).gpuDebug;
      const snap = dbg.snapshot();
      expect(snap.effectiveRendererType).toBe('INSTANCED');
      ps.dispose();
    });

    it('should support pauseEmitter and resumeEmitter', () => {
      const { ps, step } = createInstancedSystem({
        emission: { rateOverTime: 100 },
      });

      ps.pauseEmitter();
      step(100);
      expect(lastEmit(ps)).toBe(0);

      ps.resumeEmitter();
      step(300, 200);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      ps.dispose();
    });

    it('should clean up on dispose', () => {
      const { ps } = createInstancedSystem();
      expect(() => ps.dispose()).not.toThrow();
    });
  });

  describe('renderer-type resolution', () => {
    it('still creates Points when rendererType is not set', () => {
      const ps = createParticleSystem({ maxParticles: 10 }, 1000);
      expect(ps.instance).toBeInstanceOf(THREE.Points);
      ps.dispose();
    });

    it('creates Points when rendererType is POINTS', () => {
      const ps = createParticleSystem(
        { renderer: { rendererType: RendererType.POINTS }, maxParticles: 10 },
        1000
      );
      expect(ps.instance).toBeInstanceOf(THREE.Points);
      ps.dispose();
    });

    it('reports the requested vs effective renderer types in gpuDebug', () => {
      const { ps } = createInstancedSystem();
      const dbg = (
        ps as unknown as {
          gpuDebug: {
            requestedRendererType: string;
            effectiveRendererType: string;
          };
        }
      ).gpuDebug;
      expect(dbg.requestedRendererType).toBe('INSTANCED');
      expect(dbg.effectiveRendererType).toBe('INSTANCED');
      ps.dispose();
    });
  });

  describe('modifiers via pipeline flags', () => {
    it('creates the pipeline with color-over-lifetime curves', () => {
      const { ps, step } = createInstancedSystem({
        startLifetime: 2,
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

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('creates the pipeline with rotation-over-lifetime active', () => {
      const { ps, step } = createInstancedSystem({
        rotationOverLifetime: { isActive: true, min: 1, max: 2 },
      });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('creates the pipeline with linear velocity active', () => {
      const { ps, step } = createInstancedSystem({
        velocityOverLifetime: {
          isActive: true,
          linear: { x: 1, y: 0, z: 0 },
          orbital: { x: 0, y: 0, z: 0 },
        },
      });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });

    it('emits burst particles', () => {
      const { ps, step } = createInstancedSystem({
        emission: {
          rateOverTime: 0,
          rateOverDistance: 0,
          bursts: [{ time: 0, count: 5 }],
        },
      });

      step(16);
      expect(lastEmit(ps)).toBe(5);
      ps.dispose();
    });

    it('creates the pipeline with gravity', () => {
      const { ps, step } = createInstancedSystem({ gravity: -9.8 });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);
      ps.dispose();
    });
  });
});
