import * as THREE from 'three';
import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import {
  RendererType,
  SubEmitterTrigger,
} from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only contract: per-particle state lives in the vec4 storage buffers
 * (`instanceOffset`, `instanceColor`, `instanceParticleState`,
 * `instanceStartValues`). The CPU-side observable is `gpuDebug.lastEmitCount()`.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createMeshSystem = (
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
      renderer: {
        rendererType: RendererType.MESH,
        mesh: {
          geometry: new THREE.BoxGeometry(1, 1, 1),
        },
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

const getGeometry = (ps: ParticleSystem): THREE.InstancedBufferGeometry =>
  (ps.instance as THREE.Mesh).geometry as THREE.InstancedBufferGeometry;

describe('Mesh Particle Renderer (RendererType.MESH)', () => {
  describe('creation', () => {
    it('should create a THREE.Mesh instead of THREE.Points', () => {
      const { ps } = createMeshSystem();
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      expect(ps.instance).not.toBeInstanceOf(THREE.Points);
      ps.dispose();
    });

    it('should use InstancedBufferGeometry', () => {
      const { ps } = createMeshSystem();
      const geom = getGeometry(ps);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect(geom.instanceCount).toBe(50);
      ps.dispose();
    });

    it('should copy base geometry from the provided mesh', () => {
      const boxGeom = new THREE.BoxGeometry(1, 1, 1);
      const { ps } = createMeshSystem({
        renderer: {
          rendererType: RendererType.MESH,
          mesh: { geometry: boxGeom },
        },
      });
      const geom = getGeometry(ps);
      const pos = geom.getAttribute('position');
      expect(pos.count).toBe(boxGeom.getAttribute('position').count);
      expect(pos.itemSize).toBe(3);
      // The GPU contract copies only the base `position` + index from the
      // provided mesh geometry; normals/uv are not part of the contract.
      const index = geom.getIndex();
      expect(index).not.toBeNull();
      ps.dispose();
    });

    it('should have the GPU storage instance attributes', () => {
      const { ps } = createMeshSystem();
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

    it('falls back to a base quad when no mesh geometry is provided', () => {
      const ps = createParticleSystem(
        {
          maxParticles: 10,
          duration: 5,
          looping: true,
          emission: { rateOverTime: 10 },
          renderer: {
            rendererType: RendererType.MESH,
            // No mesh config
          },
        } as never,
        1000
      );
      expect(ps.instance).toBeInstanceOf(THREE.Mesh);
      const geom = getGeometry(ps);
      expect(geom.getAttribute('position').count).toBe(4); // quad
      ps.dispose();
    });

    it('should work with different geometry types', () => {
      const geometries = [
        new THREE.SphereGeometry(0.5, 8, 6),
        new THREE.IcosahedronGeometry(0.5, 0),
        new THREE.TorusGeometry(0.5, 0.2, 8, 16),
        new THREE.ConeGeometry(0.5, 1, 8),
      ];

      for (const geometry of geometries) {
        const ps = createParticleSystem(
          {
            maxParticles: 10,
            duration: 5,
            looping: true,
            emission: { rateOverTime: 10 },
            renderer: {
              rendererType: RendererType.MESH,
              mesh: { geometry },
            },
          } as never,
          1000
        );
        expect(ps.instance).toBeInstanceOf(THREE.Mesh);
        const geom = (ps.instance as THREE.Mesh)
          .geometry as THREE.InstancedBufferGeometry;
        const pos = geom.getAttribute('position');
        expect(pos.count).toBe(geometry.getAttribute('position').count);
        ps.dispose();
      }
    });
  });

  describe('emission and lifecycle', () => {
    it('should emit particles over time', () => {
      const { ps, step } = createMeshSystem({
        emission: { rateOverTime: 100 },
      });

      // After 100ms with rate 100/s → 10 particles requested.
      step(100);
      expect(lastEmit(ps)).toBe(10);
      ps.dispose();
    });

    it('should stop emitting after the non-looping duration', () => {
      const { ps, step } = createMeshSystem({
        startLifetime: 0.5,
        emission: { rateOverTime: 100 },
        looping: false,
        duration: 1,
      });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      step(1500, 1400);
      expect(lastEmit(ps)).toBe(0);

      ps.dispose();
    });

    it('should respect maxParticles in the allocator capacity', () => {
      const { ps } = createMeshSystem({ maxParticles: 10 });
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

  describe('pause and resume', () => {
    it('should stop emitting when paused', () => {
      const { ps, step } = createMeshSystem({
        emission: { rateOverTime: 100 },
      });

      step(100);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      ps.pauseEmitter();
      step(200, 100);
      expect(lastEmit(ps)).toBe(0);

      ps.dispose();
    });

    it('should resume emitting after pause', () => {
      const { ps, step } = createMeshSystem({
        emission: { rateOverTime: 100 },
      });

      ps.pauseEmitter();
      step(200);
      expect(lastEmit(ps)).toBe(0);

      ps.resumeEmitter();
      step(500, 300);
      expect(lastEmit(ps)).toBeGreaterThan(0);

      ps.dispose();
    });
  });

  describe('sub-emitters', () => {
    it('should not force MESH rendererType on sub-emitters without geometry', () => {
      const startTime = 1000;

      const ps = createParticleSystem(
        {
          maxParticles: 3,
          duration: 5,
          looping: true,
          startLifetime: 0.1,
          startSpeed: 1,
          emission: {
            rateOverTime: 0,
            rateOverDistance: 0,
            bursts: [{ time: 0, count: 2 }],
          },
          renderer: {
            rendererType: RendererType.MESH,
            mesh: { geometry: new THREE.BoxGeometry(1, 1, 1) },
          },
          subEmitters: [
            {
              trigger: SubEmitterTrigger.DEATH,
              config: {
                maxParticles: 3,
                duration: 0.5,
                looping: false,
                startLifetime: 0.1,
                startSpeed: 0,
                startSize: 0.5,
                startOpacity: 1,
                startRotation: 0,
                emission: {
                  rateOverTime: 0,
                  rateOverDistance: 0,
                  bursts: [{ time: 0, count: 1 }],
                },
              },
              maxInstances: 4,
            },
          ],
        } as never,
        startTime
      );

      ps.update({ now: startTime, delta: 0.016, elapsed: 0 });
      expect(() => {
        ps.update({ now: startTime + 200, delta: 0.2, elapsed: 0.2 });
      }).not.toThrow();

      // Sub-emitter child render objects are attached to the parent instance.
      expect(ps.instance.children.length).toBeGreaterThan(0);
      // Child has no mesh geometry → plain Points.
      expect(ps.instance.children[0]).toBeInstanceOf(THREE.Points);

      ps.dispose();
    });
  });

  describe('dispose', () => {
    it('should clean up resources on dispose', () => {
      const { ps } = createMeshSystem();
      expect(() => ps.dispose()).not.toThrow();
    });
  });
});
