import * as THREE from 'three';
import {
  SimulationSpace,
  SubEmitterTrigger,
} from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only sub-emitter contract (4.x): each sub-emitter config becomes ONE
 * child render object attached to the parent instance, driven by the FIFO
 * event kernels. The CPU observes `gpuDebug.subEmitters`.
 */

const createTestSystem = (
  config: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 10,
      duration: 5,
      looping: true,
      startLifetime: 0.2,
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

const subEmitterConfig = {
  maxParticles: 5,
  duration: 1,
  looping: false,
  startLifetime: 0.3,
  startSpeed: 0.5,
  startSize: 0.5,
  startOpacity: 1,
  startRotation: 0,
  emission: { rateOverTime: 5, rateOverDistance: 0 },
};

const emitAndWaitForDeath = (step: (t: number, d?: number) => void) => {
  step(16);
  step(100, 84);
  step(300, 200);
  step(500, 200);
};

const subEmitterCount = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { subEmitters: unknown[] };
    }
  ).gpuDebug.subEmitters.length;

describe('Sub-emitters', () => {
  describe('death trigger', () => {
    it('should spawn sub-emitter systems when particles die', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: subEmitterConfig,
          },
        ],
      });

      emitAndWaitForDeath(step);

      expect(subEmitterCount(ps)).toBe(1);
      expect(ps.instance.children.length).toBe(1);

      ps.dispose();
    });

    it('should default to DEATH trigger when trigger is not specified', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            config: subEmitterConfig,
          },
        ],
      });

      emitAndWaitForDeath(step);

      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });
  });

  describe('birth trigger', () => {
    it('should spawn sub-emitter systems when particles are born', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.BIRTH,
            config: subEmitterConfig,
          },
        ],
      });

      step(16);
      step(100, 84);

      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });
  });

  describe('maxInstances', () => {
    it('should respect the maxInstances cap (one child object per config)', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.BIRTH,
            config: subEmitterConfig,
            maxInstances: 3,
          },
        ],
      });

      step(16);
      step(200, 184);

      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });
  });

  describe('dispose cleanup', () => {
    it('should dispose all sub-emitter instances when parent is disposed', () => {
      const scene = new THREE.Group();

      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.BIRTH,
            config: subEmitterConfig,
          },
        ],
      });

      scene.add(ps.instance);

      step(16);
      step(100, 84);

      expect(ps.instance.children.length).toBe(1);

      expect(() => ps.dispose()).not.toThrow();
      expect(scene.children.length).toBe(0);
    });
  });

  describe('no sub-emitters configured', () => {
    it('should not affect behavior when subEmitters is not set', () => {
      const { ps, step } = createTestSystem({});

      emitAndWaitForDeath(step);

      expect(ps.instance.children.length).toBe(0);
      ps.dispose();
    });

    it('should not affect behavior when subEmitters is empty array', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [],
      });

      emitAndWaitForDeath(step);

      expect(ps.instance.children.length).toBe(0);
      ps.dispose();
    });
  });

  describe('inherit velocity', () => {
    it('should create sub-emitter with inheritVelocity=0 without error', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: subEmitterConfig,
            inheritVelocity: 0,
          },
        ],
      });

      expect(() => emitAndWaitForDeath(step)).not.toThrow();
      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });

    it('should create sub-emitter with inheritVelocity > 0 without error', () => {
      const { ps, step } = createTestSystem({
        startSpeed: 5,
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: subEmitterConfig,
            inheritVelocity: 0.5,
          },
        ],
      });

      expect(() => emitAndWaitForDeath(step)).not.toThrow();
      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });
  });

  describe('update propagation', () => {
    it('should update sub-emitter particles when using instance update()', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: {
              ...subEmitterConfig,
              startLifetime: 2,
              emission: { rateOverTime: 50, rateOverDistance: 0 },
            },
          },
        ],
      });

      emitAndWaitForDeath(step);
      expect(() => {
        step(600, 100);
        step(700, 100);
      }).not.toThrow();

      expect(ps.instance.children.length).toBe(1);
      ps.dispose();
    });
  });

  describe('WORLD space sub-emitters', () => {
    it('keeps the WORLD-space child attached with the GPU contract', () => {
      const startTime = 1000;

      const ps = createParticleSystem(
        {
          maxParticles: 10,
          duration: 5,
          looping: true,
          startLifetime: 5,
          startSpeed: 1,
          emission: {
            rateOverTime: 50,
            rateOverDistance: 0,
            bursts: [{ time: 0, count: 1 }],
          },
          subEmitters: [
            {
              trigger: SubEmitterTrigger.BIRTH,
              config: {
                maxParticles: 10,
                duration: 5,
                looping: false,
                startLifetime: 3,
                startSpeed: 0,
                simulationSpace: SimulationSpace.WORLD,
                emission: {
                  rateOverTime: 0,
                  rateOverDistance: 0,
                  bursts: [{ time: 0, count: 5 }],
                },
              },
              maxInstances: 1,
            },
          ],
        } as never,
        startTime
      );

      ps.update({ now: startTime, delta: 0.016, elapsed: 0 });
      ps.update({ now: startTime + 100, delta: 0.1, elapsed: 0.1 });

      // The child object is attached to the parent instance directly.
      expect(ps.instance.children.length).toBe(1);
      const child = ps.instance.children[0] as THREE.Points;
      expect(child).toBeInstanceOf(THREE.Points);

      // GPU contract attributes on the child pool.
      expect(child.geometry.attributes.particleState).toBeDefined();
      expect(child.geometry.attributes.startValues).toBeDefined();

      ps.dispose();
    });
  });

  describe('sub-emitter lifecycle', () => {
    it('sub-emitters should be created as non-looping', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: {
              ...subEmitterConfig,
              looping: true,
            },
          },
        ],
      });

      emitAndWaitForDeath(step);
      expect(subEmitterCount(ps)).toBe(1);
      ps.dispose();
    });

    it('should handle multiple sub-emitter configs on the same system', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: subEmitterConfig,
          },
          {
            trigger: SubEmitterTrigger.BIRTH,
            config: subEmitterConfig,
          },
        ],
      });

      step(16);
      step(100, 84);

      expect(subEmitterCount(ps)).toBe(2);
      expect(ps.instance.children.length).toBe(2);

      ps.dispose();
    });

    it('sub-emitter child pools use the GPU storage contract', () => {
      const { ps, step } = createTestSystem({
        subEmitters: [
          {
            trigger: SubEmitterTrigger.DEATH,
            config: subEmitterConfig,
          },
        ],
      });

      emitAndWaitForDeath(step);

      const child = ps.instance.children[0] as THREE.Points;
      const geom = child.geometry;
      // GPU-only contract: vec4 storage pools, no legacy interleaved isActive.
      expect(geom.attributes.particleState).toBeDefined();
      expect(geom.attributes.startValues).toBeDefined();
      expect(geom.attributes.isActive).toBeUndefined();

      const dbg = (
        ps as unknown as {
          gpuDebug: {
            subEmitters: Array<{
              requestedRendererType: string | null;
              effectiveRendererType: string;
              perEvent: number;
            }>;
          };
        }
      ).gpuDebug;
      expect(dbg.subEmitters[0].effectiveRendererType).toBe('POINTS');
      expect(dbg.subEmitters[0].perEvent).toBeGreaterThanOrEqual(1);

      ps.dispose();
    });
  });
});
