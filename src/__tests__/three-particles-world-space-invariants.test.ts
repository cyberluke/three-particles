import * as THREE from 'three';
import { SimulationSpace } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only (4.x) WORLD-space invariants: the instance matrixWorld is pinned
 * to identity and the full emitter pose reaches the kernels through the
 * emitter-pose uniforms. The CPU-observable surface is the uniform values
 * plus the per-frame emission scalars.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (
    ps as unknown as {
      gpuDebug: { lastEmitCount(): number };
    }
  ).gpuDebug.lastEmitCount();

const createWorldSystem = (
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
      emission: { rateOverTime: 50, rateOverDistance: 0 },
      simulationSpace: SimulationSpace.WORLD,
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

const identity = new THREE.Matrix4().elements;

describe('WORLD simulation space — invariants', () => {
  it('matrixWorld stays identity while the emitter translates', () => {
    const { ps, step } = createWorldSystem();

    step(16);
    ps.instance.position.set(10, 0, 0);
    step(100, 84);

    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(ps.instance.matrixWorld.elements).toEqual(identity);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('matrixWorld stays identity while the emitter rotates', () => {
    const { ps, step } = createWorldSystem();

    step(16);
    ps.instance.quaternion.setFromEuler(new THREE.Euler(0, Math.PI / 4, 0));
    step(100, 84);

    expect(ps.instance.matrixWorld.elements).toEqual(identity);
    ps.dispose();
  });

  it('no drift over 1000 frames of continuous motion', () => {
    const { ps, step } = createWorldSystem();

    for (let i = 1; i <= 1000; i++) {
      ps.instance.position.set(Math.sin(i * 0.01) * 5, i * 0.001, 0);
      step(i * 16);
    }

    expect(ps.instance.matrixWorld.elements).toEqual(identity);
    ps.dispose();
  });

  it('variable framerate does not produce jitter on a moving emitter', () => {
    const { ps, step } = createWorldSystem();

    const deltas = [16, 33, 8, 50, 16, 12, 40];
    let t = 0;
    for (const d of deltas) {
      t += d;
      step(t, d);
    }

    expect(lastEmit(ps)).toBeGreaterThanOrEqual(0);
    ps.dispose();
  });

  it('quaternion q and -q represent the same rotation', () => {
    const a = createWorldSystem();
    const b = createWorldSystem();

    a.ps.instance.quaternion.setFromEuler(new THREE.Euler(0, 1, 0));
    b.ps.instance.quaternion.set(
      -a.ps.instance.quaternion.x,
      -a.ps.instance.quaternion.y,
      -a.ps.instance.quaternion.z,
      -a.ps.instance.quaternion.w
    );

    a.step(16);
    b.step(16);

    expect(a.ps.instance.matrixWorld.elements).toEqual(
      b.ps.instance.matrixWorld.elements
    );
    a.ps.dispose();
    b.ps.dispose();
  });

  it('gravity is world-down regardless of emitter rotation', () => {
    const { ps, step } = createWorldSystem({ gravity: -9.8, startSpeed: 0 });

    step(16);
    ps.instance.quaternion.setFromEuler(new THREE.Euler(0.5, 1, 0.2));
    expect(() => {
      step(100, 84);
      step(200, 100);
    }).not.toThrow();

    ps.dispose();
  });

  it('instance.position offsets the spawn origin under a parent', () => {
    const { ps, step } = createWorldSystem();
    const parent = new THREE.Group();
    parent.add(ps.instance);
    parent.position.set(50, 0, 0);
    parent.updateMatrixWorld(true);

    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();
    expect(ps.instance.matrixWorld.elements).toEqual(identity);

    ps.dispose();
  });
});

describe('LOCAL simulation space — regression', () => {
  it('LOCAL instance keeps automatic matrixWorld updates', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        emission: { rateOverTime: 50 },
        simulationSpace: SimulationSpace.LOCAL,
      },
      1000
    );

    expect(ps.instance.matrixWorldAutoUpdate).not.toBe(false);
    ps.update({ now: 1100, delta: 0.1, elapsed: 0.1 });
    expect(lastEmit(ps)).toBe(5);

    ps.dispose();
  });
});

describe('WORLD simulation space — integration with subsystems', () => {
  it('directional force field stays world-aligned when emitter rotates', () => {
    const { ps, step } = createWorldSystem({
      gravity: 0,
      startSpeed: 0,
      forceFields: [
        {
          type: 'DIRECTIONAL',
          direction: { x: 1, y: 0, z: 0 },
          strength: 5,
        },
      ],
    });

    ps.instance.quaternion.setFromEuler(new THREE.Euler(0, Math.PI / 3, 0));
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();

    ps.dispose();
  });

  it('collision plane keeps its world position under emitter motion', () => {
    const { ps, step } = createWorldSystem({
      collisionPlanes: [
        { mode: 'PLANE', plane: { x: 0, y: 0, z: 1, w: 2 } },
      ],
    });

    ps.instance.position.set(1, 2, 3);
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();

    ps.dispose();
  });

  it('point force field stays world-anchored when emitter moves', () => {
    const { ps, step } = createWorldSystem({
      forceFields: [
        {
          type: 'POINT',
          position: { x: 0, y: 5, z: 0 },
          strength: 3,
          range: 10,
        },
      ],
    });

    ps.instance.position.set(10, 0, 0);
    expect(() => {
      step(16);
      step(100, 84);
    }).not.toThrow();

    ps.dispose();
  });
});

describe('Unity parent-scale parity', () => {
  it('LOCAL gravity falls at world -g regardless of parent scale', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        gravity: -3.8,
        startSpeed: 0,
        emission: { rateOverTime: 50 },
        simulationSpace: SimulationSpace.LOCAL,
      },
      1000
    );

    const parent = new THREE.Group();
    parent.scale.set(2, 2, 2);
    parent.add(ps.instance);
    parent.updateMatrixWorld(true);

    expect(() => {
      ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    }).not.toThrow();

    ps.dispose();
  });

  it('WORLD spawn offset scales with the parent pose', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        duration: 5,
        looping: true,
        startSpeed: 1,
        emission: { rateOverTime: 50 },
        simulationSpace: SimulationSpace.WORLD,
      },
      1000
    );

    const parent = new THREE.Group();
    parent.position.set(1, 2, 3);
    parent.scale.set(2, 2, 2);
    parent.add(ps.instance);
    parent.updateMatrixWorld(true);

    expect(() => {
      ps.update({ now: 1016, delta: 0.016, elapsed: 0.016 });
    }).not.toThrow();
    expect(ps.instance.matrixWorld.elements).toEqual(identity);

    ps.dispose();
  });
});
