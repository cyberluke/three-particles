<p align="center">
  <img src="assets/images/logo-colorful.png" alt="THREE Particles Logo" width="150" />
</p>

# THREE Particles

Best-in-class particle framework for Three.js. Unity-style effects with a GPU-compute backbone: Three.js TSL kernels run gravity, orbital motion, Bézier over-lifetime curves, force fields and 3D simplex noise for 350K+ particles per system at full framerate.

Author: **CyberLuke** — the single maintained line since v4.

# Features

*   Easy integration with Three.js.
*   Visual editor for fine-tuning and exporting effect configs (`@cyberluke/three-particles-editor`).
*   Highly customizable particle properties (position, velocity, size, color, alpha, rotation, etc.).
*   Support for various emitter shapes and parameters.
*   Force fields and attractors for dynamic particle behavior (point attraction/repulsion, directional wind) — up to 16 per system.
*   Collision planes — kill, clamp, or bounce particles off infinite planes (e.g., water surfaces, floors, walls). Part of the compute pass.
*   Sub-emitters triggered on particle birth or death events (GPU ping-pong event buffers).
*   Baked Bézier over-lifetime curves — 256-sample lookup arrays (`curve-bake.ts`), <0.4% max interpolation error.
*   Four renderer types (`RendererType`): `POINTS` (billboard quads), `INSTANCED` (GPU instancing, no `gl_PointSize` limit), `TRAIL` (ribbon trails with width/opacity/color tapering), `MESH` (instanced 3D meshes with full rotation and lighting).
*   Soft particles — depth-based alpha fade near opaque geometry.
*   **WebGPU compute** — all per-particle physics runs in TSL compute kernels (`SimulationBackend.GPU`); `AUTO` uses the registered WebGPU path, `CPU` maps to the identical GPU path in this GPU-only build.
*   TypeScript definitions shipped (`dist/index.d.ts`, `webgpu.d.ts`).

# Installation

```bash
npm install @cyberluke/three-particles
```

Both entry points are plain ESM (no separate CDN files; the package ships `dist/index.js`, `dist/webgpu.js` and a minified `dist/three-particles.min.js`).

# Usage

The engine is **GPU-only** (v4): a WebGPU backend is required before creating any system.

```javascript
import * as THREE from "three/webgpu";
import {
  createParticleSystem,
  updateParticleSystems,
  Shape,
} from "@cyberluke/three-particles";
import { enableWebGPU } from "@cyberluke/three-particles/webgpu";

// 1. WebGPU renderer, then register the library with it (once)
const renderer = new THREE.WebGPURenderer({ antialias: true });
await renderer.init();
enableWebGPU(renderer); // false + warning if the backend lacks compute

// 2. Create a system (same config shape the editor exports)
const system = createParticleSystem({
  maxParticles: 100000,
  gravity: -9.8,
  emission: { rateOverTime: 50 },
  shape: { shape: Shape.CONE, cone: { angle: 0.2, radius: 0.3 } },
  forceFields: [
    { type: "DIRECTIONAL", direction: { x: 1, y: 0, z: 0 }, strength: 5 },
  ],
  collisionPlanes: [
    { position: { x: 0, y: 5, z: 0 }, normal: { x: 0, y: -1, z: 0 }, mode: "KILL" },
  ],
});
scene.add(system.instance);

// 3. Render loop — dispatch compute, then render
renderer.setAnimationLoop(() => {
  const delta = clock.getDelta();
  system.update({ now: performance.now(), delta, elapsed: clock.elapsedTime });
  if (system.computeNode) renderer.compute(system.computeNode);
  renderer.render(scene, camera);
});

// or drive every created system at once:
updateParticleSystems({ now: performance.now(), delta, elapsed });
```

`updateParticleSystems(cycleData)` and per-system `system.update(cycleData)` take the same `CycleData` object: `{ now, delta, elapsed }` (see `types.ts`). `system.updateConfig(...)` applies on the next compute dispatch; `system.dispose()` releases buffers.

Note on `rendererType`: `POINTS` is the billboard-quad path (a unit quad per particle sampled with a computed point UV — there is no point-sprite mode on the GPU backend because WGSL has no `gl_PointCoord`), `INSTANCED`/`MESH` use `InstancedBufferGeometry`, `TRAIL` fills a GPU history ring (`StorageBufferAttribute`) that the ribbon material reads.

# Usage with Three.js

- **Three.js r186+** (`"three": "^0.186.0"` — the exact peer version pinned in `package.json`) with the WebGPU build (`three/webgpu`).
- A browser with [WebGPU support](https://caniuse.com/webgpu) (Chrome 113+, Edge 113+, Firefox 186+ / Nightly).
- The v4 engine ships a single WebGPU code path (`SimulationBackend.GPU`); config objects are otherwise identical to older 3.x configs.

# Usage with React Three Fiber

As of now we **do not recommend** using react-three-fiber with this engine: with react 19.3 the fiber integration is breaking (tracked upstream in [pmndrs/react-three-fiber#3915](https://github.com/pmndrs/react-three-fiber/issues/3915)). Use **three.js r186+ directly** — `WebGPURenderer` + the loop shown above is all you need:

```javascript
import * as THREE from "three/webgpu";
import { createParticleSystem } from "@cyberluke/three-particles";

const renderer = new THREE.WebGPURenderer({ antialias: true });
await renderer.init();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
// ... system = createParticleSystem(config); scene.add(system.instance);
```

# WebGPU Compute Support

Optional GPU-accelerated particle simulation via Three.js WebGPU renderer and TSL (Three Shading Language). Offloads all per-particle physics and modifiers to GPU compute shaders, enabling **50K-350K+ particles** at interactive frame rates.

The CPU path used to be the bottleneck for real 350K+ particle counts, so the whole pipeline was reworked around TSL compute kernels. Current split (grounded in `src/js/effects/three-particles/webgpu/`):

## What Runs on GPU (TSL compute kernels)

- **Core physics** (`compute-particle-update.ts`): gravity, velocity integration, position update, lifetime tracking
- **All 7 modifiers** (`compute-modifiers.ts`): size/opacity/color over lifetime, rotation, linear velocity, orbital velocity, noise (3D simplex FBM)
- **Force fields** (`compute-force-fields.ts`): point attractors/repulsors and directional forces with falloff — `MAX_FORCE_FIELDS = 16`
- **Collision planes** (`compute-collision-planes.ts`): kill/clamp/bounce evaluated inside the compute pass
- **Curves** (`curve-bake.ts`): baked into `CURVE_RESOLUTION = 256`-sample lookup arrays for fast GPU evaluation (<0.4% error)

The same kernel also fills the `TRAIL` history ring, ping-pong sub-emitter birth/death buffers and the free-list allocator — all as `StorageBufferAttribute`s read directly by the materials.

## What Stays on CPU

- **Emission** — particle activation, burst scheduling, rate-over-distance
- **Sub-emitters** — birth/death trigger spawning
- **Configuration changes** — `updateConfig()` applies on the next frame
- **Trail renderer** — TRAIL type always uses CPU simulation (other renderer types work with GPU)

# Important Notes

## Color Conventions

All RGB values in particle configs (`startColor`, `backgroundColor`) are
**sRGB** — the same convention used everywhere else in three.js. Pass the
value a color picker gives you (e.g. `{ r: 1, g: 0, b: 0 }` for pure red)
and the renderer will display it correctly.

Internally the library decodes these to linear for shader math and relies
on the renderer's standard output pass to convert back to sRGB on the way
to the framebuffer. No special `outputColorSpace` setup is required; the
three.js default (`SRGBColorSpace`) works.

User-supplied color map textures should also be tagged as sRGB
(`texture.colorSpace = THREE.SRGBColorSpace`) — this is also the
three.js default for color textures loaded via `TextureLoader`.

## Color Over Lifetime

The `colorOverLifetime` feature uses a **multiplier-based approach** (similar to Unity's particle system), where each RGB channel curve acts as a multiplier applied to the particle's `startColor`.

**Formula:** `finalColor = startColor * colorOverLifetime`

> [!IMPORTANT]
> To achieve full color transitions, set `startColor` to white `{ r: 1, g: 1, b: 1 }`. If any channel in `startColor` is set to 0, that channel cannot be modified by `colorOverLifetime`.

**Example - Rainbow effect:**
```javascript
{
  startColor: {
    min: { r: 1, g: 1, b: 1 },  // White - allows full color range
    max: { r: 1, g: 1, b: 1 }
  },
  colorOverLifetime: {
    isActive: true,
    r: {  // Red: full → half → off
      type: 'BEZIER',
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 0.5, y: 0.5, percentage: 0.5 },
        { x: 1, y: 0, percentage: 1 }
      ]
    },
    g: {  // Green: off → full → off
      type: 'BEZIER',
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 0.5, y: 1, percentage: 0.5 },
        { x: 1, y: 0, percentage: 1 }
      ]
    },
    b: {  // Blue: off → half → full
      type: 'BEZIER',
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 0.5, y: 0.5, percentage: 0.5 },
        { x: 1, y: 1, percentage: 1 }
      ]
    }
  }
}
```

## Documentation

Full API types ship with the package: `dist/index.d.ts` and `webgpu.d.ts`. A machine-readable overview is included as `llms.txt` / `llms-full.txt` in the package.
