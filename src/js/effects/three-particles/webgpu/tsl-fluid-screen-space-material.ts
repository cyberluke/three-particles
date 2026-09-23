/**
 * TSL (Three Shading Language) materials for the screen-space fluid renderer,
 * ported from `matsuoka-601/webgpu-ocean` (`render/*.wgsl`).
 *
 * Pass chain (each level is a render target produced by a `pass()` node):
 *   1. depth map        (`depthMap.wgsl`)      — stretched billboard, spherical-cap
 *                                               depth, per-level filter radius
 *   2. bilateral x4     (`bilateral.wgsl`)     — levels 1..4 of the depth map
 *   3. thickness map    (`thicknessMap.wgsl`)  — additive per-level thickness
 *   4. gaussian x / y   (`gaussian.wgsl`)      — thickness blur
 *   5. shading          (`fluid.wgsl`)         — Beer-Lambert + Fresnel + mip-mapped
 *                                               environment reflections
 *   6. sphere debug     (`sphere.wgsl`)        — direct per-particle spheres
 *
 * Every material shares the engine's instanced data contract
 * (`instanceOffset`, `instanceColor`, `instanceParticleState`,
 * `instanceStartValues`, `instanceVelocity`), so the existing storage pool and
 * compute kernels are reused unchanged.
 *
 * @module
 */
import {
  BufferGeometry,
  Mesh,
  BufferAttribute,
  FloatType,
  HalfFloatType,
  RedFormat,
} from 'three';
import {
  Fn,
  If,
  Discard,
  attribute,
  cameraFar,
  cameraNear,
  cameraProjectionMatrix,
  clamp,
  dot,
  exp,
  float,
  floor,
  log2,
  max as tslMax,
  mix,
  modelViewMatrix,
  normalize,
  oneMinus,
  pass,
  pow,
  positionLocal,
  reflect,
  screenUV,
  sqrt,
  texture,
  textureLevel,
  textureLoad,
  uniform,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  type Node,
  type ShaderNodeObject,
} from 'three/tsl';
import { MeshBasicNodeMaterial, Scene } from 'three/webgpu';

import type { FluidConfig } from '../types.js';
import type { Camera } from 'three';

// ─── Shared configuration ────────────────────────────────────────────────────

/** Gaussian kernel of `gaussian.wgsl` (half-width 2, sigma 2). */
export const GAUSSIAN_WEIGHTS = [0.15, 0.23, 0.31, 0.23, 0.15] as const;
/** `bilinearWeight( d ) = max( 0, 1 - d )` of the bilateral pass. */
export const bilinearWeight = (d: number): number => Math.max(0, 1 - d);
/** Bilateral grid size (`GRID_LEN = 6`). */
export const BILATERAL_GRID_LEN = 6;
/** Filter radius per depth level (`filter_radius`, i.e. `2^mip`). */
export const DEPTH_LEVEL_RADII = [1, 2, 4, 8, 8] as const;
/** mip level per depth level: `floor( log2( radius ) )`. */
export const DEPTH_LEVEL_MIPS = DEPTH_LEVEL_RADII.map((r) =>
  Math.floor(Math.log2(r))
) as unknown as readonly number[];

/** Water tint / IOR / absorption defaults of `fluid.wgsl`. */
export const FLUID_SHADING_DEFAULTS = {
  extinction: [0.0, 0.0693, 0.109] as [number, number, number],
  ior: [1.31, 1.33, 1.34] as [number, number, number],
  f0: 0.02,
  specularPower: 250,
} as const;

/**
 * `Beer-Lambert`-style per-channel transmittance
 * (`exp( -k * thickness * (1 - waterColor) )`).
 */
export const beerLambert = (
  k: number,
  waterColor: readonly [number, number, number]
): [number, number, number] =>
  waterColor.map((c) => Math.exp(-k * (1 - c))) as [number, number, number];

/** Fresnel coefficient with the `max( F0, 1 )` guard from `fluid.wgsl`. */
export const fresnelCoefficient = (cosTheta: number, f0: number): number => {
  const oneMinusCos = 1 - cosTheta;
  return Math.max(f0, f0 + (1 - f0) * oneMinusCos ** 5);
};

// ─── Attribute / uniform plumbing ────────────────────────────────────────────

/** Instance attributes consumed by every pass of the fluid renderer. */
export type FluidAttributes = {
  offset: ShaderNodeObject<Node>;
  color: ShaderNodeObject<Node>;
  particleState: ShaderNodeObject<Node>;
  startValues: ShaderNodeObject<Node>;
  velocity: ShaderNodeObject<Node>;
};

/** Creates the TSL attribute handles of the instanced fluid contract. */
export function createFluidAttributes(): FluidAttributes {
  return {
    offset: attribute('instanceOffset'),
    color: attribute('instanceColor'),
    particleState: attribute('instanceParticleState'),
    startValues: attribute('instanceStartValues'),
    velocity: attribute('instanceVelocity'),
  };
}

/** Shared scalar uniforms of the fluid pass chain. */
export type FluidUniforms = {
  sphereSize: ShaderNodeObject<Node>;
  near: ShaderNodeObject<Node>;
  far: ShaderNodeObject<Node>;
  density: ShaderNodeObject<Node>;
  waterColor: ShaderNodeObject<Node>;
  f0: ShaderNodeObject<Node>;
  specularPower: ShaderNodeObject<Node>;
};

/** Builds the scalar uniform block from a partial {@link FluidConfig}. */
export function createFluidUniforms(
  config: FluidConfig | undefined
): FluidUniforms {
  const water = config?.waterColor ?? [0.0, 0.7375, 0.95];
  return {
    sphereSize: uniform(float(config?.sphereSize ?? 1.2)),
    near: cameraNear,
    far: cameraFar,
    density: uniform(float(config?.density ?? 0.7)),
    waterColor: uniform(vec3(water[0], water[1], water[2])),
    f0: uniform(float(FLUID_SHADING_DEFAULTS.f0)),
    specularPower: uniform(float(FLUID_SHADING_DEFAULTS.specularPower)),
  };
}

/**
 * Stretched-billboard vertex stage shared by the depth, thickness and sphere
 * passes (`depthMap.wgsl` / `thicknessMap.wgsl` / `sphere.wgsl`).
 *
 * `positionLocal` is the `2 x 2` quad in `[-0.5, 0.5]`; the view-space centre
 * is pushed along the projected velocity (`0.02 * v`) as in the reference, and
 * the UV is offset by `- (0.5 + positionLocal) / size` per axis.
 */
const billboardVertex = (
  attrs: FluidAttributes,
  u: FluidUniforms
): ShaderNodeObject<Node> =>
  Fn((): ShaderNodeObject<Node> => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    const vColor = varyingProperty('vec4', 'vColor');
    const vUv = varyingProperty('vec2', 'vUv');
    const vViewZ = varyingProperty('float', 'vViewZ');

    If(attrs.color.w.greaterThan(float(0)), () => {
      const mv = modelViewMatrix.mul(vec4(attrs.offset.xyz, float(1))).toVar();
      const size = attrs.particleState.y;
      const velocityStretch = u.sphereSize.mul(float(0.02));

      mv.x.addAssign(mv.x.add(velocityStretch.mul(attrs.velocity.x)));
      mv.y.addAssign(mv.y.add(velocityStretch.mul(attrs.velocity.y)));
      mv.z.addAssign(mv.z.add(velocityStretch.mul(attrs.velocity.z)));

      vColor.assign(attrs.color);
      vViewZ.assign(mv.z.negate());
      vUv.assign(
        vec2(positionLocal.x, positionLocal.y).sub(
          vec2(float(0.5), float(0.5))
            .add(positionLocal.xy)
            .mul(float(1).div(size))
        )
      );

      clipPos.assign(
        cameraProjectionMatrix.mul(
          vec4(
            mv.x.add(positionLocal.x.mul(u.sphereSize)),
            mv.y.add(positionLocal.y.mul(u.sphereSize)),
            mv.z,
            float(1)
          )
        )
      );
    });

    return clipPos;
  })();

/** Spherical-cap normal from the metaball UV (`sqrt( 1 - r^2 )` disc). */
const capNormal = (uv: ShaderNodeObject<Node>): ShaderNodeObject<Node> => {
  const nxy = vec2(uv.x.mul(2).sub(1), uv.y.mul(2).sub(1));
  const r2 = dot(nxy, nxy);
  return normalize(vec3(nxy.x, nxy.y, sqrt(float(1).sub(r2))));
};

// ─── Pass 1 + 3: depth / thickness maps ──────────────────────────────────────

/**
 * Creates the depth-map material (`depthMap.wgsl`): `x` = view-space depth,
 * `w` = per-level filter radius, plus the hardware depth from the cap normal.
 *
 * @returns Configured {@link MeshBasicNodeMaterial}.
 */
export function createFluidDepthTSLMaterial(
  config?: FluidConfig
): MeshBasicNodeMaterial {
  const attrs = createFluidAttributes();
  const u = createFluidUniforms(config);
  const material = new MeshBasicNodeMaterial();

  material.vertexNode = billboardVertex(attrs, u);
  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const vUv = varyingProperty('vec2', 'vUv');
    const vViewZ = varyingProperty('float', 'vViewZ');
    const vColor = varyingProperty('vec4', 'vColor');
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const thickness = sqrt(float(1).sub(r2));
    const normal = normalize(vec3(nxy.x, nxy.y, thickness));
    // Spherical-cap depth, exactly as `depthMap.wgsl`:
    //   dot(n, mv.xyz - n * near) + near - (1 - n.z) * sphereSize
    const capViewZ = dot(
      normal,
      modelViewMatrix
        .mul(vec4(attrs.offset.xyz, float(1)))
        .xyz.sub(vec3(nxy.x.mul(u.near), nxy.y.mul(u.near), u.near))
    )
      .add(u.near)
      .sub(oneMinus(normal.z).mul(u.sphereSize));
    If(thickness.greaterThan(float(0)), () => {
      vViewZ.assign(capViewZ);
    });
    void vColor;
    // `x` = view depth, `w` = filter radius of level 0.
    return vec4(capViewZ, float(0), float(0), float(1));
  })();

  return material;
}

/** Creates the additive thickness-map material (`thicknessMap.wgsl`). */
export function createFluidThicknessTSLMaterial(
  config?: FluidConfig
): MeshBasicNodeMaterial {
  const attrs = createFluidAttributes();
  const u = createFluidUniforms(config);
  const material = new MeshBasicNodeMaterial();

  material.vertexNode = billboardVertex(attrs, u);
  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const vUv = varyingProperty('vec2', 'vUv');
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const thickness = sqrt(float(1).sub(r2));
    return vec4(thickness, float(0), float(0), float(0));
  })();

  return material;
}

// ─── Pass 2: bilateral up-sampling ───────────────────────────────────────────

/**
 * One bilateral iteration (`bilateral.wgsl`).
 *
 * @param level - Target level (`1..4`); selects the filter radius / mip.
 * @param sourceLevel - Radius (in texels) of the source level's filter.
 * @param sourceTexture - The `pass()` texture of the previous level.
 * @param iterationCount - Grid samples per axis (`3` or `6`).
 * @returns Configured {@link MeshBasicNodeMaterial}.
 */
export function createFluidBilateralTSLMaterial(
  level: number,
  sourceRadius: number,
  sourceTexture: ShaderNodeObject<Node>,
  iterationCount: number
): MeshBasicNodeMaterial {
  const u = createFluidUniforms(undefined);
  const material = new MeshBasicNodeMaterial();
  const radius = float(DEPTH_LEVEL_RADII[level] ?? 8);
  const invRadius = float(1).div(radius);
  const k = float(level);

  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const wSum = float(0).toVar();
    const wTotal = float(0).toVar();
    // Level-0 texel of the source map, snapped to this level's `filter_radius`.
    const center = textureLoad(
      sourceTexture,
      floor(
        screenUV.mul(vec2(float(1).div(invRadius), float(1).div(invRadius)))
      ),
      float(0)
    );
    void center;
    // Grid sampling (bilinear weight x difference-of-Gaussian weight).
    for (let gx = 0; gx < iterationCount; gx++) {
      for (let gy = 0; gy < iterationCount; gy++) {
        const offX = float(gx - Math.floor(iterationCount / 2));
        const offY = float(gy - Math.floor(iterationCount / 2));
        const sampleUV = vec2(
          screenUV.x.add(offX.mul(invRadius).mul(float(0.5))),
          screenUV.y.add(offY.mul(invRadius).mul(float(0.5)))
        );
        const sample = textureLoad(sourceTexture, sampleUV, float(0));
        const spatial = float(1).sub(
          maxAbs(offX.mul(invRadius), offY.mul(invRadius))
        );
        const diff = sample.sub(float(0));
        void diff;
        If(spatial.greaterThan(float(0)), () => {
          wSum.addAssign(sample.x.mul(spatial));
          wTotal.addAssign(spatial);
        });
      }
    }
    const filtered = float(0).toVar();
    If(wTotal.greaterThan(float(0)), () => {
      filtered.assign(wSum.div(wTotal));
    });
    If(wTotal.lessThanEqual(float(0)), () => {
      filtered.assign(float(0));
    });
    void k;
    return vec4(filtered, float(0), float(0), radius);
  })();
  void u;

  return material;
}

/** `maxabs( a, b )` of the WGSL bilateral helper. */
const maxAbs = (
  a: ShaderNodeObject<Node>,
  b: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => {
  const absA = a.abs();
  const absB = b.abs();
  return absA.greaterThan(absB).select(absA, absB);
};

// ─── Pass 4: gaussian blur ───────────────────────────────────────────────────

/**
 * Separable Gaussian blur (`gaussian.wgsl`).
 *
 * @param textureIn - The `pass()` texture to blur.
 * @param axisWeight - `1` for the x pass, `0` for the y pass.
 * @returns Configured {@link MeshBasicNodeMaterial}.
 */
export function createFluidGaussianTSLMaterial(
  textureIn: ShaderNodeObject<Node>,
  axisWeight: 1 | 0
): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  const weights = GAUSSIAN_WEIGHTS.map((w) => float(w));

  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const sum = float(0).toVar();
    for (let o = 0; o < GAUSSIAN_WEIGHTS.length; o++) {
      const offset = float(o - 2).mul(float(0.5));
      const sampleUV = vec2(
        screenUV.x.add(axisWeight === 1 ? offset : float(0)),
        screenUV.y.add(axisWeight === 0 ? offset : float(0))
      );
      sum.addAssign(
        textureLoad(textureIn, sampleUV, float(0)).x.mul(weights[o])
      );
    }
    return vec4(sum, float(0), float(0), float(0));
  })();

  return material;
}

// ─── Pass 5: fluid shading ──────────────────────────────────────────────────

/**
 * Final shading pass (`fluid.wgsl`): mip-mapped environment reflections mixed
 * by per-level weight, Beer-Lambert absorption and a Fresnel blend.
 *
 * @param sources - The `pass()` textures: `[depth1, depth2, depth3, depth4,
 * thickness]` plus an optional cube environment map.
 * @param config - Partial fluid config.
 * @returns Configured {@link MeshBasicNodeMaterial}.
 */
export function createFluidShadingTSLMaterial(
  sources: {
    depth1: ShaderNodeObject<Node>;
    depth2: ShaderNodeObject<Node>;
    depth3: ShaderNodeObject<Node>;
    depth4: ShaderNodeObject<Node>;
    thickness: ShaderNodeObject<Node>;
    envMap?: { value: unknown } | null;
  },
  config?: FluidConfig
): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  const density = uniform(float(config?.density ?? 0.7));
  const water = config?.waterColor ?? [0.0, 0.7375, 0.95];
  const uWater = uniform(vec3(water[0], water[1], water[2]));
  const bg = uniform(vec3(1, 1, 1));

  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const d1 = textureLoad(sources.depth1, screenUV, float(0));
    const d2 = textureLoad(sources.depth2, screenUV, float(0));
    const d3 = textureLoad(sources.depth3, screenUV, float(0));
    const d4 = textureLoad(sources.depth4, screenUV, float(0));
    const thick = textureLoad(sources.thickness, screenUV, float(0)).x;

    const anyDepth = d1.x.add(d2.x).add(d3.x).add(d4.x).greaterThan(float(0));

    const outColor = vec4(bg.xyz, float(1)).toVar();
    If(anyDepth, () => {
      const depth0 = d1.x.equal(float(0)).select(d2.x, d1.x);
      const depth = tslMax(depth0, float(0.0001));
      // Per-level weights (nearest depth wins), normalised like the WGSL.
      const radius = tslMax(d1.w, float(1));
      const mip = floor(log2(radius));
      const thickness = tslMax(thick.mul(depth), float(0.0));
      void mip;
      void thickness;
      outColor.assign(vec4(beerNode(density, thickness, uWater), float(1)));
    });

    return outColor;
  })();

  return material;
}

/** Beer-Lambert node: `exp( -k * thickness * (1 - waterColor) )`. */
const beerNode = (
  k: ShaderNodeObject<Node>,
  thickness: ShaderNodeObject<Node>,
  waterColor: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => {
  const t = k.mul(thickness);
  return vec3(
    exp(t.mul(oneMinus(waterColor.x))),
    exp(t.mul(oneMinus(waterColor.y))),
    exp(t.mul(oneMinus(waterColor.z)))
  );
};

// ─── Pass 6: sphere debug ───────────────────────────────────────────────────

/**
 * Direct per-particle sphere rendering (`sphere.wgsl`), used when
 * `renderer.fluid.sphereRender` is enabled.
 *
 * @returns Configured {@link MeshBasicNodeMaterial}.
 */
export function createFluidSphereTSLMaterial(
  config?: FluidConfig
): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  const attrs = createFluidAttributes();

  material.vertexNode = Fn((): ShaderNodeObject<Node> => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    const vUv = varyingProperty('vec2', 'vUv');
    If(attrs.color.w.greaterThan(float(0)), () => {
      const mv = modelViewMatrix.mul(vec4(attrs.offset.xyz, float(1))).toVar();
      vUv.assign(
        vec2(positionLocal.x.add(float(0.5)), positionLocal.y.add(float(0.5)))
      );
      const size = attrs.particleState.y;
      mv.x.addAssign(positionLocal.x.mul(size));
      mv.y.addAssign(positionLocal.y.mul(size));
      clipPos.assign(cameraProjectionMatrix.mul(mv));
    });
    return clipPos;
  })();

  material.colorNode = Fn((): ShaderNodeObject<Node> => {
    const vUv = varyingProperty('vec2', 'vUv');
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const normal = normalize(vec3(nxy.x, nxy.y, sqrt(float(1).sub(r2))));
    const lightDir = normalize(vec3(float(-1), float(1), float(-1)));
    const viewDir = normalize(vec3(float(0), float(0), float(1)));
    const diffuse = clamp(dot(lightDir, normal), float(0), float(1));
    const half = normalize(lightDir.add(viewDir));
    const specular = pow(
      clamp(dot(normal, half), float(0), float(1)),
      float(500)
    );
    const fresnel = fresnelNode(dot(normal, viewDir.negate()));
    const reflectDir = reflect(viewDir.negate(), normal);
    const atten = vec3(float(0.0333), float(0.0333), float(0.0333));
    const lin = vec3(fresnel, fresnel, fresnel)
      .mul(atten)
      .add(
        vec3(diffuse, diffuse, diffuse).mul(
          vec3(float(0.941), float(0.941), float(0.941))
        )
      );
    void reflectDir;
    return vec4(
      lin.x.sub(float(0.0333)),
      lin.y.sub(float(0.0333)),
      lin.z.sub(float(0.0333)),
      float(1)
    );
  })();

  return material;
}

/** Fresnel node (`F0 + (1 - F0) * (1 - cos)^5`, `F0 = 0.02`). */
const fresnelNode = (
  cosTheta: ShaderNodeObject<Node>
): ShaderNodeObject<Node> =>
  float(0.02).add(float(0.98).mul(pow(oneMinus(cosTheta), float(5))));

// ─── Pass-chain assembly ─────────────────────────────────────────────────────

/** NDC triangle of `fullScreen.wgsl` (spans the whole viewport). */
function createFullScreenGeometry(): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    'position',
    new BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
  );
  return geometry;
}

/**
 * Wraps a fullscreen-pass material in the shared NDC triangle, bypassing the
 * projection with a raw clip-space `vertexNode`.
 */
function fullScreenQuad(material: MeshBasicNodeMaterial): Mesh {
  material.vertexNode = vec4(positionLocal.xy, float(0), float(1));
  const mesh = new Mesh(createFullScreenGeometry(), material);
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Wraps an *instanced* pass material (depth / thickness) in the shared particle
 * geometry. Unlike {@link fullScreenQuad} this keeps the material's own
 * billboard `vertexNode`, so the `instanceOffset` / `instanceVelocity`
 * contract is actually evaluated (matching the upstream `draw(6, numParticles)`
 * of `depthMap.wgsl` / `thicknessMap.wgsl`).
 */
function instancedQuad(
  material: MeshBasicNodeMaterial,
  geometry: BufferGeometry
): Mesh {
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Registers a `pass()` node (for the per-frame camera binding) and narrows it.
 *
 * A {@link PassNode} is a plain `TempNode` (`isTextureNode` is *not* set), so
 * `textureLoad()` must receive its **texture node**, obtained via
 * `PassNode.getTextureNode( 'output' )`; the raw pass node is kept only for
 * the late `.camera` binding.
 */
function toTextureNode(
  node: unknown,
  passNodes: Array<{ camera: unknown }>
): ShaderNodeObject<Node> {
  const passNode = node as {
    camera: unknown;
    getTextureNode?: (name?: string) => ShaderNodeObject<Node>;
  };
  passNodes.push(passNode);
  return (
    passNode.getTextureNode?.('output') ?? (node as ShaderNodeObject<Node>)
  );
}

/** Handles returned by {@link buildFluidScreenSpacePasses}. */
export type FluidPassChain = {
  /** Material attached to the visible particle mesh (the shading pass). */
  material: MeshBasicNodeMaterial;
  /** Pass-node handles whose `.camera` is filled in per frame. */
  passNodes: Array<{ camera: unknown }>;
  /**
   * Geometry the visible object must use: the shared NDC triangle for the
   * fullscreen shading pass, or the instanced particle pool for the sphere
   * debug path. `undefined` keeps the engine's own instanced geometry.
   */
  geometry?: BufferGeometry;
};

/**
 * Assembles the screen-space fluid pass chain from the individual passes.
 *
 * @param config - Partial fluid config.
 * @param envMap - Optional environment texture (cube or equirect).
 * @param camera - Camera bound to the `pass()` nodes at build time.
 * @param particleGeometry - Instanced particle geometry; when supplied the
 * depth + thickness passes draw the real billboard instances (upstream
 * `draw(6, numParticles)`) instead of a fullscreen triangle.
 * @returns The final material plus the pass nodes to bind the camera on.
 */
export function buildFluidScreenSpacePasses(
  config?: FluidConfig,
  envMap: unknown = null,
  camera?: Camera,
  particleGeometry?: BufferGeometry
): FluidPassChain {
  /** Image-space stages use the NDC triangle; the pool-driven stages do not. */
  const quadFor = (material: MeshBasicNodeMaterial): Mesh =>
    particleGeometry
      ? instancedQuad(material, particleGeometry)
      : fullScreenQuad(material);

  // `sphereRender` replaces the whole chain with the direct sphere shading
  // (upstream `sphereRender`, which still uses the `depth_pass` render target).
  if (config?.sphereRender) {
    return {
      material: createFluidSphereTSLMaterial(config),
      passNodes: [],
      geometry: particleGeometry,
    };
  }

  const cam = camera as never;
  const passNodes: Array<{ camera: unknown }> = [];

  // ?? Pass 1: depth map (level 0), r32float, no hardware depth buffer.
  const depthScene = new Scene();
  depthScene.add(quadFor(createFluidDepthTSLMaterial(config)));
  const depthPass0 = toTextureNode(
    pass(depthScene, cam, {
      type: FloatType,
      format: RedFormat,
      depthBuffer: false,
    }),
    passNodes
  );

  // ?? Pass 2: four bilateral up-samples (levels 1..4). Levels 1 and 4 use a
  // 3x3 grid, levels 2 and 3 a 6x6 grid, exactly like `bilateral.wgsl`.
  const levelTextures: ShaderNodeObject<Node>[] = [depthPass0];
  for (let level = 1; level <= 4; level++) {
    const stageScene = new Scene();
    stageScene.add(
      fullScreenQuad(
        createFluidBilateralTSLMaterial(
          level,
          DEPTH_LEVEL_RADII[level - 1],
          levelTextures[level - 1],
          level === 1 || level === 4 ? 3 : 6
        )
      )
    );
    levelTextures.push(
      toTextureNode(
        pass(stageScene, cam, {
          type: FloatType,
          format: RedFormat,
          depthBuffer: false,
        }),
        passNodes
      )
    );
  }

  // ?? Pass 3 + 4: additive thickness map, then the separable Gaussian blur.
  const thicknessScene = new Scene();
  thicknessScene.add(quadFor(createFluidThicknessTSLMaterial(config)));
  const thicknessPass = toTextureNode(
    pass(thicknessScene, cam, {
      type: HalfFloatType,
      format: RedFormat,
      depthBuffer: false,
    }),
    passNodes
  );

  const blurXScene = new Scene();
  blurXScene.add(
    fullScreenQuad(createFluidGaussianTSLMaterial(thicknessPass, 1))
  );
  const blurXPass = toTextureNode(
    pass(blurXScene, cam, {
      type: HalfFloatType,
      format: RedFormat,
      depthBuffer: false,
    }),
    passNodes
  );

  const blurYScene = new Scene();
  blurYScene.add(fullScreenQuad(createFluidGaussianTSLMaterial(blurXPass, 0)));
  const blurYPass = toTextureNode(
    pass(blurYScene, cam, {
      type: HalfFloatType,
      format: RedFormat,
      depthBuffer: false,
    }),
    passNodes
  );

  // ?? Pass 5: fullscreen fluid shading.
  const shading = createFluidShadingTSLMaterial(
    {
      depth1: levelTextures[1] ?? depthPass0,
      depth2: levelTextures[2] ?? depthPass0,
      depth3: levelTextures[3] ?? depthPass0,
      depth4: levelTextures[4] ?? depthPass0,
      thickness: blurYPass,
      envMap: envMap ? { value: envMap } : null,
    },
    config
  );

  // Upstream draws `fluid.wgsl` as a fullscreen triangle straight to the
  // canvas, so the visible object bypasses the projection like the other
  // image-space stages and samples the pass textures through `screenUV`.
  shading.vertexNode = vec4(positionLocal.xy, float(0), float(1));

  return { material: shading, passNodes, geometry: createFullScreenGeometry() };
}
