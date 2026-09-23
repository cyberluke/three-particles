import { ObjectUtils } from '@newkrok/three-utils';
import * as THREE2 from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, mod, float, floor, dot, vec3, step, min, max, vec4, vec2, abs, uint, round, If, texture, screenUV, smoothstep, cross, length, cameraViewMatrix, normalize, mix, uniform, storage, compute, instanceIndex, atomicStore, atomicAdd, invocationLocalIndex, workgroupArray, workgroupBarrier, add, sub, pow, atomicLoad, Loop, sqrt } from 'three/tsl';

// src/js/effects/three-particles/three-particles.ts

// src/js/effects/three-particles/three-particles-bezier.ts
var cache = [];
var nCr = (n, k) => {
  let z = 1;
  for (let i = 1; i <= k; i++) z *= (n + 1 - i) / i;
  return z;
};
var createBezierCurveFunction = (particleSystemId, bezierPoints) => {
  const cacheEntry = cache.find((item) => item.bezierPoints === bezierPoints);
  if (cacheEntry) {
    if (!cacheEntry.referencedBy.includes(particleSystemId))
      cacheEntry.referencedBy.push(particleSystemId);
    return cacheEntry.curveFunction;
  }
  const entry = {
    referencedBy: [particleSystemId],
    bezierPoints,
    curveFunction: (percentage) => {
      if (percentage < 0) return bezierPoints[0].y;
      if (percentage > 1) return bezierPoints[bezierPoints.length - 1].y;
      let start = 0;
      let stop = bezierPoints.length - 1;
      for (let i = 0; i < bezierPoints.length; i++) {
        const point = bezierPoints[i];
        if (percentage < (point.percentage ?? 0)) {
          stop = i;
          break;
        }
        if (point.percentage !== void 0) start = i;
      }
      const n = stop - start;
      const calculatedPercentage = (percentage - (bezierPoints[start].percentage ?? 0)) / ((bezierPoints[stop].percentage ?? 1) - (bezierPoints[start].percentage ?? 0));
      let value = 0;
      for (let i = 0; i <= n; i++) {
        const p = bezierPoints[start + i];
        const c = nCr(n, i) * Math.pow(1 - calculatedPercentage, n - i) * Math.pow(calculatedPercentage, i);
        value += c * p.y;
      }
      return value;
    }
  };
  cache.push(entry);
  return entry.curveFunction;
};
var removeBezierCurveFunction = (particleSystemId) => {
  while (true) {
    const index = cache.findIndex(
      (item) => item.referencedBy.includes(particleSystemId)
    );
    if (index === -1) break;
    const entry = cache[index];
    entry.referencedBy = entry.referencedBy.filter(
      (id) => id !== particleSystemId
    );
    if (entry.referencedBy.length === 0) cache.splice(index, 1);
  }
};
var getBezierCacheSize = () => cache.length;

// src/js/effects/three-particles/three-particles-constants.ts
var SCALAR_STRIDE = 10;
var S_IS_ACTIVE = 0;
var S_LIFETIME = 1;
var S_START_LIFETIME = 2;
var S_START_FRAME = 3;
var S_SIZE = 4;
var S_ROTATION = 5;
var S_COLOR_R = 6;
var S_COLOR_G = 7;
var S_COLOR_B = 8;
var S_COLOR_A = 9;

// src/js/effects/three-particles/three-particles-enums.ts
var SimulationSpace = /* @__PURE__ */ ((SimulationSpace2) => {
  SimulationSpace2["LOCAL"] = "LOCAL";
  SimulationSpace2["WORLD"] = "WORLD";
  return SimulationSpace2;
})(SimulationSpace || {});
var Shape = /* @__PURE__ */ ((Shape2) => {
  Shape2["SPHERE"] = "SPHERE";
  Shape2["CONE"] = "CONE";
  Shape2["BOX"] = "BOX";
  Shape2["CIRCLE"] = "CIRCLE";
  Shape2["RECTANGLE"] = "RECTANGLE";
  return Shape2;
})(Shape || {});
var EmitFrom = /* @__PURE__ */ ((EmitFrom2) => {
  EmitFrom2["VOLUME"] = "VOLUME";
  EmitFrom2["SHELL"] = "SHELL";
  EmitFrom2["EDGE"] = "EDGE";
  return EmitFrom2;
})(EmitFrom || {});
var TimeMode = /* @__PURE__ */ ((TimeMode2) => {
  TimeMode2["LIFETIME"] = "LIFETIME";
  TimeMode2["FPS"] = "FPS";
  return TimeMode2;
})(TimeMode || {});
var LifeTimeCurve = /* @__PURE__ */ ((LifeTimeCurve2) => {
  LifeTimeCurve2["BEZIER"] = "BEZIER";
  LifeTimeCurve2["EASING"] = "EASING";
  return LifeTimeCurve2;
})(LifeTimeCurve || {});
var SubEmitterTrigger = /* @__PURE__ */ ((SubEmitterTrigger3) => {
  SubEmitterTrigger3["BIRTH"] = "BIRTH";
  SubEmitterTrigger3["DEATH"] = "DEATH";
  return SubEmitterTrigger3;
})(SubEmitterTrigger || {});
var ForceFieldType = /* @__PURE__ */ ((ForceFieldType2) => {
  ForceFieldType2["POINT"] = "POINT";
  ForceFieldType2["DIRECTIONAL"] = "DIRECTIONAL";
  return ForceFieldType2;
})(ForceFieldType || {});
var RendererType = /* @__PURE__ */ ((RendererType2) => {
  RendererType2["POINTS"] = "POINTS";
  RendererType2["INSTANCED"] = "INSTANCED";
  RendererType2["TRAIL"] = "TRAIL";
  RendererType2["MESH"] = "MESH";
  RendererType2["FLUID"] = "FLUID";
  return RendererType2;
})(RendererType || {});
var ForceFieldFalloff = /* @__PURE__ */ ((ForceFieldFalloff2) => {
  ForceFieldFalloff2["NONE"] = "NONE";
  ForceFieldFalloff2["LINEAR"] = "LINEAR";
  ForceFieldFalloff2["QUADRATIC"] = "QUADRATIC";
  return ForceFieldFalloff2;
})(ForceFieldFalloff || {});
var CollisionPlaneMode = /* @__PURE__ */ ((CollisionPlaneMode2) => {
  CollisionPlaneMode2["KILL"] = "KILL";
  CollisionPlaneMode2["CLAMP"] = "CLAMP";
  CollisionPlaneMode2["BOUNCE"] = "BOUNCE";
  return CollisionPlaneMode2;
})(CollisionPlaneMode || {});
var SimulationBackend = /* @__PURE__ */ ((SimulationBackend2) => {
  SimulationBackend2["AUTO"] = "AUTO";
  SimulationBackend2["CPU"] = "CPU";
  SimulationBackend2["GPU"] = "GPU";
  return SimulationBackend2;
})(SimulationBackend || {});

// src/js/effects/three-particles/three-particles-renderer-detect.ts
function isComputeCapableRenderer(renderer) {
  return renderer !== null && renderer !== void 0 && typeof renderer === "object" && "compute" in renderer && typeof renderer.compute === "function" && "hasFeature" in renderer && typeof renderer.hasFeature === "function";
}
function resolveSimulationBackend(renderer, preference = "AUTO" /* AUTO */) {
  const gpuCapable = isComputeCapableRenderer(renderer);
  if (preference === "CPU" /* CPU */) {
    return "CPU" /* CPU */;
  }
  if (preference === "GPU" /* GPU */) {
    return gpuCapable ? "GPU" /* GPU */ : "CPU" /* CPU */;
  }
  return gpuCapable ? "GPU" /* GPU */ : "CPU" /* CPU */;
}
var calculateRandomPositionAndVelocityOnSphere = (position, quaternion, velocity, speed, {
  radius,
  radiusThickness,
  arc
}) => {
  const u = Math.random() * (arc / 360);
  const v = Math.random();
  const randomizedDistanceRatio = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  const sinPhi = Math.sin(phi);
  const xDirection = sinPhi * Math.cos(theta);
  const yDirection = sinPhi * Math.sin(theta);
  const zDirection = Math.cos(phi);
  const normalizedThickness = 1 - radiusThickness;
  position.x = radius * normalizedThickness * xDirection + radius * radiusThickness * randomizedDistanceRatio * xDirection;
  position.y = radius * normalizedThickness * yDirection + radius * radiusThickness * randomizedDistanceRatio * yDirection;
  position.z = radius * normalizedThickness * zDirection + radius * radiusThickness * randomizedDistanceRatio * zDirection;
  position.applyQuaternion(quaternion);
  const speedMultiplierByPosition = 1 / position.length();
  velocity.set(
    position.x * speedMultiplierByPosition * speed,
    position.y * speedMultiplierByPosition * speed,
    position.z * speedMultiplierByPosition * speed
  );
  velocity.applyQuaternion(quaternion);
};
var calculateRandomPositionAndVelocityOnCone = (position, quaternion, velocity, speed, {
  radius,
  radiusThickness,
  arc,
  angle = 90
}) => {
  const theta = 2 * Math.PI * Math.random() * (arc / 360);
  const randomizedDistanceRatio = Math.random();
  const xDirection = Math.cos(theta);
  const yDirection = Math.sin(theta);
  const normalizedThickness = 1 - radiusThickness;
  position.x = radius * normalizedThickness * xDirection + radius * radiusThickness * randomizedDistanceRatio * xDirection;
  position.y = radius * normalizedThickness * yDirection + radius * radiusThickness * randomizedDistanceRatio * yDirection;
  position.z = 0;
  position.applyQuaternion(quaternion);
  const positionLength = position.length();
  const normalizedAngle = Math.abs(
    positionLength / radius * THREE2.MathUtils.degToRad(angle)
  );
  const sinNormalizedAngle = Math.sin(normalizedAngle);
  const speedMultiplierByPosition = 1 / positionLength;
  velocity.set(
    position.x * sinNormalizedAngle * speedMultiplierByPosition * speed,
    position.y * sinNormalizedAngle * speedMultiplierByPosition * speed,
    Math.cos(normalizedAngle) * speed
  );
  velocity.applyQuaternion(quaternion);
};
var calculateRandomPositionAndVelocityOnBox = (position, quaternion, velocity, speed, { scale, emitFrom }) => {
  const _scale = scale;
  switch (emitFrom) {
    case "VOLUME" /* VOLUME */:
      position.x = Math.random() * _scale.x - _scale.x / 2;
      position.y = Math.random() * _scale.y - _scale.y / 2;
      position.z = Math.random() * _scale.z - _scale.z / 2;
      break;
    case "SHELL" /* SHELL */:
      const side = Math.floor(Math.random() * 6);
      const perpendicularAxis = side % 3;
      const shellResult = [];
      shellResult[perpendicularAxis] = side > 2 ? 1 : 0;
      shellResult[(perpendicularAxis + 1) % 3] = Math.random();
      shellResult[(perpendicularAxis + 2) % 3] = Math.random();
      position.x = shellResult[0] * _scale.x - _scale.x / 2;
      position.y = shellResult[1] * _scale.y - _scale.y / 2;
      position.z = shellResult[2] * _scale.z - _scale.z / 2;
      break;
    case "EDGE" /* EDGE */:
      const side2 = Math.floor(Math.random() * 6);
      const perpendicularAxis2 = side2 % 3;
      const edge = Math.floor(Math.random() * 4);
      const edgeResult = [];
      edgeResult[perpendicularAxis2] = side2 > 2 ? 1 : 0;
      edgeResult[(perpendicularAxis2 + 1) % 3] = edge < 2 ? Math.random() : edge - 2;
      edgeResult[(perpendicularAxis2 + 2) % 3] = edge < 2 ? edge : Math.random();
      position.x = edgeResult[0] * _scale.x - _scale.x / 2;
      position.y = edgeResult[1] * _scale.y - _scale.y / 2;
      position.z = edgeResult[2] * _scale.z - _scale.z / 2;
      break;
  }
  position.applyQuaternion(quaternion);
  velocity.set(0, 0, speed);
  velocity.applyQuaternion(quaternion);
};
var calculateRandomPositionAndVelocityOnCircle = (position, quaternion, velocity, speed, {
  radius,
  radiusThickness,
  arc
}) => {
  const theta = 2 * Math.PI * Math.random() * (arc / 360);
  const randomizedDistanceRatio = Math.random();
  const xDirection = Math.cos(theta);
  const yDirection = Math.sin(theta);
  const normalizedThickness = 1 - radiusThickness;
  position.x = radius * normalizedThickness * xDirection + radius * radiusThickness * randomizedDistanceRatio * xDirection;
  position.y = radius * normalizedThickness * yDirection + radius * radiusThickness * randomizedDistanceRatio * yDirection;
  position.z = 0;
  position.applyQuaternion(quaternion);
  const positionLength = position.length();
  const speedMultiplierByPosition = 1 / positionLength;
  velocity.set(
    position.x * speedMultiplierByPosition * speed,
    position.y * speedMultiplierByPosition * speed,
    0
  );
  velocity.applyQuaternion(quaternion);
};
var calculateRandomPositionAndVelocityOnRectangle = (position, quaternion, velocity, speed, { rotation, scale }) => {
  const _scale = scale;
  const _rotation = rotation;
  const xOffset = Math.random() * _scale.x - _scale.x / 2;
  const yOffset = Math.random() * _scale.y - _scale.y / 2;
  const rotationX = THREE2.MathUtils.degToRad(_rotation.x);
  const rotationY = THREE2.MathUtils.degToRad(_rotation.y);
  position.x = xOffset * Math.cos(rotationY);
  position.y = yOffset * Math.cos(rotationX);
  position.z = xOffset * Math.sin(rotationY) - yOffset * Math.sin(rotationX);
  position.applyQuaternion(quaternion);
  velocity.set(0, 0, speed);
  velocity.applyQuaternion(quaternion);
};
var createDefaultMeshTexture = () => {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const context = canvas.getContext("2d");
    if (context) {
      context.fillStyle = "white";
      context.fillRect(0, 0, 1, 1);
      const texture8 = new THREE2.CanvasTexture(canvas);
      texture8.needsUpdate = true;
      return texture8;
    }
    return null;
  } catch {
    return null;
  }
};
var createDefaultParticleTexture = () => {
  try {
    const canvas = document.createElement("canvas");
    const size = 64;
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d");
    if (context) {
      const centerX = size / 2;
      const centerY = size / 2;
      const radius = size / 2 - 2;
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
      context.fillStyle = "white";
      context.fill();
      const texture8 = new THREE2.CanvasTexture(canvas);
      texture8.needsUpdate = true;
      return texture8;
    } else {
      console.warn(
        "Could not get 2D context to generate default particle texture."
      );
      return null;
    }
  } catch (error) {
    console.warn("Error creating default particle texture:", error);
    return null;
  }
};
var isLifeTimeCurve = (value) => {
  return typeof value !== "number" && "type" in value;
};
var getCurveFunctionFromConfig = (particleSystemId, lifetimeCurve) => {
  if (lifetimeCurve.type === "BEZIER" /* BEZIER */) {
    return createBezierCurveFunction(
      particleSystemId,
      lifetimeCurve.bezierPoints
    );
  }
  if (lifetimeCurve.type === "EASING" /* EASING */) {
    return lifetimeCurve.curveFunction;
  }
  const raw = lifetimeCurve;
  if (Array.isArray(raw.bezierPoints)) {
    return createBezierCurveFunction(
      particleSystemId,
      raw.bezierPoints
    );
  }
  if (typeof raw.curveFunction === "function") {
    return raw.curveFunction;
  }
  throw new Error(`Unsupported value type: ${lifetimeCurve}`);
};
var calculateValue = (particleSystemId, value, time = 0) => {
  if (typeof value === "number") {
    return value;
  }
  if ("min" in value && "max" in value) {
    if (value.min === value.max) {
      return value.min ?? 0;
    }
    return THREE2.MathUtils.randFloat(value.min ?? 0, value.max ?? 1);
  }
  const lifetimeCurve = value;
  return getCurveFunctionFromConfig(particleSystemId, lifetimeCurve)(time) * (lifetimeCurve.scale ?? 1);
};

// src/js/effects/three-particles/color-utils.ts
var sRGBToLinear = (c) => c < 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
var linearToSRGB = (c) => c < 31308e-7 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
var rgbSRGBToLinear = (c) => ({
  r: sRGBToLinear(c.r ?? 0),
  g: sRGBToLinear(c.g ?? 0),
  b: sRGBToLinear(c.b ?? 0)
});
var permute = Fn(({ x }) => {
  return mod(x.mul(34).add(10).mul(x), float(289));
});
var taylorInvSqrt = Fn(({ r }) => {
  return float(1.79284291400159).sub(float(0.85373472095314).mul(r));
});
var snoise3D = Fn(
  ({ v }) => {
    const ONE_THIRD = float(1 / 3);
    const ONE_SIXTH = float(1 / 6);
    const i = floor(
      v.add(dot(v, vec3(ONE_THIRD, ONE_THIRD, ONE_THIRD)))
    ).toVar();
    const x0 = v.sub(i).add(dot(i, vec3(ONE_SIXTH, ONE_SIXTH, ONE_SIXTH))).toVar();
    const g = step(x0.yzx, x0.xyz).toVar();
    const l = float(1).sub(g).toVar();
    const i1 = min(g.xyz, l.zxy).toVar();
    const i2 = max(g.xyz, l.zxy).toVar();
    const x1 = x0.sub(i1).add(ONE_SIXTH).toVar();
    const x2 = x0.sub(i2).add(ONE_SIXTH.mul(2)).toVar();
    const x3 = x0.sub(float(1)).add(ONE_SIXTH.mul(3)).toVar();
    const iw = mod(i, float(289)).toVar();
    const p0_yz = permute({
      x: permute({
        x: vec4(
          vec2(iw.z, iw.z.add(i1.z)),
          vec2(iw.z.add(i2.z), iw.z.add(1))
        )
      }).add(
        vec4(vec2(iw.y, iw.y.add(i1.y)), vec2(iw.y.add(i2.y), iw.y.add(1)))
      )
    });
    const p = permute({
      x: p0_yz.add(
        vec4(vec2(iw.x, iw.x.add(i1.x)), vec2(iw.x.add(i2.x), iw.x.add(1)))
      )
    });
    const n_ = float(0.142857142857142);
    const j = p.sub(float(49).mul(floor(p.mul(n_).mul(n_)))).toVar();
    const x_ = floor(j.mul(n_)).toVar();
    const y_ = floor(j.sub(float(7).mul(x_))).toVar();
    const NS_X = float(0.285714285714286);
    const NS_Y = float(-0.928571428571429);
    const gx = x_.mul(NS_X).add(NS_Y);
    const gy = y_.mul(NS_X).add(NS_Y);
    const gz = float(1).sub(abs(gx)).sub(abs(gy)).toVar();
    const gz_neg = step(gz, vec4(0));
    const ox = gz_neg.mul(floor(gx).add(0.5));
    const oy = gz_neg.mul(floor(gy).add(0.5));
    const gx_final = gx.sub(ox);
    const gy_final = gy.sub(oy);
    const g0 = vec3(gx_final.x, gy_final.x, gz.x).toVar();
    const g1 = vec3(gx_final.y, gy_final.y, gz.y).toVar();
    const g2 = vec3(gx_final.z, gy_final.z, gz.z).toVar();
    const g3 = vec3(gx_final.w, gy_final.w, gz.w).toVar();
    const norm = taylorInvSqrt({
      r: vec4(vec2(dot(g0, g0), dot(g1, g1)), vec2(dot(g2, g2), dot(g3, g3)))
    });
    g0.assign(g0.mul(norm.x));
    g1.assign(g1.mul(norm.y));
    g2.assign(g2.mul(norm.z));
    g3.assign(g3.mul(norm.w));
    const m = max(
      vec4(
        vec2(float(0.5).sub(dot(x0, x0)), float(0.5).sub(dot(x1, x1))),
        vec2(float(0.5).sub(dot(x2, x2)), float(0.5).sub(dot(x3, x3)))
      ),
      float(0)
    ).toVar();
    const m2 = m.mul(m).toVar();
    const m4 = m2.mul(m2).toVar();
    const gdot = vec4(
      vec2(dot(g0, x0), dot(g1, x1)),
      vec2(dot(g2, x2), dot(g3, x3))
    );
    return float(42).mul(dot(m4, gdot));
  }
);
Fn(
  ({ t }) => {
    const noiseX = snoise3D({ v: vec3(t, float(0), float(0)) });
    const noiseY = snoise3D({ v: vec3(t, t, float(0)) });
    const noiseZ = snoise3D({ v: vec3(t, t, t) });
    return vec3(noiseX, noiseY, noiseZ);
  }
);

// src/js/effects/three-particles/webgpu/compute-modifiers.ts
({
  SHAPE_A: uint(1),
  SHAPE_B: uint(2),
  SHAPE_C: uint(3),
  START_FRAME: uint(4),
  SPEED: uint(5),
  SIZE: uint(6),
  ROTATION: uint(7),
  OPACITY: uint(8),
  LIFETIME: uint(9),
  COLOR: uint(10),
  ROTOL: uint(11),
  STABLE_SEED: uint(12),
  LIN_X: uint(13),
  LIN_Y: uint(14),
  LIN_Z: uint(15),
  ORB_X: uint(16),
  ORB_Y: uint(17),
  ORB_Z: uint(18),
  NOISE_PHASE: uint(19)
});
var MLS_MPM_FIXED_POINT_MULTIPLIER = 1e7;
var MLS_MPM_SUBSTEPS = 2;
var MLS_MPM_MAX_GRID_DIM = 64;
var MLS_MPM_PARTICLE_SPACING = 0.65;
var MLS_MPM_CELL_WORDS = 4;
var MLS_MPM_C_WORDS = 3;
var MLS_MPM_DEFAULTS = {
  stiffness: 3,
  restDensity: 4,
  dynamicViscosity: 0.1,
  dt: 0.2,
  gravity: -0.3,
  sphereSize: 1.2,
  boxSize: [40, 30, 60]
};
var MLS_MPM_WALL = {
  min: 3,
  maxOffset: 4,
  stiffness: 0.3,
  extrapolationK: 3,
  clampLower: 1,
  clampUpperOffset: 2,
  cellBorder: 2,
  cellMargin: 3
};
var computeMLSMPMGridDims = (boxSize) => [
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[0])),
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[1])),
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[2]))
];
var initMLSMPMDambreak = (boxSize, capacity, spacing = MLS_MPM_PARTICLE_SPACING, random = Math.random) => {
  const slots = Math.max(1, Math.floor(capacity));
  const position = new Float32Array(slots * 4);
  const velocity = new Float32Array(slots * 4);
  const coefficients = new Float32Array(slots * MLS_MPM_C_WORDS * 4);
  const yLimit = boxSize[1] * 0.8;
  let count = 0;
  for (let y = 0; y < yLimit && count < slots; y += spacing) {
    for (let x = 3; x < boxSize[0] - 4 && count < slots; x += spacing) {
      for (let z = 3; z < boxSize[2] / 2 && count < slots; z += spacing) {
        const jitter = 2 * random();
        const base = count * 4;
        position[base] = x + jitter;
        position[base + 1] = y + jitter;
        position[base + 2] = z + jitter;
        velocity[base] = 0;
        velocity[base + 1] = 0;
        velocity[base + 2] = 0;
        const cBase = count * MLS_MPM_C_WORDS * 4;
        coefficients[cBase] = 1;
        coefficients[cBase + 5] = 1;
        coefficients[cBase + 10] = 1;
        count++;
      }
    }
  }
  return { count, position, velocity, coefficients };
};
function createMLSMPMBuffers(maxParticles, gridCount, shared) {
  const particles = Math.max(1, Math.floor(maxParticles));
  const cells = Math.max(1, Math.floor(gridCount));
  const coefficients = new Float32Array(particles * MLS_MPM_C_WORDS * 4);
  for (let i = 0; i < particles; i++) {
    const base = i * MLS_MPM_C_WORDS * 4;
    coefficients[base] = 1;
    coefficients[base + 5] = 1;
    coefficients[base + 10] = 1;
  }
  const zeroedCells = new StorageBufferAttribute(
    new Uint32Array(cells * MLS_MPM_CELL_WORDS),
    1
  );
  const freshPos = new StorageBufferAttribute(
    new Float32Array(particles * 4),
    4
  );
  const freshVel = new StorageBufferAttribute(
    new Float32Array(particles * 4),
    4
  );
  const coefficientBuffer = new StorageBufferAttribute(coefficients, 4);
  return shared ? {
    position: shared.position,
    velocity: shared.velocity,
    coefficients: coefficientBuffer,
    cells: zeroedCells
  } : {
    position: freshPos,
    velocity: freshVel,
    coefficients: coefficientBuffer,
    cells: zeroedCells
  };
}
function resolveMLSMPMParams(config, boxSize) {
  const [nx, ny, nz] = computeMLSMPMGridDims(boxSize);
  return {
    stiffness: config?.stiffness ?? MLS_MPM_DEFAULTS.stiffness,
    restDensity: config?.restDensity ?? MLS_MPM_DEFAULTS.restDensity,
    dynamicViscosity: config?.dynamicViscosity ?? MLS_MPM_DEFAULTS.dynamicViscosity,
    dt: config?.dt ?? MLS_MPM_DEFAULTS.dt,
    gravity: config?.gravity ?? MLS_MPM_DEFAULTS.gravity,
    cellSize: config?.cellSize ?? 1,
    gridSize: config?.gridSize ?? MLS_MPM_MAX_GRID_DIM,
    sphereSize: config?.sphereSize ?? MLS_MPM_DEFAULTS.sphereSize,
    boxSize: [boxSize[0], boxSize[1], boxSize[2]],
    gridDims: [nx, ny, nz],
    wallStiffness: MLS_MPM_WALL.stiffness,
    extrapolationK: MLS_MPM_WALL.extrapolationK
  };
}
var STENCIL = (
  /* 27 triples */
  Array.from(
    { length: 27 },
    (_, n) => [Math.floor(n / 9), Math.floor(n / 3) % 3, n % 3]
  )
);
var axisWeights = (diff) => {
  const minus = float(0.5).sub(diff);
  const plus = float(0.5).add(diff);
  return [
    minus.mul(minus).mul(float(0.5)),
    float(0.75).sub(diff.mul(diff)),
    plus.mul(plus).mul(float(0.5))
  ];
};
var stencilFrame = (pos) => {
  const cellIdx = floor(pos);
  const diff = pos.sub(cellIdx.add(float(0.5)));
  return {
    cellIdx,
    wx: axisWeights(diff.x),
    wy: axisWeights(diff.y),
    wz: axisWeights(diff.z)
  };
};
var stencilCell = (cellIdx, offset, wx, wy, wz) => {
  const [gx, gy, gz] = offset;
  return {
    cx: cellIdx.x.add(float(gx - 1)),
    cy: cellIdx.y.add(float(gy - 1)),
    cz: cellIdx.z.add(float(gz - 1)),
    weight: wx[gx].mul(wy[gy]).mul(wz[gz])
  };
};
var wordBase = (cx, cy, cz, ctx) => cx.mul(float(ctx.ny * ctx.nz)).add(cy.mul(float(ctx.nz))).add(cz).mul(float(MLS_MPM_CELL_WORDS));
var cellDistance = (cx, cy, cz, pos) => vec3(cx.add(0.5).sub(pos.x), cy.add(0.5).sub(pos.y), cz.add(0.5).sub(pos.z));
var decodeWord = (word, fp) => {
  const value = float(word);
  return value.lessThan(float(2147483648)).select(value, value.sub(float(4294967296))).div(fp);
};
var loadWord = (ctx, base, component) => decodeWord(atomicLoad(ctx.sCells.element(base.add(component))), ctx.fp);
var addWord = (ctx, base, component, value) => {
  atomicAdd(ctx.sCells.element(base.add(component)), uint(value.mul(ctx.fp)));
};
var createClearGridKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.gridCount)), () => {
    const base = i.mul(float(MLS_MPM_CELL_WORDS));
    atomicStore(ctx.sCells.element(base), uint(0));
    atomicStore(ctx.sCells.element(base.add(1)), uint(0));
    atomicStore(ctx.sCells.element(base.add(2)), uint(0));
    atomicStore(ctx.sCells.element(base.add(3)), uint(0));
  });
});
var createP2G1Kernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const pos = ctx.sPos.element(i).xyz.toVar();
    const vel = ctx.sVel.element(i).xyz.toVar();
    const frame = stencilFrame(pos);
    const cBase = i.mul(float(MLS_MPM_C_WORDS));
    const c0 = ctx.sC.element(cBase).xyz;
    const c1 = ctx.sC.element(cBase.add(1)).xyz;
    const c2 = ctx.sC.element(cBase.add(2)).xyz;
    for (const offset of STENCIL) {
      const cell = stencilCell(
        frame.cellIdx,
        offset,
        frame.wx,
        frame.wy,
        frame.wz
      );
      const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
      const q = c0.mul(d.x).add(c1.mul(d.y)).add(c2.mul(d.z));
      const add2 = vel.add(q).mul(cell.weight);
      const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
      addWord(ctx, base, 0, add2.x);
      addWord(ctx, base, 1, add2.y);
      addWord(ctx, base, 2, add2.z);
      addWord(ctx, base, 3, cell.weight);
    }
  });
});
var createP2G2Kernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const pos = ctx.sPos.element(i).xyz.toVar();
    const frame = stencilFrame(pos);
    const cBase = i.mul(float(MLS_MPM_C_WORDS));
    const c0 = ctx.sC.element(cBase).xyz;
    const c1 = ctx.sC.element(cBase.add(1)).xyz;
    const c2 = ctx.sC.element(cBase.add(2)).xyz;
    const density = float(0).toVar();
    for (const offset of STENCIL) {
      const cell = stencilCell(
        frame.cellIdx,
        offset,
        frame.wx,
        frame.wy,
        frame.wz
      );
      const mass = loadWord(ctx, wordBase(cell.cx, cell.cy, cell.cz, ctx), 3);
      density.assign(density.add(mass.mul(cell.weight)));
    }
    const volume = float(1).div(density);
    const pressure = max(
      ctx.k.mul(pow(density.div(ctx.d0), float(5)).sub(float(1))),
      float(0)
    );
    const negP = float(-1).mul(pressure);
    const s0 = ctx.mu.mul(c0.add(vec3(c0.x, c1.x, c2.x))).add(vec3(negP, float(0), float(0)));
    const s1 = ctx.mu.mul(c1.add(vec3(c0.y, c1.y, c2.y))).add(vec3(float(0), negP, float(0)));
    const s2 = ctx.mu.mul(c2.add(vec3(c0.z, c1.z, c2.z))).add(vec3(float(0), float(0), negP));
    const eq16 = float(-4).mul(volume).mul(ctx.dt);
    for (const offset of STENCIL) {
      const cell = stencilCell(
        frame.cellIdx,
        offset,
        frame.wx,
        frame.wy,
        frame.wz
      );
      const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
      const momentum = s0.mul(d.x).add(s1.mul(d.y)).add(s2.mul(d.z)).mul(cell.weight).mul(eq16);
      const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
      addWord(ctx, base, 0, momentum.x);
      addWord(ctx, base, 1, momentum.y);
      addWord(ctx, base, 2, momentum.z);
    }
  });
});
var createUpdateGridKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.gridCount)), () => {
    const base = i.mul(float(MLS_MPM_CELL_WORDS));
    const mass = decodeWord(
      atomicLoad(ctx.sCells.element(base.add(3))),
      ctx.fp
    );
    If(mass.greaterThan(float(0)), () => {
      const invMass = float(1).div(mass);
      const vx = loadWord(ctx, base, 0).mul(invMass);
      const vy = loadWord(ctx, base, 1).mul(invMass).add(ctx.gravity.mul(ctx.dt));
      const vz = loadWord(ctx, base, 2).mul(invMass);
      atomicStore(ctx.sCells.element(base), uint(vx.mul(ctx.fp)));
      atomicStore(ctx.sCells.element(base.add(1)), uint(vy.mul(ctx.fp)));
      atomicStore(ctx.sCells.element(base.add(2)), uint(vz.mul(ctx.fp)));
      const fi = float(i);
      const iz = fi.mod(float(ctx.nz));
      const iy = fi.div(float(ctx.nz)).floor().mod(float(ctx.ny));
      const ix = fi.div(float(ctx.ny * ctx.nz)).floor();
      zeroWallCell(ctx, base, ix, ctx.rx);
      zeroWallCell(ctx, base.add(1), iy, ctx.ry);
      zeroWallCell(ctx, base.add(2), iz, ctx.rz);
    });
  });
});
var zeroWallCell = (ctx, base, index, boxAxis) => {
  If(index.lessThan(float(MLS_MPM_WALL.cellBorder)), () => {
    atomicStore(ctx.sCells.element(base), uint(0));
  });
  If(index.greaterThan(boxAxis.sub(float(MLS_MPM_WALL.cellMargin))), () => {
    atomicStore(ctx.sCells.element(base), uint(0));
  });
};
var createG2PKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const pos = ctx.sPos.element(i).xyz.toVar();
    const frame = stencilFrame(pos);
    const cBase = i.mul(float(MLS_MPM_C_WORDS));
    const vel = vec3(float(0), float(0), float(0)).toVar();
    const b0 = vec3(float(0), float(0), float(0)).toVar();
    const b1 = vec3(float(0), float(0), float(0)).toVar();
    const b2 = vec3(float(0), float(0), float(0)).toVar();
    for (const offset of STENCIL) {
      const cell = stencilCell(
        frame.cellIdx,
        offset,
        frame.wx,
        frame.wy,
        frame.wz
      );
      const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
      const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
      const gvx = loadWord(ctx, base, 0).mul(cell.weight);
      const gvy = loadWord(ctx, base, 1).mul(cell.weight);
      const gvz = loadWord(ctx, base, 2).mul(cell.weight);
      vel.assign(vel.add(vec3(gvx, gvy, gvz)));
      b0.assign(b0.add(vec3(gvx.mul(d.x), gvy.mul(d.x), gvz.mul(d.x))));
      b1.assign(b1.add(vec3(gvx.mul(d.y), gvy.mul(d.y), gvz.mul(d.y))));
      b2.assign(b2.add(vec3(gvx.mul(d.z), gvy.mul(d.z), gvz.mul(d.z))));
    }
    const four = float(4);
    ctx.sC.element(cBase).assign(vec4(b0.mul(four), float(0)));
    ctx.sC.element(cBase.add(1)).assign(vec4(b1.mul(four), float(0)));
    ctx.sC.element(cBase.add(2)).assign(vec4(b2.mul(four), float(0)));
    const next = pos.add(vel.mul(ctx.dt));
    const lower = float(MLS_MPM_WALL.clampLower);
    const clamped = vec3(
      min(
        max(next.x, lower),
        ctx.rx.sub(float(MLS_MPM_WALL.clampUpperOffset))
      ),
      min(
        max(next.y, lower),
        ctx.ry.sub(float(MLS_MPM_WALL.clampUpperOffset))
      ),
      min(
        max(next.z, lower),
        ctx.rz.sub(float(MLS_MPM_WALL.clampUpperOffset))
      )
    );
    pos.assign(clamped);
    ctx.sPos.element(i).assign(vec4(clamped, float(0)));
    const step2 = ctx.dt.mul(ctx.extrapolationK);
    const ex = clamped.add(vel.mul(step2));
    const minWall = float(MLS_MPM_WALL.min);
    const maxOffset = float(MLS_MPM_WALL.maxOffset);
    const newVel = vel.toVar();
    If(ex.x.lessThan(minWall), () => {
      newVel.x.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.x)));
    });
    If(ex.x.greaterThan(ctx.rx.sub(maxOffset)), () => {
      newVel.x.addAssign(
        ctx.wallStiffness.mul(ctx.rx.sub(maxOffset).sub(ex.x))
      );
    });
    If(ex.y.lessThan(minWall), () => {
      newVel.y.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.y)));
    });
    If(ex.y.greaterThan(ctx.ry.sub(maxOffset)), () => {
      newVel.y.addAssign(
        ctx.wallStiffness.mul(ctx.ry.sub(maxOffset).sub(ex.y))
      );
    });
    If(ex.z.lessThan(minWall), () => {
      newVel.z.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.z)));
    });
    If(ex.z.greaterThan(ctx.rz.sub(maxOffset)), () => {
      newVel.z.addAssign(
        ctx.wallStiffness.mul(ctx.rz.sub(maxOffset).sub(ex.z))
      );
    });
    ctx.sVel.element(i).assign(vec4(newVel, float(0)));
  });
});
var layout = (name, storages, uniforms) => ({
  name,
  storageBindings: storages.length,
  uniformBindings: uniforms.length
});
function createMLSMPMPipeline(maxParticles, params, realBox, shared) {
  const count = Math.max(1, Math.floor(maxParticles));
  const [nx, ny, nz] = params.gridDims;
  const gridCount = nx * ny * nz;
  const box = realBox ?? params.boxSize;
  const buffers = createMLSMPMBuffers(count, gridCount, shared);
  const uBoxWidthRatio = uniform(
    params.boxSize[2] > 0 ? box[2] / params.boxSize[2] : 1
  );
  const sPos = storage(buffers.position, "vec4", count);
  const sVel = storage(buffers.velocity, "vec4", count);
  const sC = storage(buffers.coefficients, "vec4", count * MLS_MPM_C_WORDS);
  const sCells = storage(
    buffers.cells,
    "uint",
    gridCount * MLS_MPM_CELL_WORDS
  ).toAtomic();
  const ctx = {
    count,
    gridCount,
    ny,
    nz,
    sPos,
    sVel,
    sC,
    sCells,
    fp: float(MLS_MPM_FIXED_POINT_MULTIPLIER),
    k: float(params.stiffness),
    d0: float(params.restDensity),
    mu: float(params.dynamicViscosity),
    dt: float(params.dt),
    gravity: float(params.gravity),
    wallStiffness: float(params.wallStiffness),
    extrapolationK: float(params.extrapolationK),
    rx: float(box[0]),
    ry: float(box[1]),
    // Animated `z` extent = init extent * `uBoxWidthRatio` (`changeBoxSize`).
    rz: float(params.boxSize[2]).mul(uBoxWidthRatio)
  };
  const clearGrid = createClearGridKernel(ctx);
  const p2g1 = createP2G1Kernel(ctx);
  const p2g2 = createP2G2Kernel(ctx);
  const updateGrid = createUpdateGridKernel(ctx);
  const g2p = createG2PKernel(ctx);
  const passNames = [];
  const computeNodes = [];
  const push = (name, node) => {
    passNames.push(name);
    computeNodes.push(node);
  };
  for (let step2 = 0; step2 < MLS_MPM_SUBSTEPS; step2++) {
    const suffix = `_${step2 + 1}` ;
    push(`clearGrid${suffix}`, compute(clearGrid(), gridCount));
    push(`p2g1${suffix}`, compute(p2g1(), count));
    push(`p2g2${suffix}`, compute(p2g2(), count));
    push(`updateGrid${suffix}`, compute(updateGrid(), gridCount));
    push(`g2p${suffix}`, compute(g2p(), count));
  }
  const pool = [sPos, sVel, sC, sCells];
  return {
    computeNodes,
    passNames,
    passLayouts: passNames.map((name) => layout(name, pool, [])),
    buffers,
    gridCount,
    numParticles: count,
    uniforms: { boxWidthRatio: uBoxWidthRatio }
  };
}
var SPH_WORKGROUP_SIZE = 64;
var SPH_SUBSTEPS = 2;
var SPH_DEFAULT_KERNEL_RADIUS = 0.07;
var SPH_CELL_SIZE_FACTOR = 1;
var SPH_SENTINEL_CELLS = 4;
var SPH_MAX_HALF_BOX = 2;
var SPH_WALL_STIFFNESS = 8e3;
var SPH_R2_EPSILON = 1e-64;
var SPH_SCAN_CHUNK = SPH_WORKGROUP_SIZE;
var SPH_SCAN_STAGES = 6;
var SPH_DEFAULTS = {
  kernelRadius: SPH_DEFAULT_KERNEL_RADIUS,
  mass: 1,
  restDensity: 15e3,
  stiffness: 20,
  nearStiffness: 1,
  viscosity: 100,
  dt: 6e-3,
  gravity: -9.8,
  sphereSize: 0.08,
  halfBoxSize: [1, 2, 1]
};
var SPH_LATTICE_FACTOR = 0.5;
var SPH_LATTICE_MARGIN = 0.95;
var computeSPHGridDims = (kernelRadius = SPH_DEFAULT_KERNEL_RADIUS, halfMax = SPH_MAX_HALF_BOX) => {
  const cellSize = kernelRadius * SPH_CELL_SIZE_FACTOR;
  const dims = Math.ceil(
    (2 * halfMax + SPH_SENTINEL_CELLS * cellSize) / cellSize
  );
  return [dims, dims, dims];
};
var computeSPHOffset = (kernelRadius = SPH_DEFAULT_KERNEL_RADIUS) => SPH_SENTINEL_CELLS * kernelRadius * SPH_CELL_SIZE_FACTOR / 2;
var computeSPHScanBlocks = (cellCount) => Math.ceil(Math.max(1, cellCount) / SPH_SCAN_CHUNK);
var computeSPHScanInnerSteps = (blockCount) => Math.ceil(Math.max(1, blockCount) / SPH_SCAN_CHUNK);
var sphKernelPowers = (kernelRadius = SPH_DEFAULT_KERNEL_RADIUS) => ({
  pow2: Math.pow(kernelRadius, 2),
  pow5: Math.pow(kernelRadius, 5),
  pow6: Math.pow(kernelRadius, 6),
  pow9: Math.pow(kernelRadius, 9)
});
var sphDensityKernelScale = (powers) => 315 / (64 * Math.PI * powers.pow9);
var sphNearDensityKernelScale = (powers) => 15 / (Math.PI * powers.pow6);
var sphDensityGradientScale = (powers) => 45 / (Math.PI * powers.pow6);
var sphViscosityLaplacianScale = (powers) => 45 / (Math.PI * powers.pow6);
var initSPHDambreak = (halfBoxSize, capacity, kernelRadius = SPH_DEFAULT_KERNEL_RADIUS, random = Math.random) => {
  const slots = Math.max(1, Math.floor(capacity));
  const position = new Float32Array(slots * 4);
  const velocity = new Float32Array(slots * 4);
  const forceDensity = new Float32Array(slots * 4);
  const step2 = SPH_LATTICE_FACTOR * kernelRadius;
  const mx = SPH_LATTICE_MARGIN * halfBoxSize[0];
  const my = SPH_LATTICE_MARGIN * halfBoxSize[1];
  const mz = SPH_LATTICE_MARGIN * halfBoxSize[2];
  let count = 0;
  for (let y = -my; count < slots; y += step2) {
    for (let x = -mx; x < mx && count < slots; x += step2) {
      for (let z = -mz; z < 0 && count < slots; z += step2) {
        const jitter = 1e-3 * random();
        const base = count * 4;
        position[base] = x + jitter;
        position[base + 1] = y + jitter;
        position[base + 2] = z + jitter;
        count++;
      }
    }
  }
  return { count, position, velocity, forceDensity };
};
function createSPHBuffers(maxParticles, gridCount, shared) {
  const particles = Math.max(1, Math.floor(maxParticles));
  const cells = Math.max(1, Math.floor(gridCount));
  const blocks = computeSPHScanBlocks(cells + 1);
  const particleVec4 = () => new StorageBufferAttribute(new Float32Array(particles * 4), 4);
  const pos = shared ? shared.position : particleVec4();
  const vel = shared ? shared.velocity : particleVec4();
  return {
    position: pos,
    velocity: vel,
    forceDensity: particleVec4(),
    sortedPosition: particleVec4(),
    sortedVelocity: particleVec4(),
    sortedForceDensity: particleVec4(),
    cellCounts: new StorageBufferAttribute(new Uint32Array(cells), 1),
    prefixSums: new StorageBufferAttribute(new Float32Array(cells + 1), 1),
    particleCellOffsets: new StorageBufferAttribute(
      new Uint32Array(particles),
      1
    ),
    blockPartials: new StorageBufferAttribute(new Float32Array(blocks), 1),
    blockInclusive: new StorageBufferAttribute(new Float32Array(blocks), 1),
    blockOffsets: new StorageBufferAttribute(new Float32Array(blocks), 1)
  };
}
function resolveSPHParams(config, halfBoxSize, realHalfBox) {
  const kernelRadius = config?.kernelRadius ?? SPH_DEFAULTS.kernelRadius;
  const [nx, ny, nz] = computeSPHGridDims(kernelRadius, SPH_MAX_HALF_BOX);
  const powers = sphKernelPowers(kernelRadius);
  const box = halfBoxSize;
  return {
    kernelRadius,
    mass: config?.mass ?? SPH_DEFAULTS.mass,
    restDensity: config?.restDensity ?? SPH_DEFAULTS.restDensity,
    stiffness: config?.stiffness ?? SPH_DEFAULTS.stiffness,
    nearStiffness: config?.nearStiffness ?? SPH_DEFAULTS.nearStiffness,
    viscosity: config?.viscosity ?? SPH_DEFAULTS.viscosity,
    dt: config?.dt ?? SPH_DEFAULTS.dt,
    gravity: config?.gravity ?? SPH_DEFAULTS.gravity,
    sphereSize: config?.sphereSize ?? SPH_DEFAULTS.sphereSize,
    halfBoxSize: [halfBoxSize[0], halfBoxSize[1], halfBoxSize[2]],
    realHalfBox: [box[0], box[1], box[2]],
    cellSize: kernelRadius * SPH_CELL_SIZE_FACTOR,
    offset: computeSPHOffset(kernelRadius),
    gridDims: [nx, ny, nz],
    powers,
    densityScale: sphDensityKernelScale(powers),
    nearDensityScale: sphNearDensityKernelScale(powers),
    gradientScale: sphDensityGradientScale(powers),
    laplacianScale: sphViscosityLaplacianScale(powers)
  };
}
var cellCoords = (pos, ctx) => floor(
  pos.add(vec3(ctx.halfX, ctx.halfY, ctx.halfZ)).add(ctx.offset).mul(ctx.cellSizeInv)
);
var cellIdOf = (coords, ctx) => coords.x.add(coords.y.mul(float(ctx.xGrids))).add(coords.z.mul(float(ctx.xGrids * ctx.yGrids)));
var insideLattice = (coords, ctx) => coords.x.greaterThanEqual(float(0)).and(coords.y.greaterThanEqual(float(0))).and(coords.z.greaterThanEqual(float(0))).and(coords.x.lessThan(float(ctx.xGrids))).and(coords.y.lessThan(float(ctx.yGrids))).and(coords.z.lessThan(float(ctx.zGrids)));
var slabExtent = (coord, grids) => min(coord, float(grids).sub(coord).sub(float(1)));
var cellCountAt = (ctx, index) => float(atomicLoad(ctx.sCells.element(index)));
var slabTermIds = (coords, dx, dy, dz, ctx) => {
  const xg = float(ctx.xGrids);
  const yg = float(ctx.yGrids);
  const xy = xg.mul(yg);
  const first = coords.x.sub(dx).add(coords.y.sub(dy).mul(xg)).add(coords.z.sub(dz).mul(xy));
  const last = coords.x.add(dx).add(coords.y.add(dy).mul(xg)).add(coords.z.add(dz).mul(xy));
  return { first, last };
};
var forEachSlabCell = (coords, ex, ey, ez, ctx, body) => {
  for (let dz = 0; dz < 3; dz++) {
    for (let dy = 0; dy < 3; dy++) {
      for (let dx = 0; dx < 3; dx++) {
        const kx = float(dx);
        const ky = float(dy);
        const kz = float(dz);
        const ids = slabTermIds(
          coords,
          min(kx, ex),
          min(ky, ey),
          min(kz, ez),
          ctx
        );
        const inRange = kx.lessThanEqual(ex).and(ky.lessThanEqual(ey).and(kz.lessThanEqual(ez)));
        If(inRange, () => {
          body(
            ctx.sPrefix.element(ids.first),
            ctx.sPrefix.element(ids.last.add(1))
          );
        });
      }
    }
  }
};
var createGridClearKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.gridCount)), () => {
    atomicStore(ctx.sCells.element(i), uint(0));
  });
});
var createGridBuildKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const coords = cellCoords(ctx.sPos.element(i).xyz, ctx);
    If(insideLattice(coords, ctx), () => {
      const old = atomicAdd(
        ctx.sCells.element(cellIdOf(coords, ctx)),
        uint(1)
      );
      ctx.sOffsets.element(i).assign(old);
    });
  });
});
var createScanPartialsKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.scanBlocks)), () => {
    const total = float(0).toVar();
    for (let k = 0; k < SPH_SCAN_CHUNK; k++) {
      const idx = float(i).mul(float(SPH_SCAN_CHUNK)).add(float(k));
      If(idx.lessThan(float(ctx.gridCount)), () => {
        total.assign(total.add(cellCountAt(ctx, idx)));
      });
    }
    ctx.sPartials.element(i).assign(total);
  });
});
var createBlockScanKernel = (ctx) => Fn(() => {
  const lane = invocationLocalIndex;
  const steps = float(ctx.scanSteps);
  const laneTotal = float(0).toVar();
  for (let k = 0; k < ctx.scanSteps; k++) {
    const idx = float(lane).mul(steps).add(float(k));
    If(idx.lessThan(float(ctx.scanBlocks)), () => {
      laneTotal.assign(laneTotal.add(ctx.sPartials.element(idx)));
      ctx.sInclusive.element(idx).assign(laneTotal);
    });
  }
  const a = workgroupArray("float", SPH_SCAN_CHUNK);
  const b = workgroupArray("float", SPH_SCAN_CHUNK);
  a.element(lane).assign(laneTotal);
  workgroupBarrier();
  let src = a;
  let dst = b;
  for (let stage = 0; stage < SPH_SCAN_STAGES; stage++) {
    const s = float(Math.pow(2, stage));
    If(float(lane).greaterThanEqual(s), () => {
      dst.element(lane).assign(
        add(
          float(src.element(lane)),
          float(src.element(float(lane).sub(s)))
        )
      );
    });
    If(float(lane).lessThan(s), () => {
      dst.element(lane).assign(src.element(lane));
    });
    workgroupBarrier();
    const next = dst;
    dst = src;
    src = next;
  }
  const exclusive = float(0).toVar();
  exclusive.assign(sub(float(src.element(lane)), laneTotal));
  for (let k = 0; k < ctx.scanSteps; k++) {
    const idx = float(lane).mul(steps).add(float(k));
    If(idx.lessThan(float(ctx.scanBlocks)), () => {
      ctx.sBlockOffsets.element(idx).assign(
        ctx.sInclusive.element(idx).sub(ctx.sPartials.element(idx)).add(exclusive)
      );
    });
  }
});
var createScanApplyKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.scanBlocks)), () => {
    const base = float(i).mul(float(SPH_SCAN_CHUNK));
    const blockStart = ctx.sBlockOffsets.element(i);
    const running = float(0).toVar();
    ctx.sPrefix.element(base).assign(blockStart);
    for (let k = 0; k < SPH_SCAN_CHUNK; k++) {
      const idx = base.add(float(k));
      If(idx.lessThan(float(ctx.gridCount + 1)), () => {
        If(idx.lessThan(float(ctx.gridCount)), () => {
          running.assign(running.add(cellCountAt(ctx, idx)));
        });
        ctx.sPrefix.element(idx.add(1)).assign(blockStart.add(running));
      });
    }
  });
});
var reorderBody = (ctx, writeFields) => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const coords = cellCoords(ctx.sPos.element(i).xyz, ctx);
    If(insideLattice(coords, ctx), () => {
      const target = ctx.sPrefix.element(cellIdOf(coords, ctx).add(1)).sub(ctx.sOffsets.element(i)).sub(float(1));
      If(target.lessThan(float(ctx.count)), () => {
        writeFields(target);
      });
    });
  });
};
var createReorderPositionKernel = (ctx) => Fn(() => {
  reorderBody(ctx, (target) => {
    const i = instanceIndex;
    ctx.sSortedPos.element(target).assign(ctx.sPos.element(i));
    ctx.sSortedVel.element(target).assign(ctx.sVel.element(i));
  });
});
var createReorderForceKernel = (ctx) => Fn(() => {
  reorderBody(ctx, (target) => {
    ctx.sSortedForce.element(target).assign(ctx.sForce.element(instanceIndex));
  });
});
var createDensityKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const pos = ctx.sPos.element(i).xyz;
    const coords = cellCoords(pos, ctx);
    If(insideLattice(coords, ctx), () => {
      const density = float(0).toVar();
      const nearDensity = float(0).toVar();
      const ex = slabExtent(coords.x, ctx.xGrids);
      const ey = slabExtent(coords.y, ctx.yGrids);
      const ez = slabExtent(coords.z, ctx.zGrids);
      forEachSlabCell(coords, ex, ey, ez, ctx, (start, end) => {
        Loop(end.sub(start), ({ i: j }) => {
          const other = ctx.sSortedPos.element(start.add(j)).xyz;
          const delta = pos.sub(other);
          const r2 = dot(delta, delta);
          If(r2.lessThan(ctx.radiusPow2), () => {
            const r = sqrt(r2);
            const gap = ctx.radius.sub(r);
            density.assign(
              density.add(
                ctx.mass.mul(ctx.densityScale).mul(cube(ctx.radiusPow2.sub(r2)))
              )
            );
            nearDensity.assign(
              nearDensity.add(
                ctx.mass.mul(ctx.nearDensityScale).mul(cube(gap))
              )
            );
          });
        });
      });
      const posVec = ctx.sPos.element(i).toVar();
      posVec.w.assign(nearDensity);
      ctx.sPos.element(i).assign(posVec);
      const forceVec = ctx.sForce.element(i).toVar();
      forceVec.w.assign(density);
      ctx.sForce.element(i).assign(forceVec);
    });
  });
});
var createForceKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const posVec = ctx.sPos.element(i);
    const posI = posVec.xyz;
    const velI = ctx.sVel.element(i).xyz;
    const densityI = ctx.sForce.element(i).w;
    const nearI = posVec.w;
    const coords = cellCoords(posI, ctx);
    const fPress = vec3(float(0), float(0), float(0)).toVar();
    const fVisc = vec3(float(0), float(0), float(0)).toVar();
    If(insideLattice(coords, ctx), () => {
      const ex = slabExtent(coords.x, ctx.xGrids);
      const ey = slabExtent(coords.y, ctx.yGrids);
      const ez = slabExtent(coords.z, ctx.zGrids);
      forEachSlabCell(coords, ex, ey, ez, ctx, (start, end) => {
        Loop(end.sub(start), ({ i: j }) => {
          const slot = start.add(j);
          const densityJ = ctx.sSortedForce.element(slot).w;
          const posJ = ctx.sSortedPos.element(slot).xyz;
          const nearJ = ctx.sSortedPos.element(slot).w;
          const velJ = ctx.sSortedVel.element(slot).xyz;
          const delta = posI.sub(posJ);
          const r2 = dot(delta, delta);
          If(
            densityJ.greaterThan(float(0)).and(nearJ.greaterThan(float(0))),
            () => {
              If(
                r2.greaterThan(ctx.r2Epsilon).and(r2.lessThan(ctx.radiusPow2)),
                () => {
                  const r = sqrt(r2);
                  const gap = ctx.radius.sub(r);
                  const pressureI = ctx.stiffness.mul(
                    densityI.sub(ctx.restDensity)
                  );
                  const pressureJ = ctx.stiffness.mul(
                    densityJ.sub(ctx.restDensity)
                  );
                  const nearPressureI = ctx.nearStiffness.mul(nearI);
                  const nearPressureJ = ctx.nearStiffness.mul(nearJ);
                  const dir = normalize(posJ.sub(posI));
                  const grad = ctx.gradientScale.mul(gap.mul(gap));
                  const laplacian = ctx.laplacianScale.mul(gap);
                  const shared = float(0.5).mul(pressureI.add(pressureJ));
                  const nearShared = float(0.5).mul(
                    nearPressureI.add(nearPressureJ)
                  );
                  const pressTerm = dir.mul(shared.mul(grad).mul(ctx.mass).div(densityJ)).negate();
                  const nearTerm = dir.mul(nearShared.mul(grad).mul(ctx.mass).div(nearJ)).negate();
                  const viscTerm = velJ.sub(velI).mul(laplacian.mul(ctx.mass).div(densityJ));
                  fPress.assign(fPress.add(pressTerm).add(nearTerm));
                  fVisc.assign(fVisc.add(viscTerm));
                }
              );
            }
          );
        });
      });
    });
    const gravityVec = vec3(float(0), ctx.gravity.mul(densityI), float(0));
    const force = fPress.add(fVisc.mul(ctx.viscosity)).add(gravityVec);
    ctx.sForce.element(i).assign(vec4(force.x, force.y, force.z, densityI));
  });
});
var createIntegrateKernel = (ctx) => Fn(() => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const posVec = ctx.sPos.element(i).toVar();
    const forceVec = ctx.sForce.element(i);
    const density = forceVec.w;
    If(density.notEqual(float(0)), () => {
      const accel = vec4(
        forceVec.x.div(density),
        forceVec.y.div(density),
        forceVec.z.div(density),
        float(0)
      ).toVar();
      const wall = float(SPH_WALL_STIFFNESS);
      signedWall(accel.x, wall, ctx.halfX.sub(posVec.x));
      signedWall(accel.x, wall, ctx.halfX.add(posVec.x));
      signedWall(accel.y, wall, ctx.halfY.sub(posVec.y));
      signedWall(accel.y, wall, ctx.halfY.add(posVec.y));
      signedWall(accel.z, wall, ctx.halfZ.sub(posVec.z));
      signedWall(accel.z, wall, ctx.halfZ.add(posVec.z));
      const velVec = ctx.sVel.element(i).toVar();
      velVec.x.addAssign(accel.x.mul(ctx.dt));
      velVec.y.addAssign(accel.y.mul(ctx.dt));
      velVec.z.addAssign(accel.z.mul(ctx.dt));
      posVec.x.addAssign(velVec.x.mul(ctx.dt));
      posVec.y.addAssign(velVec.y.mul(ctx.dt));
      posVec.z.addAssign(velVec.z.mul(ctx.dt));
      ctx.sVel.element(i).assign(velVec);
      ctx.sPos.element(i).assign(posVec);
    });
  });
});
var signedWall = (axis, stiffness, distance) => {
  axis.addAssign(stiffness.mul(min(distance, float(0))));
};
var cube = (n) => n.mul(n).mul(n);
var layout2 = (name, storages, uniforms) => ({
  name,
  storageBindings: storages.length,
  uniformBindings: uniforms.length
});
function createSPHPipeline(maxParticles, params, realHalfBox, shared) {
  const count = Math.max(1, Math.floor(maxParticles));
  const [nx, ny, nz] = params.gridDims;
  const gridCount = nx * ny * nz;
  const scanBlocks = computeSPHScanBlocks(gridCount + 1);
  const scanSteps = computeSPHScanInnerSteps(scanBlocks);
  const box = realHalfBox ?? params.realHalfBox;
  const buffers = createSPHBuffers(count, gridCount, shared);
  const uBoxWidthRatio = uniform(
    params.halfBoxSize[2] > 0 ? box[2] / params.halfBoxSize[2] : 1
  );
  const sPos = storage(buffers.position, "vec4", count);
  const sVel = storage(buffers.velocity, "vec4", count);
  const sForce = storage(buffers.forceDensity, "vec4", count);
  const sSortedPos = storage(buffers.sortedPosition, "vec4", count);
  const sSortedVel = storage(buffers.sortedVelocity, "vec4", count);
  const sSortedForce = storage(buffers.sortedForceDensity, "vec4", count);
  const sCells = storage(buffers.cellCounts, "uint", gridCount).toAtomic();
  const sPrefix = storage(buffers.prefixSums, "float", gridCount + 1);
  const sOffsets = storage(buffers.particleCellOffsets, "uint", count);
  const sPartials = storage(buffers.blockPartials, "float", scanBlocks);
  const sInclusive = storage(buffers.blockInclusive, "float", scanBlocks);
  const sBlockOffsets = storage(buffers.blockOffsets, "float", scanBlocks);
  const ctx = {
    count,
    gridCount,
    xGrids: nx,
    yGrids: ny,
    zGrids: nz,
    scanBlocks,
    scanSteps,
    sPos,
    sVel,
    sForce,
    sSortedPos,
    sSortedVel,
    sSortedForce,
    sCells,
    sPrefix,
    sOffsets,
    sPartials,
    sInclusive,
    sBlockOffsets,
    cellSizeInv: float(1 / params.cellSize),
    offset: float(params.offset),
    // One half-max set feeds both the lattice coordinates and the walls
    // (`xHalfMax` / `yHalfMax` / `zHalfMax` of the reference params block);
    // the `z` axis carries the animated `boxWidthRatio` squeeze.
    halfX: float(box[0]),
    halfY: float(box[1]),
    halfZ: float(params.halfBoxSize[2]).mul(uBoxWidthRatio),
    radius: float(params.kernelRadius),
    radiusPow2: float(params.powers.pow2),
    r2Epsilon: float(SPH_R2_EPSILON),
    mass: float(params.mass),
    stiffness: float(params.stiffness),
    nearStiffness: float(params.nearStiffness),
    restDensity: float(params.restDensity),
    viscosity: float(params.viscosity),
    dt: float(params.dt),
    gravity: float(params.gravity),
    densityScale: float(params.densityScale),
    nearDensityScale: float(params.nearDensityScale),
    gradientScale: float(params.gradientScale),
    laplacianScale: float(params.laplacianScale),
    pool: [
      sPos,
      sVel,
      sForce,
      sSortedPos,
      sSortedVel,
      sSortedForce,
      sCells,
      sPrefix,
      sOffsets,
      sPartials,
      sInclusive,
      sBlockOffsets
    ]
  };
  const subsets = {
    gridClear: [sCells],
    gridBuild: [sPos, sCells, sOffsets],
    scanPartials: [sCells, sPartials],
    scanBlocks: [sPartials, sInclusive, sBlockOffsets],
    scanApply: [sCells, sPrefix, sBlockOffsets],
    reorderPosition: [sPos, sVel, sSortedPos, sSortedVel, sPrefix, sOffsets],
    reorderForce: [sPos, sForce, sSortedForce, sPrefix, sOffsets],
    density: [sPos, sForce, sSortedPos, sPrefix],
    force: [sPos, sVel, sForce, sSortedPos, sSortedVel, sSortedForce, sPrefix],
    integrate: [sPos, sVel, sForce]
  };
  const clear = createGridClearKernel(ctx);
  const build = createGridBuildKernel(ctx);
  const scan1 = createScanPartialsKernel(ctx);
  const scan2 = createBlockScanKernel(ctx);
  const scan3 = createScanApplyKernel(ctx);
  const reorderPos = createReorderPositionKernel(ctx);
  const reorderForce = createReorderForceKernel(ctx);
  const density = createDensityKernel(ctx);
  const force = createForceKernel(ctx);
  const integrate = createIntegrateKernel(ctx);
  const passNames = [];
  const computeNodes = [];
  const passLayouts = [];
  const push = (name, node, subset) => {
    passNames.push(name);
    computeNodes.push(node);
    passLayouts.push(
      layout2(name, subset === "pool" ? ctx.pool : subsets[subset], [])
    );
  };
  for (let step2 = 0; step2 < SPH_SUBSTEPS; step2++) {
    const suffix = `_${step2 + 1}` ;
    push(`gridClear${suffix}`, compute(clear(), gridCount), "gridClear");
    push(`gridBuild${suffix}`, compute(build(), count), "gridBuild");
    push("scanPartials", compute(scan1(), scanBlocks), "scanPartials");
    push("scanBlocks", compute(scan2(), SPH_SCAN_CHUNK), "scanBlocks");
    push("scanApply", compute(scan3(), scanBlocks), "scanApply");
    push(
      `reorderPosition${suffix}`,
      compute(reorderPos(), count),
      "reorderPosition"
    );
    push(
      `reorderForce${suffix}`,
      compute(reorderForce(), count),
      "reorderForce"
    );
    push(`density${suffix}`, compute(density(), count), "density");
    push(
      `reorderPositionB${suffix}`,
      compute(reorderPos(), count),
      "reorderPosition"
    );
    push(
      `reorderForceB${suffix}`,
      compute(reorderForce(), count),
      "reorderForce"
    );
    push(`force${suffix}`, compute(force(), count), "force");
    push(`integrate${suffix}`, compute(integrate(), count), "integrate");
  }
  return {
    computeNodes,
    passNames,
    passLayouts,
    buffers,
    gridCount,
    numParticles: count,
    uniforms: { boxWidthRatio: uBoxWidthRatio }
  };
}
Fn(
  ({
    vLifetime,
    vStartLifetime,
    vStartFrame,
    uFps,
    uUseFPSForFrameIndex,
    uTiles
  }) => {
    const totalFrames = uTiles.x.mul(uTiles.y);
    const lifePercent = min(vLifetime.div(vStartLifetime), float(1));
    const fpsBased = max(vLifetime.div(1e3).mul(uFps), float(0));
    const lifetimeBased = max(
      min(floor(lifePercent.mul(totalFrames)), totalFrames.sub(1)),
      float(0)
    );
    const fpsResult = uFps.equal(0).select(float(0), fpsBased);
    const frameOffset = uUseFPSForFrameIndex.greaterThan(0.5).select(fpsResult, lifetimeBased);
    return round(vStartFrame).add(frameOffset);
  }
);
Fn(
  ({ baseUV, frameIndex, uTiles }) => {
    const spriteX = floor(mod(frameIndex, uTiles.x));
    const spriteY = floor(mod(frameIndex.div(uTiles.x), uTiles.y));
    return vec2(
      baseUV.x.div(uTiles.x).add(spriteX.div(uTiles.x)),
      baseUV.y.div(uTiles.y).add(spriteY.div(uTiles.y))
    );
  }
);
var linearizeDepth = Fn(
  ({ depthSample, near, far }) => {
    const zNdc = depthSample.mul(2).sub(1);
    return near.mul(2).mul(far).div(far.add(near).sub(zNdc.mul(far.sub(near))));
  }
);
Fn(
  ({
    viewZ,
    uSoftEnabled,
    uSoftIntensity,
    uSceneDepthTex,
    uCameraNearFar
  }) => {
    const softFade = float(1).toVar();
    If(uSoftEnabled.greaterThan(0.5), () => {
      const depthSample = texture(uSceneDepthTex, screenUV).x;
      const sceneDepthLinear = linearizeDepth({
        depthSample,
        near: uCameraNearFar.x,
        far: uCameraNearFar.y
      });
      const depthDiff = sceneDepthLinear.sub(viewZ);
      softFade.assign(smoothstep(float(0), uSoftIntensity, depthDiff));
    });
    return softFade;
  }
);
Fn(
  ({ tangent, viewDir }) => {
    const rawPerp = cross(tangent, viewDir).toVar();
    const perpLen = length(rawPerp);
    const camRight = vec3(
      cameraViewMatrix.element(0).element(0),
      cameraViewMatrix.element(1).element(0),
      cameraViewMatrix.element(2).element(0)
    );
    const camRightDotTangent = dot(camRight, tangent);
    const fallbackPerp = normalize(
      camRight.sub(tangent.mul(camRightDotTangent))
    );
    return normalize(
      perpLen.lessThan(1e-4).select(
        fallbackPerp,
        normalize(
          mix(
            fallbackPerp,
            normalize(rawPerp),
            smoothstep(float(0), float(0.7), perpLen)
          )
        )
      )
    );
  }
);
Fn(
  ({ v, q }) => {
    const t = cross(q.xyz, v).mul(2);
    return v.add(t.mul(q.w)).add(cross(q.xyz, t));
  }
);

// src/js/effects/three-particles/webgpu/tsl-materials.ts
function createFluidSimPipeline(solver, shared, maxParticles, normalizedConfig) {
  const renderer = normalizedConfig.renderer;
  const isSPH = solver === "SPH";
  const sharedPair = {
    position: shared.position,
    velocity: shared.velocity
  };
  if (isSPH) {
    const cfg2 = renderer.sph;
    const halfBox = [
      ...cfg2?.halfBoxSize ?? SPH_DEFAULTS.halfBoxSize
    ];
    const ratio2 = typeof cfg2?.boxWidthRatio === "number" && Number.isFinite(cfg2.boxWidthRatio) ? cfg2.boxWidthRatio : 1;
    const state2 = initSPHDambreak(halfBox, Math.max(1, maxParticles));
    shared.position.array.set(state2.position);
    shared.velocity.array.set(state2.velocity);
    const sph = createSPHPipeline(
      state2.count,
      resolveSPHParams(cfg2, halfBox),
      [halfBox[0], halfBox[1], halfBox[2] * ratio2],
      sharedPair
    );
    return {
      computeNodes: sph.computeNodes,
      passNames: sph.passNames,
      passLayouts: sph.passLayouts,
      buffers: sph.buffers,
      uniforms: sph.uniforms,
      gridCount: sph.gridCount,
      numParticles: sph.numParticles
    };
  }
  const cfg = renderer.mlsMpm;
  const box = [
    ...cfg?.boxSize ?? MLS_MPM_DEFAULTS.boxSize
  ];
  const ratio = typeof cfg?.boxWidthRatio === "number" && Number.isFinite(cfg.boxWidthRatio) ? cfg.boxWidthRatio : 1;
  const state = initMLSMPMDambreak(box, Math.max(1, maxParticles));
  shared.position.array.set(state.position);
  shared.velocity.array.set(state.velocity);
  const mls = createMLSMPMPipeline(
    state.count,
    resolveMLSMPMParams(cfg, box),
    [box[0], box[1], box[2] * ratio],
    sharedPair
  );
  return {
    computeNodes: mls.computeNodes,
    passNames: mls.passNames,
    passLayouts: mls.passLayouts,
    buffers: mls.buffers,
    uniforms: mls.uniforms,
    gridCount: mls.gridCount,
    numParticles: mls.numParticles
  };
}

// src/js/effects/three-particles/three-particles.ts
function resolveWebGPUEffectiveRendererType(requested) {
  switch (requested) {
    case "INSTANCED" /* INSTANCED */:
      return "INSTANCED" /* INSTANCED */;
    case "TRAIL" /* TRAIL */:
      return "TRAIL" /* TRAIL */;
    case "MESH" /* MESH */:
      return "MESH" /* MESH */;
    case "FLUID" /* FLUID */:
      return "FLUID" /* FLUID */;
    case "POINTS" /* POINTS */:
    default:
      return "POINTS" /* POINTS */;
  }
}
var _particleSystemId = 0;
var createdParticleSystems = [];
var _tslMaterialFactory = null;
var _rendererBackendIsGPU = true;
var _cpuPreferenceWarned = false;
var _cpuPreferencePreferenceWarn = () => {
  _cpuPreferenceWarned = true;
  console.warn(
    "three-particles: simulationBackend 'CPU' maps to the GPU kernel in 4.0.0 (GPU-only build)."
  );
};
var registerTSLMaterialFactory = (factory, options) => {
  if (options && "renderer" in options && !isComputeCapableRenderer(options.renderer)) {
    console.warn(
      "three-particles: registerTSLMaterialFactory skipped ??? the provided renderer does not support compute dispatches (expected THREE.WebGPURenderer). Particle systems will use the CPU/GLSL path."
    );
    return false;
  }
  _tslMaterialFactory = factory;
  if (options && "renderer" in options) {
    _rendererBackendIsGPU = !!options.renderer?.backend?.isWebGPUBackend;
  } else {
    _rendererBackendIsGPU = true;
  }
  return true;
};
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Euler(0, 0, 0, "XYZ");
var _lastWorldPositionSnapshot = new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Quaternion();
var assertNamed = (cond, message) => {
  if (!cond) {
    throw new Error(`three-particles: ${message}`);
  }
};
var normalizeVector2Value = (raw, fallback, label) => {
  if (raw === void 0 || raw === null) {
    return new THREE2.Vector2(fallback[0], fallback[1]);
  }
  if (raw instanceof THREE2.Vector2) return raw;
  let n1;
  let n2;
  if (Array.isArray(raw)) {
    n1 = Number(raw[0]);
    n2 = Number(raw[1]);
  } else if (typeof raw === "object") {
    const o = raw;
    n1 = o.x !== void 0 ? Number(o.x) : o.u !== void 0 ? Number(o.u) : void 0;
    n2 = o.y !== void 0 ? Number(o.y) : o.v !== void 0 ? Number(o.v) : void 0;
  }
  assertNamed(
    n1 !== void 0 && n2 !== void 0 && Number.isFinite(n1) && Number.isFinite(n2),
    `${label} must be one of: Vector2, [x,y], [u,v], {x,y} or {u,v}`
  );
  return new THREE2.Vector2(n1, n2);
};
var normalizeTextureValue = (raw, label) => {
  if (raw === void 0 || raw === null) return null;
  assertNamed(
    typeof raw === "object" && "image" in raw,
    `${label} must be null or a texture object with .image (got ${String(raw)})`
  );
  return raw;
};
var normalizeDepthTextureValue = (raw, label) => {
  if (raw === void 0 || raw === null) return null;
  assertNamed(
    typeof raw === "object" && "image" in raw,
    `${label} must be a texture object with .image when set (got ${String(raw)})`
  );
  return raw;
};
var normalizeBackgroundToVector3 = (raw, label) => {
  if (raw === void 0 || raw === null) return new THREE2.Vector3(1, 1, 1);
  if (typeof raw === "number") {
    const c = new THREE2.Color(raw);
    return new THREE2.Vector3(c.r, c.g, c.b);
  }
  if (typeof raw === "string") {
    const s = raw.trim();
    const c = new THREE2.Color(s.startsWith("#") ? s : `#${s}`);
    assertNamed(
      Number.isFinite(c.r) && Number.isFinite(c.g) && Number.isFinite(c.b),
      `${label} is not a valid hex color string`
    );
    return new THREE2.Vector3(c.r, c.g, c.b);
  }
  if (Array.isArray(raw)) {
    const [r, g, b] = raw;
    assertNamed(
      Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b),
      `${label} array must contain three finite numbers`
    );
    return new THREE2.Vector3(r, g, b);
  }
  const o = raw;
  assertNamed(
    Number.isFinite(Number(o.r)) && Number.isFinite(Number(o.g)) && Number.isFinite(Number(o.b)),
    `${label} object must provide finite r/g/b`
  );
  return new THREE2.Vector3(Number(o.r), Number(o.g), Number(o.b));
};
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Vector3();
new THREE2.Vector2();
var toVector3 = (v, fallback) => v ? new THREE2.Vector3(v.x ?? 0, v.y ?? 0, v.z ?? 0) : fallback.clone();
var normalizeForceFields = (rawForceFields) => (rawForceFields ?? []).map((ff) => ({
  isActive: ff.isActive ?? true,
  type: ff.type ?? "POINT" /* POINT */,
  position: toVector3(ff.position, new THREE2.Vector3(0, 0, 0)),
  direction: toVector3(ff.direction, new THREE2.Vector3(0, 1, 0)).normalize(),
  strength: ff.strength ?? 1,
  range: Math.max(0, ff.range ?? Infinity),
  falloff: ff.falloff ?? "LINEAR" /* LINEAR */
}));
var normalizeCollisionPlanes = (rawPlanes) => (rawPlanes ?? []).map((cp) => ({
  isActive: cp.isActive ?? true,
  position: toVector3(cp.position, new THREE2.Vector3(0, 0, 0)),
  normal: toVector3(cp.normal, new THREE2.Vector3(0, 1, 0)).normalize(),
  mode: cp.mode ?? "KILL" /* KILL */,
  dampen: Math.max(0, Math.min(1, cp.dampen ?? 0.5)),
  lifetimeLoss: Math.max(0, Math.min(1, cp.lifetimeLoss ?? 0))
}));
var blendingMap = {
  "THREE.NoBlending": THREE2.NoBlending,
  "THREE.NormalBlending": THREE2.NormalBlending,
  "THREE.AdditiveBlending": THREE2.AdditiveBlending,
  "THREE.SubtractiveBlending": THREE2.SubtractiveBlending,
  "THREE.MultiplyBlending": THREE2.MultiplyBlending
};
var toBlendingConstant = (v) => {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const key = v.startsWith("THREE.") ? v : `THREE.${v}`;
    const mapped = blendingMap[key];
    if (mapped !== void 0) return mapped;
  }
  return THREE2.NormalBlending;
};
var getDefaultParticleSystemConfig = () => JSON.parse(JSON.stringify(DEFAULT_PARTICLE_SYSTEM_CONFIG));
var DEFAULT_PARTICLE_SYSTEM_CONFIG = {
  transform: {
    position: new THREE2.Vector3(),
    rotation: new THREE2.Vector3(),
    scale: new THREE2.Vector3(1, 1, 1)
  },
  duration: 5,
  looping: true,
  startDelay: 0,
  startLifetime: 5,
  startSpeed: 1,
  startSize: 1,
  startOpacity: 1,
  startRotation: 0,
  startColor: {
    min: { r: 1, g: 1, b: 1 },
    max: { r: 1, g: 1, b: 1 }
  },
  gravity: 0,
  simulationSpace: "LOCAL" /* LOCAL */,
  simulationBackend: "AUTO" /* AUTO */,
  maxParticles: 100,
  emission: {
    rateOverTime: 10,
    rateOverDistance: 0,
    bursts: []
  },
  shape: {
    shape: "SPHERE" /* SPHERE */,
    sphere: {
      radius: 1,
      radiusThickness: 1,
      arc: 360
    },
    cone: {
      angle: 25,
      radius: 1,
      radiusThickness: 1,
      arc: 360
    },
    circle: {
      radius: 1,
      radiusThickness: 1,
      arc: 360
    },
    rectangle: {
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 }
    },
    box: {
      scale: { x: 1, y: 1, z: 1 },
      emitFrom: "VOLUME" /* VOLUME */
    }
  },
  map: void 0,
  renderer: {
    blending: THREE2.NormalBlending,
    discardBackgroundColor: false,
    backgroundColorTolerance: 1,
    backgroundColor: { r: 1, g: 1, b: 1 },
    transparent: true,
    depthTest: true,
    depthWrite: false,
    softParticles: {
      enabled: false,
      intensity: 1
    }
  },
  velocityOverLifetime: {
    isActive: false,
    linear: {
      x: 0,
      y: 0,
      z: 0
    },
    orbital: {
      x: 0,
      y: 0,
      z: 0
    }
  },
  sizeOverLifetime: {
    isActive: false,
    lifetimeCurve: {
      type: "BEZIER" /* BEZIER */,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 1, y: 1, percentage: 1 }
      ]
    }
  },
  colorOverLifetime: {
    isActive: false,
    r: {
      type: "BEZIER" /* BEZIER */,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 }
      ]
    },
    g: {
      type: "BEZIER" /* BEZIER */,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 }
      ]
    },
    b: {
      type: "BEZIER" /* BEZIER */,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 1, percentage: 0 },
        { x: 1, y: 1, percentage: 1 }
      ]
    }
  },
  opacityOverLifetime: {
    isActive: false,
    lifetimeCurve: {
      type: "BEZIER" /* BEZIER */,
      scale: 1,
      bezierPoints: [
        { x: 0, y: 0, percentage: 0 },
        { x: 1, y: 1, percentage: 1 }
      ]
    }
  },
  rotationOverLifetime: {
    isActive: false,
    min: 0,
    max: 0
  },
  noise: {
    isActive: false,
    useRandomOffset: false,
    strength: 1,
    frequency: 0.5,
    octaves: 1,
    positionAmount: 1,
    rotationAmount: 0,
    sizeAmount: 0
  },
  textureSheetAnimation: {
    tiles: new THREE2.Vector2(1, 1),
    timeMode: "LIFETIME" /* LIFETIME */,
    fps: 30,
    startFrame: 0
  },
  forceFields: [],
  collisionPlanes: []
};
var destroyParticleSystem = (particleSystem) => {
  createdParticleSystems = createdParticleSystems.filter(
    ({
      particleSystem: savedParticleSystem,
      trailMesh,
      generalData: { particleSystemId }
    }) => {
      if (savedParticleSystem !== particleSystem) {
        return true;
      }
      removeBezierCurveFunction(particleSystemId);
      if (trailMesh) {
        trailMesh.geometry.dispose();
        if (Array.isArray(trailMesh.material))
          trailMesh.material.forEach((m) => m.dispose());
        else trailMesh.material.dispose();
        if (trailMesh.parent) trailMesh.parent.remove(trailMesh);
      }
      savedParticleSystem.geometry.dispose();
      if (Array.isArray(savedParticleSystem.material))
        savedParticleSystem.material.forEach((material) => material.dispose());
      else savedParticleSystem.material.dispose();
      if (savedParticleSystem.parent)
        savedParticleSystem.parent.remove(savedParticleSystem);
      return false;
    }
  );
};
var _defaultTexture = null;
var getDefaultTexture = () => {
  if (_defaultTexture) return _defaultTexture;
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 1, 1);
  }
  _defaultTexture = new THREE2.Texture(canvas);
  _defaultTexture.needsUpdate = true;
  return _defaultTexture;
};
var prefillFluidState = (buffers, count, cfg) => {
  const pairOf = (v, fb) => {
    if (typeof v === "number" && Number.isFinite(v)) return [v, v];
    if (v && typeof v === "object") {
      const m = v.min;
      const x = v.max;
      const mn = typeof m === "number" && Number.isFinite(m) ? m : fb;
      const mx = typeof x === "number" && Number.isFinite(x) ? x : fb;
      return [mn, mx];
    }
    return [fb, fb];
  };
  const mid = (p) => (p[0] + p[1]) * 0.5;
  const lifeMid = mid(pairOf(cfg.startLifetime, 5)) * 1e3;
  const sizeMid = mid(pairOf(cfg.startSize, 1));
  const rotMid = mid(pairOf(cfg.startRotation, 0));
  const opMid = mid(pairOf(cfg.startOpacity, 1));
  const cMin = cfg.startColor?.min ?? { r: 1, g: 1, b: 1 };
  const cMax = cfg.startColor?.max ?? { r: 1, g: 1, b: 1 };
  const cr = ((cMin.r ?? 1) + (cMax.r ?? 1)) * 0.5;
  const cg = ((cMin.g ?? 1) + (cMax.g ?? 1)) * 0.5;
  const cb = ((cMin.b ?? 1) + (cMax.b ?? 1)) * 0.5;
  const w = (a) => a?.array ?? null;
  const col = w(buffers.color);
  const ps = w(buffers.particleState);
  const sv = w(buffers.startValues);
  const ex = w(buffers.startColorsExt);
  const oi = w(buffers.orbitalIsActive);
  for (let i = 0; i < count; i++) {
    const b = i * 4;
    if (col) {
      col[b] = cr;
      col[b + 1] = cg;
      col[b + 2] = cb;
      col[b + 3] = opMid;
    }
    if (ps) {
      ps[b] = 0;
      ps[b + 1] = sizeMid;
      ps[b + 2] = rotMid;
      ps[b + 3] = 0;
    }
    if (sv) {
      sv[b] = lifeMid;
      sv[b + 1] = sizeMid;
      sv[b + 2] = opMid;
      sv[b + 3] = cr;
    }
    if (ex) {
      ex[b] = cg;
      ex[b + 1] = cb;
      ex[b + 2] = 0;
      ex[b + 3] = i;
    }
    if (oi) {
      oi[b] = 1;
    }
  }
};
var createParticleSystem = (config = DEFAULT_PARTICLE_SYSTEM_CONFIG, externalNow) => {
  const now = externalNow || Date.now();
  const useTSL = _tslMaterialFactory !== null;
  if (!useTSL) {
    throw new Error(
      "three-particles: WebGPU TSL material factory not registered. Call enableWebGPU(renderer) immediately after creating a WebGPURenderer. @cyberluke/three-particles 4.0.0 is GPU-only - no CPU fallback path exists."
    );
  }
  if (!_rendererBackendIsGPU) {
    throw new Error(
      "three-particles: renderer is not a native WebGPU backend. This build has no WebGL2 fallback. Use a new THREE.WebGPURenderer()."
    );
  }
  const factory = _tslMaterialFactory;
  if (!factory.createComputePipeline) {
    throw new Error(
      "three-particles: active WebGPU renderer does not provide a complete TSL compute pipeline (createComputePipeline missing). No CPU fallback exists; install a WebGPU-capable backend."
    );
  }
  const maxParticles = config.maxParticles || DEFAULT_PARTICLE_SYSTEM_CONFIG.maxParticles;
  const normalizedConfig = ObjectUtils.deepMerge(
    DEFAULT_PARTICLE_SYSTEM_CONFIG,
    config,
    { applyToFirstObject: false, skippedProperties: [] }
  );
  if (normalizedConfig.simulationBackend === "CPU") {
    if (!_cpuPreferenceWarned) {
      _cpuPreferencePreferenceWarn();
    }
    normalizedConfig.simulationBackend = "GPU" /* GPU */;
  }
  const requestedRendererType = normalizedConfig.renderer.rendererType || "POINTS" /* POINTS */;
  const effectiveRendererType = resolveWebGPUEffectiveRendererType(
    requestedRendererType
  );
  const rrType = effectiveRendererType;
  const useInstancing = effectiveRendererType === "INSTANCED" /* INSTANCED */ || effectiveRendererType === "MESH" /* MESH */ || effectiveRendererType === "FLUID" /* FLUID */;
  const trailConfig = normalizedConfig.renderer.trail;
  const trailLength = Math.max(2, Math.round(trailConfig?.length ?? 20));
  const trailHistoryAttribute = rrType === "TRAIL" /* TRAIL */ ? new StorageBufferAttribute(
    new Float32Array(maxParticles * (trailLength + 1) * 4),
    4
  ) : null;
  const trailDesc = trailHistoryAttribute ? {
    attribute: trailHistoryAttribute,
    meta: null,
    length: trailLength,
    minVertexDistance: trailConfig?.minVertexDistance ?? 0,
    maxTime: (trailConfig?.maxTime ?? 0) * 1e3
  } : null;
  const subEmitterConfigs = normalizedConfig.subEmitters ?? [];
  const fifos = subEmitterConfigs.map((se) => {
    const capacity = Math.max(1, Math.round(se.maxInstances ?? 32));
    const f = factory.createSubEmitterFifoAttribute(capacity);
    f.trigger = se.trigger === "BIRTH" ? 0 : 1;
    return f;
  });
  const fifoBaseStride = fifos.reduce((m, f) => Math.max(m, f.windowSize), 0);
  const forceFields = normalizeForceFields(normalizedConfig.forceFields);
  const collisionPlanes = normalizeCollisionPlanes(
    normalizedConfig.collisionPlanes
  );
  const pipeline = factory.createComputePipeline(
    maxParticles,
    useInstancing,
    normalizedConfig,
    _particleSystemId,
    // pre-increment inside generalData below would be off by 1; use the raw next id
    forceFields.length,
    collisionPlanes.length,
    fifos,
    trailDesc ?? void 0
  );
  let fluidHighWater = 0;
  let fluidSolverId = null;
  let fluidBoxWidthRatioValue = 1;
  if (rrType === "FLUID" /* FLUID */) {
    const sharedPos = pipeline.buffers.position;
    const sharedVel = pipeline.buffers.velocity;
    const pb = pipeline.buffers;
    const solverName = String(
      normalizedConfig.renderer.fluid?.solver ?? ""
    ).trim().toUpperCase();
    const isSPHSolver = solverName === "SPH";
    const solverId = isSPHSolver ? "SPH" : "MLS-MPM";
    const solverPrefix = isSPHSolver ? "sph" : "mlsmpm";
    const solver = createFluidSimPipeline(
      solverId,
      {
        position: sharedPos,
        velocity: sharedVel
      },
      maxParticles,
      normalizedConfig
    );
    if (isSPHSolver) {
      pb.fluidForceDensity = solver.buffers.forceDensity;
      pb.fluidSortedPosition = solver.buffers.sortedPosition;
      pb.fluidSortedVelocity = solver.buffers.sortedVelocity;
      pb.fluidSortedForceDensity = solver.buffers.sortedForceDensity;
      pb.fluidCellCounts = solver.buffers.cellCounts;
      pb.fluidPrefixSums = solver.buffers.prefixSums;
      pb.fluidParticleCellOffsets = solver.buffers.particleCellOffsets;
      pb.fluidBlockPartials = solver.buffers.blockPartials;
      pb.fluidBlockInclusive = solver.buffers.blockInclusive;
      pb.fluidBlockOffsets = solver.buffers.blockOffsets;
    } else {
      pb.fluidCoefficients = solver.buffers.coefficients;
      pb.fluidCells = solver.buffers.cells;
    }
    fluidHighWater = solver.numParticles;
    fluidSolverId = solverId;
    fluidBoxWidthRatioValue = isSPHSolver ? normalizedConfig.renderer.sph?.boxWidthRatio ?? 1 : normalizedConfig.renderer.mlsMpm?.boxWidthRatio ?? 1;
    prefillFluidState(pipeline.buffers, solver.numParticles, normalizedConfig);
    if (pipeline.uniforms.emitCount) {
      pipeline.uniforms.emitCount.value = 0;
    }
    pipeline.uniforms.fluidBoxWidthRatio = solver.uniforms.boxWidthRatio;
    const fl = pipeline;
    fl.computeNodes = [
      ...fl.computeNodes ?? (pipeline.emitNode && pipeline.simNode ? [pipeline.emitNode, pipeline.simNode] : []),
      ...solver.computeNodes
    ];
    fl.passNames = [
      ...fl.passNames ?? ["emit", "simulate"],
      ...solver.passNames.map((n) => `${solverPrefix}:${n}`)
    ];
    fl.passLayouts = [
      ...fl.passLayouts ?? [],
      ...solver.passLayouts.map((p) => ({
        ...p,
        name: `${solverPrefix}:${p.name}`
      }))
    ];
  }
  const ribbonPipeline = trailDesc ? factory.createTrailRibbonUpdate({
    position: new StorageBufferAttribute(
      new Float32Array(maxParticles * trailLength * 2 * 4),
      4
    ),
    next: new StorageBufferAttribute(
      new Float32Array(maxParticles * trailLength * 2 * 4),
      4
    ),
    uvColorA: new StorageBufferAttribute(
      new Float32Array(maxParticles * trailLength * 2 * 4),
      4
    ),
    colorB: new StorageBufferAttribute(
      new Float32Array(maxParticles * trailLength * 2 * 4),
      4
    ),
    history: trailDesc.attribute,
    meta: trailDesc.meta,
    particleColor: pipeline.buffers.color,
    curveFns: {
      width: trailConfig?.widthOverTrail ? getCurveFunctionFromConfig(
        _particleSystemId,
        trailConfig.widthOverTrail
      ) : void 0,
      opacity: trailConfig?.opacityOverTrail ? getCurveFunctionFromConfig(
        _particleSystemId,
        trailConfig.opacityOverTrail
      ) : void 0,
      colorR: trailConfig?.colorOverTrail?.isActive ? getCurveFunctionFromConfig(
        _particleSystemId,
        trailConfig.colorOverTrail.r
      ) : void 0,
      colorG: trailConfig?.colorOverTrail?.isActive ? getCurveFunctionFromConfig(
        _particleSystemId,
        trailConfig.colorOverTrail.g
      ) : void 0,
      colorB: trailConfig?.colorOverTrail?.isActive ? getCurveFunctionFromConfig(
        _particleSystemId,
        trailConfig.colorOverTrail.b
      ) : void 0
    },
    width: trailConfig?.width ?? 1,
    length: trailLength,
    maxTime: trailDesc.maxTime,
    maxParticles
  }) : null;
  const subEntries = [];
  for (let fi = 0; fi < subEmitterConfigs.length; fi++) {
    const se = subEmitterConfigs[fi];
    const fifo = fifos[fi];
    const childCfg = ObjectUtils.deepMerge(
      getDefaultParticleSystemConfig(),
      se.config ?? {},
      { applyToFirstObject: false, skippedProperties: [] }
    );
    const firstBurst = childCfg.emission?.bursts?.[0];
    const burstCount = firstBurst ? Math.max(
      1,
      Math.ceil(
        calculateValue(_particleSystemId + 1 + fi, firstBurst.count, 0) * (firstBurst.cycles ?? 1)
      )
    ) : 1;
    const perEvent = Math.min(burstCount, fifo.capacity);
    const childMax = Math.max(2, Math.min(perEvent * fifo.capacity, 65536));
    const childRequestedRendererType = childCfg.renderer?.rendererType;
    const childEffectiveRendererType = resolveWebGPUEffectiveRendererType(
      childRequestedRendererType
    );
    const childInstanced = childEffectiveRendererType === "INSTANCED" /* INSTANCED */ || childEffectiveRendererType === "MESH" /* MESH */;
    const childPipeline = factory.createComputePipeline(
      childMax,
      childInstanced,
      childCfg,
      _particleSystemId + 1 + fi,
      0,
      0,
      [],
      void 0
    );
    const childShapeParams = factory.encodeShapeEmitParams(
      childCfg,
      _particleSystemId + 1 + fi
    );
    const childVel = childCfg.velocityOverLifetime;
    const init = factory.createSubEmitterInitUpdate(
      childPipeline.buffers,
      childMax,
      childShapeParams,
      pipeline.buffers,
      maxParticles,
      fifo,
      se.inheritVelocity ?? 0,
      perEvent,
      {
        linear: [
          childVel?.linear?.x,
          childVel?.linear?.y,
          childVel?.linear?.z
        ],
        orbital: [
          childVel?.orbital?.x,
          childVel?.orbital?.y,
          childVel?.orbital?.z
        ]
      }
    );
    subEntries.push({
      fifo,
      pipeline: childPipeline,
      init,
      instanced: childInstanced,
      requestedRendererType: childRequestedRendererType,
      effectiveRendererType: childEffectiveRendererType,
      cfg: childCfg,
      object: null,
      perEvent,
      gravity: childCfg.gravity,
      noise: childCfg.noise?.isActive ? {
        isActive: true,
        strength: childCfg.noise.strength,
        noisePower: 0.15 * childCfg.noise.strength,
        frequency: childCfg.noise.frequency,
        positionAmount: childCfg.noise.positionAmount,
        rotationAmount: childCfg.noise.rotationAmount,
        sizeAmount: childCfg.noise.sizeAmount,
        fbmMax: 2 - Math.pow(2, -childCfg.noise.octaves)
      } : null,
      rate: childCfg.emission?.rateOverTime ? calculateValue(
        _particleSystemId + 1 + fi,
        childCfg.emission.rateOverTime,
        0
      ) : 0,
      acc: 0,
      lastEmit: 0,
      poseFrom: "self",
      selfPose: {
        x: 0,
        y: 0,
        z: 0,
        qx: 0,
        qy: 0,
        qz: 0,
        qw: 1,
        sx: 1,
        sy: 1,
        sz: 1,
        isWorld: childCfg.simulationSpace === "WORLD" /* WORLD */ ? 1 : 0
      }
    });
  }
  const cameraNearFarSource = normalizedConfig.renderer.cameraNearFar;
  const tilesSource = normalizedConfig.textureSheetAnimation?.tiles;
  const elapsedUniform = { value: 0 };
  const sharedUniforms = {
    elapsed: elapsedUniform,
    viewportHeight: { value: 720 },
    cameraNearFar: {
      value: normalizeVector2Value(
        cameraNearFarSource,
        [0.1, 1e3],
        "renderer.cameraNearFar"
      )
    },
    useInstancing: { value: useInstancing },
    softParticlesEnabled: {
      value: !!normalizedConfig.renderer.softParticles?.enabled
    },
    softParticlesIntensity: {
      value: Math.max(
        normalizedConfig.renderer.softParticles?.intensity ?? 1,
        1e-3
      )
    },
    sceneDepthTexture: {
      value: normalizeDepthTextureValue(
        normalizedConfig.renderer.softParticles?.depthTexture,
        "renderer.softParticles.depthTexture"
      )
    },
    discardBackgroundColor: {
      value: !!normalizedConfig.renderer.discardBackgroundColor
    },
    backgroundColor: { value: new THREE2.Color(16777215) },
    backgroundColorTolerance: {
      value: normalizedConfig.renderer.backgroundColorTolerance ?? 0
    },
    map: {
      value: normalizeTextureValue(
        normalizedConfig.map ?? getDefaultTexture(),
        "map"
      )
    },
    startLifetime: { value: 0 },
    startSize: { value: 1 },
    startRotation: { value: 0 },
    startOpacity: { value: 1 },
    startColor: { value: new THREE2.Color(1, 1, 1) },
    lifetime: { value: 0 },
    color: { value: new THREE2.Color(1, 1, 1) },
    // Sprite-sheet animation fields consumed by tsl-shared.createParticleUniforms.
    fps: { value: normalizedConfig.textureSheetAnimation?.fps || 30 },
    useFPSForFrameIndex: {
      value: normalizedConfig.textureSheetAnimation?.timeMode === "FPS" /* FPS */
    },
    tiles: {
      // The ONLY normalizer: `tiles` reaches the TSL factory as a Vector2
      // (also {u,v} pairs are accepted per §10). The engine's own default is
      // already (1,1) via the merged default config.
      value: normalizeVector2Value(
        tilesSource,
        [1, 1],
        "textureSheetAnimation.tiles"
      )
    },
    // FLUID metaball renderer parameters (consumed by createFluidTSLMaterial).
    fluidStretch: {
      value: typeof normalizedConfig.renderer.fluid?.stretch === "number" && Number.isFinite(normalizedConfig.renderer.fluid.stretch) ? normalizedConfig.renderer.fluid.stretch : 1
    },
    fluidAbsorption: {
      value: typeof normalizedConfig.renderer.fluid?.absorption === "number" && Number.isFinite(normalizedConfig.renderer.fluid.absorption) ? normalizedConfig.renderer.fluid.absorption : 1.44
    },
    fluidIor: {
      value: typeof normalizedConfig.renderer.fluid?.ior === "number" && Number.isFinite(normalizedConfig.renderer.fluid.ior) ? normalizedConfig.renderer.fluid.ior : 1.33
    },
    // Extra knobs consumed by the screen-space pass chain
    // (`tsl-fluid-screen-space-material.ts`). Missing entries fall back to
    // the documented {@link FluidConfig} defaults; all three are stored as
    // plain scalars / fixed-length tuples so the TSL side can read them
    // without per-frame normalization.
    fluidSphereSize: {
      value: typeof normalizedConfig.renderer.fluid?.sphereSize === "number" && Number.isFinite(normalizedConfig.renderer.fluid.sphereSize) ? normalizedConfig.renderer.fluid.sphereSize : 1.2
    },
    fluidDensity: {
      value: typeof normalizedConfig.renderer.fluid?.density === "number" && Number.isFinite(normalizedConfig.renderer.fluid.density) ? normalizedConfig.renderer.fluid.density : 0.7
    },
    fluidWaterColor: {
      value: Array.isArray(normalizedConfig.renderer.fluid?.waterColor) && normalizedConfig.renderer.fluid.waterColor.length === 3 ? [
        Number(
          normalizedConfig.renderer.fluid.waterColor[0]
        ) || 0,
        Number(
          normalizedConfig.renderer.fluid.waterColor[1]
        ) || 0,
        Number(
          normalizedConfig.renderer.fluid.waterColor[2]
        ) || 0
      ] : [0, 0.7375, 0.95]
    },
    fluidSphereRender: {
      value: !!normalizedConfig.renderer.fluid?.sphereRender
    }
  };
  const bgVec = normalizeBackgroundToVector3(
    normalizedConfig.renderer.backgroundColor,
    "renderer.backgroundColor"
  );
  sharedUniforms.backgroundColor.value.setRGB(
    bgVec.x,
    bgVec.y,
    bgVec.z
  );
  const rendererConfig = {
    transparent: !!normalizedConfig.renderer.transparent,
    blending: toBlendingConstant(normalizedConfig.renderer.blending),
    depthTest: normalizedConfig.renderer.depthTest !== false,
    depthWrite: normalizedConfig.renderer.depthWrite !== false
  };
  const buffers = pipeline.buffers;
  let geometry;
  if (useInstancing) {
    const g = new THREE2.InstancedBufferGeometry();
    const meshGeometry = normalizedConfig.renderer.mesh?.geometry;
    const baseGeometry = rrType === "MESH" /* MESH */ && meshGeometry ? meshGeometry : new THREE2.BufferGeometry();
    if (rrType !== "MESH" /* MESH */ || !meshGeometry) {
      const quad = new Float32Array([
        -0.5,
        -0.5,
        0,
        0.5,
        -0.5,
        0,
        0.5,
        0.5,
        0,
        -0.5,
        0.5,
        0
      ]);
      const quadUV = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
      const quadNormal = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
      const idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
      baseGeometry.setAttribute("position", new THREE2.BufferAttribute(quad, 3));
      baseGeometry.setAttribute("uv", new THREE2.BufferAttribute(quadUV, 2));
      baseGeometry.setAttribute(
        "normal",
        new THREE2.BufferAttribute(quadNormal, 3)
      );
      baseGeometry.setIndex(new THREE2.BufferAttribute(idx, 1));
    }
    g.setAttribute("position", baseGeometry.getAttribute("position"));
    if (baseGeometry.index !== null) g.setIndex(baseGeometry.index);
    g.instanceCount = maxParticles;
    g.setAttribute("instanceOffset", buffers.position);
    g.setAttribute("instanceColor", buffers.color);
    g.setAttribute("instanceParticleState", buffers.particleState);
    g.setAttribute("instanceStartValues", buffers.startValues);
    if (rrType === "FLUID" /* FLUID */) {
      g.setAttribute("instanceVelocity", buffers.velocity);
    }
    geometry = g;
  } else {
    const g = new THREE2.BufferGeometry();
    g.setAttribute("position", buffers.position);
    g.setAttribute("color", buffers.color);
    g.setAttribute("particleState", buffers.particleState);
    g.setAttribute("startValues", buffers.startValues);
    g.setDrawRange(0, maxParticles);
    geometry = g;
    g.instanceCount = maxParticles;
  }
  const material = factory.createTSLParticleMaterial(
    rrType,
    sharedUniforms,
    rendererConfig,
    true,
    geometry
  );
  let trailGeometry = null;
  if (ribbonPipeline && trailDesc) {
    const rb = ribbonPipeline.buffers;
    const g = new THREE2.BufferGeometry();
    g.setAttribute("position", rb.position);
    g.setAttribute("trailNext", rb.next);
    g.setAttribute("trailUVColor", rb.uvColorA);
    g.setAttribute("trailColorBA", rb.colorB);
    const idx = new Uint32Array(maxParticles * (trailLength - 1) * 6);
    let o = 0;
    for (let pIdx = 0; pIdx < maxParticles; pIdx++) {
      for (let s = 0; s < trailLength - 1; s++) {
        const b = pIdx * trailLength * 2 + s * 2;
        idx[o++] = b;
        idx[o++] = b + 1;
        idx[o++] = b + 2;
        idx[o++] = b + 1;
        idx[o++] = b + 3;
        idx[o++] = b + 2;
      }
    }
    g.setIndex(new THREE2.BufferAttribute(idx, 1));
    g.setDrawRange(0, maxParticles * trailLength * 2);
    trailGeometry = g;
  }
  const trailMaterial = trailGeometry ? factory.createTSLTrailMaterial(
    {
      map: {
        value: normalizedConfig.map ?? getDefaultTexture()
      },
      useMap: { value: !!normalizedConfig.map },
      discardBackgroundColor: {
        value: !!normalizedConfig.renderer.discardBackgroundColor
      },
      backgroundColor: {
        value: normalizedConfig.renderer.backgroundColor ?? {
          r: 1,
          g: 1,
          b: 1
        }
      },
      backgroundColorTolerance: {
        value: normalizedConfig.renderer.backgroundColorTolerance ?? 0
      },
      softParticlesEnabled: {
        value: !!normalizedConfig.renderer.softParticles?.enabled
      },
      softParticlesIntensity: {
        value: Math.max(
          normalizedConfig.renderer.softParticles?.intensity ?? 1,
          1e-3
        )
      },
      sceneDepthTexture: {
        value: normalizedConfig.renderer.softParticles?.depthTexture ?? null
      },
      cameraNearFar: { value: new THREE2.Vector2(0.1, 1e3) }
    },
    {
      transparent: !!normalizedConfig.renderer.transparent,
      blending: toBlendingConstant(normalizedConfig.renderer.blending),
      depthTest: normalizedConfig.renderer.depthTest !== false,
      depthWrite: normalizedConfig.renderer.depthWrite !== false
    }
  ) : null;
  const fluidPassGeometry = material.__fluidPassGeometry;
  const particleSystem = trailGeometry ? new THREE2.Mesh(trailGeometry, trailMaterial) : useInstancing ? new THREE2.Mesh(fluidPassGeometry ?? geometry, material) : new THREE2.Points(geometry, material);
  particleSystem.frustumCulled = false;
  for (const e of subEntries) {
    const cb = e.pipeline.buffers;
    const childMax = e.pipeline.allocatorCount - 1;
    const childGeometry = e.instanced ? (() => {
      const g = new THREE2.InstancedBufferGeometry();
      const quad = new Float32Array([
        -0.5,
        -0.5,
        0,
        0.5,
        -0.5,
        0,
        0.5,
        0.5,
        0,
        -0.5,
        0.5,
        0
      ]);
      const idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
      g.setAttribute("position", new THREE2.BufferAttribute(quad, 3));
      g.setIndex(new THREE2.BufferAttribute(idx, 1));
      g.instanceCount = childMax;
      g.setAttribute("instanceOffset", cb.position);
      g.setAttribute("instanceColor", cb.color);
      g.setAttribute("instanceParticleState", cb.particleState);
      g.setAttribute("instanceStartValues", cb.startValues);
      if (e.effectiveRendererType === "FLUID" /* FLUID */) {
        g.setAttribute("instanceVelocity", cb.velocity);
      }
      return g;
    })() : (() => {
      const g = new THREE2.BufferGeometry();
      g.setAttribute("position", cb.position);
      g.setAttribute("color", cb.color);
      g.setAttribute("particleState", cb.particleState);
      g.setAttribute("startValues", cb.startValues);
      g.setDrawRange(0, childMax);
      return g;
    })();
    const childUniforms = {
      ...sharedUniforms,
      useInstancing: { value: e.instanced }
    };
    const childMaterial = factory.createTSLParticleMaterial(
      e.effectiveRendererType,
      childUniforms,
      rendererConfig,
      true
    );
    const childObject = e.instanced ? new THREE2.Mesh(childGeometry, childMaterial) : new THREE2.Points(childGeometry, childMaterial);
    childObject.frustumCulled = false;
    particleSystem.add(childObject);
    e.object = childObject;
  }
  if (import.meta.env?.DEV !== false) {
    const required = useInstancing ? [
      "position",
      // quad / mesh vertex positions
      "instanceOffset",
      // GPU particle position
      "instanceColor",
      // GPU particle RGBA
      "instanceParticleState",
      // GPU packed state vec4
      "instanceStartValues"
      // GPU packed initial-state vec4
    ] : ["position", "color", "particleState", "startValues"];
    for (const name of required) {
      if (!geometry.getAttribute(name)) {
        throw new Error(
          "three-particles: " + (useInstancing ? "instanced" : "POINTS") + " geometry " + name + " is missing its required contract attribute."
        );
      }
    }
    const contractIdentity = useInstancing ? [
      ["instanceOffset", buffers.position],
      ["instanceColor", buffers.color],
      ["instanceParticleState", buffers.particleState],
      ["instanceStartValues", buffers.startValues]
    ] : [
      ["position", buffers.position],
      ["color", buffers.color],
      ["particleState", buffers.particleState],
      ["startValues", buffers.startValues]
    ];
    for (const [name, buf] of contractIdentity) {
      if (geometry.getAttribute(name) !== buf) {
        throw new Error(
          `three-particles: attribute "${name}" is not the compute-owned storage buffer.`
        );
      }
    }
    const kind = pipeline.shapeUniforms.shapeKind.value;
    if (!(kind >= 0 && kind <= 4)) {
      throw new Error(
        `three-particles: gpuShapeKind ${kind} outside 0..4 (SPHERE..BOX).`
      );
    }
    if (!(maxParticles > 0)) {
      throw new Error("three-particles: maxParticles must be > 0.");
    }
    if (pipeline.allocatorCount !== maxParticles + 1) {
      throw new Error(
        "three-particles: allocator capacity must equal maxParticles + 1."
      );
    }
    const passLayouts = [
      ...pipeline.passLayouts ?? [],
      ...ribbonPipeline?.passLayouts ?? [],
      ...subEntries.flatMap((e) => [
        ...e.init.passLayouts ?? [],
        ...(e.pipeline.passLayouts ?? []).map(
          (p) => ({ ...p, name: `child:${p.name}` })
        )
      ])
    ];
    for (const pass2 of passLayouts) {
      if (pass2.storageBindings > 8) {
        throw new Error(
          `${pass2.name}: ${pass2.storageBindings} storage buffers > guaranteed limit 8`
        );
      }
    }
    if (trailDesc && trailDesc.meta !== pipeline.trailMeta) {
      throw new Error("three-particles: trail ring meta buffer mismatch.");
    }
    for (const f of fifos) {
      const n = f.counter.array.length;
      if (n !== 2) {
        throw new Error(
          "three-particles: sub-emitter FIFO must expose exactly 2 ping-pong counter slots."
        );
      }
      const p = f.payload.array.length;
      if (p !== 2 * 6 * f.capacity) {
        throw new Error(
          "three-particles: sub-emitter FIFO payload length must be 2 * 6 * capacity."
        );
      }
    }
  }
  const _numOr = (v, d) => typeof v === "number" && Number.isFinite(v) ? v : d;
  const xform = normalizedConfig.transform;
  if (xform?.position) {
    particleSystem.position.set(
      _numOr(xform.position.x, 0),
      _numOr(xform.position.y, 0),
      _numOr(xform.position.z, 0)
    );
  }
  if (xform?.rotation) {
    particleSystem.rotation.set(
      THREE2.MathUtils.degToRad(_numOr(xform.rotation.x, 0)),
      THREE2.MathUtils.degToRad(_numOr(xform.rotation.y, 0)),
      THREE2.MathUtils.degToRad(_numOr(xform.rotation.z, 0))
    );
  }
  if (xform?.scale) {
    particleSystem.scale.set(
      _numOr(xform.scale.x, 1),
      _numOr(xform.scale.y, 1),
      _numOr(xform.scale.z, 1)
    );
  }
  particleSystem.updateMatrix();
  particleSystem.updateMatrixWorld(true);
  if (normalizedConfig.simulationSpace === "WORLD" /* WORLD */) {
    particleSystem.matrixWorldAutoUpdate = false;
    particleSystem.matrixWorld.identity();
  }
  const generalData = {
    particleSystemId: _particleSystemId++,
    normalizedLifetimePercentage: 0,
    distanceFromLastEmitByDistance: 0,
    lastWorldPosition: new THREE2.Vector3(-99999),
    currentWorldPosition: new THREE2.Vector3(-99999),
    worldPositionChange: new THREE2.Vector3(),
    sourceWorldMatrix: new THREE2.Matrix4(),
    worldQuaternion: new THREE2.Quaternion(),
    wrapperQuaternion: new THREE2.Quaternion(),
    worldScale: new THREE2.Vector3(1, 1, 1),
    worldEuler: new THREE2.Euler(),
    gravityVelocity: new THREE2.Vector3(0, 0, 0),
    startValues: {},
    linearVelocityData: void 0,
    orbitalVelocityData: void 0,
    lifetimeValues: {},
    creationTimes: new Float32Array(0),
    cpuDirtyParticleWatermark: -1,
    highWaterIndex: fluidHighWater,
    // FLUID solver snapshots (§2a); `null` keeps the single-pass metaball path.
    fluidSolver: fluidSolverId,
    fluidBoxWidthRatio: fluidBoxWidthRatioValue,
    noise: {
      isActive: normalizedConfig.noise.isActive,
      strength: normalizedConfig.noise.strength,
      // Oracle `0.15 * strength`; the single fbmMax division lives inside the
      // FBM sum (CPU: FBM.get3; GPU: the octave loop amp / fbmMax).
      noisePower: 0.15 * normalizedConfig.noise.strength,
      frequency: normalizedConfig.noise.frequency,
      positionAmount: normalizedConfig.noise.positionAmount,
      rotationAmount: normalizedConfig.noise.rotationAmount,
      sizeAmount: normalizedConfig.noise.sizeAmount,
      fbmMax: 2 - Math.pow(2, -normalizedConfig.noise.octaves)
    },
    isEnabled: true,
    burstStates: normalizedConfig.emission.bursts?.length ? normalizedConfig.emission.bursts.map(() => ({
      cyclesExecuted: 0,
      lastCycleTime: 0,
      probabilityPassed: false
    })) : void 0
  };
  const props = {
    particleSystem,
    mappedAttributes: {
      position: buffers.position,
      isActive: buffers.orbitalIsActive,
      lifetime: buffers.particleState,
      startLifetime: buffers.startValues,
      startFrame: buffers.particleState,
      size: buffers.particleState,
      rotation: buffers.particleState,
      color: buffers.color
    },
    // ?? Deprecated zero-size sentinels (GPU-only v4) ????
    // These legacy CPU particle-state fields are not authoritative anymore: the
    // compute kernels own the state in GPU storage. Only the TRAIL path (which
    // throws in v4) consumed them, so they are 0-length placeholders.
    scalarArray: new Float32Array(0),
    scalarInterleavedBuffer: new THREE2.InterleavedBuffer(
      new Float32Array(0),
      SCALAR_STRIDE
    ),
    elapsedUniform,
    generalData,
    onUpdate: () => {
    },
    onComplete: () => {
    },
    creationTime: now + (normalizedConfig.startDelay || 0),
    lastEmissionTime: now,
    emissionAccumulator: 0,
    duration: normalizedConfig.duration,
    looping: normalizedConfig.looping,
    simulationSpace: normalizedConfig.simulationSpace,
    gravity: normalizedConfig.gravity,
    normalizedForceFields: forceFields,
    normalizedCollisionPlanes: collisionPlanes,
    emission: normalizedConfig.emission,
    normalizedConfig,
    iterationCount: 0,
    velocities: [],
    freeList: [],
    deactivateParticle: () => {
    },
    killParticle: () => {
    },
    activateParticle: () => {
    },
    computePipeline: pipeline,
    useGPUCompute: true,
    computeDispatchReady: false,
    maxParticles,
    material,
    geometry,
    rrType,
    requestedRendererType,
    effectiveRendererType: rrType,
    sharedUniforms,
    allComputeNodes: [
      ...pipeline.computeNodes ?? [],
      ...ribbonPipeline ? [ribbonPipeline.ribbonNode] : [],
      ...subEntries.flatMap((e) => [
        e.init.commandBuildNode,
        e.init.childInitNode,
        ...e.init.counterClearNode !== null && e.init.counterClearNode !== void 0 ? [e.init.counterClearNode] : [],
        ...e.pipeline.computeNodes ?? []
      ])
    ],
    passNames: [
      ...pipeline.passNames ?? ["emit", "simulate"],
      ...ribbonPipeline ? ["trail-ribbon"] : [],
      ...subEntries.flatMap((e, ei) => [
        `sub${ei}:command-build`,
        `sub${ei}:child-init`,
        `sub${ei}:counter-clear`,
        `sub${ei}:child-emit`,
        `sub${ei}:child-sim`
      ])
    ],
    fifoBaseStride,
    ribbonUniforms: ribbonPipeline ? ribbonPipeline.uniforms : void 0,
    ribbonBuffers: ribbonPipeline ? ribbonPipeline.buffers : void 0,
    frameParity: 0,
    subEntries: subEntries.map((e) => ({
      fifo: { capacity: e.fifo.capacity, windowSize: e.fifo.windowSize },
      requestedRendererType: e.requestedRendererType,
      effectiveRendererType: e.effectiveRendererType,
      pipeline: e.pipeline,
      init: e.init,
      gravity: e.gravity,
      noise: e.noise,
      rate: e.rate,
      acc: 0,
      isWorld: e.selfPose.isWorld,
      quat: [e.selfPose.qx, e.selfPose.qy, e.selfPose.qz, e.selfPose.qw],
      scale: [e.selfPose.sx, e.selfPose.sy, e.selfPose.sz],
      position: [
        _numOr(
          e.cfg.transform?.position?.x,
          0
        ),
        _numOr(
          e.cfg.transform?.position?.y,
          0
        ),
        _numOr(
          e.cfg.transform?.position?.z,
          0
        )
      ]
    }))
  };
  for (const e of subEntries) {
    if (!e.object) continue;
    const tf = e.cfg.transform;
    if (tf?.position) {
      e.object.position.set(
        _numOr(tf.position.x, 0),
        _numOr(tf.position.y, 0),
        _numOr(tf.position.z, 0)
      );
    }
    if (tf?.rotation) {
      e.object.rotation.set(
        THREE2.MathUtils.degToRad(_numOr(tf.rotation.x, 0)),
        THREE2.MathUtils.degToRad(_numOr(tf.rotation.y, 0)),
        THREE2.MathUtils.degToRad(_numOr(tf.rotation.z, 0))
      );
    }
    if (tf?.scale) {
      e.object.scale.set(
        _numOr(tf.scale.x, 1),
        _numOr(tf.scale.y, 1),
        _numOr(tf.scale.z, 1)
      );
    }
    e.object.updateMatrix();
    const q = new THREE2.Quaternion().setFromEuler(
      new THREE2.Euler(
        THREE2.MathUtils.degToRad(
          _numOr(tf?.rotation?.x, 0)
        ),
        THREE2.MathUtils.degToRad(
          _numOr(tf?.rotation?.y, 0)
        ),
        THREE2.MathUtils.degToRad(
          _numOr(tf?.rotation?.z, 0)
        ),
        "XYZ"
      )
    );
    const entry = props.subEntries?.[subEntries.indexOf(e)];
    if (entry) {
      entry.quat = [q.x, q.y, q.z, q.w];
      entry.scale = [
        _numOr(tf?.scale?.x, 1),
        _numOr(tf?.scale?.y, 1),
        _numOr(tf?.scale?.z, 1)
      ];
    }
  }
  createdParticleSystems.push(props);
  const _dbgPassCounts = [
    ...(pipeline.passLayouts ?? []).map((p) => [p.name, p.storageBindings]),
    ...(ribbonPipeline?.passLayouts ?? []).map((p) => [p.name, p.storageBindings]),
    ...subEntries.flatMap((e, ei) => [
      ...(e.init.passLayouts ?? []).map(
        (p) => [`sub${ei}:${p.name}`, p.storageBindings]
      ),
      ...(e.pipeline.passLayouts ?? []).map(
        (p) => [`sub${ei}:${p.name}`, p.storageBindings]
      )
    ])
  ];
  const _dbgMaxPass = _dbgPassCounts.reduce((m, p) => Math.max(m, p[1]), 0);
  if (typeof console !== "undefined" && console.log) {
    const logCfg = normalizedConfig;
    const shpU = pipeline.shapeUniforms;
    const sv = logCfg.startValues;
    const u = pipeline.uniforms;
    console.log(`[PS:create] system #${generalData.particleSystemId}`, {
      rendererType: rrType,
      requestedRendererType,
      effectiveRendererType: rrType,
      simulationSpace: normalizedConfig.simulationSpace,
      maxParticles,
      useInstancing
    });
    console.log(`[PS:config] system #${generalData.particleSystemId}`, {
      shape: {
        publicKind: logCfg.shape?.shape ?? null,
        gpuShapeKind: shpU.shapeKind?.value ?? 0,
        radius: shpU.radius?.value ?? logCfg.shape?.radius ?? null,
        radiusThickness: shpU.radiusThickness?.value ?? null,
        arcDeg: shpU.arcDeg?.value ?? null,
        coneAngleDeg: shpU.coneAngleDeg?.value ?? null,
        rectScale: [shpU.rectScaleX?.value, shpU.rectScaleY?.value],
        rectRotationDeg: [shpU.rectRotXDeg?.value, shpU.rectRotYDeg?.value],
        boxScale: [shpU.boxSX?.value, shpU.boxSY?.value, shpU.boxSZ?.value],
        boxEmitFrom: shpU.boxEmitFrom?.value ?? null
      },
      transform: {
        position: xform?.position ?? null,
        rotation: xform?.rotation ?? null,
        scale: xform?.scale ?? null
      },
      emission: {
        rateOverTime: logCfg.emission?.rateOverTime ?? 0,
        rateOverDistance: logCfg.emission?.rateOverDistance ?? 0,
        bursts: logCfg.emission?.bursts?.length ?? 0
      },
      startValues: {
        lifetime: sv?.startLifetime ?? null,
        speed: sv?.startSpeed ?? null,
        size: sv?.startSize ?? null,
        rotation: sv?.startRotation ?? null,
        color: sv?.startColor ?? null,
        opacity: sv?.startOpacity ?? null
      },
      textureId: config.textureId ?? config._editorData?.textureId ?? null,
      textureResolved: !!normalizedConfig.map,
      forceFieldCount: forceFields.length,
      collisionPlaneCount: collisionPlanes.length,
      subEmitterCount: (normalizedConfig.subEmitters ?? []).length,
      trailEnabled: !!trailDesc,
      modifiers: {
        linearVelocity: !!logCfg.velocityOverLifetime?.isActive && (u.linearVelX !== void 0 || u.axisLinXMin !== void 0 || !!(logCfg.velocityOverLifetime?.linear && Object.values(logCfg.velocityOverLifetime.linear).some(
          (value) => value !== void 0 && value !== 0
        ))),
        orbitalVelocity: !!logCfg.velocityOverLifetime?.isActive && !!(logCfg.velocityOverLifetime?.orbital && Object.values(logCfg.velocityOverLifetime.orbital).some(
          (value) => value !== void 0 && value !== 0
        )),
        sizeOverLifetime: !!normalizedConfig.sizeOverLifetime?.isActive,
        opacityOverLifetime: !!normalizedConfig.opacityOverLifetime?.isActive,
        colorOverLifetime: !!normalizedConfig.colorOverLifetime?.isActive,
        rotationOverLifetime: !!normalizedConfig.rotationOverLifetime?.isActive,
        noise: !!normalizedConfig.noise?.isActive
      }
    });
    console.log(
      `[PS:pipeline] system #${generalData.particleSystemId}: ${(props.passNames ?? []).join(" -> ") || "emit -> simulate"} | storageBindings=${_dbgPassCounts.map((p) => `${p[0]}=${p[1]}\u22648`).join(" ")} | packedFloats=${pipeline.buffers.packedData?.length ?? 0}`
    );
  }
  const update = (cycleData) => {
    updateParticleSystemInstance(props, cycleData);
  };
  const resumeEmitter = () => {
    generalData.isEnabled = true;
  };
  const pauseEmitter = () => {
    generalData.isEnabled = false;
  };
  const dispose = () => {
    destroyParticleSystem(particleSystem);
  };
  const updateConfig = (partial) => {
    ObjectUtils.deepMerge(normalizedConfig, partial, {
      applyToFirstObject: true,
      skippedProperties: []
    });
  };
  return {
    instance: particleSystem,
    resumeEmitter,
    pauseEmitter,
    dispose,
    update,
    updateConfig,
    /**
     * ?? Deprecated synchronous active count ????
     * Returns -1 (= unsupported) in the GPU-only engine: the authoritative count
     * is `maxParticles - allocator[0]` which lives in GPU storage and is only
     * available through an explicit (throttled) `getArrayBufferAsync` read-back.
     */
    getActiveParticleCount: () => -1,
    computeNode: props.allComputeNodes && props.allComputeNodes.length > 0 ? props.allComputeNodes : pipeline.computeNodes ?? pipeline.computeNode,
    /**
         * ?? Temporary one-shot GPU debug handle (deprecated, no per-frame cost) ????
         * getActiveParticleCount() stays -1; this object is the raw material for an
         * explicit 
    enderer.getArrayBufferAsync(...) read-back (bytes, multiples of 4).
         * lastEmitCount() mirrors uEmitCount, the u32 count written per frame.
         */
    gpuDebug: {
      maxParticles,
      allocatorCount: pipeline.allocatorCount,
      /** Canonical requested vs effective GPU renderer classes (§2). */
      requestedRendererType,
      effectiveRendererType: rrType,
      /** u32 birth system seed for this pipeline (written ONCE at create). */
      systemSeed: pipeline.uniforms.seed.value,
      buffers: pipeline.buffers,
      emitNode: pipeline.emitNode,
      simNode: pipeline.simNode,
      passNames: pipeline.passNames ?? ["emit", "simulate"],
      allPassNames: props.passNames ?? [],
      storageBindingCount: _dbgMaxPass,
      passBindingCounts: _dbgPassCounts,
      lastEmitCount: () => pipeline.uniforms.emitCount.value,
      /**
       * Per-sub-emitter-child canonical pairs (§2/§21): each child pool's own
       * requested vs effective renderer class + its events-per-frame.
       */
      subEmitters: (subEntries ?? []).map((e) => ({
        requestedRendererType: e.requestedRendererType ?? null,
        effectiveRendererType: e.effectiveRendererType,
        perEvent: e.perEvent
      })),
      /** Decode summary for the `[PS:config]` / `[PS:pipeline]` logs. */
      snapshot: () => {
        const shp = normalizedConfig.shape;
        const branch = shp.shape === "CONE" ? shp.cone : shp.shape === "CIRCLE" ? shp.circle : shp.sphere;
        const tex = normalizedConfig.map;
        return {
          systemId: generalData.particleSystemId,
          // Canonical effective + original requested renderer classes (§2).
          effectiveRendererType: rrType,
          requestedRendererType,
          rendererType: rrType,
          simulationSpace: normalizedConfig.simulationSpace,
          maxParticles,
          shape: {
            publicShape: shp.shape,
            gpuShapeKind: pipeline.shapeUniforms?.shapeKind?.value ?? 0,
            radius: branch?.radius ?? null,
            radiusThickness: branch?.radiusThickness ?? null,
            arcDeg: branch?.arc ?? null,
            coneAngleDeg: shp.shape === "CONE" ? shp.cone?.angle ?? null : null,
            rectScale: shp.rectangle?.scale ?? null,
            rectRotation: shp.rectangle?.rotation ?? null,
            boxScale: shp.box?.scale ?? null,
            boxEmitFrom: shp.box?.emitFrom ?? null
          },
          textureId: config.textureId ?? config._editorData?.textureId ?? null,
          textureResolved: !!normalizedConfig.map,
          textureDimensions: tex?.image ? [tex.image.width ?? 0, tex.image.height ?? 0] : null,
          forceFieldCount: (normalizedConfig.forceFields ?? []).length,
          collisionPlaneCount: (normalizedConfig.collisionPlanes ?? []).length,
          subEmitterCount: (normalizedConfig.subEmitters ?? []).length,
          trailEnabled: !!normalizedConfig.renderer.trail
        };
      }
    }
  };
};
var _lastUploadStampMap = /* @__PURE__ */ new WeakMap();
var _cmdUploadSeen = /* @__PURE__ */ new WeakSet();
var updateParticleSystemInstance = (props, { now, delta, elapsed }) => {
  const {
    generalData,
    normalizedConfig,
    particleSystem,
    elapsedUniform,
    creationTime,
    normalizedForceFields,
    normalizedCollisionPlanes,
    emission,
    computePipeline: pipeline,
    maxParticles = 0,
    allComputeNodes,
    subEntries,
    fifoBaseStride = 0,
    ribbonUniforms
  } = props;
  if (!pipeline) return;
  const u = pipeline.uniforms;
  const dur = normalizedConfig.duration;
  const lifetime = now - creationTime;
  const loop = normalizedConfig.looping;
  const iterationTimeMs = loop ? lifetime % (dur * 1e3) : lifetime;
  generalData.normalizedLifetimePercentage = Math.max(
    Math.min(iterationTimeMs / 1e3 / dur, 1),
    0
  );
  elapsedUniform.value = elapsed;
  const gv = generalData.gravityVelocity;
  gv.set(0, normalizedConfig.gravity, 0);
  if (normalizedConfig.simulationSpace === "WORLD" /* WORLD */) {
    particleSystem.updateMatrix();
    _tmpM1.copy(particleSystem.matrix);
    if (particleSystem.parent) {
      particleSystem.parent.updateMatrixWorld();
      _tmpM1.premultiply(particleSystem.parent.matrixWorld);
    }
    _tmpM1.decompose(
      generalData.currentWorldPosition,
      generalData.worldQuaternion,
      generalData.worldScale
    );
  } else {
    particleSystem.updateMatrixWorld();
    particleSystem.getWorldPosition(generalData.currentWorldPosition);
    particleSystem.getWorldQuaternion(generalData.worldQuaternion);
    particleSystem.getWorldScale(generalData.worldScale);
    _tmpQ1.copy(generalData.worldQuaternion).invert();
    gv.applyQuaternion(_tmpQ1);
    gv.x /= generalData.worldScale.x || 1;
    gv.y /= generalData.worldScale.y || 1;
    gv.z /= generalData.worldScale.z || 1;
  }
  if (generalData.lastWorldPosition.x !== -99999) {
    _lastWorldPositionSnapshot.copy(generalData.lastWorldPosition);
    generalData.distanceFromLastEmitByDistance += _lastWorldPositionSnapshot.distanceTo(generalData.currentWorldPosition);
  }
  generalData.lastWorldPosition.copy(generalData.currentWorldPosition);
  let emitCount = 0;
  if (generalData.isEnabled && (loop || iterationTimeMs < dur * 1e3)) {
    const lastEmit = props.lastEmissionTime;
    const emissionDelta = now - lastEmit;
    if (emissionDelta > 0) {
      props.lastEmissionTime = now;
      if (emission.rateOverTime) {
        props.emissionAccumulator += calculateValue(
          generalData.particleSystemId,
          emission.rateOverTime,
          generalData.normalizedLifetimePercentage
        ) * (emissionDelta / 1e3);
      }
    }
    emitCount += Math.floor(props.emissionAccumulator);
    if (emitCount > 0) props.emissionAccumulator -= emitCount;
    if (emission.rateOverDistance && generalData.distanceFromLastEmitByDistance > 0) {
      const r = calculateValue(
        generalData.particleSystemId,
        emission.rateOverDistance,
        generalData.normalizedLifetimePercentage
      );
      if (r > 0) {
        const n2 = Math.floor(generalData.distanceFromLastEmitByDistance * r);
        emitCount += n2;
        generalData.distanceFromLastEmitByDistance = Math.max(
          generalData.distanceFromLastEmitByDistance - n2 / r,
          0
        );
      }
    }
    if (emission.bursts && generalData.burstStates) {
      const bursts = emission.bursts;
      const states = generalData.burstStates;
      const tSec = iterationTimeMs / 1e3;
      for (let i = 0; i < bursts.length; i++) {
        const b = bursts[i];
        const s = states[i];
        const cyc = b.cycles ?? 1;
        const iv = b.interval ?? 0;
        const prob = b.probability ?? 1;
        if (loop && tSec < (b.time ?? 0) && s.cyclesExecuted > 0) {
          s.cyclesExecuted = 0;
          s.lastCycleTime = 0;
          s.probabilityPassed = false;
        }
        if (s.cyclesExecuted >= cyc) continue;
        const next = (b.time ?? 0) + s.cyclesExecuted * iv;
        if (tSec >= next) {
          if (s.cyclesExecuted === 0)
            s.probabilityPassed = Math.random() < prob;
          if (s.probabilityPassed) {
            emitCount += Math.floor(
              calculateValue(
                generalData.particleSystemId,
                b.count,
                generalData.normalizedLifetimePercentage
              )
            );
          }
          s.cyclesExecuted++;
          s.lastCycleTime = tSec;
        }
      }
    }
    if (emitCount > maxParticles) emitCount = maxParticles;
  }
  u.delta.value = delta;
  u.deltaMs.value = delta * 1e3;
  u.gravityVelocity.value.copy(gv);
  u.emitCount.value = emitCount;
  pipeline.emitNode.count = Math.max(
    1,
    emitCount
  );
  if (pipeline.subBirthEventsNode) {
    pipeline.subBirthEventsNode.count = Math.max(1, emitCount);
  }
  const n = generalData.noise;
  if (u.noiseStrength)
    u.noiseStrength.value = n.strength;
  if (u.noisePower) u.noisePower.value = n.noisePower;
  if (u.noiseFrequency)
    u.noiseFrequency.value = n.frequency;
  if (u.noisePositionAmount)
    u.noisePositionAmount.value = n.positionAmount;
  if (u.noiseRotationAmount)
    u.noiseRotationAmount.value = n.rotationAmount;
  if (u.noiseSizeAmount)
    u.noiseSizeAmount.value = n.sizeAmount;
  if (generalData.fluidSolver) {
    u.emitCount.value = 0;
    pipeline.emitNode.count = 1;
    if (u.fluidBoxWidthRatio) {
      u.fluidBoxWidthRatio.value = generalData.fluidBoxWidthRatio ?? 1;
    }
  }
  const pose = pipeline.emitterPose;
  if (pose) {
    if (normalizedConfig.simulationSpace === "WORLD" /* WORLD */) {
      particleSystem.updateMatrix();
      _tmpM1.copy(particleSystem.matrix);
      if (particleSystem.parent) {
        particleSystem.parent.updateMatrixWorld();
        _tmpM1.premultiply(particleSystem.parent.matrixWorld);
      }
      _tmpM1.decompose(_tmpV1, _tmpQ1, _tmpV2);
      pose.positionW.value.set(_tmpV1.x, _tmpV1.y, _tmpV1.z, 1);
      pose.wrapperQuat.value.set(_tmpQ1.x, _tmpQ1.y, _tmpQ1.z, _tmpQ1.w);
      pose.worldScale.value.set(_tmpV2.x || 1, _tmpV2.y || 1, _tmpV2.z || 1);
    } else {
      pose.positionW.value.set(0, 0, 0, 0);
      pose.wrapperQuat.value.set(0, 0, 0, 1);
      pose.worldScale.value.set(1, 1, 1);
    }
  }
  const parity = (props.frameParity ?? 0) % 2;
  const fifoBase = parity;
  if (u.fifoBase) u.fifoBase.value = fifoBase;
  if (u.nowMs) u.nowMs.value = now;
  if (ribbonUniforms?.nowMs) ribbonUniforms.nowMs.value = now;
  for (const e of subEntries ?? []) {
    const cp = e.pipeline;
    if (!cp) continue;
    const cu = cp.uniforms;
    if (cu.delta) cu.delta.value = delta;
    if (cu.deltaMs) cu.deltaMs.value = delta * 1e3;
    if (cu.nowMs) cu.nowMs.value = now;
    if (cu.gravityVelocity) {
      cu.gravityVelocity.value.set(
        0,
        e.gravity,
        0
      );
    }
    if (e.noise) {
      if (cu.noiseStrength)
        cu.noiseStrength.value = e.noise.strength;
      if (cu.noisePower)
        cu.noisePower.value = e.noise.noisePower;
      if (cu.noiseFrequency)
        cu.noiseFrequency.value = e.noise.frequency;
      if (cu.noisePositionAmount)
        cu.noisePositionAmount.value = e.noise.positionAmount;
      if (cu.noiseRotationAmount)
        cu.noiseRotationAmount.value = e.noise.rotationAmount;
      if (cu.noiseSizeAmount)
        cu.noiseSizeAmount.value = e.noise.sizeAmount;
    }
    if (cu.fifoBase) cu.fifoBase.value = fifoBase;
    if (e.init.uniforms.fifoBase) e.init.uniforms.fifoBase.value = fifoBase;
    let childEmit = 0;
    if (e.rate > 0) {
      e.acc += e.rate * delta / 1;
      childEmit = Math.floor(e.acc);
      if (childEmit > 0) e.acc -= childEmit;
    }
    const childCapacity = Math.max(2, (cp.allocatorCount ?? 2) - 1);
    if (childEmit > childCapacity) childEmit = childCapacity;
    if (cp.emitNode) cp.emitNode.count = Math.max(1, childEmit);
    if (cu.emitCount) cu.emitCount.value = childEmit;
    const cpose = cp.emitterPose;
    if (cpose) {
      if (e.isWorld === 1) {
        cpose.positionW.value.set(
          e.position[0],
          e.position[1],
          e.position[2],
          1
        );
        cpose.wrapperQuat.value.set(e.quat[0], e.quat[1], e.quat[2], e.quat[3]);
        cpose.worldScale.value.set(e.scale[0], e.scale[1], e.scale[2]);
      } else {
        cpose.positionW.value.set(0, 0, 0, 0);
        cpose.wrapperQuat.value.set(0, 0, 0, 1);
        cpose.worldScale.value.set(1, 1, 1);
      }
    }
    const ip = e.init.uniforms;
    if (ip.positionW && ip.wrapperQuat) {
      if (e.isWorld === 1) {
        ip.positionW.value.set(e.position[0], e.position[1], e.position[2], 1);
        ip.wrapperQuat.value.set(e.quat[0], e.quat[1], e.quat[2], e.quat[3]);
      } else {
        ip.positionW.value.set(0, 0, 0, 0);
        ip.wrapperQuat.value.set(0, 0, 0, 1);
      }
    }
  }
  const ffInfo = pipeline.forceFieldInfo;
  const cInfo = pipeline.collisionPlaneInfo ?? null;
  if ((ffInfo || cInfo) && _tslMaterialFactory) {
    const cdArr = pipeline.buffers.packedData;
    const cdNode = pipeline.packedDataNode;
    if (ffInfo && normalizedForceFields.length > 0) {
      const encFF = _tslMaterialFactory.encodeForceFieldsForGPU(
        normalizedForceFields,
        generalData.particleSystemId,
        generalData.normalizedLifetimePercentage
      );
      let changedFF = false;
      for (let k = 0; k < encFF.length; k++)
        if (cdArr[ffInfo.offset + k] !== encFF[k]) {
          changedFF = true;
          break;
        }
      if (changedFF) {
        cdArr.set(encFF, ffInfo.offset);
        cdNode.addUpdateRange(ffInfo.offset, encFF.length);
        cdNode.needsUpdate = true;
      }
      ffInfo.countUniform.value = normalizedForceFields.length;
    }
    if (cInfo && normalizedCollisionPlanes.length > 0) {
      const encCP = _tslMaterialFactory.encodeCollisionPlanesForGPU(
        normalizedCollisionPlanes
      );
      let changedCP = false;
      for (let k = 0; k < encCP.length; k++)
        if (cdArr[cInfo.offset + k] !== encCP[k]) {
          changedCP = true;
          break;
        }
      if (changedCP) {
        cdArr.set(encCP, cInfo.offset);
        cdNode.addUpdateRange(cInfo.offset, encCP.length);
        cdNode.needsUpdate = true;
      }
      cInfo.countUniform.value = normalizedCollisionPlanes.length;
    }
  }
  const bufs = pipeline.buffers;
  let stamp = _lastUploadStampMap.get(bufs);
  if (stamp === void 0 || stamp === 0) {
    for (const key of Object.keys(bufs)) {
      const a = bufs[key];
      if (a && "needsUpdate" in a) a.needsUpdate = true;
    }
    _lastUploadStampMap.set(bufs, 1);
  } else {
    _lastUploadStampMap.set(bufs, stamp + 1);
  }
  for (const e of subEntries ?? []) {
    const cb = e.pipeline?.buffers;
    if (cb && !_lastUploadStampMap.has(
      cb
    )) {
      for (const key of Object.keys(cb)) {
        const a = cb[key];
        if (a && "needsUpdate" in a) a.needsUpdate = true;
      }
      _lastUploadStampMap.set(
        cb,
        1
      );
    }
    const cmd = e.init.commandBuffer;
    if (cmd && "needsUpdate" in cmd && !_cmdUploadSeen.has(cmd)) {
      cmd.needsUpdate = true;
      _cmdUploadSeen.add(cmd);
    }
  }
  const rb = props.ribbonBuffers;
  if (rb && !_lastUploadStampMap.has(rb)) {
    for (const key of Object.keys(rb)) {
      const a = rb[key];
      if (a && "needsUpdate" in a) a.needsUpdate = true;
    }
    _lastUploadStampMap.set(rb, 1);
  }
  props.computeDispatchReady = true;
  props.iterationCount++;
  props.frameParity = (props.frameParity ?? 0) ^ 1;
  if (props.trailMesh) updateTrailGeometry(props, now);
};
var _tmpQ1 = new THREE2.Quaternion();
var _tmpV1 = new THREE2.Vector3();
var _tmpV2 = new THREE2.Vector3();
var _tmpM1 = new THREE2.Matrix4();
var catmullRom = (out, outIdx, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, t) => {
  const t2 = t * t;
  const t3 = t2 * t;
  out[outIdx] = 0.5 * (2 * p1x + (-p0x + p2x) * t + (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 + (-p0x + 3 * p1x - 3 * p2x + p3x) * t3);
  out[outIdx + 1] = 0.5 * (2 * p1y + (-p0y + p2y) * t + (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 + (-p0y + 3 * p1y - 3 * p2y + p3y) * t3);
  out[outIdx + 2] = 0.5 * (2 * p1z + (-p0z + p2z) * t + (2 * p0z - 5 * p1z + 4 * p2z - p3z) * t2 + (-p0z + 3 * p1z - 3 * p2z + p3z) * t3);
};
var clearTrailVertex = (vIdx, cIdx, aIdx, uvIdx, trailPosArr, trailNextArr, trailHalfWidthArr, trailUVArr, trailAlphaArr, trailColorArr, fallbackX, fallbackY, fallbackZ) => {
  trailPosArr[vIdx] = fallbackX;
  trailPosArr[vIdx + 1] = fallbackY;
  trailPosArr[vIdx + 2] = fallbackZ;
  trailPosArr[vIdx + 3] = fallbackX;
  trailPosArr[vIdx + 4] = fallbackY;
  trailPosArr[vIdx + 5] = fallbackZ;
  trailNextArr[vIdx] = fallbackX;
  trailNextArr[vIdx + 1] = fallbackY;
  trailNextArr[vIdx + 2] = fallbackZ;
  trailNextArr[vIdx + 3] = fallbackX;
  trailNextArr[vIdx + 4] = fallbackY;
  trailNextArr[vIdx + 5] = fallbackZ;
  trailHalfWidthArr[aIdx] = 0;
  trailHalfWidthArr[aIdx + 1] = 0;
  trailUVArr[uvIdx] = 0;
  trailUVArr[uvIdx + 1] = 0;
  trailUVArr[uvIdx + 2] = 0;
  trailUVArr[uvIdx + 3] = 0;
  trailAlphaArr[aIdx] = 0;
  trailAlphaArr[aIdx + 1] = 0;
  trailColorArr[cIdx] = 0;
  trailColorArr[cIdx + 1] = 0;
  trailColorArr[cIdx + 2] = 0;
  trailColorArr[cIdx + 3] = 0;
  trailColorArr[cIdx + 4] = 0;
  trailColorArr[cIdx + 5] = 0;
  trailColorArr[cIdx + 6] = 0;
  trailColorArr[cIdx + 7] = 0;
};
var writeTrailVertex = (vIdx, cIdx, aIdx, uvIdx, hx, hy, hz, nx, ny, nz, halfWidth, t, alpha, fr, fg, fb, ca, trailPosArr, trailNextArr, trailHalfWidthArr, trailUVArr, trailAlphaArr, trailColorArr) => {
  trailPosArr[vIdx] = hx;
  trailPosArr[vIdx + 1] = hy;
  trailPosArr[vIdx + 2] = hz;
  trailPosArr[vIdx + 3] = hx;
  trailPosArr[vIdx + 4] = hy;
  trailPosArr[vIdx + 5] = hz;
  trailNextArr[vIdx] = nx;
  trailNextArr[vIdx + 1] = ny;
  trailNextArr[vIdx + 2] = nz;
  trailNextArr[vIdx + 3] = nx;
  trailNextArr[vIdx + 4] = ny;
  trailNextArr[vIdx + 5] = nz;
  trailHalfWidthArr[aIdx] = halfWidth;
  trailHalfWidthArr[aIdx + 1] = halfWidth;
  trailUVArr[uvIdx] = 0;
  trailUVArr[uvIdx + 1] = t;
  trailUVArr[uvIdx + 2] = 1;
  trailUVArr[uvIdx + 3] = t;
  trailAlphaArr[aIdx] = alpha;
  trailAlphaArr[aIdx + 1] = alpha;
  trailColorArr[cIdx] = fr;
  trailColorArr[cIdx + 1] = fg;
  trailColorArr[cIdx + 2] = fb;
  trailColorArr[cIdx + 3] = ca;
  trailColorArr[cIdx + 4] = fr;
  trailColorArr[cIdx + 5] = fg;
  trailColorArr[cIdx + 6] = fb;
  trailColorArr[cIdx + 7] = ca;
};
var _rawPoints = null;
var _rawPointsSize = 0;
var _smoothedPoints = null;
var _smoothedPointsSize = 0;
var _ribbonIndices = null;
var _ribbonIndicesSize = 0;
var _ribbonCount = 0;
var updateTrailGeometry = (props, now) => {
  const {
    generalData,
    trailPositionAttr,
    trailAlphaAttr,
    trailColorAttr,
    trailNextAttr: trailNextAttrCached,
    trailHalfWidthAttr: trailHalfWidthAttrCached,
    trailUVAttr: trailUVAttrCached,
    trailWidthCurveFn,
    trailOpacityCurveFn,
    trailColorOverTrailFns,
    trailConfig,
    mappedAttributes: ma
  } = props;
  if (!trailPositionAttr || !trailAlphaAttr || !trailColorAttr || !trailNextAttrCached || !trailHalfWidthAttrCached || !trailUVAttrCached || !trailWidthCurveFn || !trailOpacityCurveFn || !trailConfig || !generalData.positionHistory || !generalData.positionHistoryIndex || !generalData.positionHistoryCount)
    return;
  const trailLength = trailConfig.length;
  const positionHistory = generalData.positionHistory;
  const historyIndex = generalData.positionHistoryIndex;
  const historyCount = generalData.positionHistoryCount;
  const sampleTimes = generalData.trailSampleTimes;
  const lastSampledPos = generalData.trailLastSampledPosition;
  const prevNormal = generalData.trailPrevNormal;
  const minVertexDist = trailConfig.minVertexDistance;
  const minVertexDistSq = minVertexDist * minVertexDist;
  const maxTime = trailConfig.maxTime;
  const maxTimeMs = maxTime * 1e3;
  const useSmoothing = trailConfig.smoothing;
  const subdivisions = trailConfig.smoothingSubdivisions;
  const useTwistPrevention = trailConfig.twistPrevention;
  const ribbonId = trailConfig.ribbonId;
  const trailScalarArr = props.scalarArray;
  const positionArr = ma.position.array;
  const prevFilled = generalData.trailPrevFilledCount;
  const trailPosArr = trailPositionAttr.array;
  const trailAlphaArr = trailAlphaAttr.array;
  const trailColorArr = trailColorAttr.array;
  const trailNextArr = trailNextAttrCached.array;
  const trailUVArr = trailUVAttrCached.array;
  const trailHalfWidthArr = trailHalfWidthAttrCached.array;
  const verticesPerParticle = trailLength * 2;
  const hwm = generalData.highWaterIndex;
  const creationTimesLength = hwm > 0 ? hwm : generalData.creationTimes.length;
  let hasUpdates = false;
  const useRibbon = ribbonId !== void 0;
  let ribbonLeader = -1;
  if (useRibbon) {
    if (!_ribbonIndices || _ribbonIndicesSize < creationTimesLength) {
      _ribbonIndices = new Uint32Array(creationTimesLength);
      _ribbonIndicesSize = creationTimesLength;
    }
    _ribbonCount = 0;
    for (let i = 0; i < creationTimesLength; i++) {
      if (trailScalarArr[i * SCALAR_STRIDE + S_IS_ACTIVE])
        _ribbonIndices[_ribbonCount++] = i;
    }
    for (let i = 1; i < _ribbonCount; i++) {
      const key = _ribbonIndices[i];
      const keyTime = generalData.creationTimes[key];
      let j = i - 1;
      while (j >= 0 && generalData.creationTimes[_ribbonIndices[j]] > keyTime) {
        _ribbonIndices[j + 1] = _ribbonIndices[j];
        j--;
      }
      _ribbonIndices[j + 1] = key;
    }
    if (_ribbonCount > 0) ribbonLeader = _ribbonIndices[0];
  }
  for (let index = 0; index < creationTimesLength; index++) {
    const vertBase = index * verticesPerParticle;
    if (trailScalarArr[index * SCALAR_STRIDE + S_IS_ACTIVE]) {
      if (useRibbon && _ribbonCount >= 2 && index !== ribbonLeader) {
        const posIdx2 = index * 3;
        const px2 = positionArr[posIdx2];
        const py2 = positionArr[posIdx2 + 1];
        const pz2 = positionArr[posIdx2 + 2];
        const histBase = (index * trailLength + historyIndex[index]) * 3;
        positionHistory[histBase] = px2;
        positionHistory[histBase + 1] = py2;
        positionHistory[histBase + 2] = pz2;
        if (sampleTimes) {
          sampleTimes[index * trailLength + historyIndex[index]] = now;
        }
        historyIndex[index] = (historyIndex[index] + 1) % trailLength;
        if (historyCount[index] < trailLength) historyCount[index]++;
        continue;
      }
      hasUpdates = true;
      const posIdx = index * 3;
      const px = positionArr[posIdx];
      const py = positionArr[posIdx + 1];
      const pz = positionArr[posIdx + 2];
      let shouldSample = true;
      if (minVertexDist > 0 && lastSampledPos && historyCount[index] > 0) {
        const lsIdx = index * 3;
        const dx = px - lastSampledPos[lsIdx];
        const dy = py - lastSampledPos[lsIdx + 1];
        const dz = pz - lastSampledPos[lsIdx + 2];
        if (dx * dx + dy * dy + dz * dz < minVertexDistSq) {
          shouldSample = false;
        }
      }
      if (shouldSample) {
        const histBase = (index * trailLength + historyIndex[index]) * 3;
        positionHistory[histBase] = px;
        positionHistory[histBase + 1] = py;
        positionHistory[histBase + 2] = pz;
        if (sampleTimes) {
          sampleTimes[index * trailLength + historyIndex[index]] = now;
        }
        historyIndex[index] = (historyIndex[index] + 1) % trailLength;
        if (historyCount[index] < trailLength) historyCount[index]++;
        if (lastSampledPos) {
          const lsIdx = index * 3;
          lastSampledPos[lsIdx] = px;
          lastSampledPos[lsIdx + 1] = py;
          lastSampledPos[lsIdx + 2] = pz;
        }
      }
      let rawCount = historyCount[index];
      let effectiveCount = rawCount;
      if (maxTime > 0 && sampleTimes && rawCount > 0) {
        const sampleBase = index * trailLength;
        effectiveCount = 0;
        for (let s = 0; s < rawCount; s++) {
          const sampleSlot = (historyIndex[index] - 1 - s + trailLength * 2) % trailLength;
          const age = now - sampleTimes[sampleBase + sampleSlot];
          if (age <= maxTimeMs) {
            effectiveCount++;
          } else {
            break;
          }
        }
      }
      const count = effectiveCount;
      const ribbonWidth = trailConfig.width;
      const trailBase = index * SCALAR_STRIDE;
      const cr = trailScalarArr[trailBase + S_COLOR_R];
      const cg = trailScalarArr[trailBase + S_COLOR_G];
      const cb = trailScalarArr[trailBase + S_COLOR_B];
      const ca = trailScalarArr[trailBase + S_COLOR_A];
      const ringOff = index * trailLength * 3;
      const rawPtsSize = count * 3;
      if (!_rawPoints || _rawPointsSize < rawPtsSize) {
        _rawPoints = new Float32Array(rawPtsSize);
        _rawPointsSize = rawPtsSize;
      }
      const rawPts = _rawPoints;
      for (let s = 0; s < count; s++) {
        const histSlot = (historyIndex[index] - 1 - s + trailLength * 2) % trailLength * 3 + ringOff;
        rawPts[s * 3] = positionHistory[histSlot];
        rawPts[s * 3 + 1] = positionHistory[histSlot + 1];
        rawPts[s * 3 + 2] = positionHistory[histSlot + 2];
      }
      let finalPts;
      let finalCount;
      if (useSmoothing && count >= 3) {
        const segmentCount = count - 1;
        finalCount = segmentCount * subdivisions + 1;
        const neededSize = finalCount * 3;
        if (!_smoothedPoints || _smoothedPointsSize < neededSize) {
          _smoothedPoints = new Float32Array(neededSize);
          _smoothedPointsSize = neededSize;
        }
        finalPts = _smoothedPoints;
        for (let seg = 0; seg < segmentCount; seg++) {
          const i0 = Math.max(0, seg - 1);
          const i1 = seg;
          const i2 = Math.min(count - 1, seg + 1);
          const i3 = Math.min(count - 1, seg + 2);
          const p0x = rawPts[i0 * 3], p0y = rawPts[i0 * 3 + 1], p0z = rawPts[i0 * 3 + 2];
          const p1x = rawPts[i1 * 3], p1y = rawPts[i1 * 3 + 1], p1z = rawPts[i1 * 3 + 2];
          const p2x = rawPts[i2 * 3], p2y = rawPts[i2 * 3 + 1], p2z = rawPts[i2 * 3 + 2];
          const p3x = rawPts[i3 * 3], p3y = rawPts[i3 * 3 + 1], p3z = rawPts[i3 * 3 + 2];
          for (let sub2 = 0; sub2 < subdivisions; sub2++) {
            const t = sub2 / subdivisions;
            const outIdx = (seg * subdivisions + sub2) * 3;
            catmullRom(
              finalPts,
              outIdx,
              p0x,
              p0y,
              p0z,
              p1x,
              p1y,
              p1z,
              p2x,
              p2y,
              p2z,
              p3x,
              p3y,
              p3z,
              t
            );
          }
        }
        const lastOutIdx = (finalCount - 1) * 3;
        finalPts[lastOutIdx] = rawPts[(count - 1) * 3];
        finalPts[lastOutIdx + 1] = rawPts[(count - 1) * 3 + 1];
        finalPts[lastOutIdx + 2] = rawPts[(count - 1) * 3 + 2];
      } else {
        finalPts = rawPts;
        finalCount = count;
      }
      if (finalCount > trailLength) finalCount = trailLength;
      if (useSmoothing && finalCount >= 2) {
        const MIN_SEG_DIST_SQ = 1e-4 * 1e-4;
        for (let d = 1; d < finalCount; d++) {
          const pi = (d - 1) * 3;
          const ci = d * 3;
          const dx = finalPts[ci] - finalPts[pi];
          const dy = finalPts[ci + 1] - finalPts[pi + 1];
          const dz = finalPts[ci + 2] - finalPts[pi + 2];
          if (dx * dx + dy * dy + dz * dz < MIN_SEG_DIST_SQ) {
            finalPts[ci] = finalPts[pi];
            finalPts[ci + 1] = finalPts[pi + 1];
            finalPts[ci + 2] = finalPts[pi + 2];
          }
        }
      }
      const prevFilledSlots = prevFilled ? prevFilled[index] : trailLength;
      if (prevFilled) prevFilled[index] = finalCount;
      for (let s = 0; s < trailLength; s++) {
        const vIdx = (vertBase + s * 2) * 3;
        const cIdx = (vertBase + s * 2) * 4;
        const aIdx = vertBase + s * 2;
        const uvIdxBase = (vertBase + s * 2) * 2;
        if (s >= finalCount) {
          if (s >= prevFilledSlots) break;
          clearTrailVertex(
            vIdx,
            cIdx,
            aIdx,
            uvIdxBase,
            trailPosArr,
            trailNextArr,
            trailHalfWidthArr,
            trailUVArr,
            trailAlphaArr,
            trailColorArr,
            px,
            py,
            pz
          );
          continue;
        }
        const hx = finalPts[s * 3];
        const hy = finalPts[s * 3 + 1];
        const hz = finalPts[s * 3 + 2];
        let nx, ny, nz;
        if (s > 0 && s < finalCount - 1) {
          const px2 = finalPts[(s - 1) * 3];
          const py2 = finalPts[(s - 1) * 3 + 1];
          const pz2 = finalPts[(s - 1) * 3 + 2];
          const nx2 = finalPts[(s + 1) * 3];
          const ny2 = finalPts[(s + 1) * 3 + 1];
          const nz2 = finalPts[(s + 1) * 3 + 2];
          const atx = nx2 - px2;
          const aty = ny2 - py2;
          const atz = nz2 - pz2;
          const atLen = Math.sqrt(atx * atx + aty * aty + atz * atz);
          if (atLen > 1e-4) {
            nx = hx + atx / atLen;
            ny = hy + aty / atLen;
            nz = hz + atz / atLen;
          } else {
            nx = finalPts[(s + 1) * 3];
            ny = finalPts[(s + 1) * 3 + 1];
            nz = finalPts[(s + 1) * 3 + 2];
          }
        } else if (s < finalCount - 1) {
          nx = finalPts[(s + 1) * 3];
          ny = finalPts[(s + 1) * 3 + 1];
          nz = finalPts[(s + 1) * 3 + 2];
        } else if (finalCount >= 2) {
          const prevX = finalPts[(s - 1) * 3];
          const prevY = finalPts[(s - 1) * 3 + 1];
          const prevZ = finalPts[(s - 1) * 3 + 2];
          nx = hx + (hx - prevX);
          ny = hy + (hy - prevY);
          nz = hz + (hz - prevZ);
        } else {
          nx = hx;
          ny = hy + 1e-3;
          nz = hz;
        }
        const t = finalCount > 1 ? s / (finalCount - 1) : 0;
        let timeFade = 1;
        if (maxTime > 0 && sampleTimes && effectiveCount > 0) {
          const sampleBase = index * trailLength;
          if (useSmoothing && rawCount >= 2) {
            const rawF = s / Math.max(finalCount - 1, 1) * (rawCount - 1);
            const rawLo = Math.min(Math.floor(rawF), rawCount - 1);
            const rawHi = Math.min(rawLo + 1, rawCount - 1);
            const frac = rawF - rawLo;
            const slotLo = (historyIndex[index] - 1 - rawLo + trailLength * 2) % trailLength;
            const slotHi = (historyIndex[index] - 1 - rawHi + trailLength * 2) % trailLength;
            const ageLo = now - sampleTimes[sampleBase + slotLo];
            const ageHi = now - sampleTimes[sampleBase + slotHi];
            const age = ageLo + (ageHi - ageLo) * frac;
            timeFade = 1 - Math.min(age / maxTimeMs, 1);
          } else {
            const rawS = Math.min(s, rawCount - 1);
            const sampleSlot = (historyIndex[index] - 1 - rawS + trailLength * 2) % trailLength;
            const age = now - sampleTimes[sampleBase + sampleSlot];
            timeFade = 1 - Math.min(age / maxTimeMs, 1);
          }
        }
        const widthScale = trailWidthCurveFn(t);
        const opacityScale = trailOpacityCurveFn(t);
        const halfWidth = ribbonWidth * widthScale * 0.5;
        const alpha = ca * opacityScale * timeFade;
        const fr = trailColorOverTrailFns ? cr * trailColorOverTrailFns.r(t) : cr;
        const fg = trailColorOverTrailFns ? cg * trailColorOverTrailFns.g(t) : cg;
        const fb = trailColorOverTrailFns ? cb * trailColorOverTrailFns.b(t) : cb;
        writeTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          hx,
          hy,
          hz,
          nx,
          ny,
          nz,
          halfWidth,
          t,
          alpha,
          fr,
          fg,
          fb,
          ca,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr
        );
      }
      if (useTwistPrevention && prevNormal && finalCount >= 2) {
        const nIdx = index * 3;
        const tx = finalPts[3] - finalPts[0];
        const ty = finalPts[4] - finalPts[1];
        const tz = finalPts[5] - finalPts[2];
        const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (tLen > 1e-4) {
          const ntx = tx / tLen;
          const nty = ty / tLen;
          const ntz = tz / tLen;
          let upx = 0, upy = 1, upz = 0;
          const dot9 = ntx * upx + nty * upy + ntz * upz;
          if (Math.abs(dot9) > 0.999) {
            upx = 1;
            upy = 0;
            upz = 0;
          }
          let cnx = nty * upz - ntz * upy;
          let cny = ntz * upx - ntx * upz;
          let cnz = ntx * upy - nty * upx;
          const cnLen = Math.sqrt(cnx * cnx + cny * cny + cnz * cnz);
          if (cnLen > 1e-4) {
            cnx /= cnLen;
            cny /= cnLen;
            cnz /= cnLen;
          }
          const prevNx = prevNormal[nIdx];
          const prevNy = prevNormal[nIdx + 1];
          const prevNz = prevNormal[nIdx + 2];
          const hasPrev = prevNx !== 0 || prevNy !== 0 || prevNz !== 0;
          if (hasPrev) {
            const normalDot = cnx * prevNx + cny * prevNy + cnz * prevNz;
            if (normalDot < 0) {
              for (let s = 0; s < Math.min(finalCount, trailLength); s++) {
                const aIdx = vertBase + s * 2;
                const hw = trailHalfWidthArr[aIdx];
                trailHalfWidthArr[aIdx] = -hw;
                trailHalfWidthArr[aIdx + 1] = -hw;
              }
              cnx = -cnx;
              cny = -cny;
              cnz = -cnz;
            }
          }
          prevNormal[nIdx] = cnx;
          prevNormal[nIdx + 1] = cny;
          prevNormal[nIdx + 2] = cnz;
        }
      }
    } else if (historyCount[index] > 0 || prevFilled && prevFilled[index] > 0) {
      hasUpdates = true;
      historyCount[index] = 0;
      historyIndex[index] = 0;
      const clearSlots = prevFilled ? prevFilled[index] : trailLength;
      if (prevFilled) prevFilled[index] = 0;
      for (let s = 0; s < clearSlots; s++) {
        const vIdx = (vertBase + s * 2) * 3;
        const cIdx = (vertBase + s * 2) * 4;
        const aIdx = vertBase + s * 2;
        const uvIdxBase = (vertBase + s * 2) * 2;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
      }
    }
  }
  if (useRibbon && _ribbonCount >= 2 && _ribbonIndices) {
    hasUpdates = true;
    const leader = _ribbonIndices[0];
    const leaderVertBase = leader * verticesPerParticle;
    const controlCount = _ribbonCount;
    const filledCount = Math.min(
      trailLength,
      Math.max(controlCount * 4, controlCount)
    );
    const chainSize = filledCount * 3;
    if (!_rawPoints || _rawPointsSize < chainSize) {
      _rawPoints = new Float32Array(chainSize);
      _rawPointsSize = chainSize;
    }
    if (controlCount === 2) {
      const p0Idx = _ribbonIndices[0] * 3;
      const p1Idx = _ribbonIndices[1] * 3;
      for (let i = 0; i < filledCount; i++) {
        const t = i / (filledCount - 1);
        _rawPoints[i * 3] = positionArr[p0Idx] + t * (positionArr[p1Idx] - positionArr[p0Idx]);
        _rawPoints[i * 3 + 1] = positionArr[p0Idx + 1] + t * (positionArr[p1Idx + 1] - positionArr[p0Idx + 1]);
        _rawPoints[i * 3 + 2] = positionArr[p0Idx + 2] + t * (positionArr[p1Idx + 2] - positionArr[p0Idx + 2]);
      }
    } else {
      const segments = controlCount - 1;
      const ptsPerSeg = Math.max(1, Math.floor((filledCount - 1) / segments));
      let wi = 0;
      for (let seg = 0; seg < segments && wi < filledCount; seg++) {
        const i0 = Math.max(0, seg - 1);
        const i1 = seg;
        const i2 = Math.min(controlCount - 1, seg + 1);
        const i3 = Math.min(controlCount - 1, seg + 2);
        const p0i = _ribbonIndices[i0] * 3;
        const p1i = _ribbonIndices[i1] * 3;
        const p2i = _ribbonIndices[i2] * 3;
        const p3i = _ribbonIndices[i3] * 3;
        const subCount = seg === segments - 1 ? filledCount - wi : ptsPerSeg;
        for (let sub2 = 0; sub2 < subCount && wi < filledCount; sub2++) {
          const t = sub2 / subCount;
          catmullRom(
            _rawPoints,
            wi * 3,
            positionArr[p0i],
            positionArr[p0i + 1],
            positionArr[p0i + 2],
            positionArr[p1i],
            positionArr[p1i + 1],
            positionArr[p1i + 2],
            positionArr[p2i],
            positionArr[p2i + 1],
            positionArr[p2i + 2],
            positionArr[p3i],
            positionArr[p3i + 1],
            positionArr[p3i + 2],
            t
          );
          wi++;
        }
      }
      if (wi > 0) {
        const lastPIdx = _ribbonIndices[controlCount - 1] * 3;
        _rawPoints[(wi - 1) * 3] = positionArr[lastPIdx];
        _rawPoints[(wi - 1) * 3 + 1] = positionArr[lastPIdx + 1];
        _rawPoints[(wi - 1) * 3 + 2] = positionArr[lastPIdx + 2];
      }
    }
    const leaderBase = leader * SCALAR_STRIDE;
    const leaderCr = trailScalarArr[leaderBase + S_COLOR_R];
    const leaderCg = trailScalarArr[leaderBase + S_COLOR_G];
    const leaderCb = trailScalarArr[leaderBase + S_COLOR_B];
    const leaderCa = trailScalarArr[leaderBase + S_COLOR_A];
    const leaderPrevFilled = prevFilled ? prevFilled[leader] : trailLength;
    if (prevFilled) prevFilled[leader] = filledCount;
    for (let s = 0; s < trailLength; s++) {
      const vIdx = (leaderVertBase + s * 2) * 3;
      const cIdx = (leaderVertBase + s * 2) * 4;
      const aIdx = leaderVertBase + s * 2;
      const uvIdxBase = (leaderVertBase + s * 2) * 2;
      if (s >= filledCount) {
        if (s >= leaderPrevFilled) break;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
        continue;
      }
      const ptIdx = s * 3;
      const ptx = _rawPoints[ptIdx];
      const pty = _rawPoints[ptIdx + 1];
      const ptz = _rawPoints[ptIdx + 2];
      let nx, ny, nz;
      if (s > 0 && s < filledCount - 1) {
        const px2 = _rawPoints[(s - 1) * 3];
        const py2 = _rawPoints[(s - 1) * 3 + 1];
        const pz2 = _rawPoints[(s - 1) * 3 + 2];
        const nx2 = _rawPoints[(s + 1) * 3];
        const ny2 = _rawPoints[(s + 1) * 3 + 1];
        const nz2 = _rawPoints[(s + 1) * 3 + 2];
        const atx = nx2 - px2;
        const aty = ny2 - py2;
        const atz = nz2 - pz2;
        const atLen = Math.sqrt(atx * atx + aty * aty + atz * atz);
        if (atLen > 1e-4) {
          nx = ptx + atx / atLen;
          ny = pty + aty / atLen;
          nz = ptz + atz / atLen;
        } else {
          nx = _rawPoints[(s + 1) * 3];
          ny = _rawPoints[(s + 1) * 3 + 1];
          nz = _rawPoints[(s + 1) * 3 + 2];
        }
      } else if (s < filledCount - 1) {
        nx = _rawPoints[(s + 1) * 3];
        ny = _rawPoints[(s + 1) * 3 + 1];
        nz = _rawPoints[(s + 1) * 3 + 2];
      } else if (filledCount >= 2) {
        const prevX = _rawPoints[(s - 1) * 3];
        const prevY = _rawPoints[(s - 1) * 3 + 1];
        const prevZ = _rawPoints[(s - 1) * 3 + 2];
        nx = ptx + (ptx - prevX);
        ny = pty + (pty - prevY);
        nz = ptz + (ptz - prevZ);
      } else {
        nx = ptx;
        ny = pty + 1e-3;
        nz = ptz;
      }
      const t = filledCount > 1 ? s / (filledCount - 1) : 0;
      let ribbonTimeFade = 1;
      if (maxTime > 0 && controlCount >= 2) {
        const ctrlF = t * (controlCount - 1);
        const ctrlLo = Math.min(Math.floor(ctrlF), controlCount - 1);
        const ctrlHi = Math.min(ctrlLo + 1, controlCount - 1);
        const frac = ctrlF - ctrlLo;
        const ageLo = now - generalData.creationTimes[_ribbonIndices[ctrlLo]];
        const ageHi = now - generalData.creationTimes[_ribbonIndices[ctrlHi]];
        const age = ageLo + (ageHi - ageLo) * frac;
        ribbonTimeFade = 1 - Math.min(age / maxTimeMs, 1);
      }
      const widthScale = trailWidthCurveFn(t);
      const opacityScale = trailOpacityCurveFn(t);
      const halfWidth = trailConfig.width * widthScale * 0.5;
      const alpha = leaderCa * opacityScale * ribbonTimeFade;
      const fr = trailColorOverTrailFns ? leaderCr * trailColorOverTrailFns.r(t) : leaderCr;
      const fg = trailColorOverTrailFns ? leaderCg * trailColorOverTrailFns.g(t) : leaderCg;
      const fb = trailColorOverTrailFns ? leaderCb * trailColorOverTrailFns.b(t) : leaderCb;
      writeTrailVertex(
        vIdx,
        cIdx,
        aIdx,
        uvIdxBase,
        ptx,
        pty,
        ptz,
        nx,
        ny,
        nz,
        halfWidth,
        t,
        alpha,
        fr,
        fg,
        fb,
        leaderCa,
        trailPosArr,
        trailNextArr,
        trailHalfWidthArr,
        trailUVArr,
        trailAlphaArr,
        trailColorArr
      );
    }
    if (useTwistPrevention && prevNormal && filledCount >= 2) {
      const nIdx = leader * 3;
      const tx = _rawPoints[3] - _rawPoints[0];
      const ty = _rawPoints[4] - _rawPoints[1];
      const tz = _rawPoints[5] - _rawPoints[2];
      const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
      if (tLen > 1e-4) {
        const ntx = tx / tLen;
        const nty = ty / tLen;
        const ntz = tz / tLen;
        let upx = 0, upy = 1, upz = 0;
        const dot9 = ntx * upx + nty * upy + ntz * upz;
        if (Math.abs(dot9) > 0.999) {
          upx = 1;
          upy = 0;
          upz = 0;
        }
        let cnx = nty * upz - ntz * upy;
        let cny = ntz * upx - ntx * upz;
        let cnz = ntx * upy - nty * upx;
        const cnLen = Math.sqrt(cnx * cnx + cny * cny + cnz * cnz);
        if (cnLen > 1e-4) {
          cnx /= cnLen;
          cny /= cnLen;
          cnz /= cnLen;
        }
        const prevNx = prevNormal[nIdx];
        const prevNy = prevNormal[nIdx + 1];
        const prevNz = prevNormal[nIdx + 2];
        const hasPrev = prevNx !== 0 || prevNy !== 0 || prevNz !== 0;
        if (hasPrev) {
          const normalDot = cnx * prevNx + cny * prevNy + cnz * prevNz;
          if (normalDot < 0) {
            for (let s = 0; s < Math.min(filledCount, trailLength); s++) {
              const aIdx = leaderVertBase + s * 2;
              const hw = trailHalfWidthArr[aIdx];
              trailHalfWidthArr[aIdx] = -hw;
              trailHalfWidthArr[aIdx + 1] = -hw;
            }
            cnx = -cnx;
            cny = -cny;
            cnz = -cnz;
          }
        }
        prevNormal[nIdx] = cnx;
        prevNormal[nIdx + 1] = cny;
        prevNormal[nIdx + 2] = cnz;
      }
    }
    for (let ri = 1; ri < _ribbonCount; ri++) {
      const pIdx = _ribbonIndices[ri];
      const pVertBase = pIdx * verticesPerParticle;
      const pClearSlots = prevFilled ? prevFilled[pIdx] : trailLength;
      if (prevFilled) prevFilled[pIdx] = 0;
      for (let s = 0; s < pClearSlots; s++) {
        const vIdx = (pVertBase + s * 2) * 3;
        const cIdx = (pVertBase + s * 2) * 4;
        const aIdx = pVertBase + s * 2;
        const uvIdxBase = (pVertBase + s * 2) * 2;
        clearTrailVertex(
          vIdx,
          cIdx,
          aIdx,
          uvIdxBase,
          trailPosArr,
          trailNextArr,
          trailHalfWidthArr,
          trailUVArr,
          trailAlphaArr,
          trailColorArr,
          0,
          0,
          0
        );
      }
    }
  }
  if (hasUpdates) {
    trailPositionAttr.needsUpdate = true;
    trailAlphaAttr.needsUpdate = true;
    trailColorAttr.needsUpdate = true;
    trailNextAttrCached.needsUpdate = true;
    trailHalfWidthAttrCached.needsUpdate = true;
    trailUVAttrCached.needsUpdate = true;
  }
};
var updateParticleSystems = (cycleData) => {
  createdParticleSystems.forEach(
    (props) => updateParticleSystemInstance(props, cycleData)
  );
};

export { CollisionPlaneMode, EmitFrom, ForceFieldFalloff, ForceFieldType, LifeTimeCurve, RendererType, SCALAR_STRIDE, S_COLOR_A, S_COLOR_B, S_COLOR_G, S_COLOR_R, S_IS_ACTIVE, S_LIFETIME, S_ROTATION, S_SIZE, S_START_FRAME, S_START_LIFETIME, Shape, SimulationBackend, SimulationSpace, SubEmitterTrigger, TimeMode, assertNamed, blendingMap, calculateRandomPositionAndVelocityOnBox, calculateRandomPositionAndVelocityOnCircle, calculateRandomPositionAndVelocityOnCone, calculateRandomPositionAndVelocityOnRectangle, calculateRandomPositionAndVelocityOnSphere, calculateValue, createBezierCurveFunction, createDefaultMeshTexture, createDefaultParticleTexture, createParticleSystem, getBezierCacheSize, getCurveFunctionFromConfig, getDefaultParticleSystemConfig, isComputeCapableRenderer, isLifeTimeCurve, linearToSRGB, normalizeBackgroundToVector3, normalizeDepthTextureValue, normalizeTextureValue, normalizeVector2Value, prefillFluidState, registerTSLMaterialFactory, removeBezierCurveFunction, resolveSimulationBackend, resolveWebGPUEffectiveRendererType, rgbSRGBToLinear, sRGBToLinear, updateParticleSystems };
//# sourceMappingURL=chunk-GC74DZXV.js.map
//# sourceMappingURL=chunk-GC74DZXV.js.map