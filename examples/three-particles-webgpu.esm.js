import { registerTSLMaterialFactory, registerElectricArcGPUFactory } from '@cyberluke/three-particles';
export { assertNamed, normalizeBackgroundToVector3, normalizeDepthTextureValue, normalizeTextureValue, normalizeVector2Value, resolveWebGPUEffectiveRendererType } from '@cyberluke/three-particles';
import * as THREE2 from 'three';
import { Vector3, Vector4, RedFormat, FloatType, HalfFloatType, Mesh, BufferGeometry, BufferAttribute, DoubleSide, DataTexture, AdditiveBlending } from 'three';
import { Fn, mod, float, floor, dot, vec3, step, min, max, vec4, vec2, abs, uint, round, If, texture, screenUV, smoothstep, cross, length, cameraViewMatrix, normalize, mix, uniform, storage, atomicStore, compute, atomicLoad, instanceIndex, atomicAdd, sqrt, buffer, attribute, varyingProperty, positionLocal, modelViewMatrix, cameraProjectionMatrix, Discard, exp2, pow, cameraFar, cameraNear, oneMinus, textureLoad, log2, exp, clamp, reflect, pass, cos, sin, fract, invocationLocalIndex, workgroupArray, workgroupBarrier, add, sub, normalLocal, uv, cameraPosition, Loop, Continue, PI } from 'three/tsl';
import { StorageBufferAttribute, StorageInstancedBufferAttribute, MeshBasicNodeMaterial, Scene, PointsNodeMaterial } from 'three/webgpu';

// src/webgpu.ts

// src/js/effects/electric-arc/electric-arc-math.ts
var u32 = (n) => n >>> 0;
var pcgRawU32Scalar = (seedU) => {
  const s = u32(Math.imul(u32(seedU), 747796405) + 2891336453);
  const shifted = s >>> (s >>> 28) + 4;
  let word = u32(shifted ^ s);
  word = u32(Math.imul(word, 277803737));
  return u32(word >>> 22 ^ word);
};
var pcg01Scalar = (seedU) => pcgRawU32Scalar(seedU) * (1 / 4294967296);
var mixSeedScalar = (a, b, c) => u32(u32(Math.imul(u32(a), 2654435761)) ^ u32(b) ^ u32(c));
var globalFlicker = (seed, epoch) => 0.78 + 0.27 * pcg01Scalar(mixSeedScalar(seed, epoch, 7));

// src/js/effects/three-particles/color-utils.ts
var sRGBToLinear = (c) => c < 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
var PLANE_STRIDE = 12;
var MAX_COLLISION_PLANES = 16;
var COLLISION_PLANE_DATA_SIZE = MAX_COLLISION_PLANES * PLANE_STRIDE;
var _encodeBuf = null;
function encodeCollisionPlanesForGPU(planes) {
  if (!_encodeBuf || _encodeBuf.length !== COLLISION_PLANE_DATA_SIZE) {
    _encodeBuf = new Float32Array(COLLISION_PLANE_DATA_SIZE);
  }
  const data = _encodeBuf;
  data.fill(0);
  const count = Math.min(planes.length, MAX_COLLISION_PLANES);
  for (let i = 0; i < count; i++) {
    const cp = planes[i];
    const base = i * PLANE_STRIDE;
    data[base] = cp.isActive ? 1 : 0;
    let modeCode = 0;
    if (cp.mode === "CLAMP" /* CLAMP */) modeCode = 1;
    else if (cp.mode === "BOUNCE" /* BOUNCE */) modeCode = 2;
    data[base + 1] = modeCode;
    data[base + 2] = cp.position.x;
    data[base + 3] = cp.position.y;
    data[base + 4] = cp.position.z;
    data[base + 5] = cp.normal.x;
    data[base + 6] = cp.normal.y;
    data[base + 7] = cp.normal.z;
    data[base + 8] = cp.dampen;
    data[base + 9] = cp.lifetimeLoss;
    data[base + 10] = 0;
    data[base + 11] = 0;
  }
  return data;
}
function createCollisionPlaneTSL(sCurveData, collisionPlaneOffset, collisionPlaneCount) {
  const count = Math.min(collisionPlaneCount, MAX_COLLISION_PLANES);
  const uCollisionPlaneCount = uniform(float(count));
  const cpBase = collisionPlaneOffset;
  const applyCollisionPlanesTSL = Fn(
    ({
      pos,
      vel,
      oiaVec,
      sColorNode,
      ps,
      startLife,
      particleIdx,
      sOrbitalIsActiveNode
    }) => {
      Loop(uCollisionPlaneCount, ({ i }) => {
        const base = i.mul(PLANE_STRIDE).add(cpBase);
        const isActive = sCurveData.element(base);
        If(isActive.lessThan(0.5), () => {
          Continue();
        });
        const mode = sCurveData.element(base.add(1));
        const planePos = vec3(
          sCurveData.element(base.add(2)),
          sCurveData.element(base.add(3)),
          sCurveData.element(base.add(4))
        );
        const planeNormal = vec3(
          sCurveData.element(base.add(5)),
          sCurveData.element(base.add(6)),
          sCurveData.element(base.add(7))
        );
        const dampen = sCurveData.element(base.add(8));
        const lifetimeLoss = sCurveData.element(base.add(9));
        const toParticle = pos.sub(planePos);
        const signedDist = dot(toParticle, planeNormal);
        If(signedDist.lessThan(0), () => {
          If(mode.lessThan(0.5), () => {
            ps.x.assign(startLife.add(float(1)));
          }).ElseIf(mode.lessThan(1.5), () => {
            pos.assign(pos.sub(planeNormal.mul(signedDist)));
            const velDotN = dot(vel, planeNormal);
            If(velDotN.lessThan(0), () => {
              vel.assign(vel.sub(planeNormal.mul(velDotN)));
            });
          }).Else(() => {
            pos.assign(pos.sub(planeNormal.mul(signedDist)));
            const vDotN = dot(vel, planeNormal);
            const reflected = vel.sub(planeNormal.mul(vDotN.mul(2)));
            vel.assign(reflected.mul(dampen));
            If(lifetimeLoss.greaterThan(0), () => {
              ps.x.assign(ps.x.add(lifetimeLoss.mul(startLife).mul(1e3)));
            });
          });
        });
      });
    },
    "void"
  );
  return {
    /** Uniform for the active collision plane count. */
    countUniform: uCollisionPlaneCount,
    /** TSL function to call in the compute kernel: apply({ pos, vel, ... }) */
    apply: applyCollisionPlanesTSL
  };
}

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

// src/js/effects/three-particles/three-particles-utils.ts
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

// src/js/effects/three-particles/webgpu/compute-force-fields.ts
var FIELD_STRIDE = 12;
var MAX_FORCE_FIELDS = 16;
var FORCE_FIELD_DATA_SIZE = MAX_FORCE_FIELDS * FIELD_STRIDE;
var GPU_INFINITY = 1e10;
var _encodeBuf2 = null;
function encodeForceFieldsForGPU(forceFields, particleSystemId, systemLifetimePercentage) {
  if (!_encodeBuf2 || _encodeBuf2.length !== FORCE_FIELD_DATA_SIZE) {
    _encodeBuf2 = new Float32Array(FORCE_FIELD_DATA_SIZE);
  }
  const data = _encodeBuf2;
  data.fill(0);
  const count = Math.min(forceFields.length, MAX_FORCE_FIELDS);
  for (let i = 0; i < count; i++) {
    const ff = forceFields[i];
    const base = i * FIELD_STRIDE;
    data[base] = ff.isActive ? 1 : 0;
    data[base + 1] = ff.type === "POINT" /* POINT */ ? 0 : 1;
    data[base + 2] = ff.position.x;
    data[base + 3] = ff.position.y;
    data[base + 4] = ff.position.z;
    data[base + 5] = ff.direction.x;
    data[base + 6] = ff.direction.y;
    data[base + 7] = ff.direction.z;
    data[base + 8] = calculateValue(
      particleSystemId,
      ff.strength,
      systemLifetimePercentage
    );
    data[base + 9] = ff.range === Infinity ? GPU_INFINITY : ff.range;
    let falloffCode = 0;
    if (ff.falloff === "LINEAR" /* LINEAR */) falloffCode = 1;
    else if (ff.falloff === "QUADRATIC" /* QUADRATIC */) falloffCode = 2;
    data[base + 10] = falloffCode;
    data[base + 11] = 0;
  }
  return data;
}
function createForceFieldTSL(sCurveData, forceFieldOffset, forceFieldCount) {
  const count = Math.min(forceFieldCount, MAX_FORCE_FIELDS);
  const uForceFieldCount = uniform(float(count));
  const ffBase = forceFieldOffset;
  const applyForceFieldsTSL = Fn(
    ({
      pos,
      vel,
      delta
    }) => {
      Loop(uForceFieldCount, ({ i }) => {
        const base = i.mul(FIELD_STRIDE).add(ffBase);
        const isActive = sCurveData.element(base);
        If(isActive.lessThan(0.5), () => {
          Continue();
        });
        const fieldType = sCurveData.element(base.add(1));
        const fieldPos = vec3(
          sCurveData.element(base.add(2)),
          sCurveData.element(base.add(3)),
          sCurveData.element(base.add(4))
        );
        const fieldDir = vec3(
          sCurveData.element(base.add(5)),
          sCurveData.element(base.add(6)),
          sCurveData.element(base.add(7))
        );
        const strength = sCurveData.element(base.add(8));
        const range = sCurveData.element(base.add(9));
        const falloffType = sCurveData.element(base.add(10));
        If(strength.equal(0), () => {
          Continue();
        });
        If(fieldType.greaterThan(0.5), () => {
          const force = strength.mul(delta);
          vel.assign(vel.add(fieldDir.mul(force)));
        });
        If(fieldType.lessThan(0.5), () => {
          const toField = fieldPos.sub(pos);
          const dist = length(toField);
          If(dist.greaterThan(1e-4), () => {
            const inRange = dist.lessThan(range);
            If(inRange, () => {
              const dir = normalize(toField);
              const normDist = dist.div(range);
              const falloffNone = float(1);
              const falloffLinear = float(1).sub(normDist);
              const falloffQuadratic = float(1).sub(normDist.mul(normDist));
              const useLinear = falloffType.greaterThan(0.5);
              const useQuadratic = falloffType.greaterThan(1.5);
              const falloff = useQuadratic.select(
                falloffQuadratic,
                useLinear.select(falloffLinear, falloffNone)
              );
              const force = strength.mul(falloff).mul(delta);
              vel.assign(vel.add(dir.mul(force)));
            });
          });
        });
      });
    },
    "void"
  );
  return {
    /** Uniform for the active force field count. */
    countUniform: uForceFieldCount,
    /** TSL function to call in the compute kernel: apply({ pos, vel, delta }) */
    apply: applyForceFieldsTSL
  };
}

// src/js/effects/three-particles/webgpu/curve-bake.ts
var CURVE_RESOLUTION = 256;
function bakeCurve(curveFn, resolution = CURVE_RESOLUTION) {
  const samples = new Float32Array(resolution);
  const lastIndex = resolution - 1;
  for (let i = 0; i < resolution; i++) {
    const t = lastIndex === 0 ? 0 : i / lastIndex;
    samples[i] = curveFn(t);
  }
  return samples;
}
function bakeCurveIntoBuffer(buffer2, writeOffset, particleSystemId, curve) {
  const curveFn = getCurveFunctionFromConfig(particleSystemId, curve);
  const lastIndex = CURVE_RESOLUTION - 1;
  for (let i = 0; i < CURVE_RESOLUTION; i++) {
    const t = i / lastIndex;
    buffer2[writeOffset + i] = curveFn(t);
  }
  return writeOffset + CURVE_RESOLUTION;
}
function bakeVelocityAxisIntoBuffer(buffer2, writeOffset, particleSystemId, value) {
  if (isLifeTimeCurve(value)) {
    return bakeCurveIntoBuffer(buffer2, writeOffset, particleSystemId, value);
  }
  return writeOffset;
}
function bakeParticleSystemCurves(normalizedConfig, particleSystemId) {
  let curveCount = 0;
  const {
    sizeOverLifetime,
    opacityOverLifetime,
    colorOverLifetime,
    velocityOverLifetime
  } = normalizedConfig;
  const hasSizeOverLifetime = sizeOverLifetime.isActive;
  const hasOpacityOverLifetime = opacityOverLifetime.isActive;
  const hasColorOverLifetime = colorOverLifetime.isActive;
  const isVelActive = velocityOverLifetime.isActive;
  const isCurveAxis = (v) => isVelActive && v !== void 0 && isLifeTimeCurve(v);
  const hasLinearVelX = isCurveAxis(velocityOverLifetime.linear.x);
  const hasLinearVelY = isCurveAxis(velocityOverLifetime.linear.y);
  const hasLinearVelZ = isCurveAxis(velocityOverLifetime.linear.z);
  const hasOrbitalVelX = isCurveAxis(velocityOverLifetime.orbital.x);
  const hasOrbitalVelY = isCurveAxis(velocityOverLifetime.orbital.y);
  const hasOrbitalVelZ = isCurveAxis(velocityOverLifetime.orbital.z);
  if (hasSizeOverLifetime) curveCount++;
  if (hasOpacityOverLifetime) curveCount++;
  if (hasColorOverLifetime) curveCount += 3;
  if (hasLinearVelX) curveCount++;
  if (hasLinearVelY) curveCount++;
  if (hasLinearVelZ) curveCount++;
  if (hasOrbitalVelX) curveCount++;
  if (hasOrbitalVelY) curveCount++;
  if (hasOrbitalVelZ) curveCount++;
  const data = new Float32Array(curveCount * CURVE_RESOLUTION);
  let writeOffset = 0;
  let nextIndex = 0;
  let sizeOverLifetimeIdx = -1;
  let opacityOverLifetimeIdx = -1;
  let colorRIdx = -1;
  let colorGIdx = -1;
  let colorBIdx = -1;
  let linearVelXIdx = -1;
  let linearVelYIdx = -1;
  let linearVelZIdx = -1;
  let orbitalVelXIdx = -1;
  let orbitalVelYIdx = -1;
  let orbitalVelZIdx = -1;
  if (hasSizeOverLifetime) {
    sizeOverLifetimeIdx = nextIndex++;
    writeOffset = bakeCurveIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      sizeOverLifetime.lifetimeCurve
    );
  }
  if (hasOpacityOverLifetime) {
    opacityOverLifetimeIdx = nextIndex++;
    writeOffset = bakeCurveIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      opacityOverLifetime.lifetimeCurve
    );
  }
  if (hasColorOverLifetime) {
    colorRIdx = nextIndex++;
    writeOffset = bakeCurveIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      colorOverLifetime.r
    );
    colorGIdx = nextIndex++;
    writeOffset = bakeCurveIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      colorOverLifetime.g
    );
    colorBIdx = nextIndex++;
    writeOffset = bakeCurveIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      colorOverLifetime.b
    );
  }
  if (hasLinearVelX) {
    linearVelXIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.linear.x
    );
  }
  if (hasLinearVelY) {
    linearVelYIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.linear.y
    );
  }
  if (hasLinearVelZ) {
    linearVelZIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.linear.z
    );
  }
  if (hasOrbitalVelX) {
    orbitalVelXIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.orbital.x
    );
  }
  if (hasOrbitalVelY) {
    orbitalVelYIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.orbital.y
    );
  }
  if (hasOrbitalVelZ) {
    orbitalVelZIdx = nextIndex++;
    writeOffset = bakeVelocityAxisIntoBuffer(
      data,
      writeOffset,
      particleSystemId,
      velocityOverLifetime.orbital.z
    );
  }
  return {
    data,
    curveCount,
    sizeOverLifetime: sizeOverLifetimeIdx,
    opacityOverLifetime: opacityOverLifetimeIdx,
    colorR: colorRIdx,
    colorG: colorGIdx,
    colorB: colorBIdx,
    linearVelX: linearVelXIdx,
    linearVelY: linearVelYIdx,
    linearVelZ: linearVelZIdx,
    orbitalVelX: orbitalVelXIdx,
    orbitalVelY: orbitalVelYIdx,
    orbitalVelZ: orbitalVelZIdx
  };
}
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
var SUB_EMITTER_EVENT_STRIDE = 6;
var subEmitterWindowSize = (capacity) => SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity);
var asU32 = (n) => n.nodeType === "uint" ? n : n.toUint();
var pcgRawU32 = (seedU) => {
  const stateU = asU32(seedU).mul(uint(747796405)).add(uint(2891336453));
  const wordU = stateU.shiftRight(stateU.shiftRight(uint(28)).add(uint(4))).bitXor(stateU).mul(uint(277803737));
  return wordU.shiftRight(uint(22)).bitXor(wordU);
};
var pcg01 = (seedU) => pcgRawU32(seedU).toFloat().mul(float(1 / 4294967296));
var mixBirthSeed = (birthNoU, systemSeedU, channelU) => asU32(birthNoU).mul(uint(2654435761)).bitXor(asU32(systemSeedU)).bitXor(asU32(channelU));
var randomChannel = (birthNoU, systemSeedU, channelU) => pcg01(mixBirthSeed(birthNoU, systemSeedU, channelU));
var stableSeedU32 = (birthNoU, systemSeedU) => pcgRawU32(mixBirthSeed(birthNoU, systemSeedU, CH.STABLE_SEED)).bitAnd(
  uint(16777215)
);
var stableSeedFromExt = (extW) => extW.toUint();
var nextSystemSeed = () => Math.floor(Math.random() * 4294967296) >>> 0;
var CH = {
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
};
var createSubEmitterFifoAttribute = (capacity) => ({
  counter: new StorageBufferAttribute(new Uint32Array(2), 1),
  payload: new StorageBufferAttribute(
    new Float32Array(2 * SUB_EMITTER_EVENT_STRIDE * Math.max(1, capacity)),
    1
  ),
  trigger: 1,
  capacity: Math.max(1, capacity),
  windowSize: subEmitterWindowSize(capacity)
});
function createModifierStorageBuffers(maxParticles, instanced, curveData, hasForceFields = false, hasCollisionPlanes = false, trailLength = 0) {
  const Cls = instanced ? StorageInstancedBufferAttribute : StorageBufferAttribute;
  const curveLen = Math.max(curveData.length, 1);
  const ffSize = hasForceFields ? FORCE_FIELD_DATA_SIZE : 0;
  const cpSize = hasCollisionPlanes ? COLLISION_PLANE_DATA_SIZE : 0;
  const packedData = new Float32Array(curveLen + ffSize + cpSize);
  packedData.set(curveData, 0);
  const allocatorCount = maxParticles + 1;
  const allocatorData = new Uint32Array(allocatorCount);
  allocatorData[0] = 0;
  for (let i = 0; i < maxParticles; i++) allocatorData[i + 1] = i;
  const trailMeta = trailLength > 0 ? new StorageBufferAttribute(
    new Uint32Array(Math.max(1, maxParticles) * 2),
    1
  ) : null;
  return {
    buffers: {
      position: new Cls(new Float32Array(maxParticles * 4), 4),
      velocity: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      color: new Cls(new Float32Array(maxParticles * 4), 4),
      particleState: new Cls(new Float32Array(maxParticles * 4), 4),
      startValues: new Cls(new Float32Array(maxParticles * 4), 4),
      startColorsExt: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      orbitalIsActive: new StorageBufferAttribute(
        new Float32Array(maxParticles * 4),
        4
      ),
      allocator: new StorageBufferAttribute(allocatorData, 1),
      trailMeta,
      packedData
    },
    allocatorCount
  };
}
function createCurveLookup(sCurveData) {
  return Fn(
    ({
      curveIndex,
      t
    }) => {
      const clamped = min(t, float(1));
      const pos = clamped.mul(CURVE_RESOLUTION - 1);
      const idx0 = floor(pos);
      const f = fract(pos);
      const base = curveIndex.mul(CURVE_RESOLUTION);
      const v0 = sCurveData.element(base.add(idx0));
      const v1 = sCurveData.element(
        base.add(min(idx0.add(1), float(CURVE_RESOLUTION - 1)))
      );
      return mix(v0, v1, f);
    }
  );
}
function shapeEmitNodes(u, rA, rB, rC, rSpeed) {
  const DEG2 = float(0.01745329);
  const uRadius = u.radius;
  const uRadiusThickness = u.thickness;
  const thetaS = rA.mul(u.arcDeg).mul(DEG2);
  const cosPhi = rB.mul(float(2)).sub(float(1));
  const sinPhi = sqrt(float(1).sub(cosPhi.mul(cosPhi)));
  const dirSx = sinPhi.mul(cos(thetaS));
  const dirSy = sinPhi.mul(sin(thetaS));
  const dirSz = cosPhi;
  const distS = uRadius.mul(float(1).sub(uRadiusThickness)).add(uRadius.mul(uRadiusThickness).mul(rC));
  const pSx = dirSx.mul(distS);
  const pSy = dirSy.mul(distS);
  const pSz = dirSz.mul(distS);
  const spS = mix(u.speedMin, u.speedMax, rSpeed);
  const vSx = dirSx.mul(spS);
  const vSy = dirSy.mul(spS);
  const vSz = dirSz.mul(spS);
  const thetaB = rA.mul(u.arcDeg).mul(DEG2);
  const dirBx = cos(thetaB);
  const dirBy = sin(thetaB);
  const distB = uRadius.mul(float(1).sub(uRadiusThickness)).add(uRadius.mul(uRadiusThickness).mul(rB));
  const pBx = dirBx.mul(distB);
  const pBy = dirBy.mul(distB);
  const pBz = float(0);
  const nAngle = distB.div(uRadius.max(float(1e-6))).mul(u.coneAngleDeg.mul(DEG2));
  const spB = mix(u.speedMin, u.speedMax, rSpeed);
  const sinNA = sin(nAngle);
  const vCx = dirBx.mul(sinNA).mul(spB);
  const vCy = dirBy.mul(sinNA).mul(spB);
  const vCz = cos(nAngle).mul(spB);
  const vIx = dirBx.mul(spB);
  const vIy = dirBy.mul(spB);
  const vIz = float(0);
  const rxOff = rA.mul(u.rectSX).sub(u.rectSX.mul(float(0.5)));
  const ryOff = rB.mul(u.rectSY).sub(u.rectSY.mul(float(0.5)));
  const rotXr = u.rectRX.mul(DEG2);
  const rotYr = u.rectRY.mul(DEG2);
  const pRx = rxOff.mul(cos(rotYr));
  const pRy = ryOff.mul(cos(rotXr));
  const pRz = rxOff.mul(sin(rotYr)).sub(ryOff.mul(sin(rotXr)));
  const halfX = u.boxSX.mul(float(0.5));
  const halfY = u.boxSY.mul(float(0.5));
  const halfZ = u.boxSZ.mul(float(0.5));
  const pVx = rA.mul(u.boxSX).sub(halfX);
  const pVy = rB.mul(u.boxSY).sub(halfY);
  const pVz = rC.mul(u.boxSZ).sub(halfZ);
  const side = floor(rA.mul(float(6))).min(float(5));
  const pa = side.sub(floor(side.div(float(3))).mul(float(3)));
  const a0 = side.greaterThan(float(2)).select(float(1), float(0));
  const isPa0 = pa.equal(float(0));
  const isPa1 = pa.equal(float(1));
  const shX = isPa0.select(a0, isPa1.select(rC, rB));
  const shY = isPa0.select(rB, isPa1.select(a0, rC));
  const shZ = isPa0.select(rC, isPa1.select(rB, a0));
  const pShx = shX.mul(u.boxSX).sub(halfX);
  const pShy = shY.mul(u.boxSY).sub(halfY);
  const pShz = shZ.mul(u.boxSZ).sub(halfZ);
  const edge = floor(rB.mul(float(4))).min(float(3));
  const lowEdge = edge.lessThan(float(2));
  const e1 = lowEdge.select(rC, edge.sub(float(2)));
  const e2 = lowEdge.select(edge, rC);
  const edX = isPa0.select(a0, isPa1.select(e2, e1));
  const edY = isPa0.select(e1, isPa1.select(a0, e2));
  const edZ = isPa0.select(e2, isPa1.select(e1, a0));
  const pEdx = edX.mul(u.boxSX).sub(halfX);
  const pEdy = edY.mul(u.boxSY).sub(halfY);
  const pEdz = edZ.mul(u.boxSZ).sub(halfZ);
  const boxFrom = u.boxFrom;
  const pBX = boxFrom.equal(float(0)).select(pVx, boxFrom.equal(float(1)).select(pShx, pEdx));
  const pBY = boxFrom.equal(float(0)).select(pVy, boxFrom.equal(float(1)).select(pShy, pEdy));
  const pBZ = boxFrom.equal(float(0)).select(pVz, boxFrom.equal(float(1)).select(pShz, pEdz));
  const vPlaneZ = mix(u.speedMin, u.speedMax, rSpeed);
  const kind = u.kind;
  const isSphereKind = kind.equal(float(0));
  const isConeKind = kind.equal(float(1));
  const isCircleKind = kind.equal(float(2));
  const isRectKind = kind.equal(float(3));
  const planeX = isRectKind.select(pRx, pBX);
  const planeY = isRectKind.select(pRy, pBY);
  const planeZ = isRectKind.select(pRz, pBZ);
  const isDiscKind = isConeKind.or(isCircleKind);
  const discX = isDiscKind.select(pBx, planeX);
  const discY = isDiscKind.select(pBy, planeY);
  const discZ = isDiscKind.select(pBz, planeZ);
  const px = isSphereKind.select(pSx, discX);
  const py = isSphereKind.select(pSy, discY);
  const pz = isSphereKind.select(pSz, discZ);
  const cOrI_X = isConeKind.select(vCx, vIx);
  const cOrI_Y = isConeKind.select(vCy, vIy);
  const cOrI_Z = isConeKind.select(vCz, vIz);
  const isDisc = isConeKind.or(isCircleKind);
  const nonSphereVX = isDisc.select(cOrI_X, float(0));
  const nonSphereVY = isDisc.select(cOrI_Y, float(0));
  const nonSphereVZ = isDisc.select(cOrI_Z, vPlaneZ);
  const vx = isSphereKind.select(vSx, nonSphereVX);
  const vy = isSphereKind.select(vSy, nonSphereVY);
  const vz = isSphereKind.select(vSz, nonSphereVZ);
  return { px, py, pz, vx, vy, vz };
}
function quatRotateNodes(x, y, z, q) {
  const qx = q.x;
  const qy = q.y;
  const qz = q.z;
  const qw = q.w;
  const projD = qx.mul(x).add(qy.mul(y)).add(qz.mul(z)).mul(float(2));
  const scaleV = qw.mul(qw).mul(float(2)).sub(float(1));
  const twoW = qw.mul(float(2));
  return [
    x.mul(scaleV).add(qx.mul(projD)).add(qy.mul(z).sub(qz.mul(y)).mul(twoW)),
    y.mul(scaleV).add(qy.mul(projD)).add(qz.mul(x).sub(qx.mul(z)).mul(twoW)),
    z.mul(scaleV).add(qz.mul(projD)).add(qx.mul(y).sub(qy.mul(x)).mul(twoW))
  ];
}
function createModifierComputeUpdate(buffers, maxParticles, curveMap, flags, shapeParams, forceFieldCount = 0, collisionPlaneCount = 0, subFifos = [], trailDesc, velocityValues) {
  const uDelta = uniform(float(0));
  const uDeltaMs = uniform(float(0));
  const uNowMs = uniform(float(0));
  const uGravityVelocity = uniform(new Vector3(0, 0, 0));
  const uSystemSeed = uniform(nextSystemSeed(), "uint");
  const uSeed = uSystemSeed;
  const uEmitCount = uniform(0, "uint");
  const uNoiseStrength = uniform(float(0));
  const uNoisePower = uniform(float(0));
  const uNoiseFrequency = uniform(float(1));
  const uNoisePosAmount = uniform(float(0));
  const uNoiseRotAmount = uniform(float(0));
  const uNoiseSizeAmount = uniform(float(0));
  const shapeUniforms = {};
  const sh = (name, v) => {
    const u = uniform(float(v));
    shapeUniforms[name] = u;
    return u;
  };
  const uShape = sh("shapeKind", shapeParams.shapeKind);
  const uRadius = sh("radius", shapeParams.radius);
  const uRadiusThickness = sh("radiusThickness", shapeParams.radiusThickness);
  const uArcDeg = sh("arcDeg", shapeParams.arcDeg);
  const uConeAngleDeg = sh("coneAngleDeg", shapeParams.coneAngleDeg);
  const uRectRX = sh("rectangleRotXDeg", shapeParams.rectangleRotXDeg);
  const uRectRY = sh("rectangleRotYDeg", shapeParams.rectangleRotYDeg);
  const uRectSX = sh("rectangleScaleX", shapeParams.rectangleScaleX);
  const uRectSY = sh("rectangleScaleY", shapeParams.rectangleScaleY);
  const uBoxSX = sh("boxScaleX", shapeParams.boxScaleX);
  const uBoxSY = sh("boxScaleY", shapeParams.boxScaleY);
  const uBoxSZ = sh("boxScaleZ", shapeParams.boxScaleZ);
  const uBoxEmitFrom = sh("boxEmitFrom", shapeParams.boxEmitFrom);
  const uEmitterPos = uniform(new Vector4(0, 0, 0, 0));
  const uWrapperQuat = uniform(new Vector4(0, 0, 0, 1));
  const uWorldScale = uniform(new Vector3(1, 1, 1));
  const uSpeedMin = sh("speedMin", shapeParams.speedMin);
  const uSpeedMax = sh("speedMax", shapeParams.speedMax);
  const uSizeMin = sh("sizeMin", shapeParams.sizeMin);
  const uSizeMax = sh("sizeMax", shapeParams.sizeMax);
  const uRotMin = sh("rotMin", shapeParams.rotMin);
  const uRotMax = sh("rotMax", shapeParams.rotMax);
  const uRotOLMin = sh("rotOverLifeMin", shapeParams.rotOverLifeMin);
  const uRotOLMax = sh("rotOverLifeMax", shapeParams.rotOverLifeMax);
  const uOpMin = sh("opacityMin", shapeParams.opacityMin);
  const uOpMax = sh("opacityMax", shapeParams.opacityMax);
  const uLifeMin = sh("lifeMin", shapeParams.lifeMin);
  const uLifeMax = sh("lifeMax", shapeParams.lifeMax);
  const uCRR = sh("colorRMin", sRGBToLinear(shapeParams.colorRMin));
  const uCRX = sh("colorRMax", sRGBToLinear(shapeParams.colorRMax));
  const uCGR = sh("colorGMin", sRGBToLinear(shapeParams.colorGMin));
  const uCGX = sh("colorGMax", sRGBToLinear(shapeParams.colorGMax));
  const uCBR = sh("colorBMin", sRGBToLinear(shapeParams.colorBMin));
  const uCBX = sh("colorBMax", sRGBToLinear(shapeParams.colorBMax));
  const uFrMin = sh("startFrameMin", shapeParams.startFrameMin);
  const uFrMax = sh("startFrameMax", shapeParams.startFrameMax);
  const sPos = storage(buffers.position, "vec4", maxParticles);
  const sVel = storage(buffers.velocity, "vec4", maxParticles);
  const sCol = storage(buffers.color, "vec4", maxParticles);
  const sPS = storage(buffers.particleState, "vec4", maxParticles);
  const sSV = storage(buffers.startValues, "vec4", maxParticles);
  const sEx = storage(buffers.startColorsExt, "vec4", maxParticles);
  const sOIA = storage(buffers.orbitalIsActive, "vec4", maxParticles);
  const allocatorCount = maxParticles + 1;
  float(maxParticles);
  const ringModU = uint(maxParticles);
  const sAllocator = storage(
    buffers.allocator,
    "uint",
    Math.max(1, allocatorCount)
  ).toAtomic();
  const sCD = buffer(buffers.packedData, "float", buffers.packedData.length);
  const lookupCurve = createCurveLookup(sCD);
  const uFifoBase = uniform(float(0), "uint");
  const fifoNodes = subFifos.map((f) => ({
    trigger: f.trigger,
    capacity: Math.max(1, f.capacity),
    windowSize: subEmitterWindowSize(Math.max(1, f.capacity)),
    count: storage(
      f.counter,
      "uint",
      Math.max(1, f.counter.array.length)
    ).toAtomic(),
    payload: storage(
      f.payload,
      "float",
      Math.max(1, f.payload.array.length)
    )
  }));
  const birthFifos = fifoNodes.filter((f) => f.trigger === 0);
  const deathFifos = fifoNodes.filter((f) => f.trigger === 1);
  const hasDeathFifo = deathFifos.length > 0;
  const writeFifoEvent = (f, x, y, z, vx, vy, vz) => {
    const winBase = uFifoBase.mul(float(f.windowSize)).toVar();
    const oldCount = float(
      atomicAdd(f.count.element(uFifoBase), uint(1))
    ).toVar();
    If(oldCount.lessThan(float(f.capacity)), () => {
      const b = winBase.add(oldCount.mul(float(SUB_EMITTER_EVENT_STRIDE)));
      f.payload.element(b).assign(x);
      f.payload.element(b.add(float(1))).assign(y);
      f.payload.element(b.add(float(2))).assign(z);
      f.payload.element(b.add(float(3))).assign(vx);
      f.payload.element(b.add(float(4))).assign(vy);
      f.payload.element(b.add(float(5))).assign(vz);
    });
  };
  const trailRows = trailDesc ? trailDesc.length + 1 : 0;
  const sTrail = trailDesc ? storage(trailDesc.attribute, "vec4", trailRows * maxParticles) : null;
  const sTrailMeta = trailDesc && buffers.trailMeta ? storage(buffers.trailMeta, "uint", maxParticles * 2).toAtomic() : null;
  const curveLen = Math.max(curveMap.data.length, 1);
  const forceFieldOffset = curveLen;
  const collisionOffset = forceFieldOffset + (flags.forceFields ? FORCE_FIELD_DATA_SIZE : 0);
  const ffNodes = flags.forceFields ? createForceFieldTSL(sCD, forceFieldOffset, forceFieldCount) : null;
  const cpNodes = flags.collisionPlanes ? createCollisionPlaneTSL(sCD, collisionOffset, collisionPlaneCount) : null;
  const parseAxis = (rawAxis, curveIdx) => {
    if (curveIdx >= 0) {
      return { ci: curveIdx, min: 0, max: 0, isRange: false };
    }
    if (rawAxis && typeof rawAxis === "object" && "min" in rawAxis && "max" in rawAxis) {
      const mn = Number(rawAxis.min) || 0;
      const mx = Number(rawAxis.max) || 0;
      return { ci: -1, min: mn, max: mx, isRange: mn !== mx };
    }
    const c = typeof rawAxis === "number" ? rawAxis : 0;
    return { ci: -1, min: c, max: c, isRange: false };
  };
  const vv = velocityValues ?? {
    linear: [void 0, void 0, void 0],
    orbital: [void 0, void 0, void 0]
  };
  const linAxes = [
    parseAxis(vv.linear[0], curveMap.linearVelX),
    parseAxis(vv.linear[1], curveMap.linearVelY),
    parseAxis(vv.linear[2], curveMap.linearVelZ)
  ];
  const orbAxes = [
    parseAxis(vv.orbital[0], curveMap.orbitalVelX),
    parseAxis(vv.orbital[1], curveMap.orbitalVelY),
    parseAxis(vv.orbital[2], curveMap.orbitalVelZ)
  ];
  const axisUniforms = /* @__PURE__ */ new Map();
  for (const a of [...linAxes, ...orbAxes]) {
    if (a.isRange) {
      axisUniforms.set(a, [uniform(float(a.min)), uniform(float(a.max))]);
    }
  }
  const simAxis = (a, lifePct, particleSeed, salt) => {
    if (a.ci >= 0) {
      return lookupCurve({
        curveIndex: float(a.ci),
        t: lifePct
      });
    }
    if (a.isRange) {
      const [mn, mx] = axisUniforms.get(a);
      return mix(
        mn,
        mx,
        pcg01(particleSeed.toUint().mul(uint(2654435761)).bitXor(salt))
      );
    }
    return float(a.min);
  };
  const emitKernel = Fn(() => {
    const i = instanceIndex;
    If(i.lessThan(uEmitCount), () => {
      const birthNo = atomicAdd(sAllocator.element(0), uint(1)).toVar();
      const slotIdx = birthNo.mod(ringModU).toVar();
      const rcA = randomChannel(birthNo, uSystemSeed, CH.SHAPE_A);
      const rcB = randomChannel(birthNo, uSystemSeed, CH.SHAPE_B);
      const rcC = randomChannel(birthNo, uSystemSeed, CH.SHAPE_C);
      const rcSpeed = randomChannel(birthNo, uSystemSeed, CH.SPEED);
      const rcSize = randomChannel(birthNo, uSystemSeed, CH.SIZE);
      const rcRot = randomChannel(birthNo, uSystemSeed, CH.ROTATION);
      const rcOpacity = randomChannel(birthNo, uSystemSeed, CH.OPACITY);
      const rcSheet = randomChannel(birthNo, uSystemSeed, CH.START_FRAME);
      const rcLife = randomChannel(birthNo, uSystemSeed, CH.LIFETIME);
      const rcColor = randomChannel(birthNo, uSystemSeed, CH.COLOR);
      const rcRotOl = randomChannel(birthNo, uSystemSeed, CH.ROTOL);
      const shE = shapeEmitNodes(
        {
          kind: uShape,
          radius: uRadius,
          thickness: uRadiusThickness,
          arcDeg: uArcDeg,
          coneAngleDeg: uConeAngleDeg,
          rectRX: uRectRX,
          rectRY: uRectRY,
          rectSX: uRectSX,
          rectSY: uRectSY,
          boxSX: uBoxSX,
          boxSY: uBoxSY,
          boxSZ: uBoxSZ,
          boxFrom: uBoxEmitFrom,
          speedMin: uSpeedMin,
          speedMax: uSpeedMax
        },
        rcA,
        rcB,
        rcC,
        rcSpeed
      );
      const pxL = shE.px;
      const pyL = shE.py;
      const pzL = shE.pz;
      const vxL = shE.vx;
      const vyL = shE.vy;
      const vzL = shE.vz;
      const [rotPX, rotPY, rotPZ] = quatRotateNodes(
        pxL,
        pyL,
        pzL,
        uWrapperQuat
      );
      const [rotVX, rotVY, rotVZ] = quatRotateNodes(
        vxL,
        vyL,
        vzL,
        uWrapperQuat
      );
      const isWorld = uEmitterPos.w.greaterThan(0.5);
      const sxf = isWorld.select(uWorldScale.x, float(1));
      const syf = isWorld.select(uWorldScale.y, float(1));
      const szf = isWorld.select(uWorldScale.z, float(1));
      const ox = rotPX.mul(sxf).add(uEmitterPos.x);
      const oy = rotPY.mul(syf).add(uEmitterPos.y);
      const oz = rotPZ.mul(szf).add(uEmitterPos.z);
      const opac = mix(uOpMin, uOpMax, rcOpacity);
      const clR = mix(uCRR, uCRX, rcColor);
      const clG = mix(uCGR, uCGX, rcColor);
      const clB = mix(uCBR, uCBX, rcColor);
      const slife = mix(uLifeMin, uLifeMax, rcLife).mul(float(1e3));
      const ssize = mix(uSizeMin, uSizeMax, rcSize);
      const srot = mix(uRotMin, uRotMax, rcRot);
      const startFrame = floor(mix(uFrMin, uFrMax, rcSheet)).toVar();
      const rotSpeed = mix(uRotOLMin, uRotOLMax, rcRotOl);
      const stableSeedU = stableSeedU32(birthNo, uSystemSeed);
      sPos.element(slotIdx).assign(vec4(ox, oy, oz, float(0)));
      sVel.element(slotIdx).assign(vec4(rotVX, rotVY, rotVZ, float(0)));
      sCol.element(slotIdx).assign(vec4(clR, clG, clB, opac));
      sPS.element(slotIdx).assign(vec4(float(0), ssize, srot, startFrame));
      sSV.element(slotIdx).assign(vec4(slife, ssize, opac, clR));
      sEx.element(slotIdx).assign(vec4(clG, clB, rotSpeed, stableSeedU.toFloat()));
      sOIA.element(slotIdx).assign(vec4(rotPX, rotPY, rotPZ, float(1)));
    });
  });
  const emitNode = compute(emitKernel(), maxParticles);
  const noiseOctavesCount = Math.max(
    1,
    Math.round(shapeParams.noiseOctaves || 1)
  );
  const noiseFbmMax = 2 - Math.pow(2, -noiseOctavesCount);
  if (!Number.isFinite(noiseFbmMax) || noiseFbmMax <= 0) {
    throw new Error(
      `three-particles: invalid FBM normalization ${noiseFbmMax}`
    );
  }
  const simKernel = Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(maxParticles)), () => {
      const oiaVec = sOIA.element(i).toVar();
      If(oiaVec.w.greaterThanEqual(float(0.5)), () => {
        const pos = sPos.element(i).xyz.toVar();
        const vel = sVel.element(i).xyz.toVar();
        const ps = sPS.element(i).toVar();
        const sv = sSV.element(i);
        const ex = sEx.element(i);
        const startLife = sv.x;
        const life = ps.x;
        const lifePct = min(life.div(startLife), float(1));
        vel.assign(vel.sub(vec3(uGravityVelocity).mul(uDelta)));
        if (ffNodes) ffNodes.apply({ pos, vel, delta: uDelta });
        pos.assign(pos.add(vel.mul(uDelta)));
        if (cpNodes)
          cpNodes.apply({
            pos,
            vel,
            oiaVec,
            sColorNode: sCol,
            ps,
            startLife,
            particleIdx: i,
            sOrbitalIsActiveNode: sOIA
          });
        if (flags.linearVelocity) {
          const lvx = simAxis(
            linAxes[0],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_X
          );
          const lvy = simAxis(
            linAxes[1],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_Y
          );
          const lvz = simAxis(
            linAxes[2],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.LIN_Z
          );
          pos.assign(pos.add(vec3(lvx, lvy, lvz).mul(uDelta)));
        }
        if (flags.orbitalVelocity) {
          const offset = vec3(oiaVec.x, oiaVec.y, oiaVec.z).toVar();
          pos.assign(pos.sub(offset));
          const oX = simAxis(
            orbAxes[0],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_X
          );
          const oY = simAxis(
            orbAxes[1],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_Y
          );
          const oZ = simAxis(
            orbAxes[2],
            lifePct,
            stableSeedFromExt(ex.w),
            CH.ORB_Z
          );
          const angX = oX.mul(uDelta);
          const angY = oZ.mul(uDelta);
          const angZ = oY.mul(uDelta);
          const c3 = cos(angZ), s3 = sin(angZ);
          const zx = offset.x.mul(c3).sub(offset.y.mul(s3));
          const zy = offset.x.mul(s3).add(offset.y.mul(c3));
          const zz = offset.z;
          const c2 = cos(angY), s2 = sin(angY);
          const yx = zx.mul(c2).add(zz.mul(s2));
          const yz = zx.mul(s2).negate().add(zz.mul(c2));
          const yy = zy;
          const c1 = cos(angX), s1 = sin(angX);
          const fx = yx;
          const fy = yy.mul(c1).sub(yz.mul(s1));
          const fz = yy.mul(s1).add(yz.mul(c1));
          pos.assign(pos.add(vec3(fx, fy, fz)));
          oiaVec.assign(vec4(fx, fy, fz, oiaVec.w));
        }
        if (flags.sizeOverLifetime) {
          const s = lookupCurve({
            curveIndex: float(curveMap.sizeOverLifetime),
            t: lifePct
          });
          ps.y.assign(s.mul(sv.y));
        }
        if (flags.opacityOverLifetime) {
          const op = lookupCurve({
            curveIndex: float(curveMap.opacityOverLifetime),
            t: lifePct
          });
          const col = sCol.element(i).toVar();
          col.w.assign(op.mul(sv.z));
          sCol.element(i).assign(col);
        }
        if (flags.colorOverLifetime) {
          const col = sCol.element(i).toVar();
          const sce = sEx.element(i);
          if (curveMap.colorR >= 0) {
            col.x.assign(
              sv.w.mul(
                lookupCurve({ curveIndex: float(curveMap.colorR), t: lifePct })
              )
            );
          }
          if (curveMap.colorG >= 0) {
            col.y.assign(
              sce.x.mul(
                lookupCurve({ curveIndex: float(curveMap.colorG), t: lifePct })
              )
            );
          }
          if (curveMap.colorB >= 0) {
            col.z.assign(
              sce.y.mul(
                lookupCurve({ curveIndex: float(curveMap.colorB), t: lifePct })
              )
            );
          }
          sCol.element(i).assign(col);
        }
        if (flags.rotationOverLifetime) {
          ps.z.assign(ps.z.add(ex.z.mul(uDelta).mul(float(0.02))));
        }
        if (flags.noise) {
          const noiseOffset = shapeParams.noiseUseRandomOffset ? pcg01(
            stableSeedFromExt(ex.w).toUint().mul(uint(2654435761)).bitXor(CH.NOISE_PHASE)
          ).mul(float(100)) : float(0);
          const np = lifePct.add(noiseOffset).mul(float(10)).mul(uNoiseStrength).mul(uNoiseFrequency);
          let noiseX = float(0).toVar();
          let noiseY = float(0).toVar();
          let noiseZ = float(0).toVar();
          let amp = 1;
          let lac = 1;
          for (let o = 0; o < noiseOctavesCount; o++) {
            const t = np.mul(float(lac));
            const sc = float(amp / noiseFbmMax);
            noiseX.assign(
              noiseX.add(snoise3D({ v: vec3(t, float(0), float(0)) }).mul(sc))
            );
            noiseY.assign(
              noiseY.add(snoise3D({ v: vec3(t, t, float(0)) }).mul(sc))
            );
            noiseZ.assign(noiseZ.add(snoise3D({ v: vec3(t, t, t) }).mul(sc)));
            amp *= 0.5;
            lac *= 2;
          }
          If(uNoisePosAmount.greaterThan(float(1e-3)), () => {
            pos.assign(
              pos.add(
                vec3(noiseX, noiseY, noiseZ).mul(uNoisePower).mul(uNoisePosAmount)
              )
            );
          });
          If(uNoiseRotAmount.greaterThan(float(1e-3)), () => {
            ps.z.assign(ps.z.add(noiseX.mul(uNoisePower).mul(uNoiseRotAmount)));
          });
          If(uNoiseSizeAmount.greaterThan(float(1e-3)), () => {
            ps.y.assign(
              ps.y.add(noiseX.mul(uNoisePower).mul(uNoiseSizeAmount))
            );
          });
        }
        ps.x.assign(ps.x.add(uDeltaMs));
        sPos.element(i).assign(vec4(pos, float(0)));
        sVel.element(i).assign(vec4(vel, float(0)));
        sPS.element(i).assign(ps);
        sOIA.element(i).assign(oiaVec);
        If(ps.x.greaterThan(startLife), () => {
          const inactive = sOIA.element(i).toVar();
          sOIA.element(i).assign(
            vec4(
              inactive.x,
              inactive.y,
              inactive.z,
              hasDeathFifo ? float(-1) : float(0)
            )
          );
          sCol.element(i).assign(vec4(float(0), float(0), float(0), float(0)));
        });
      });
    });
  });
  const simNode = compute(simKernel(), maxParticles);
  let trailHistoryNode = null;
  if (sTrail && sTrailMeta && trailDesc) {
    const trailHistoryKernel = Fn(() => {
      const i = instanceIndex;
      If(float(i).lessThan(float(maxParticles)), () => {
        const oiaVec = sOIA.element(i).toVar();
        const activeNow = oiaVec.w.greaterThanEqual(float(0.5));
        const pendingDeath = oiaVec.w.lessThan(float(-0.5));
        If(activeNow.or(pendingDeath), () => {
          const pos = sPos.element(i).toVar();
          const L = float(trailDesc.length);
          float(trailRows);
          const curIdx = i.mul(uint(2));
          const cursor = float(atomicLoad(sTrailMeta.element(curIdx))).toVar();
          const count = float(
            atomicLoad(sTrailMeta.element(curIdx.add(uint(1))))
          ).toVar();
          const baseI = i.mul(float(trailRows));
          const prev = sTrail.element(baseI.add(cursor)).toVar();
          const ddx = pos.x.sub(prev.x);
          const ddy = pos.y.sub(prev.y);
          const ddz = pos.z.sub(prev.z);
          const dist = sqrt(ddx.mul(ddx).add(ddy.mul(ddy)).add(ddz.mul(ddz)));
          const firstSample = count.lessThan(float(0.5));
          const farEnough = dist.greaterThanEqual(
            float(Math.max(1e-6, trailDesc.minVertexDistance))
          );
          If(firstSample.or(farEnough), () => {
            const newCursor = cursor.greaterThanEqual(L.sub(float(1))).select(float(0), cursor.add(float(1)));
            sTrail.element(baseI.add(newCursor)).assign(vec4(pos.x, pos.y, pos.z, uNowMs));
            atomicStore(sTrailMeta.element(curIdx), newCursor.toUint());
            atomicStore(
              sTrailMeta.element(curIdx.add(uint(1))),
              min(count.add(float(1)), float(trailDesc.length)).toUint()
            );
          });
        });
      });
    });
    trailHistoryNode = compute(trailHistoryKernel(), maxParticles);
  }
  let subBirthEventsNode = null;
  if (birthFifos.length > 0) {
    const subBirthKernel = Fn(() => {
      const i = instanceIndex;
      If(i.lessThan(uEmitCount), () => {
        const counterAfter = atomicLoad(sAllocator.element(0)).toVar();
        const birthNo = counterAfter.sub(uEmitCount).add(i).toVar();
        const slot = birthNo.mod(ringModU).toVar();
        const p = sPos.element(slot).toVar();
        const v = sVel.element(slot).toVar();
        for (const f of birthFifos) {
          writeFifoEvent(f, p.x, p.y, p.z, v.x, v.y, v.z);
        }
      });
    });
    subBirthEventsNode = compute(subBirthKernel(), maxParticles);
  }
  let subDeathEventsNode = null;
  if (deathFifos.length > 0) {
    const subDeathKernel = Fn(() => {
      const i = instanceIndex;
      If(float(i).lessThan(float(maxParticles)), () => {
        const oiaVec = sOIA.element(i).toVar();
        If(oiaVec.w.equal(float(-1)), () => {
          const p = sPos.element(i).toVar();
          const v = sVel.element(i).toVar();
          for (const f of deathFifos) {
            writeFifoEvent(f, p.x, p.y, p.z, v.x, v.y, v.z);
          }
          sOIA.element(i).assign(vec4(oiaVec.x, oiaVec.y, oiaVec.z, float(0)));
        });
      });
    });
    subDeathEventsNode = compute(subDeathKernel(), maxParticles);
  }
  const layout3 = (name, storageNodes, uniformNodes) => ({
    name,
    storageBindings: new Set(
      storageNodes.filter((n) => n !== null && n !== void 0)
    ).size,
    uniformBindings: new Set(
      uniformNodes.filter((n) => n !== null && n !== void 0)
    ).size
  });
  const basePool = [sPos, sVel, sCol, sPS, sSV, sEx, sOIA, sAllocator];
  const emitUniforms = [
    uEmitCount,
    uSeed,
    uShape,
    uRadius,
    uRadiusThickness,
    uArcDeg,
    uConeAngleDeg,
    uRectRX,
    uRectRY,
    uRectSX,
    uRectSY,
    uBoxSX,
    uBoxSY,
    uBoxSZ,
    uBoxEmitFrom,
    uSpeedMin,
    uSpeedMax,
    uSizeMin,
    uSizeMax,
    uRotMin,
    uRotMax,
    uOpMin,
    uOpMax,
    uLifeMin,
    uLifeMax,
    uCRR,
    uCRX,
    uCGR,
    uCGX,
    uCBR,
    uCBX,
    uFrMin,
    uFrMax,
    uRotOLMin,
    uRotOLMax,
    uWrapperQuat,
    uEmitterPos,
    uWorldScale
  ];
  const simUniforms = [
    uDelta,
    uDeltaMs,
    uGravityVelocity,
    uNoiseStrength,
    uNoisePower,
    uNoiseFrequency,
    uNoisePosAmount,
    uNoiseRotAmount,
    uNoiseSizeAmount,
    sCD,
    ...Array.from(axisUniforms.values()).flat()
  ];
  const passLayouts = [
    layout3("emit", basePool, emitUniforms),
    layout3("simulate", basePool, simUniforms)
  ];
  if (trailHistoryNode) {
    passLayouts.push(
      layout3("trail-history", [sPos, sOIA, sTrail, sTrailMeta], [uNowMs])
    );
  }
  if (subBirthEventsNode) {
    passLayouts.push(
      layout3(
        "sub-birth-events",
        [
          sAllocator,
          sPos,
          sVel,
          ...birthFifos.flatMap((f) => [f.count, f.payload])
        ],
        [uEmitCount, uFifoBase]
      )
    );
  }
  if (subDeathEventsNode) {
    passLayouts.push(
      layout3(
        "sub-death-events",
        [sPos, sVel, sOIA, ...deathFifos.flatMap((f) => [f.count, f.payload])],
        [uFifoBase]
      )
    );
  }
  return {
    emitNode,
    simNode,
    trailHistoryNode,
    subBirthEventsNode,
    subDeathEventsNode,
    computeNodes: [
      emitNode,
      ...subBirthEventsNode ? [subBirthEventsNode] : [],
      simNode,
      ...subDeathEventsNode ? [subDeathEventsNode] : [],
      ...trailHistoryNode ? [trailHistoryNode] : []
    ],
    passLayouts,
    uniforms: {
      delta: uDelta,
      deltaMs: uDeltaMs,
      nowMs: uNowMs,
      gravityVelocity: uGravityVelocity,
      noiseStrength: uNoiseStrength,
      noisePower: uNoisePower,
      noiseFrequency: uNoiseFrequency,
      noisePositionAmount: uNoisePosAmount,
      noiseRotationAmount: uNoiseRotAmount,
      noiseSizeAmount: uNoiseSizeAmount,
      emitCount: uEmitCount,
      seed: uSeed,
      /** Ping-pong FIFO window index for this frame (0 or 1). */
      fifoBase: uFifoBase
    },
    shapeUniforms,
    buffers,
    allocatorCount,
    packedDataNode: sCD,
    passNames: [
      "emit",
      ...subBirthEventsNode ? ["sub-birth-events"] : [],
      "simulate",
      ...subDeathEventsNode ? ["sub-death-events"] : [],
      ...trailHistoryNode ? ["trail-history"] : []
    ],
    trailMeta: buffers.trailMeta,
    // Emitter-pose uniforms, refreshed once per frame by the CPU (scalar only).
    emitterPose: {
      positionW: uEmitterPos,
      wrapperQuat: uWrapperQuat,
      worldScale: uWorldScale
    },
    forceFieldInfo: ffNodes ? { offset: forceFieldOffset, countUniform: ffNodes.countUniform } : null,
    collisionPlaneInfo: cpNodes ? { offset: collisionOffset, countUniform: cpNodes.countUniform } : null
  };
}
function createSubEmitterInitUpdate(child, childMax, childParams, parent, parentMax, fifo, inheritVelocity, particlesPerEvent, childVelValues) {
  const capacity = Math.max(1, fifo.capacity);
  const perEvent = Math.max(1, particlesPerEvent);
  const windowSize = subEmitterWindowSize(capacity);
  const uSystemSeed = uniform(nextSystemSeed(), "uint");
  const uSeed = uSystemSeed;
  const uInherit = uniform(float(Math.max(0, inheritVelocity)));
  const uFifoBase = uniform(float(0), "uint");
  const uWrapperQuat = uniform(new Vector4(0, 0, 0, 1));
  const uEmitterPos = uniform(new Vector4(0, 0, 0, 0));
  const uWorldScale = uniform(new Vector3(1, 1, 1));
  const cRadius = uniform(float(childParams.radius));
  const cThickness = uniform(float(childParams.radiusThickness));
  const cArc = uniform(float(childParams.arcDeg));
  const cAngle = uniform(float(childParams.coneAngleDeg));
  const cRRX = uniform(float(childParams.rectangleRotXDeg));
  const cRRY = uniform(float(childParams.rectangleRotYDeg));
  const cRSX = uniform(float(childParams.rectangleScaleX));
  const cRSY = uniform(float(childParams.rectangleScaleY));
  const cBSX = uniform(float(childParams.boxScaleX));
  const cBSY = uniform(float(childParams.boxScaleY));
  const cBSZ = uniform(float(childParams.boxScaleZ));
  const cBF = uniform(float(childParams.boxEmitFrom));
  const cKind = uniform(float(childParams.shapeKind));
  const cSpeedMin = uniform(float(childParams.speedMin));
  const cSpeedMax = uniform(float(childParams.speedMax));
  const cSizeMin = uniform(float(childParams.sizeMin));
  const cSizeMax = uniform(float(childParams.sizeMax));
  const cRotMin = uniform(float(childParams.rotMin));
  const cRotMax = uniform(float(childParams.rotMax));
  const cOpMin = uniform(float(childParams.opacityMin));
  const cOpMax = uniform(float(childParams.opacityMax));
  const cLifeMin = uniform(float(childParams.lifeMin));
  const cLifeMax = uniform(float(childParams.lifeMax));
  const cCRR = uniform(float(sRGBToLinear(childParams.colorRMin)));
  const cCRX = uniform(float(sRGBToLinear(childParams.colorRMax)));
  const cCGR = uniform(float(sRGBToLinear(childParams.colorGMin)));
  const cCGX = uniform(float(sRGBToLinear(childParams.colorGMax)));
  const cCBR = uniform(float(sRGBToLinear(childParams.colorBMin)));
  const cCBX = uniform(float(sRGBToLinear(childParams.colorBMax)));
  const cFrMin = uniform(float(childParams.startFrameMin));
  const cFrMax = uniform(float(childParams.startFrameMax));
  const cPos = storage(child.position, "vec4", childMax);
  const cVel = storage(child.velocity, "vec4", childMax);
  const cCol = storage(child.color, "vec4", childMax);
  const cPS = storage(child.particleState, "vec4", childMax);
  const cSV = storage(child.startValues, "vec4", childMax);
  const cEx = storage(child.startColorsExt, "vec4", childMax);
  const cOIA = storage(child.orbitalIsActive, "vec4", childMax);
  const cAlloc = storage(
    child.allocator,
    "uint",
    Math.max(1, childMax + 1)
  ).toAtomic();
  float(childMax);
  const cRingModU = uint(childMax);
  const commandBuffer = new StorageBufferAttribute(
    new Float32Array(4 * (1 + capacity * perEvent)),
    4
  );
  const fifoCounter = storage(
    fifo.counter,
    "uint",
    Math.max(1, fifo.counter.array.length)
  ).toAtomic();
  const fifoPayload = storage(
    fifo.payload,
    "float",
    Math.max(1, fifo.payload.array.length)
  );
  const sCmd = storage(commandBuffer, "vec4", 1 + capacity * perEvent);
  const cParseAxis = (rawAxis, ci) => {
    if (rawAxis && typeof rawAxis === "object" && "min" in rawAxis) {
      const mn = Number(rawAxis.min) || 0;
      const mx = Number(rawAxis.max) || 0;
      return { min: mn, max: mx, isRange: mn !== mx };
    }
    const c = typeof rawAxis === "number" ? rawAxis : 0;
    return { min: c, max: c, isRange: false };
  };
  const cVv = childVelValues ?? {
    linear: [void 0, void 0, void 0],
    orbital: [void 0, void 0, void 0]
  };
  [0, 1, 2].map((k) => cParseAxis(cVv.linear[k]));
  [0, 1, 2].map((k) => cParseAxis(cVv.orbital[k]));
  const otherIdx = uFifoBase.equal(float(0)).select(uint(1), uint(0));
  const counterClearKernel = Fn(() => {
    atomicStore(fifoCounter.element(otherIdx), uint(0));
  });
  const counterClearNode = compute(counterClearKernel(), 1);
  const commandBuildKernel = Fn(() => {
    const i = instanceIndex;
    const winBase = uFifoBase.mul(float(windowSize)).toVar();
    const count = float(atomicLoad(fifoCounter.element(uFifoBase))).toVar();
    If(float(i).equal(float(0)), () => {
      sCmd.element(float(0)).assign(vec4(count, float(0), float(0), float(0)));
    });
    If(float(i).lessThan(count), () => {
      const eb = winBase.add(float(i).mul(float(SUB_EMITTER_EVENT_STRIDE)));
      const eX = fifoPayload.element(eb).toVar();
      const eY = fifoPayload.element(eb.add(float(1))).toVar();
      const eZ = fifoPayload.element(eb.add(float(2))).toVar();
      const vX = fifoPayload.element(eb.add(float(3))).toVar();
      const vY = fifoPayload.element(eb.add(float(4))).toVar();
      const vZ = fifoPayload.element(eb.add(float(5))).toVar();
      for (let jj = 0; jj < perEvent; jj++) {
        const birthNo = atomicAdd(cAlloc.element(0), uint(1)).toVar();
        const slot = birthNo.mod(cRingModU).toVar();
        const m = float(i.mul(float(perEvent)).add(float(jj)));
        sCmd.element(float(1).add(m.mul(float(2)))).assign(vec4(slot.toFloat(), eX, eY, eZ));
        sCmd.element(float(2).add(m.mul(float(2)))).assign(vec4(vX, vY, vZ, float(0)));
      }
    });
  });
  const commandBuildNode = compute(commandBuildKernel(), capacity);
  const childInitKernel = Fn(() => {
    const i = instanceIndex;
    const header = sCmd.element(float(0)).toVar();
    If(float(i).lessThan(header.x.mul(float(perEvent))), () => {
      const m = float(i);
      const mU = i;
      const c0 = sCmd.element(float(1).add(m.mul(float(2)))).toVar();
      const c1 = sCmd.element(float(2).add(m.mul(float(2)))).toVar();
      const slot = c0.x;
      const eX = c0.y;
      const eY = c0.z;
      const eZ = c0.w;
      const vX = c1.x;
      const vY = c1.y;
      const vZ = c1.z;
      const parentSpeed = sqrt(
        vX.mul(vX).add(vY.mul(vY)).add(vZ.mul(vZ))
      ).toVar();
      const spAdd = parentSpeed.mul(uInherit);
      {
        const rcA = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_A)
        );
        const rcB = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_B)
        );
        const rcC = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SHAPE_C)
        );
        const rcSpeed = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SPEED)
        );
        const rcSize = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.SIZE)
        );
        const rcRot = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.ROTATION)
        );
        const rcOpacity = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.OPACITY)
        );
        const rcSheet = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.START_FRAME)
        );
        const rcLife = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.LIFETIME)
        );
        const rcColor = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.COLOR)
        );
        const rcRotOl = pcg01(
          mU.mul(uint(2654435761)).bitXor(uSystemSeed).bitXor(CH.ROTOL)
        );
        const shE = shapeEmitNodes(
          {
            kind: cKind,
            radius: cRadius,
            thickness: cThickness,
            arcDeg: cArc,
            coneAngleDeg: cAngle,
            rectRX: cRRX,
            rectRY: cRRY,
            rectSX: cRSX,
            rectSY: cRSY,
            boxSX: cBSX,
            boxSY: cBSY,
            boxSZ: cBSZ,
            boxFrom: cBF,
            speedMin: cSpeedMin.add(spAdd),
            speedMax: cSpeedMax.add(spAdd)
          },
          rcA,
          rcB,
          rcC,
          rcSpeed
        );
        const [rx, ry, rz] = quatRotateNodes(
          shE.px,
          shE.py,
          shE.pz,
          uWrapperQuat
        );
        const [rvx, rvy, rvz] = quatRotateNodes(
          shE.vx,
          shE.vy,
          shE.vz,
          uWrapperQuat
        );
        const isWorld = uEmitterPos.w.greaterThan(0.5);
        const sxf = isWorld.select(uWorldScale.x, float(1));
        const syf = isWorld.select(uWorldScale.y, float(1));
        const szf = isWorld.select(uWorldScale.z, float(1));
        const px = rx.mul(sxf).add(eX);
        const py = ry.mul(syf).add(eY);
        const pz = rz.mul(szf).add(eZ);
        const opac = mix(cOpMin, cOpMax, rcOpacity);
        const clR = mix(cCRR, cCRX, rcColor);
        const clG = mix(cCGR, cCGX, rcColor);
        const clB = mix(cCBR, cCBX, rcColor);
        const slife = mix(cLifeMin, cLifeMax, rcLife).mul(float(1e3));
        const ssize = mix(cSizeMin, cSizeMax, rcSize);
        const srot = mix(cRotMin, cRotMax, rcRot);
        const startFrame = floor(mix(cFrMin, cFrMax, rcSheet)).toVar();
        const rotSpeed = mix(
          float(childParams.rotOverLifeMin),
          float(childParams.rotOverLifeMax),
          rcRotOl
        );
        const stableSeedU = pcgRawU32(
          mixBirthSeed(mU, uSystemSeed, CH.STABLE_SEED)
        ).bitAnd(uint(16777215)).toVar();
        cPos.element(slot).assign(vec4(px, py, pz, float(0)));
        cVel.element(slot).assign(vec4(rvx, rvy, rvz, float(0)));
        cCol.element(slot).assign(vec4(clR, clG, clB, opac));
        cPS.element(slot).assign(vec4(float(0), ssize, srot, startFrame));
        cSV.element(slot).assign(vec4(slife, ssize, opac, clR));
        cEx.element(slot).assign(vec4(clG, clB, rotSpeed, stableSeedU.toFloat()));
        cOIA.element(slot).assign(vec4(rx, ry, rz, float(1)));
      }
    });
  });
  const childInitNode = compute(
    childInitKernel(),
    Math.max(1, capacity * perEvent)
  );
  const initPassLayouts = [
    {
      name: "sub-command-build",
      storageBindings: 4,
      // fifoCounter, fifoPayload, allocator, commands
      uniformBindings: 1
      // uFifoBase
    },
    {
      // command buffer + the 7 child pools; NO allocator here.
      name: "sub-child-init",
      storageBindings: 8,
      uniformBindings: 35
      // seed/inherit/pose + 30 child shape scalars
    },
    {
      name: "sub-counter-clear",
      storageBindings: 1,
      uniformBindings: 1
    }
  ];
  return {
    commandBuildNode,
    childInitNode,
    counterClearNode,
    commandBuffer,
    passLayouts: initPassLayouts,
    passName: fifo.trigger === 0 ? "sub-birth" : "sub-death",
    counterClearPassName: "fifo-counter-clear",
    uniforms: {
      seed: uSeed,
      fifoBase: uFifoBase,
      inherit: uInherit,
      positionW: uEmitterPos,
      wrapperQuat: uWrapperQuat
    },
    buffers: child
  };
}
function createTrailRibbonUpdate(desc) {
  const L = desc.length;
  const rows = L + 1;
  const vertexCount = Math.max(1, desc.maxParticles) * Math.max(1, L);
  const uNowMs = uniform(float(0));
  const activeFns = [
    desc.curveFns.width ?? ((t) => t),
    desc.curveFns.opacity ?? ((t) => t),
    desc.curveFns.colorR ?? ((t) => t),
    desc.curveFns.colorG ?? ((t) => t),
    desc.curveFns.colorB ?? ((t) => t)
  ];
  const curveData = new Float32Array(activeFns.length * CURVE_RESOLUTION);
  activeFns.forEach((fn, k) => {
    curveData.set(bakeCurve(fn, CURVE_RESOLUTION), k * CURVE_RESOLUTION);
  });
  const IDX_WIDTH = 0;
  const IDX_OPACITY = 1;
  const IDX_CR = 2;
  const IDX_CG = 3;
  const IDX_CB = 4;
  const aPos = storage(desc.position, "vec4", vertexCount * 2);
  const aNext = storage(desc.next, "vec4", vertexCount * 2);
  const aUVA = storage(desc.uvColorA, "vec4", vertexCount * 2);
  const aColB = storage(desc.colorB, "vec4", vertexCount * 2);
  const hist = storage(desc.history, "vec4", rows * desc.maxParticles);
  const sMeta = storage(
    desc.meta,
    "uint",
    Math.max(1, desc.meta.array.length)
  ).toAtomic();
  const pColor = storage(desc.particleColor, "vec4", desc.maxParticles);
  const sCD = buffer(curveData, "float", curveData.length);
  const lookupCurve = createCurveLookup(sCD);
  const halfWidthBase = float(desc.width * 0.5);
  const kernel = Fn(() => {
    const idx = instanceIndex;
    const lf = float(L);
    const rowF = float(rows);
    const i = floor(float(idx).div(lf));
    const s = float(idx).sub(i.mul(lf));
    If(i.lessThan(float(desc.maxParticles)), () => {
      const base = i.mul(rowF);
      const cursor = float(atomicLoad(sMeta.element(i.mul(float(2))))).toVar();
      const count = float(
        atomicLoad(sMeta.element(i.mul(float(2)).add(float(1))))
      ).toVar();
      If(count.greaterThan(float(0.5)), () => {
        const raw = cursor.sub(s).add(lf);
        const si = raw.sub(floor(raw.div(lf)).mul(lf));
        const sample = hist.element(base.add(si)).toVar();
        const nextRaw = si.add(float(1));
        const ni = nextRaw.sub(floor(nextRaw.div(lf)).mul(lf));
        const nextSample = hist.element(base.add(ni)).toVar();
        const t = count.greaterThan(float(1.5)).select(s.div(max(count.sub(float(1)), float(1))), float(0));
        const wScale = lookupCurve({
          curveIndex: float(IDX_WIDTH),
          t
        });
        const oScale = lookupCurve({
          curveIndex: float(IDX_OPACITY),
          t
        });
        const cr = lookupCurve({ curveIndex: float(IDX_CR), t });
        const cg = lookupCurve({ curveIndex: float(IDX_CG), t });
        const cb = lookupCurve({ curveIndex: float(IDX_CB), t });
        const pcol = pColor.element(i).toVar();
        const inRange = s.lessThan(count);
        const ageOk = desc.maxTime > 0 ? uNowMs.sub(sample.w).lessThanEqual(float(desc.maxTime)) : inRange;
        const alive = inRange.and(ageOk);
        const hw = alive.select(halfWidthBase.mul(wScale), float(0));
        const alpha = alive.select(oScale.mul(pcol.w), float(0));
        const nPos = count.greaterThan(float(1.5)).select(nextSample.xyz, sample.xyz);
        for (let side = 0; side < 2; side++) {
          const vi = float(idx).mul(float(2)).add(float(side));
          aPos.element(vi).assign(vec4(sample.x, sample.y, sample.z, hw));
          aNext.element(vi).assign(vec4(nPos.x, nPos.y, nPos.z, alpha));
          aUVA.element(vi).assign(vec4(float(side), t, cr.mul(pcol.x), cg.mul(pcol.y)));
          aColB.element(vi).assign(vec4(cb.mul(pcol.z), alpha, float(0), float(0)));
        }
      });
    });
  });
  const ribbonNode = compute(kernel(), vertexCount);
  return {
    ribbonNode,
    // 4 ribbon streams + history + meta + particle color = 7 storage bindings.
    passLayouts: [
      { name: "trail-ribbon", storageBindings: 7, uniformBindings: 2 }
    ],
    uniforms: { nowMs: uNowMs },
    buffers: {
      position: desc.position,
      next: desc.next,
      uvColorA: desc.uvColorA,
      colorB: desc.colorB,
      history: desc.history
    }
  };
}

// src/js/effects/electric-arc/webgpu/compute-electric-arc.ts
var ARC_BRANCH_SAMPLES = 8;
function createElectricArcCompute(cfg) {
  const segments = cfg.segments;
  const branchCount = cfg.branches.enabled ? Math.max(0, Math.min(8, Math.round(cfg.branches.maxCount))) : 0;
  const totalSamples = segments + branchCount * ARC_BRANCH_SAMPLES;
  const arcBuffer = new StorageBufferAttribute(
    new Float32Array(totalSamples * 4),
    4
  );
  const widthBuffer = new StorageBufferAttribute(
    new Float32Array(totalSamples),
    1
  );
  const nStart = uniform(new Vector3().copy(cfg.start));
  const nEnd = uniform(new Vector3().copy(cfg.end));
  const nU = uniform(new Vector3(1, 0, 0));
  const nV = uniform(new Vector3(0, 0, 1));
  const nTime = uniform(float(0));
  const nPin = uniform(float(cfg.endpointPinning));
  const nAmp = uniform(float(cfg.amplitude));
  const nKnots = uniform(float(cfg.coarseKnots));
  const nMicroF = uniform(float(cfg.microFrequency));
  const nEpoch = uniform(float(0));
  const nSeed = uniform(float(cfg.seed));
  const nSeedInv = uniform(float(cfg.seed * 1e-3 + 1));
  const nBvar = uniform(float(cfg.brightnessVariation));
  const nGlobalF = uniform(float(1));
  const nIntensity = uniform(float(cfg.intensity));
  const nSegLast = uniform(float(segments - 1));
  const nBranchBase = uniform(float(branchCount > 0 ? segments : 0));
  const nLen0 = uniform(
    float(
      cfg.branches.length[0] * Math.max(cfg.end.distanceTo(cfg.start), 0.01)
    )
  );
  const nLen1 = uniform(
    float(
      cfg.branches.length[1] * Math.max(cfg.end.distanceTo(cfg.start), 0.01)
    )
  );
  const nProb = uniform(
    float(cfg.branchProbability || cfg.branches.probability)
  );
  const coarseOf = Fn(
    ({
      tN,
      axisU
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const cellF = tN.mul(nKnots);
      const c0 = floor(cellF);
      const f = cellF.sub(c0);
      const c1 = c0.add(1);
      const h0 = pcg01(mixBirthSeed(c0.toUint(), seedU, epochU).bitXor(axisU)).mul(2).sub(1);
      const h1 = pcg01(mixBirthSeed(c1.toUint(), seedU, epochU).bitXor(axisU)).mul(2).sub(1);
      return mix(h0, h1, f);
    }
  );
  const pulseWidth = (iu, seedU, epochU, wXor) => pcg01(mixBirthSeed(iu.add(uint(1)), seedU, epochU).bitXor(wXor)).mul(0.7).add(0.3);
  const coarsePulseOf = Fn(
    ({
      tN,
      axisU
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const ax = axisU.toUint().sub(uint(1));
      const wXor = uint(11).mul(ax.add(uint(1)));
      const lvlXor = ax.add(uint(2));
      const clsXor = ax.add(uint(5));
      const degXor = ax.add(uint(9));
      const nU2 = nKnots.toUint();
      const total = float(0).toVar();
      Loop(21, ({ i }) => {
        If(i.lessThan(nKnots), () => {
          total.addAssign(pulseWidth(i.toUint(), seedU, epochU, wXor));
        });
      });
      const inv = float(1).div(total.max(float(1e-6)));
      const acc0 = float(0).toVar();
      const cellU = uint(0).toVar();
      const fN = float(0).toVar();
      const found = float(0).toVar();
      Loop(21, ({ i }) => {
        If(i.lessThan(nKnots).and(found.lessThan(0.5)), () => {
          const wN = pulseWidth(i.toUint(), seedU, epochU, wXor).mul(inv);
          const acc1 = acc0.add(wN);
          If(tN.lessThan(acc1), () => {
            found.assign(1);
            cellU.assign(i.toUint());
            fN.assign(
              tN.sub(acc0).div(wN.max(float(1e-6))).max(float(0)).min(float(1))
            );
          });
          acc0.assign(acc1);
        });
      });
      If(tN.equal(float(1)), () => {
        cellU.assign(nU2.sub(uint(1)));
        fN.assign(float(1));
      });
      const deg01 = pcg01(
        mixBirthSeed(cellU.add(uint(900)), seedU, epochU).bitXor(degXor)
      );
      const degOn = deg01.lessThan(float(0.25));
      const slotFrac = degOn.select(float(0.5), float(1));
      const fS = fN.div(slotFrac).min(float(1));
      const dead = fN.greaterThan(slotFrac).and(fN.lessThan(float(1)));
      const level = pcg01(mixBirthSeed(cellU, seedU, epochU).bitXor(lvlXor)).mul(2).sub(1);
      const cls = pcg01(
        mixBirthSeed(cellU.add(uint(128)), seedU, epochU).bitXor(clsXor)
      );
      const riseV = fS.div(float(0.15)).min(float(1));
      const tailV = exp(float(-3).mul(fS.sub(float(0.15))));
      const spike = level.mul(fS.lessThan(float(0.15)).select(riseV, tailV));
      const cls3 = cls.lessThan(float(0.2)).select(float(0), cls.lessThan(float(0.4)).select(level, spike));
      const val = dead.select(float(0), cls3);
      return degOn.select(val.mul(float(0.5)), val);
    }
  );
  const coarseOrganicOf = Fn(
    ({
      tN,
      axisU
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const ax = axisU.toUint().sub(uint(1));
      const lvlXor = ax.add(uint(2));
      const holdXor = uint(31).mul(ax.add(uint(1)));
      const nU2 = nKnots.toUint();
      const cellF = tN.mul(nKnots);
      const c0f = floor(cellF);
      const f = cellF.sub(c0f);
      const c0u = c0f.toUint().min(nU2);
      const c1u = c0u.add(uint(1)).min(nU2);
      const l0 = pcg01(mixBirthSeed(c0u, seedU, epochU).bitXor(lvlXor)).mul(2).sub(1);
      const l1 = pcg01(mixBirthSeed(c1u, seedU, epochU).bitXor(lvlXor)).mul(2).sub(1);
      const holdFrac = pcg01(
        mixBirthSeed(c0u.add(uint(700)), seedU, epochU).bitXor(holdXor)
      ).mul(float(0.4)).add(float(0.45));
      const travel = f.div(holdFrac).min(float(1));
      const e = travel.mul(travel).mul(float(3).sub(travel.mul(float(2))));
      const eased = l0.add(l1.sub(l0).mul(e));
      return f.greaterThanEqual(holdFrac).select(l1, eased);
    }
  );
  const coarseFn = cfg.chaosAlgorithm === "pulse" ? coarsePulseOf : cfg.chaosAlgorithm === "organic" ? coarseOrganicOf : coarseOf;
  const microOf = Fn(
    ({
      tN,
      ch
    }) => {
      const p = vec3(tN.mul(nMicroF), nTime, ch.mul(nSeedInv));
      return snoise3D({ v: p }).mul(float(0.72)).add(snoise3D({ v: p.mul(2.13) }).mul(float(0.28)));
    }
  );
  const impulseOf = Fn(
    ({ tN, ch }) => snoise3D({
      v: vec3(
        tN.mul(nMicroF).mul(2.7).add(float(11.37)),
        nTime.mul(1.7).add(float(3.1)),
        ch.add(float(9.7))
      )
    })
  );
  const mainPosAt = Fn(({ tN }) => {
    const base = mix(nStart, nEnd, tN);
    const sT = tN.equal(float(1)).select(float(0.9999975), tN).toVar();
    const env = pow(sin(PI.mul(sT)), nPin);
    const ou = coarseFn({ tN: sT, axisU: uint(1) }).mul(float(0.68)).add(microOf({ tN: sT, ch: uint(1) }).mul(float(0.24))).add(impulseOf({ tN: sT, ch: uint(1) }).mul(float(0.08)));
    const ov = coarseFn({ tN: sT, axisU: uint(2) }).mul(float(0.68)).add(microOf({ tN: sT, ch: uint(2) }).mul(float(0.24))).add(impulseOf({ tN: sT, ch: uint(2) }).mul(float(0.08)));
    const ampEnv = nAmp.mul(env);
    const x = base.x.add(nU.x.mul(ou).add(nV.x.mul(ov)).mul(ampEnv));
    const y = base.y.add(nU.y.mul(ou).add(nV.y.mul(ov)).mul(ampEnv));
    const z = base.z.add(nU.z.mul(ou).add(nV.z.mul(ov)).mul(ampEnv));
    return vec3(
      tN.equal(float(0)).select(nStart.x, tN.equal(float(1)).select(nEnd.x, x)),
      tN.equal(float(0)).select(nStart.y, tN.equal(float(1)).select(nEnd.y, y)),
      tN.equal(float(0)).select(nStart.z, tN.equal(float(1)).select(nEnd.z, z))
    );
  });
  const rotMagAt = Fn(({ tN }) => {
    const sT = tN.equal(float(1)).select(float(0.9999975), tN).toVar();
    const env = pow(sin(PI.mul(sT)), nPin).toVar();
    const ou = coarseFn({ tN: sT, axisU: uint(1) }).mul(float(0.68)).add(microOf({ tN: sT, ch: uint(1) }).mul(float(0.24))).add(impulseOf({ tN: sT, ch: uint(1) }).mul(float(0.08))).mul(env);
    const ov = coarseFn({ tN: sT, axisU: uint(2) }).mul(float(0.68)).add(microOf({ tN: sT, ch: uint(2) }).mul(float(0.24))).add(impulseOf({ tN: sT, ch: uint(2) }).mul(float(0.08))).mul(env);
    return sqrt(ou.mul(ou).add(ov.mul(ov))).min(float(1));
  });
  const kernel = Fn(() => {
    const i = instanceIndex;
    const iF = i.toFloat();
    const sArc = storage(arcBuffer, "vec4", totalSamples);
    const sW = storage(widthBuffer, "float", totalSamples);
    const isMain = iF.lessThan(float(segments));
    const tN = iF.div(nSegLast);
    const mainPos = mainPosAt({ tN });
    const hb = pcg01(mixBirthSeed(i, nSeed.toUint(), uint(5)));
    const mainBrightness = nGlobalF.mul(
      float(1).add(nBvar.mul(hb.mul(2).sub(1)))
    );
    const rel = iF.sub(nBranchBase);
    const bN = floor(rel.div(float(ARC_BRANCH_SAMPLES)));
    const jN = rel.sub(bN.mul(float(ARC_BRANCH_SAMPLES)));
    const tB = jN.div(float(ARC_BRANCH_SAMPLES - 1));
    const bh = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(3)));
    const active = bh.lessThan(nProb).select(float(1), float(0));
    const dir1r = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(13))).mul(2).sub(1);
    const dir2r = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(14))).mul(2).sub(1);
    const dir3r = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(15))).mul(2).sub(1);
    const dLen = sqrt(
      dir1r.mul(dir1r).add(dir2r.mul(dir2r)).add(dir3r.mul(dir3r))
    ).max(float(1e-6));
    const dir1 = dir1r.div(dLen);
    const dir2 = dir2r.div(dLen);
    const dir3 = dir3r.div(dLen);
    const hLen = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(16)));
    const len = mix(nLen0, nLen1, hLen).mul(active);
    const tS = floor(bh.mul(nSegLast)).div(nSegLast);
    const origin = mainPosAt({ tN: tS });
    const bend = pcg01(mixBirthSeed(bN, nSeed.toUint(), uint(18))).mul(2).sub(1).mul(float(0.28));
    const cX = origin.x.add(
      dir1.mul(len).add(nU.x.mul(bend).mul(len)).mul(float(0.5))
    );
    const cY = origin.y.add(
      dir2.mul(len).add(nU.y.mul(bend).mul(len)).mul(float(0.5))
    );
    const cZ = origin.z.add(
      dir3.mul(len).add(nU.z.mul(bend).mul(len)).mul(float(0.5))
    );
    const eX = origin.x.add(dir1.mul(len));
    const eY = origin.y.add(dir2.mul(len));
    const eZ = origin.z.add(dir3.mul(len));
    const om = float(1).sub(tB);
    const w0 = om.mul(om);
    const w1 = om.mul(tB).mul(2);
    const w2 = tB.mul(tB);
    const bx = origin.x.mul(w0).add(cX.mul(w1)).add(eX.mul(w2));
    const by = origin.y.mul(w0).add(cY.mul(w1)).add(eY.mul(w2));
    const bz = origin.z.mul(w0).add(cZ.mul(w1)).add(eZ.mul(w2));
    const x = isMain.select(mainPos.x, bx);
    const y = isMain.select(mainPos.y, by);
    const z = isMain.select(mainPos.z, bz);
    const bright = isMain.select(mainBrightness, active);
    sArc.element(i).assign(vec4(x, y, z, bright));
    const rotMag = rotMagAt({ tN });
    const wThin = float(1).sub(float(0.45).mul(rotMag));
    const wMain = isMain.select(wThin, float(1));
    sW.element(i).assign(wMain);
  });
  const computeNode = compute(kernel(), totalSamples);
  let disposed = false;
  return {
    computeNode,
    arcBuffer,
    widthBuffer,
    totalSamples,
    mainCount: segments,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      arcBuffer.array = new Float32Array(0);
      widthBuffer.array = new Float32Array(0);
    },
    uniforms: {
      start: nStart,
      end: nEnd,
      basisU: nU,
      basisV: nV,
      time: nTime,
      pin: nPin,
      amp: nAmp,
      knots: nKnots,
      microF: nMicroF,
      epoch: nEpoch,
      seed: nSeed,
      seedInv: nSeedInv,
      brightnessVar: nBvar,
      globalFlicker: nGlobalF,
      intensity: nIntensity,
      segLast: nSegLast,
      branchBase: nBranchBase,
      branchLen0: nLen0,
      branchLen1: nLen1,
      branchProb: nProb
    }
  };
}
var _dummyTexture = null;
function getDummyTexture() {
  if (!_dummyTexture) {
    _dummyTexture = new DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    _dummyTexture.needsUpdate = true;
  }
  return _dummyTexture;
}
function createParticleUniforms(sharedUniforms) {
  const dummy = getDummyTexture();
  const map = sharedUniforms.map.value ?? dummy;
  return {
    uMap: map,
    uElapsed: uniform(float(sharedUniforms.elapsed.value)),
    uFps: uniform(float(sharedUniforms.fps.value)),
    uUseFPSForFrameIndex: uniform(
      float(sharedUniforms.useFPSForFrameIndex.value ? 1 : 0)
    ),
    uTiles: uniform(sharedUniforms.tiles.value),
    uDiscardBg: uniform(
      float(sharedUniforms.discardBackgroundColor.value ? 1 : 0)
    ),
    uBgColor: uniform(
      new Vector3(
        sharedUniforms.backgroundColor.value.r,
        sharedUniforms.backgroundColor.value.g,
        sharedUniforms.backgroundColor.value.b
      )
    ),
    uBgTolerance: uniform(float(sharedUniforms.backgroundColorTolerance.value)),
    uSoftEnabled: uniform(
      float(sharedUniforms.softParticlesEnabled.value ? 1 : 0)
    ),
    uSoftIntensity: uniform(float(sharedUniforms.softParticlesIntensity.value)),
    uSceneDepthTex: sharedUniforms.sceneDepthTexture.value ?? dummy,
    uCameraNearFar: uniform(sharedUniforms.cameraNearFar.value)
  };
}
var computeFrameIndex = Fn(
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
var computeSpriteSheetUV = Fn(
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
var computeSoftParticleFade = Fn(
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
var billboardPerp = Fn(
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
function applyBackgroundDiscard({
  texColor,
  uDiscardBg,
  uBgColor,
  uBgTolerance
}) {
  const diff = vec3(
    texColor.x.sub(uBgColor.x),
    texColor.y.sub(uBgColor.y),
    texColor.z.sub(uBgColor.z)
  );
  Discard(
    uDiscardBg.greaterThan(0.5).and(abs(length(diff)).lessThan(uBgTolerance))
  );
}

// src/js/effects/electric-arc/webgpu/tsl-electric-arc-material.ts
var profileAt = (d2, dAbs, uProfile, k) => {
  const g = exp(d2.mul(float(-k)));
  const t = float(1).sub(dAbs.mul(Math.sqrt(0.55 * k))).max(float(0));
  return mix(g, t, uProfile);
};
function createElectricArcRibbonMaterial(arcBuffer, widthBuffer, totalSamples, params) {
  const uCore = uniform(params.coreColor);
  const uArc = uniform(params.arcColor);
  const uHalf = uniform(float(params.halfWidth));
  const uIntensity = uniform(float(params.intensity));
  const uGlow = uniform(float(params.glowIntensity));
  const uHalo = uniform(float(params.haloIntensity));
  const uProfile = uniform(float(params.profileMode));
  const aPacked = attribute("position", "vec4");
  const sArc = storage(arcBuffer, "vec4", totalSamples);
  const sW = storage(widthBuffer, "float", totalSamples);
  const vAcross = varyingProperty("float", "vAcross");
  const vBright = varyingProperty("float", "vBright");
  const positionNode = Fn(() => {
    const iF = aPacked.x;
    const side = aPacked.y;
    const pI = aPacked.z;
    const nI = aPacked.w;
    const cur = sArc.element(iF.toUint());
    const prev = sArc.element(pI.toUint());
    const next = sArc.element(nI.toUint());
    vAcross.assign(side);
    vBright.assign(cur.w);
    const rawTan = next.sub(prev);
    const tanLen = length(rawTan);
    const tangent = normalize(
      tanLen.lessThan(float(1e-4)).select(vec3(float(0), float(1), float(0)), rawTan)
    );
    const curPos = vec3(cur.x, cur.y, cur.z);
    const viewDir = normalize(cameraPosition.sub(curPos));
    const perp = billboardPerp({ tangent, viewDir });
    const wid = sW.element(iF.toUint());
    return curPos.add(perp.mul(side).mul(uHalf.mul(wid)));
  })();
  const colorNode = Fn(() => {
    const dAbs = vAcross.abs();
    const d2 = dAbs.mul(dAbs);
    const c = profileAt(d2, dAbs, uProfile, 70);
    const i = profileAt(d2, dAbs, uProfile, 8);
    const h = profileAt(d2, dAbs, uProfile, 2);
    const isCore = params.layers === "core";
    const isSheath = params.layers === "sheath";
    const k = isCore ? c.mul(uIntensity) : isSheath ? i.mul(uGlow) : h.mul(uHalo);
    const kk = k.mul(vBright);
    const rgb0 = isCore || isSheath ? uCore : uArc;
    const r = rgb0.x.mul(kk);
    const g = rgb0.y.mul(kk);
    const b = rgb0.z.mul(kk);
    const alpha = kk.mul(float(0.5)).min(float(1));
    Discard(alpha.lessThan(float(1e-3)));
    return vec4(r, g, b, alpha);
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.blending = AdditiveBlending;
  material.depthTest = true;
  material.depthWrite = false;
  material.toneMapped = false;
  material.fog = false;
  material.side = DoubleSide;
  material.positionNode = positionNode;
  material.colorNode = colorNode;
  return {
    material,
    halfWidth: uHalf,
    intensity: uIntensity,
    glowIntensity: uGlow,
    haloIntensity: uHalo,
    profileMode: uProfile
  };
}
function createElectricContactMaterial(params) {
  const uColor = uniform(params.color);
  const uCore = uniform(params.coreColor);
  const uCenter = uniform(params.center);
  const uHalf = uniform(float(params.halfSize));
  const uIntensity = uniform(float(params.intensity));
  const vUv = varyingProperty("vec2", "vContactUV");
  const positionNode = Fn(() => {
    const corner = attribute("position", "vec2");
    vUv.assign(attribute("uv", "vec2"));
    const camRight = vec3(
      cameraViewMatrix.element(0).element(0),
      cameraViewMatrix.element(1).element(0),
      cameraViewMatrix.element(2).element(0)
    );
    const camUp = vec3(
      cameraViewMatrix.element(0).element(1),
      cameraViewMatrix.element(1).element(1),
      cameraViewMatrix.element(2).element(1)
    );
    return uCenter.add(camRight.mul(corner.x.mul(uHalf))).add(camUp.mul(corner.y.mul(uHalf)));
  })();
  const colorNode = Fn(() => {
    const x = vUv.x.sub(float(0.5)).mul(float(2));
    const y = vUv.y.sub(float(0.5)).mul(float(2));
    const d2 = x.mul(x).add(y.mul(y));
    const core = exp(d2.mul(float(-9)));
    const corona = exp(d2.mul(float(-2.2)));
    const kc = core.mul(uIntensity).mul(params.flicker);
    const kw = corona.mul(params.flicker);
    const r = uCore.x.mul(kc).add(uColor.x.mul(kw));
    const g = uCore.y.mul(kc).add(uColor.y.mul(kw));
    const b = uCore.z.mul(kc).add(uColor.z.mul(kw));
    const alpha = core.add(corona.mul(float(0.6)));
    Discard(alpha.lessThan(float(1e-3)));
    return vec4(r, g, b, alpha);
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.blending = AdditiveBlending;
  material.depthTest = true;
  material.depthWrite = false;
  material.toneMapped = false;
  material.fog = false;
  material.side = DoubleSide;
  material.positionNode = positionNode;
  material.colorNode = colorNode;
  return { material, halfSize: uHalf, intensity: uIntensity };
}

// src/js/effects/electric-arc/webgpu/electric-arc-webgpu.ts
var _dir = new Vector3();
var _helper = new Vector3();
var _ub = new Vector3();
var _vb = new Vector3();
var nBasis = (out, a, b) => {
  out.crossVectors(a, b);
  const l = out.length() || 1;
  out.multiplyScalar(1 / l);
};
function ribbonGeometry(start, count, u0, u1) {
  const vertCount = count * 2;
  const geometry = new THREE2.BufferGeometry();
  const pos = new Float32Array(vertCount * 4);
  const uv2 = new Float32Array(vertCount * 2);
  const idx = new Uint16Array((count - 1) * 6);
  const inv = 1 / (count - 1);
  for (let j = 0; j < count; j++) {
    const i = start + j;
    const prev = j === 0 ? i : i - 1;
    const next = j === count - 1 ? i : i + 1;
    const li = j * 2;
    const ri = li + 1;
    pos[li * 4] = i;
    pos[li * 4 + 1] = -1;
    pos[li * 4 + 2] = prev;
    pos[li * 4 + 3] = next;
    pos[ri * 4] = i;
    pos[ri * 4 + 1] = 1;
    pos[ri * 4 + 2] = prev;
    pos[ri * 4 + 3] = next;
    const u = u0 + (u1 - u0) * j * inv;
    uv2[li * 2] = u;
    uv2[li * 2 + 1] = 0;
    uv2[ri * 2] = u;
    uv2[ri * 2 + 1] = 1;
  }
  for (let j = 0; j < count - 1; j++) {
    const l0 = j * 2;
    const r0 = l0 + 1;
    const l1 = l0 + 2;
    const r1 = l1 + 1;
    const o = j * 6;
    idx[o] = l0;
    idx[o + 1] = r0;
    idx[o + 2] = l1;
    idx[o + 3] = r0;
    idx[o + 4] = r1;
    idx[o + 5] = l1;
  }
  geometry.setAttribute("position", new THREE2.BufferAttribute(pos, 4));
  geometry.setAttribute("uv", new THREE2.BufferAttribute(uv2, 2));
  geometry.setIndex(new THREE2.BufferAttribute(idx, 1));
  geometry.boundingSphere = new THREE2.Sphere(new Vector3(), 8);
  return geometry;
}
function contactGeometry() {
  const geometry = new THREE2.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE2.BufferAttribute(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), 2)
  );
  geometry.setAttribute(
    "uv",
    new THREE2.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2)
  );
  geometry.setIndex(
    new THREE2.BufferAttribute(new Uint16Array([0, 1, 2, 1, 3, 2]), 1)
  );
  geometry.boundingSphere = new THREE2.Sphere(new Vector3(), 1);
  return geometry;
}
var setNum = (u, v) => {
  u.value = v;
};
function createElectricArcGPU(cfg) {
  const root = new THREE2.Group();
  root.name = "electric-arc-gpu";
  const pipeline = createElectricArcCompute(cfg);
  const { arcBuffer, widthBuffer, totalSamples, uniforms, mainCount } = pipeline;
  const coreColorVec = new Vector3();
  const arcColorVec = new Vector3();
  const setColors = () => {
    const c = new THREE2.Color(cfg.coreColor);
    coreColorVec.set(c.r, c.g, c.b);
    const a = new THREE2.Color(cfg.color);
    arcColorVec.set(a.r, a.g, a.b);
  };
  setColors();
  const coreHalf = Math.max(cfg.thickness, 5e-4) * 0.5;
  const mainGeo = ribbonGeometry(0, mainCount, 0, 1);
  const prof = cfg.glow.profile === "triangle" ? 1 : 0;
  const ribbonSpecs = [
    {
      layers: "core",
      halfWidth: coreHalf,
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity,
      haloIntensity: cfg.glow.intensity * 0.4
    },
    {
      layers: "sheath",
      halfWidth: coreHalf * 2,
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity * 0.8,
      haloIntensity: cfg.glow.intensity * 0.4
    },
    {
      layers: "halo",
      halfWidth: coreHalf * Math.max(2, cfg.glow.width),
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity * 0.8,
      haloIntensity: cfg.glow.intensity * 0.4
    }
  ];
  const mainMeshes = [];
  for (const spec of ribbonSpecs) {
    const handles = createElectricArcRibbonMaterial(
      arcBuffer,
      widthBuffer,
      totalSamples,
      {
        coreColor: coreColorVec,
        arcColor: arcColorVec,
        halfWidth: spec.halfWidth,
        intensity: spec.intensity,
        glowIntensity: spec.glowIntensity,
        haloIntensity: spec.haloIntensity,
        layers: spec.layers,
        profileMode: prof
      }
    );
    const mesh = new THREE2.Mesh(mainGeo, handles.material);
    mesh.frustumCulled = false;
    root.add(mesh);
    mainMeshes.push({ mesh, handles, layer: spec.layers });
  }
  const branchNum = cfg.branches.enabled && cfg.branches.maxCount > 0 ? Math.min(8, Math.round(cfg.branches.maxCount)) : 0;
  const branchMeshes = [];
  const branchGeos = [];
  if (branchNum > 0) {
    const [ts0, ts1] = cfg.branches.thicknessScale;
    const wMid = (ts0 + ts1) * 0.5;
    const start0 = mainCount;
    const geo = ribbonGeometry(start0, branchNum * ARC_BRANCH_SAMPLES, 0, 1);
    const branchHandles = createElectricArcRibbonMaterial(
      arcBuffer,
      widthBuffer,
      totalSamples,
      {
        coreColor: coreColorVec,
        arcColor: arcColorVec,
        halfWidth: coreHalf * wMid * 2,
        intensity: cfg.intensity * 0.8,
        glowIntensity: cfg.glow.intensity * 0.7,
        haloIntensity: cfg.glow.intensity * 0.3,
        layers: "core",
        profileMode: prof
      }
    );
    const mesh = new THREE2.Mesh(geo, branchHandles.material);
    mesh.frustumCulled = false;
    root.add(mesh);
    branchGeos.push(geo);
    branchMeshes.push({ mesh, handles: branchHandles });
  }
  const contactA = new Vector3(cfg.start.x, cfg.start.y, cfg.start.z);
  const contactB = new Vector3(cfg.end.x, cfg.end.y, cfg.end.z);
  const contactMeshes = [];
  const contactGeo = contactGeometry();
  if (cfg.contact.enabled) {
    for (const center of [contactA, contactB]) {
      const handles = createElectricContactMaterial({
        color: arcColorVec,
        coreColor: coreColorVec,
        center,
        halfSize: Math.max(cfg.contact.radius, 5e-3),
        intensity: cfg.contact.intensity,
        flicker: uniforms.globalFlicker
      });
      const mesh = new THREE2.Mesh(contactGeo, handles.material);
      mesh.frustumCulled = false;
      root.add(mesh);
      contactMeshes.push({ mesh, handles, center });
    }
  }
  const update = (cycle, start, end) => {
    uniforms.start.value.copy(start);
    uniforms.end.value.copy(end);
    _dir.subVectors(end, start);
    const dist = Math.max(_dir.length(), 1e-4);
    _dir.multiplyScalar(1 / dist);
    if (_dir.y < 0.85 && _dir.y > -0.85) _helper.set(0, 1, 0);
    else _helper.set(1, 0, 0);
    nBasis(_ub, _dir, _helper);
    nBasis(_vb, _dir, _ub);
    uniforms.basisU.value.copy(_ub);
    uniforms.basisV.value.copy(_vb);
    const epoch = Math.floor(cycle.elapsed * cfg.flickerHz * cfg.speed);
    const flicker = globalFlicker(cfg.seed, epoch);
    setNum(uniforms.time, cycle.elapsed * cfg.speed);
    setNum(uniforms.epoch, epoch);
    setNum(uniforms.globalFlicker, flicker);
    setNum(uniforms.amp, cfg.amplitude);
    setNum(uniforms.knots, cfg.coarseKnots);
    setNum(uniforms.microF, cfg.microFrequency);
    setNum(uniforms.pin, cfg.endpointPinning);
    setNum(uniforms.brightnessVar, cfg.brightnessVariation);
    setNum(uniforms.intensity, cfg.intensity);
    setNum(
      uniforms.branchProb,
      cfg.branchProbability || cfg.branches.probability
    );
    if (contactMeshes.length === 2) {
      contactMeshes[0].center.copy(start);
      contactMeshes[1].center.copy(end);
    }
    return flicker;
  };
  const updateLive = (patch) => {
    setColors();
    const pMode = cfg.glow.profile === "triangle" ? 1 : 0;
    for (const m of mainMeshes) {
      const h = m.handles;
      setNum(h.profileMode, pMode);
      if (m.layer === "core") {
        setNum(h.halfWidth, coreHalfOf());
        setNum(h.intensity, cfg.intensity);
      } else if (m.layer === "sheath") {
        setNum(h.halfWidth, coreHalfOf() * 2);
        setNum(h.glowIntensity, cfg.glow.intensity * 0.8);
        setNum(h.haloIntensity, cfg.glow.intensity * 0.4);
      } else {
        setNum(h.halfWidth, coreHalfOf() * Math.max(2, cfg.glow.width));
        setNum(h.glowIntensity, cfg.glow.intensity * 0.8);
        setNum(h.haloIntensity, cfg.glow.intensity * 0.4);
      }
    }
    for (const b of branchMeshes) {
      setNum(b.handles.halfWidth, coreHalfOf() * midThicknessScale() * 2);
      setNum(b.handles.intensity, cfg.intensity * 0.8);
      setNum(b.handles.profileMode, pMode);
    }
    for (const c of contactMeshes) {
      setNum(c.handles.intensity, cfg.contact.intensity);
      setNum(c.handles.halfSize, Math.max(cfg.contact.radius, 5e-3));
    }
  };
  const coreHalfOf = () => Math.max(cfg.thickness, 5e-4) * 0.5;
  const midThicknessScale = () => {
    const [ts0, ts1] = cfg.branches.thicknessScale;
    return (ts0 + ts1) * 0.5;
  };
  let disposed = false;
  return {
    root,
    update,
    updateLive,
    backend: "GPU" /* GPU */,
    computeNode: pipeline.computeNode,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const m of mainMeshes) m.handles.material.dispose();
      mainGeo.dispose();
      for (const b of branchMeshes) {
        b.handles.material.dispose();
      }
      for (const geo of branchGeos) geo.dispose();
      for (const c of contactMeshes) c.handles.material.dispose();
      contactGeo.dispose();
      pipeline.dispose();
    }
  };
}
var MLS_MPM_WORKGROUP_SIZE = 64;
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
  fov: Math.PI / 4,
  minBoxSize: 1,
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
var encodeFixedPoint = (value, multiplier = MLS_MPM_FIXED_POINT_MULTIPLIER) => Math.trunc(value * multiplier);
var decodeFixedPoint = (encoded, multiplier = MLS_MPM_FIXED_POINT_MULTIPLIER) => encoded / multiplier;
var computeMLSMPMGridDims = (boxSize) => [
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[0])),
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[1])),
  Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[2]))
];
var computeMLSMPMGridCount = (boxSize) => {
  const [nx, ny, nz] = computeMLSMPMGridDims(boxSize);
  return nx * ny * nz;
};
var mlsmpmQuadraticWeights = (diff) => {
  const minus = 0.5 - diff;
  const plus = 0.5 + diff;
  return [0.5 * minus * minus, 0.75 - diff * diff, 0.5 * plus * plus];
};
var mlsmpmCellIndex = (ix, iy, iz, ny, nz) => ix * ny * nz + iy * nz + iz;
var mlsmpmCellWordBase = (ix, iy, iz, ny, nz) => mlsmpmCellIndex(ix, iy, iz, ny, nz) * MLS_MPM_CELL_WORDS;
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
var countMLSMPMDambreak = (boxSize, capacity, spacing = MLS_MPM_PARTICLE_SPACING) => initMLSMPMDambreak(boxSize, capacity, spacing, () => 0).count;
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
var SPH_SLAB_RADIUS = 1;
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
  fov: Math.PI / 4,
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
var computeSPHGridCount = (kernelRadius, halfMax) => {
  const [x, y, z] = computeSPHGridDims(kernelRadius, halfMax);
  return x * y * z;
};
var computeSPHOffset = (kernelRadius = SPH_DEFAULT_KERNEL_RADIUS) => SPH_SENTINEL_CELLS * kernelRadius * SPH_CELL_SIZE_FACTOR / 2;
var sphCellId = (xi, yi, zi, xGrids, yGrids) => xi + yi * xGrids + zi * xGrids * yGrids;
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
var countSPHDambreak = (halfBoxSize, capacity, kernelRadius) => initSPHDambreak(halfBoxSize, capacity, kernelRadius, () => 0).count;
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
  const box = realHalfBox ?? halfBoxSize;
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

// src/js/effects/three-particles/three-particles-constants.ts
var POINT_SIZE_SCALE = 100;
var ALPHA_DISCARD_THRESHOLD = 1e-3;

// src/js/effects/three-particles/webgpu/tsl-fluid-metaball-material.ts
function createFluidTSLMaterial(sharedUniforms, rendererConfig, gpuCompute = false, stretchFactor = 1, absorption = 1.44, ior = 1.33) {
  const u = createParticleUniforms(sharedUniforms);
  const uViewportHeight = uniform(
    typeof sharedUniforms.viewportHeight?.value === "number" ? sharedUniforms.viewportHeight.value : 1
  );
  sharedUniforms.viewportHeight = uViewportHeight;
  const uStretch = uniform(float(stretchFactor));
  const uAbsorbK = uniform(float(absorption * 1.442695));
  const f0 = Math.pow(ior - 1, 2) / Math.pow(ior + 1, 2);
  const uF0 = uniform(float(f0));
  const aInstanceOffset = attribute("instanceOffset");
  const aColor = attribute("instanceColor");
  const aVelocity = attribute("instanceVelocity");
  const aParticleState = gpuCompute ? attribute("instanceParticleState") : null;
  const aStartValues = gpuCompute ? attribute("instanceStartValues") : null;
  const aSize = gpuCompute ? null : attribute("instanceSize");
  const aLifetime = gpuCompute ? null : attribute("instanceLifetime");
  const aStartLifetime = gpuCompute ? null : attribute("instanceStartLifetime");
  const aStartFrame = gpuCompute ? null : attribute("instanceStartFrame");
  const vColor = varyingProperty("vec4", "vColor");
  const vLifetime = varyingProperty("float", "vLifetime");
  const vStartLifetime = varyingProperty("float", "vStartLifetime");
  const vStartFrame = varyingProperty("float", "vStartFrame");
  const vUv = varyingProperty("vec2", "vUv");
  const vVelXY = varyingProperty("vec2", "vVelXY");
  const vVelZ = varyingProperty("float", "vVelZ");
  const vViewZ = varyingProperty("float", "vViewZ");
  const vertexNode = Fn(() => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    If(aColor.w.greaterThan(0), () => {
      vColor.assign(aColor.toVar());
      if (gpuCompute) {
        vLifetime.assign(aParticleState.x);
        vStartLifetime.assign(aStartValues.x);
        vStartFrame.assign(aParticleState.w);
      } else {
        vLifetime.assign(aLifetime);
        vStartLifetime.assign(aStartLifetime);
        vStartFrame.assign(aStartFrame);
      }
      vUv.assign(
        vec2(positionLocal.x.add(0.5), float(0.5).sub(positionLocal.y))
      );
      const mvPos = modelViewMatrix.mul(vec4(aInstanceOffset.xyz, 1)).toVar();
      const mvVel = modelViewMatrix.mul(vec4(aVelocity.xyz, 0)).xyz;
      vVelXY.assign(vec2(mvVel.x, mvVel.y));
      vVelZ.assign(mvVel.z);
      const dist = sqrt(
        mvPos.x.mul(mvPos.x).add(mvPos.y.mul(mvPos.y)).add(mvPos.z.mul(mvPos.z))
      );
      const sizeVal = gpuCompute ? aParticleState.y : aSize;
      const pointSizePx = sizeVal.mul(POINT_SIZE_SCALE).div(dist);
      const projY = cameraProjectionMatrix.element(1).element(1);
      const halfExtent = pointSizePx.mul(mvPos.z.negate()).div(projY.mul(uViewportHeight).mul(0.5));
      const vlen = sqrt(mvVel.x.mul(mvVel.x).add(mvVel.y.mul(mvVel.y)));
      const hasVel = vlen.greaterThan(1e-4);
      const invVlen = float(1).div(hasVel.select(vlen, float(1)));
      const tx = hasVel.select(mvVel.x.mul(invVlen), float(1));
      const ty = hasVel.select(mvVel.y.mul(invVlen), float(0));
      const stretch = float(1).add(min(vlen.mul(uStretch), 3));
      const ox = positionLocal.x.mul(halfExtent).mul(stretch);
      const oy = positionLocal.y.mul(halfExtent);
      mvPos.x.addAssign(tx.mul(ox).add(ty.mul(oy).negate()));
      mvPos.y.addAssign(tx.mul(oy).add(ty.mul(ox)));
      vViewZ.assign(mvPos.z.negate());
      clipPos.assign(cameraProjectionMatrix.mul(mvPos));
    });
    return clipPos;
  })();
  const fragmentColor = Fn(() => {
    const p = vUv.mul(2).sub(vec2(1, 1));
    const r2 = p.x.mul(p.x).add(p.y.mul(p.y));
    If(r2.greaterThan(1), () => {
      Discard();
    });
    const nz = sqrt(float(1).sub(r2));
    const N = vec3(p.x, p.y, nz);
    const speed = sqrt(
      vVelXY.x.mul(vVelXY.x).add(vVelXY.y.mul(vVelXY.y)).add(vVelZ.mul(vVelZ))
    );
    const speedBoost = float(1).add(min(speed.mul(0.15), 0.5));
    const frameIndex = computeFrameIndex({
      vLifetime,
      vStartLifetime,
      vStartFrame,
      uFps: u.uFps,
      uUseFPSForFrameIndex: u.uUseFPSForFrameIndex,
      uTiles: u.uTiles
    });
    const uvPoint = computeSpriteSheetUV({
      baseUV: vUv,
      frameIndex,
      uTiles: u.uTiles
    });
    const texColor = texture(u.uMap, uvPoint);
    const base = vColor.mul(texColor);
    const NdotL = max(dot(N, vec3(0, 0, 1)), float(0));
    const diffuse = float(0.5).add(float(0.5).mul(NdotL));
    const thickness = nz.mul(2);
    const absorb = exp2(thickness.mul(uAbsorbK).negate());
    const oneMinusNz = float(1).sub(nz);
    const fresnel = uF0.add(float(1).sub(uF0).mul(pow(oneMinusNz, 5)));
    const refr = base.rgb.mul(absorb).mul(diffuse).mul(speedBoost);
    const reflColor = base.rgb.add(vec3(0.08, 0.08, 0.1));
    const mixedColor = refr.mul(float(1).sub(fresnel)).add(reflColor.mul(fresnel));
    const outColor = vec4(
      mixedColor,
      vColor.w.mul(float(1).sub(exp2(thickness.mul(uAbsorbK).negate())))
    );
    const softFade = computeSoftParticleFade({
      viewZ: vViewZ,
      uSoftEnabled: u.uSoftEnabled,
      uSoftIntensity: u.uSoftIntensity,
      uSceneDepthTex: u.uSceneDepthTex,
      uCameraNearFar: u.uCameraNearFar
    });
    outColor.assign(vec4(outColor.xyz, outColor.w.mul(softFade)));
    applyBackgroundDiscard({
      texColor: outColor,
      uDiscardBg: u.uDiscardBg,
      uBgColor: u.uBgColor,
      uBgTolerance: u.uBgTolerance
    });
    Discard(outColor.w.lessThan(ALPHA_DISCARD_THRESHOLD));
    return outColor;
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.vertexNode = vertexNode;
  material.colorNode = fragmentColor;
  return material;
}
var GAUSSIAN_WEIGHTS = [0.15, 0.23, 0.31, 0.23, 0.15];
var bilinearWeight = (d) => Math.max(0, 1 - d);
var BILATERAL_GRID_LEN = 6;
var DEPTH_LEVEL_RADII = [1, 2, 4, 8, 8];
var DEPTH_LEVEL_MIPS = DEPTH_LEVEL_RADII.map(
  (r) => Math.floor(Math.log2(r))
);
var FLUID_SHADING_DEFAULTS = {
  extinction: [0, 0.0693, 0.109],
  ior: [1.31, 1.33, 1.34],
  f0: 0.02,
  specularPower: 250
};
var beerLambert = (k, waterColor) => waterColor.map((c) => Math.exp(-k * (1 - c)));
var fresnelCoefficient = (cosTheta, f0) => {
  const oneMinusCos = 1 - cosTheta;
  return Math.max(f0, f0 + (1 - f0) * oneMinusCos ** 5);
};
function createFluidAttributes() {
  return {
    offset: attribute("instanceOffset"),
    color: attribute("instanceColor"),
    particleState: attribute("instanceParticleState"),
    startValues: attribute("instanceStartValues"),
    velocity: attribute("instanceVelocity")
  };
}
function createFluidUniforms(config) {
  const water = config?.waterColor ?? [0, 0.7375, 0.95];
  return {
    sphereSize: uniform(float(config?.sphereSize ?? 1.2)),
    near: cameraNear,
    far: cameraFar,
    density: uniform(float(config?.density ?? 0.7)),
    waterColor: uniform(vec3(water[0], water[1], water[2])),
    f0: uniform(float(FLUID_SHADING_DEFAULTS.f0)),
    specularPower: uniform(float(FLUID_SHADING_DEFAULTS.specularPower))
  };
}
var billboardVertex = (attrs, u) => Fn(() => {
  const clipPos = vec4(0, 0, 0, -1).toVar();
  const vColor = varyingProperty("vec4", "vColor");
  const vUv = varyingProperty("vec2", "vUv");
  const vViewZ = varyingProperty("float", "vViewZ");
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
        vec2(float(0.5), float(0.5)).add(positionLocal.xy).mul(float(1).div(size))
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
function createFluidDepthTSLMaterial(config) {
  const attrs = createFluidAttributes();
  const u = createFluidUniforms(config);
  const material = new MeshBasicNodeMaterial();
  material.vertexNode = billboardVertex(attrs, u);
  material.colorNode = Fn(() => {
    const vUv = varyingProperty("vec2", "vUv");
    const vViewZ = varyingProperty("float", "vViewZ");
    varyingProperty("vec4", "vColor");
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const thickness = sqrt(float(1).sub(r2));
    const normal = normalize(vec3(nxy.x, nxy.y, thickness));
    const capViewZ = dot(
      normal,
      modelViewMatrix.mul(vec4(attrs.offset.xyz, float(1))).xyz.sub(vec3(nxy.x.mul(u.near), nxy.y.mul(u.near), u.near))
    ).add(u.near).sub(oneMinus(normal.z).mul(u.sphereSize));
    If(thickness.greaterThan(float(0)), () => {
      vViewZ.assign(capViewZ);
    });
    return vec4(capViewZ, float(0), float(0), float(1));
  })();
  return material;
}
function createFluidThicknessTSLMaterial(config) {
  const attrs = createFluidAttributes();
  const u = createFluidUniforms(config);
  const material = new MeshBasicNodeMaterial();
  material.vertexNode = billboardVertex(attrs, u);
  material.colorNode = Fn(() => {
    const vUv = varyingProperty("vec2", "vUv");
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const thickness = sqrt(float(1).sub(r2));
    return vec4(thickness, float(0), float(0), float(0));
  })();
  return material;
}
function createFluidBilateralTSLMaterial(level, sourceRadius, sourceTexture, iterationCount) {
  createFluidUniforms(void 0);
  const material = new MeshBasicNodeMaterial();
  const radius = float(DEPTH_LEVEL_RADII[level] ?? 8);
  const invRadius = float(1).div(radius);
  float(level);
  material.colorNode = Fn(() => {
    const wSum = float(0).toVar();
    const wTotal = float(0).toVar();
    textureLoad(
      sourceTexture,
      floor(
        screenUV.mul(vec2(float(1).div(invRadius), float(1).div(invRadius)))
      ),
      float(0)
    );
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
        sample.sub(float(0));
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
    return vec4(filtered, float(0), float(0), radius);
  })();
  return material;
}
var maxAbs = (a, b) => {
  const absA = a.abs();
  const absB = b.abs();
  return absA.greaterThan(absB).select(absA, absB);
};
function createFluidGaussianTSLMaterial(textureIn, axisWeight) {
  const material = new MeshBasicNodeMaterial();
  const weights = GAUSSIAN_WEIGHTS.map((w) => float(w));
  material.colorNode = Fn(() => {
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
function createFluidShadingTSLMaterial(sources, config) {
  const material = new MeshBasicNodeMaterial();
  const density = uniform(float(config?.density ?? 0.7));
  const water = config?.waterColor ?? [0, 0.7375, 0.95];
  const uWater = uniform(vec3(water[0], water[1], water[2]));
  const bg = uniform(vec3(1, 1, 1));
  material.colorNode = Fn(() => {
    const d1 = textureLoad(sources.depth1, screenUV, float(0));
    const d2 = textureLoad(sources.depth2, screenUV, float(0));
    const d3 = textureLoad(sources.depth3, screenUV, float(0));
    const d4 = textureLoad(sources.depth4, screenUV, float(0));
    const thick = textureLoad(sources.thickness, screenUV, float(0)).x;
    const anyDepth = d1.x.add(d2.x).add(d3.x).add(d4.x).greaterThan(float(0));
    const outColor = vec4(bg.xyz, float(1)).toVar();
    If(anyDepth, () => {
      const depth0 = d1.x.equal(float(0)).select(d2.x, d1.x);
      const depth = max(depth0, float(1e-4));
      const radius = max(d1.w, float(1));
      floor(log2(radius));
      const thickness = max(thick.mul(depth), float(0));
      outColor.assign(vec4(beerNode(density, thickness, uWater), float(1)));
    });
    return outColor;
  })();
  return material;
}
var beerNode = (k, thickness, waterColor) => {
  const t = k.mul(thickness);
  return vec3(
    exp(t.mul(oneMinus(waterColor.x))),
    exp(t.mul(oneMinus(waterColor.y))),
    exp(t.mul(oneMinus(waterColor.z)))
  );
};
function createFluidSphereTSLMaterial(config) {
  const material = new MeshBasicNodeMaterial();
  const attrs = createFluidAttributes();
  material.vertexNode = Fn(() => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    const vUv = varyingProperty("vec2", "vUv");
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
  material.colorNode = Fn(() => {
    const vUv = varyingProperty("vec2", "vUv");
    const nxy = vec2(vUv.x.mul(2).sub(1), vUv.y.mul(2).sub(1));
    const r2 = dot(nxy, nxy);
    Discard(r2.greaterThan(float(1)));
    const normal = normalize(vec3(nxy.x, nxy.y, sqrt(float(1).sub(r2))));
    const lightDir = normalize(vec3(float(-1), float(1), float(-1)));
    const viewDir = normalize(vec3(float(0), float(0), float(1)));
    const diffuse = clamp(dot(lightDir, normal), float(0), float(1));
    const half = normalize(lightDir.add(viewDir));
    pow(
      clamp(dot(normal, half), float(0), float(1)),
      float(500)
    );
    const fresnel = fresnelNode(dot(normal, viewDir.negate()));
    reflect(viewDir.negate(), normal);
    const atten = vec3(float(0.0333), float(0.0333), float(0.0333));
    const lin = vec3(fresnel, fresnel, fresnel).mul(atten).add(
      vec3(diffuse, diffuse, diffuse).mul(
        vec3(float(0.941), float(0.941), float(0.941))
      )
    );
    return vec4(
      lin.x.sub(float(0.0333)),
      lin.y.sub(float(0.0333)),
      lin.z.sub(float(0.0333)),
      float(1)
    );
  })();
  return material;
}
var fresnelNode = (cosTheta) => float(0.02).add(float(0.98).mul(pow(oneMinus(cosTheta), float(5))));
function createFullScreenGeometry() {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
  );
  return geometry;
}
function fullScreenQuad(material) {
  material.vertexNode = vec4(positionLocal.xy, float(0), float(1));
  const mesh = new Mesh(createFullScreenGeometry(), material);
  mesh.frustumCulled = false;
  return mesh;
}
function instancedQuad(material, geometry) {
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}
function toTextureNode(node, passNodes) {
  passNodes.push(node);
  return node;
}
function buildFluidScreenSpacePasses(config, envMap = null, camera, particleGeometry) {
  const quadFor = (material) => particleGeometry ? instancedQuad(material, particleGeometry) : fullScreenQuad(material);
  if (config?.sphereRender) {
    return {
      material: createFluidSphereTSLMaterial(),
      passNodes: [],
      geometry: particleGeometry
    };
  }
  const cam = camera;
  const passNodes = [];
  const depthScene = new Scene();
  depthScene.add(quadFor(createFluidDepthTSLMaterial(config)));
  const depthPass0 = toTextureNode(
    pass(depthScene, cam, {
      type: FloatType,
      format: RedFormat,
      depthBuffer: false
    }),
    passNodes
  );
  const levelTextures = [depthPass0];
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
          depthBuffer: false
        }),
        passNodes
      )
    );
  }
  const thicknessScene = new Scene();
  thicknessScene.add(quadFor(createFluidThicknessTSLMaterial(config)));
  const thicknessPass = toTextureNode(
    pass(thicknessScene, cam, {
      type: HalfFloatType,
      format: RedFormat,
      depthBuffer: false
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
      depthBuffer: false
    }),
    passNodes
  );
  const blurYScene = new Scene();
  blurYScene.add(fullScreenQuad(createFluidGaussianTSLMaterial(blurXPass, 0)));
  const blurYPass = toTextureNode(
    pass(blurYScene, cam, {
      type: HalfFloatType,
      format: RedFormat,
      depthBuffer: false
    }),
    passNodes
  );
  const shading = createFluidShadingTSLMaterial(
    {
      depth1: levelTextures[1] ?? depthPass0,
      depth2: levelTextures[2] ?? depthPass0,
      depth3: levelTextures[3] ?? depthPass0,
      depth4: levelTextures[4] ?? depthPass0,
      thickness: blurYPass},
    config
  );
  shading.vertexNode = vec4(positionLocal.xy, float(0), float(1));
  return { material: shading, passNodes, geometry: createFullScreenGeometry() };
}
function createInstancedBillboardTSLMaterial(sharedUniforms, rendererConfig, gpuCompute = false) {
  const u = createParticleUniforms(sharedUniforms);
  const uViewportHeight = uniform(
    typeof sharedUniforms.viewportHeight?.value === "number" ? sharedUniforms.viewportHeight.value : 1
  );
  sharedUniforms.viewportHeight = uViewportHeight;
  const aInstanceOffset = attribute("instanceOffset");
  const aColor = attribute("instanceColor");
  const aParticleState = gpuCompute ? attribute("instanceParticleState") : null;
  const aStartValues = gpuCompute ? attribute("instanceStartValues") : null;
  const aSize = gpuCompute ? null : attribute("instanceSize");
  const aLifetime = gpuCompute ? null : attribute("instanceLifetime");
  const aStartLifetime = gpuCompute ? null : attribute("instanceStartLifetime");
  const aRotation = gpuCompute ? null : attribute("instanceRotation");
  const aStartFrame = gpuCompute ? null : attribute("instanceStartFrame");
  const vColor = varyingProperty("vec4", "vColor");
  const vLifetime = varyingProperty("float", "vLifetime");
  const vStartLifetime = varyingProperty("float", "vStartLifetime");
  const vRotation = varyingProperty("float", "vRotation");
  const vStartFrame = varyingProperty("float", "vStartFrame");
  const vUv = varyingProperty("vec2", "vUv");
  const vViewZ = varyingProperty("float", "vViewZ");
  const vertexNode = Fn(() => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    If(aColor.w.greaterThan(0), () => {
      vColor.assign(aColor.toVar());
      if (gpuCompute) {
        vLifetime.assign(aParticleState.x);
        vStartLifetime.assign(aStartValues.x);
        vRotation.assign(aParticleState.z);
        vStartFrame.assign(aParticleState.w);
      } else {
        vLifetime.assign(aLifetime);
        vStartLifetime.assign(aStartLifetime);
        vRotation.assign(aRotation);
        vStartFrame.assign(aStartFrame);
      }
      vUv.assign(
        vec2(positionLocal.x.add(0.5), float(0.5).sub(positionLocal.y))
      );
      const mvPosition = modelViewMatrix.mul(vec4(aInstanceOffset.xyz, 1)).toVar();
      const dist = length(mvPosition.xyz);
      const sizeVal = gpuCompute ? aParticleState.y : aSize;
      const pointSizePx = sizeVal.mul(POINT_SIZE_SCALE).div(dist);
      const projY = cameraProjectionMatrix.element(1).element(1);
      const perspectiveSize = pointSizePx.mul(mvPosition.z.negate()).div(projY.mul(uViewportHeight).mul(0.5));
      mvPosition.x.addAssign(positionLocal.x.mul(perspectiveSize));
      mvPosition.y.addAssign(positionLocal.y.mul(perspectiveSize));
      vViewZ.assign(mvPosition.z.negate());
      clipPos.assign(cameraProjectionMatrix.mul(mvPosition));
    });
    return clipPos;
  })();
  const fragmentColor = Fn(() => {
    const outColor = vColor.toVar();
    const center = vec2(0.5, 0.5);
    const centered = vUv.sub(center);
    const cosR = cos(vRotation);
    const sinR = sin(vRotation);
    const rotated = vec2(
      centered.x.mul(cosR).add(centered.y.mul(sinR)),
      centered.x.mul(sinR).negate().add(centered.y.mul(cosR))
    );
    const rotatedUV = rotated.add(center);
    const dist = length(rotatedUV.sub(center));
    If(dist.greaterThan(0.5), () => {
      Discard();
    });
    const frameIndex = computeFrameIndex({
      vLifetime,
      vStartLifetime,
      vStartFrame,
      uFps: u.uFps,
      uUseFPSForFrameIndex: u.uUseFPSForFrameIndex,
      uTiles: u.uTiles
    });
    const uvPoint = computeSpriteSheetUV({
      baseUV: rotatedUV,
      frameIndex,
      uTiles: u.uTiles
    });
    const texColor = texture(u.uMap, uvPoint);
    outColor.assign(outColor.mul(texColor));
    applyBackgroundDiscard({
      texColor,
      uDiscardBg: u.uDiscardBg,
      uBgColor: u.uBgColor,
      uBgTolerance: u.uBgTolerance
    });
    const softFade = computeSoftParticleFade({
      viewZ: vViewZ,
      uSoftEnabled: u.uSoftEnabled,
      uSoftIntensity: u.uSoftIntensity,
      uSceneDepthTex: u.uSceneDepthTex,
      uCameraNearFar: u.uCameraNearFar
    });
    outColor.assign(vec4(outColor.xyz, outColor.w.mul(softFade)));
    Discard(outColor.w.lessThan(ALPHA_DISCARD_THRESHOLD));
    return outColor;
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.vertexNode = vertexNode;
  material.colorNode = fragmentColor;
  return material;
}
var applyQuaternion = Fn(
  ({ v, q }) => {
    const t = cross(q.xyz, v).mul(2);
    return v.add(t.mul(q.w)).add(cross(q.xyz, t));
  }
);
function createMeshParticleTSLMaterial(sharedUniforms, rendererConfig, gpuCompute = false) {
  const u = createParticleUniforms(sharedUniforms);
  const aInstanceOffset = attribute("instanceOffset");
  const aColor = attribute("instanceColor");
  const aParticleState = gpuCompute ? attribute("instanceParticleState") : null;
  const aStartValues = gpuCompute ? attribute("instanceStartValues") : null;
  const aInstanceQuat = gpuCompute ? null : attribute("instanceQuat");
  const aSize = gpuCompute ? null : attribute("instanceSize");
  const aLifetime = gpuCompute ? null : attribute("instanceLifetime");
  const aStartLifetime = gpuCompute ? null : attribute("instanceStartLifetime");
  const aRotation = gpuCompute ? null : attribute("instanceRotation");
  const aStartFrame = gpuCompute ? null : attribute("instanceStartFrame");
  const vColor = varyingProperty("vec4", "vColor");
  const vLifetime = varyingProperty("float", "vLifetime");
  const vStartLifetime = varyingProperty("float", "vStartLifetime");
  const vStartFrame = varyingProperty("float", "vStartFrame");
  const vRotation = varyingProperty("float", "vRotation");
  const vNormal = varyingProperty("vec3", "vNormal");
  const vViewZ = varyingProperty("float", "vViewZ");
  const vertexSetup = Fn(() => {
    const clipPos = vec4(0, 0, 0, -1).toVar();
    If(aColor.w.greaterThan(0), () => {
      vColor.assign(aColor.toVar());
      if (gpuCompute) {
        vLifetime.assign(aParticleState.x);
        vStartLifetime.assign(aStartValues.x);
        vStartFrame.assign(aParticleState.w);
        vRotation.assign(aParticleState.z);
      } else {
        vLifetime.assign(aLifetime);
        vStartLifetime.assign(aStartLifetime);
        vStartFrame.assign(aStartFrame);
        vRotation.assign(aRotation);
      }
      let quat;
      if (gpuCompute) {
        const halfZ = aParticleState.z.mul(0.5);
        quat = vec4(0, 0, sin(halfZ), cos(halfZ));
      } else {
        quat = aInstanceQuat;
      }
      const rotatedPos = applyQuaternion({
        v: positionLocal,
        q: quat
      });
      const scaledPos = rotatedPos.mul(gpuCompute ? aParticleState.y : aSize);
      const worldPos = scaledPos.add(aInstanceOffset.xyz);
      const mvPos = modelViewMatrix.mul(vec4(worldPos, 1));
      vViewZ.assign(mvPos.z.negate());
      const rotatedNormal = applyQuaternion({
        v: normalLocal,
        q: quat
      });
      const mvNormal = modelViewMatrix.mul(vec4(rotatedNormal, 0)).xyz;
      vNormal.assign(mvNormal.normalize());
      clipPos.assign(cameraProjectionMatrix.mul(mvPos));
    });
    return clipPos;
  })();
  const fragmentColor = Fn(() => {
    const outColor = vColor.toVar();
    const uvPoint = vec2(uv()).toVar();
    If(u.uTiles.x.greaterThan(1).or(u.uTiles.y.greaterThan(1)), () => {
      const frameIndex = computeFrameIndex({
        vLifetime,
        vStartLifetime,
        vStartFrame,
        uFps: u.uFps,
        uUseFPSForFrameIndex: u.uUseFPSForFrameIndex,
        uTiles: u.uTiles
      });
      uvPoint.assign(
        computeSpriteSheetUV({
          baseUV: uv(),
          frameIndex,
          uTiles: u.uTiles
        })
      );
    });
    const texColor = texture(u.uMap, uvPoint);
    outColor.assign(outColor.mul(texColor));
    applyBackgroundDiscard({
      texColor,
      uDiscardBg: u.uDiscardBg,
      uBgColor: u.uBgColor,
      uBgTolerance: u.uBgTolerance
    });
    const lightIntensity = float(0.5).add(
      float(0.5).mul(max(dot(vNormal, vec3(0, 0, 1)), float(0)))
    );
    outColor.assign(vec4(outColor.xyz.mul(lightIntensity), outColor.w));
    const softFade = computeSoftParticleFade({
      viewZ: vViewZ,
      uSoftEnabled: u.uSoftEnabled,
      uSoftIntensity: u.uSoftIntensity,
      uSceneDepthTex: u.uSceneDepthTex,
      uCameraNearFar: u.uCameraNearFar
    });
    outColor.assign(vec4(outColor.xyz, outColor.w.mul(softFade)));
    Discard(outColor.w.lessThan(ALPHA_DISCARD_THRESHOLD));
    return outColor;
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.vertexNode = vertexSetup;
  material.colorNode = fragmentColor;
  return material;
}
function createPointSpriteTSLMaterial(sharedUniforms, rendererConfig, gpuCompute = false) {
  const u = createParticleUniforms(sharedUniforms);
  const aColor = attribute("color");
  const aParticleState = gpuCompute ? attribute("particleState") : null;
  const aStartValues = gpuCompute ? attribute("startValues") : null;
  const aSize = gpuCompute ? null : attribute("size");
  const aLifetime = gpuCompute ? null : attribute("lifetime");
  const aStartLifetime = gpuCompute ? null : attribute("startLifetime");
  const aRotation = gpuCompute ? null : attribute("rotation");
  const aStartFrame = gpuCompute ? null : attribute("startFrame");
  const mvPos = modelViewMatrix.mul(vec4(positionLocal, 1));
  const sizeVal = gpuCompute ? aParticleState.y : aSize;
  const sizeNode = aColor.w.greaterThan(0).select(sizeVal.mul(POINT_SIZE_SCALE).div(length(mvPos.xyz)), float(0));
  const vColor = varyingProperty("vec4", "vColor");
  const vLifetime = varyingProperty("float", "vLifetime");
  const vStartLifetime = varyingProperty("float", "vStartLifetime");
  const vRotation = varyingProperty("float", "vRotation");
  const vStartFrame = varyingProperty("float", "vStartFrame");
  const vViewZ = varyingProperty("float", "vViewZ");
  const vertexSetup = Fn(() => {
    If(aColor.w.greaterThan(0), () => {
      vColor.assign(aColor.toVar());
      if (gpuCompute) {
        vLifetime.assign(aParticleState.x);
        vStartLifetime.assign(aStartValues.x);
        vRotation.assign(aParticleState.z);
        vStartFrame.assign(aParticleState.w);
      } else {
        vLifetime.assign(aLifetime);
        vStartLifetime.assign(aStartLifetime);
        vRotation.assign(aRotation);
        vStartFrame.assign(aStartFrame);
      }
      vViewZ.assign(mvPos.z.negate());
    });
    return positionLocal;
  })();
  const fragmentColor = Fn(() => {
    const outColor = vColor.toVar();
    const frameIndex = computeFrameIndex({
      vLifetime,
      vStartLifetime,
      vStartFrame,
      uFps: u.uFps,
      uUseFPSForFrameIndex: u.uUseFPSForFrameIndex,
      uTiles: u.uTiles
    });
    const center = vec2(0.5, 0.5);
    const centered = vec2(0, 0);
    const cosR = cos(vRotation);
    const sinR = sin(vRotation);
    const rotated = vec2(
      centered.x.mul(cosR).add(centered.y.mul(sinR)),
      centered.x.mul(sinR).negate().add(centered.y.mul(cosR))
    );
    const rotatedUV = rotated.add(center);
    const dist = length(rotatedUV.sub(center));
    Discard(dist.greaterThan(0.5));
    const uvPoint = computeSpriteSheetUV({
      baseUV: rotatedUV,
      frameIndex,
      uTiles: u.uTiles
    });
    const texColor = texture(u.uMap, uvPoint);
    outColor.assign(outColor.mul(texColor));
    applyBackgroundDiscard({
      texColor,
      uDiscardBg: u.uDiscardBg,
      uBgColor: u.uBgColor,
      uBgTolerance: u.uBgTolerance
    });
    const softFade = computeSoftParticleFade({
      viewZ: vViewZ,
      uSoftEnabled: u.uSoftEnabled,
      uSoftIntensity: u.uSoftIntensity,
      uSceneDepthTex: u.uSceneDepthTex,
      uCameraNearFar: u.uCameraNearFar
    });
    outColor.assign(vec4(outColor.xyz, outColor.w.mul(softFade)));
    Discard(outColor.w.lessThan(ALPHA_DISCARD_THRESHOLD));
    return outColor;
  })();
  const material = new PointsNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.sizeNode = sizeNode;
  material.positionNode = vertexSetup;
  material.colorNode = fragmentColor;
  return material;
}
function createTrailUniforms(trailUniforms) {
  const dummy = getDummyTexture();
  const map = trailUniforms.map.value ?? dummy;
  return {
    uMap: map,
    uUseMap: uniform(float(trailUniforms.useMap.value ? 1 : 0)),
    uDiscardBg: uniform(
      float(trailUniforms.discardBackgroundColor.value ? 1 : 0)
    ),
    uBgColor: uniform(
      new trailUniforms.cameraNearFar.value.constructor(
        trailUniforms.backgroundColor.value.r,
        trailUniforms.backgroundColor.value.g,
        trailUniforms.backgroundColor.value.b
      )
    ),
    uBgTolerance: uniform(float(trailUniforms.backgroundColorTolerance.value)),
    uSoftEnabled: uniform(
      float(trailUniforms.softParticlesEnabled.value ? 1 : 0)
    ),
    uSoftIntensity: uniform(float(trailUniforms.softParticlesIntensity.value)),
    uSceneDepthTex: trailUniforms.sceneDepthTexture.value ?? dummy,
    uCameraNearFar: uniform(trailUniforms.cameraNearFar.value)
  };
}
function createTrailRibbonTSLMaterial(trailUniforms, rendererConfig) {
  const u = createTrailUniforms(trailUniforms);
  const aPosPacked = attribute("position", "vec4");
  const aNextPacked = attribute("trailNext", "vec4");
  const aUvColorA = attribute("trailUVColor", "vec4");
  const aColorBA = attribute("trailColorBA", "vec4");
  const aTrailAlpha = aNextPacked.w;
  const aTrailColor = vec4(aUvColorA.z, aUvColorA.w, aColorBA.x, aColorBA.y);
  const aTrailOffset = aUvColorA.x.sub(float(0.5));
  const aTrailHalfWidth = aPosPacked.w;
  const aTrailNext = vec3(aNextPacked.x, aNextPacked.y, aNextPacked.z);
  const aTrailUV = vec2(aUvColorA.x, aUvColorA.y);
  const vAlpha = varyingProperty("float", "vAlpha");
  const vColor = varyingProperty("vec4", "vColor");
  const vUv = varyingProperty("vec2", "vUv");
  const vViewZ = varyingProperty("float", "vViewZ");
  const positionNode = Fn(() => {
    vAlpha.assign(aTrailAlpha);
    vColor.assign(aTrailColor);
    vUv.assign(aTrailUV);
    const current = vec3(aPosPacked.x, aPosPacked.y, aPosPacked.z);
    const next = vec3(aTrailNext);
    const rawTangent = next.sub(current);
    const tangentLen = length(rawTangent);
    const tangent = normalize(
      tangentLen.lessThan(1e-4).select(vec3(0, 1, 0), rawTangent)
    );
    modelViewMatrix.mul(vec4(current, 1));
    const viewDir = normalize(cameraPosition.sub(current));
    const perp = billboardPerp({ tangent, viewDir });
    const offsetPos = current.add(perp.mul(aTrailOffset).mul(aTrailHalfWidth));
    const mvOffset = modelViewMatrix.mul(vec4(offsetPos, 1));
    vViewZ.assign(mvOffset.z.negate());
    return offsetPos;
  })();
  const colorNode = Fn(() => {
    const outColor = vColor.toVar();
    const edgeDist = float(1).sub(abs(vUv.x.mul(2).sub(1)));
    const edgeFade = smoothstep(float(0), float(0.4), edgeDist);
    If(u.uUseMap.greaterThan(0.5), () => {
      const texColor = texture(u.uMap, vUv);
      const texBrightness = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
      outColor.rgb.assign(
        outColor.rgb.mul(float(0.5).add(texBrightness.mul(0.5)))
      );
      outColor.a.assign(outColor.a.mul(texColor.a));
    });
    outColor.a.assign(outColor.a.mul(vAlpha).mul(edgeFade));
    Discard(outColor.a.lessThan(ALPHA_DISCARD_THRESHOLD));
    If(u.uSoftEnabled.greaterThan(0.5), () => {
      const depthSample = texture(u.uSceneDepthTex, screenUV).x;
      const sceneDepthLinear = linearizeDepth({
        depthSample,
        near: u.uCameraNearFar.x,
        far: u.uCameraNearFar.y
      });
      const depthDiff = sceneDepthLinear.sub(vViewZ);
      const softFade = smoothstep(float(0), u.uSoftIntensity, depthDiff);
      outColor.a.assign(outColor.a.mul(softFade));
    });
    Discard(outColor.a.lessThan(ALPHA_DISCARD_THRESHOLD));
    const diff = vec3(
      outColor.r.sub(u.uBgColor.x),
      outColor.g.sub(u.uBgColor.y),
      outColor.b.sub(u.uBgColor.z)
    );
    Discard(
      u.uDiscardBg.greaterThan(0.5).and(abs(length(diff)).lessThan(u.uBgTolerance))
    );
    return outColor;
  })();
  const material = new MeshBasicNodeMaterial();
  material.transparent = rendererConfig.transparent;
  material.blending = rendererConfig.blending;
  material.depthTest = rendererConfig.depthTest;
  material.depthWrite = rendererConfig.depthWrite;
  material.toneMapped = false;
  material.fog = false;
  material.side = DoubleSide;
  material.positionNode = positionNode;
  material.colorNode = colorNode;
  return material;
}

// src/js/effects/three-particles/webgpu/tsl-materials.ts
function createTSLParticleMaterial(rendererType, sharedUniforms, rendererConfig, gpuCompute = false, particleGeometry) {
  switch (rendererType) {
    case "INSTANCED" /* INSTANCED */:
      return createInstancedBillboardTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
    case "MESH" /* MESH */:
      return createMeshParticleTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
    case "FLUID" /* FLUID */: {
      const fluidCfg = {
        stretch: readFluidScalar(sharedUniforms, "fluidStretch", 1),
        absorption: readFluidScalar(sharedUniforms, "fluidAbsorption", 1.44),
        ior: readFluidScalar(sharedUniforms, "fluidIor", 1.33),
        sphereSize: readFluidScalar(sharedUniforms, "fluidSphereSize", 1.2),
        density: readFluidScalar(sharedUniforms, "fluidDensity", 0.7),
        waterColor: readFluidWaterColor(sharedUniforms),
        sphereRender: readFluidFlag(sharedUniforms, "fluidSphereRender")
      };
      const chain = buildFluidScreenSpacePasses(
        fluidCfg,
        sharedUniforms.envMap?.value ?? null,
        void 0,
        particleGeometry
      );
      chain.material.__fluidPassNodes = chain.passNodes;
      if (chain.geometry) {
        chain.material.__fluidPassGeometry = chain.geometry;
      }
      return chain.material;
    }
    case "POINTS" /* POINTS */:
    default:
      return createPointSpriteTSLMaterial(
        sharedUniforms,
        rendererConfig,
        gpuCompute
      );
  }
}
var readFluidScalar = (sharedUniforms, key, fallback) => {
  const raw = sharedUniforms[key]?.value;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
};
var readFluidFlag = (sharedUniforms, key) => {
  const raw = sharedUniforms[key]?.value;
  return raw === true;
};
var readFluidWaterColor = (sharedUniforms) => {
  const raw = sharedUniforms.fluidWaterColor?.value;
  if (Array.isArray(raw) && raw.length === 3) {
    return [Number(raw[0]) || 0, Number(raw[1]) || 0, Number(raw[2]) || 0];
  }
  return [0, 0.7375, 0.95];
};
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
function createTSLTrailMaterial(trailUniforms, rendererConfig) {
  return createTrailRibbonTSLMaterial(trailUniforms, rendererConfig);
}
var pair = (v) => {
  if (typeof v === "number") return [v, v];
  if (v && typeof v === "object") {
    const o = v;
    return [Number(o.min) || 0, Number(o.max) || 0];
  }
  return [0, 0];
};
var shapeKindOf = (t) => {
  switch (t) {
    case "SPHERE":
      return 0;
    case "CONE":
      return 1;
    case "CIRCLE":
      return 2;
    case "RECTANGLE":
      return 3;
    case "BOX":
      return 4;
    default:
      return 0;
  }
};
var boxEmitFromOf = (e) => e === "SHELL" ? 1 : e === "EDGE" ? 2 : 0;
function encodeShapeEmitParams(normalizedConfig, particleSystemId) {
  const bakedCurves = bakeParticleSystemCurves(
    normalizedConfig,
    particleSystemId
  );
  const pairLocal = pair;
  const [lifeMin, lifeMax] = pairLocal(normalizedConfig.startLifetime);
  const [spdMin, spdMax] = pairLocal(normalizedConfig.startSpeed);
  const [szMin, szMax] = pairLocal(normalizedConfig.startSize);
  const [rotMin, rotMax] = pairLocal(normalizedConfig.startRotation);
  const [opMin, opMax] = pairLocal(normalizedConfig.startOpacity);
  const cMin = normalizedConfig.startColor.min || { r: 1, g: 1, b: 1 };
  const cMax = normalizedConfig.startColor.max || { r: 1, g: 1, b: 1 };
  const sf = normalizedConfig.textureSheetAnimation && normalizedConfig.textureSheetAnimation.startFrame || 0;
  const sfPair = pairLocal(sf);
  const shp = normalizedConfig.shape;
  const sph = shp.sphere;
  const cone = shp.cone;
  const circ = shp.circle;
  const rect = shp.rectangle;
  const bx = shp.box;
  const num = (v, fallback) => typeof v === "number" && Number.isFinite(v) ? v : fallback;
  const kind = shapeKindOf(shp.shape);
  return {
    shapeKind: kind,
    radius: shp.shape === "CONE" ? num(cone?.radius, 1) : shp.shape === "CIRCLE" ? num(circ?.radius, 1) : num(sph?.radius, 1),
    radiusThickness: shp.shape === "CONE" ? num(cone?.radiusThickness, 1) : shp.shape === "CIRCLE" ? num(circ?.radiusThickness, 1) : num(sph?.radiusThickness, 1),
    arcDeg: shp.shape === "CONE" ? num(cone?.arc, 360) : shp.shape === "CIRCLE" ? num(circ?.arc, 360) : num(sph?.arc, 360),
    coneAngleDeg: num(cone?.angle, 90),
    rectangleRotXDeg: num(rect?.rotation?.x, 0),
    rectangleRotYDeg: num(rect?.rotation?.y, 0),
    rectangleScaleX: num(rect?.scale?.x, 1),
    rectangleScaleY: num(rect?.scale?.y, 1),
    boxScaleX: num(bx?.scale?.x, 1),
    boxScaleY: num(bx?.scale?.y, 1),
    boxScaleZ: num(bx?.scale?.z, 1),
    boxEmitFrom: boxEmitFromOf(bx?.emitFrom),
    speedMin: spdMin,
    speedMax: spdMax,
    sizeMin: szMin,
    sizeMax: szMax,
    rotMin,
    rotMax,
    opacityMin: opMin,
    opacityMax: opMax,
    lifeMin,
    lifeMax,
    colorRMin: cMin.r,
    colorRMax: cMax.r,
    colorGMin: cMin.g,
    colorGMax: cMax.g,
    colorBMin: cMin.b,
    colorBMax: cMax.b,
    startFrameMin: sfPair[0],
    startFrameMax: sfPair[1],
    // Separate rotationOverLifetime range (never the startRotation pair).
    rotOverLifeMin: (() => {
      const rol = normalizedConfig.rotationOverLifetime;
      return typeof rol?.min === "number" && Number.isFinite(rol.min) ? rol.min : 0;
    })(),
    rotOverLifeMax: (() => {
      const rol = normalizedConfig.rotationOverLifetime;
      return typeof rol?.max === "number" && Number.isFinite(rol.max) ? rol.max : 0;
    })(),
    noiseOctaves: num(normalizedConfig.noise?.octaves, 1),
    noiseUseRandomOffset: !!normalizedConfig.noise?.useRandomOffset,
    rotationCurveActive: normalizedConfig.rotationOverLifetime.isActive,
    rotationalXCurve: bakedCurves.orbitalVelX ?? -1,
    rotationalYCurve: bakedCurves.orbitalVelY ?? -1,
    rotationalZCurve: bakedCurves.orbitalVelZ ?? -1,
    linearXCurve: bakedCurves.linearVelX ?? -1,
    linearYCurve: bakedCurves.linearVelY ?? -1,
    linearZCurve: bakedCurves.linearVelZ ?? -1
  };
}
function createComputePipeline(maxParticles, instanced, normalizedConfig, particleSystemId, forceFieldCount, collisionPlaneCount = 0, subFifos, trailDesc) {
  const bakedCurves = bakeParticleSystemCurves(
    normalizedConfig,
    particleSystemId
  );
  const v = normalizedConfig.velocityOverLifetime;
  const flags = {
    sizeOverLifetime: normalizedConfig.sizeOverLifetime.isActive,
    opacityOverLifetime: normalizedConfig.opacityOverLifetime.isActive,
    colorOverLifetime: normalizedConfig.colorOverLifetime.isActive,
    rotationOverLifetime: normalizedConfig.rotationOverLifetime.isActive,
    linearVelocity: v.isActive && (isLifeTimeCurve(v.linear.x ?? 0) || isLifeTimeCurve(v.linear.y ?? 0) || isLifeTimeCurve(v.linear.z ?? 0) || v.linear.x !== 0 || v.linear.y !== 0 || v.linear.z !== 0),
    orbitalVelocity: v.isActive && (isLifeTimeCurve(v.orbital.x ?? 0) || isLifeTimeCurve(v.orbital.y ?? 0) || isLifeTimeCurve(v.orbital.z ?? 0) || v.orbital.x !== 0 || v.orbital.y !== 0 || v.orbital.z !== 0),
    noise: normalizedConfig.noise.isActive,
    forceFields: forceFieldCount > 0,
    collisionPlanes: collisionPlaneCount > 0
  };
  const shapeParams = encodeShapeEmitParams(
    normalizedConfig,
    particleSystemId
  );
  const velocityValues = {
    linear: [v.linear.x, v.linear.y, v.linear.z],
    orbital: [v.orbital.x, v.orbital.y, v.orbital.z]
  };
  if (trailDesc && !trailDesc.meta) {
    trailDesc.meta = new StorageBufferAttribute(
      new Uint32Array(Math.max(1, maxParticles) * 2),
      1
    );
  }
  const built = createModifierStorageBuffers(
    maxParticles,
    instanced,
    bakedCurves.data,
    flags.forceFields,
    flags.collisionPlanes,
    trailDesc ? trailDesc.length : 0
  );
  if (trailDesc && built.buffers.trailMeta) {
    trailDesc.meta = built.buffers.trailMeta;
  }
  return createModifierComputeUpdate(
    built.buffers,
    maxParticles,
    bakedCurves,
    flags,
    shapeParams,
    forceFieldCount,
    collisionPlaneCount,
    subFifos ?? [],
    trailDesc,
    velocityValues
  );
}

// src/webgpu.ts
function enableWebGPU(renderer) {
  const factory = {
    createTSLParticleMaterial,
    createTSLTrailMaterial,
    createComputePipeline,
    encodeForceFieldsForGPU,
    encodeCollisionPlanesForGPU,
    createSubEmitterFifoAttribute,
    createSubEmitterInitUpdate,
    createTrailRibbonUpdate,
    encodeShapeEmitParams
  };
  const registered = registerTSLMaterialFactory(
    factory,
    renderer !== void 0 ? { renderer } : void 0
  );
  if (renderer !== void 0 && registered) {
    registerElectricArcGPUFactory({ create: createElectricArcGPU }, renderer);
  } else {
    registerElectricArcGPUFactory(null);
  }
  return registered;
}

export { BILATERAL_GRID_LEN, CH, DEPTH_LEVEL_MIPS, DEPTH_LEVEL_RADII, FLUID_SHADING_DEFAULTS, GAUSSIAN_WEIGHTS, MLS_MPM_CELL_WORDS, MLS_MPM_C_WORDS, MLS_MPM_DEFAULTS, MLS_MPM_FIXED_POINT_MULTIPLIER, MLS_MPM_MAX_GRID_DIM, MLS_MPM_PARTICLE_SPACING, MLS_MPM_SUBSTEPS, MLS_MPM_WALL, MLS_MPM_WORKGROUP_SIZE, SPH_CELL_SIZE_FACTOR, SPH_DEFAULTS, SPH_DEFAULT_KERNEL_RADIUS, SPH_LATTICE_FACTOR, SPH_LATTICE_MARGIN, SPH_MAX_HALF_BOX, SPH_R2_EPSILON, SPH_SCAN_CHUNK, SPH_SCAN_STAGES, SPH_SENTINEL_CELLS, SPH_SLAB_RADIUS, SPH_SUBSTEPS, SPH_WALL_STIFFNESS, SPH_WORKGROUP_SIZE, beerLambert, bilinearWeight, buildFluidScreenSpacePasses, computeMLSMPMGridCount, computeMLSMPMGridDims, computeSPHGridCount, computeSPHGridDims, computeSPHOffset, computeSPHScanBlocks, computeSPHScanInnerSteps, countMLSMPMDambreak, countSPHDambreak, createComputePipeline, createFluidAttributes, createFluidBilateralTSLMaterial, createFluidDepthTSLMaterial, createFluidGaussianTSLMaterial, createFluidShadingTSLMaterial, createFluidSimPipeline, createFluidSphereTSLMaterial, createFluidTSLMaterial, createFluidThicknessTSLMaterial, createFluidUniforms, createMLSMPMBuffers, createMLSMPMPipeline, createModifierStorageBuffers, createSPHBuffers, createSPHPipeline, createSubEmitterFifoAttribute, createSubEmitterInitUpdate, createTSLParticleMaterial, createTSLTrailMaterial, createTrailRibbonUpdate, decodeFixedPoint, enableWebGPU, encodeCollisionPlanesForGPU, encodeFixedPoint, encodeForceFieldsForGPU, encodeShapeEmitParams, fresnelCoefficient, initMLSMPMDambreak, initSPHDambreak, mixBirthSeed, mlsmpmCellIndex, mlsmpmCellWordBase, mlsmpmQuadraticWeights, nextSystemSeed, pcg01, pcgRawU32, randomChannel, resolveMLSMPMParams, resolveSPHParams, sphCellId, sphDensityGradientScale, sphDensityKernelScale, sphKernelPowers, sphNearDensityKernelScale, sphViscosityLaplacianScale, subEmitterWindowSize };
//# sourceMappingURL=webgpu.js.map
//# sourceMappingURL=webgpu.js.map