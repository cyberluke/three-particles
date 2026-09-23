/**
 * Static ribbon geometry for the electric arc (§13).
 *
 * One indexed camera-facing strip: for every centerline sample a LEFT and a
 * RIGHT vertex; triangles `L0 R0 L1` / `R0 R1 L1`. The index and the
 * scalar attributes (arcIndex / arcU / side) never change; only the
 * `position` array is rewritten per frame — no geometry recreation, no
 * per-frame allocation.
 *
 * @module
 */
import * as THREE from 'three';

export type RibbonGeometry = {
  geometry: THREE.BufferGeometry;
  positionArray: Float32Array;
  /** Static scalar attributes (also used by the TSL material). */
  arcIndex: THREE.BufferAttribute;
  arcSide: THREE.BufferAttribute;
};

export function buildRibbonGeometry(segments: number): RibbonGeometry {
  const vertexCount = segments * 2;
  const geometry = new THREE.BufferGeometry();
  const positionArray = new Float32Array(vertexCount * 3);
  const uvArray = new Float32Array(vertexCount * 2);
  const arcIndexArr = new Float32Array(vertexCount);
  const arcSideArr = new Float32Array(vertexCount);
  const index = new Uint16Array((segments - 1) * 6);

  const inv = 1 / (segments - 1);
  for (let s = 0; s < segments; s++) {
    const u = s * inv;
    const li = s * 2;
    const ri = li + 1;
    uvArray[li * 2] = u;
    uvArray[li * 2 + 1] = 0;
    uvArray[ri * 2] = u;
    uvArray[ri * 2 + 1] = 1;
    arcIndexArr[li] = s;
    arcIndexArr[ri] = s;
    arcSideArr[li] = -1;
    arcSideArr[ri] = 1;
  }
  for (let s = 0; s < segments - 1; s++) {
    const l0 = s * 2;
    const r0 = l0 + 1;
    const l1 = l0 + 2;
    const r1 = l1 + 1;
    const o = s * 6;
    index[o] = l0;
    index[o + 1] = r0;
    index[o + 2] = l1;
    index[o + 3] = r0;
    index[o + 4] = r1;
    index[o + 5] = l1;
  }

  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(positionArray, 3)
  );
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvArray, 2));
  const arcIndex = new THREE.BufferAttribute(arcIndexArr, 1);
  const arcSide = new THREE.BufferAttribute(arcSideArr, 1);
  geometry.setAttribute('arcIndex', arcIndex);
  geometry.setAttribute('arcSide', arcSide);
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 8);

  return { geometry, positionArray, arcIndex, arcSide };
}

/** Contact billboard quad (4 verts, 2 triangles). */
export function buildContactGeometry(): {
  geometry: THREE.BufferGeometry;
  positionArray: Float32Array;
} {
  const geometry = new THREE.BufferGeometry();
  // corner coordinates (x,y in [-1,1]) packed into the position attribute
  const positionArray = new Float32Array([
    -1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0,
  ]);
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(positionArray, 3)
  );
  geometry.setAttribute(
    'uv',
    new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2)
  );
  geometry.setIndex(
    new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 1, 3, 2]), 1)
  );
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1);
  return { geometry, positionArray };
}
