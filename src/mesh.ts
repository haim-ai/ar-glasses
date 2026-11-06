import * as THREE from "three";
import { FaceLandmarker } from "@mediapipe/tasks-vision";

type Conn = { start: number; end: number };
const N = 478; // 468 face points + 10 iris points

const toIndex = (list: Conn[]) => list.flatMap((c) => [c.start, c.end]);

/**
 * A minimal face scan: tiny round dots on every landmark, a faint triangle web,
 * and brighter contours (face outline, eyes, brows, lips, irises). All three share one position buffer.
 */
export function createFaceMesh(accent: string) {
  const positions = new Float32Array(N * 3);
  const pos = new THREE.BufferAttribute(positions, 3);
  pos.setUsage(THREE.DynamicDrawUsage);

  const dotsGeo = new THREE.BufferGeometry();
  dotsGeo.setAttribute("position", pos);
  const dots = new THREE.Points(
    dotsGeo,
    new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(accent) }, uSize: { value: 3.6 * Math.min(devicePixelRatio || 1, 2) } },
      vertexShader: /* glsl */ `
        uniform float uSize;
        void main() {
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uSize;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          gl_FragColor = vec4(uColor, smoothstep(0.5, 0.25, d));
        }`,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    }),
  );

  const webGeo = new THREE.BufferGeometry();
  webGeo.setAttribute("position", pos);
  webGeo.setIndex(toIndex(FaceLandmarker.FACE_LANDMARKS_TESSELATION));
  const web = new THREE.LineSegments(webGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.26, depthTest: false, depthWrite: false }));

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", pos);
  lineGeo.setIndex(toIndex([...FaceLandmarker.FACE_LANDMARKS_CONTOURS, ...FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS]));
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.95, depthTest: false, depthWrite: false }));

  const group = new THREE.Group();
  group.add(web, lines, dots);
  // drawn first and without depth, so it lies on the face and the glasses always sit on top of it
  group.traverse((o) => (o.renderOrder = 0));
  group.visible = false;
  for (const g of [dotsGeo, webGeo, lineGeo]) g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

  let fresh = true;
  return {
    group,
    /** landmarks → screen units; a light follow keeps it calm without lag */
    update(lm: { x: number; y: number; z: number }[], W: number, H: number) {
      const k = fresh ? 1 : 0.65;
      const n = Math.min(N, lm.length);
      for (let i = 0; i < n; i++) {
        const j = i * 3;
        positions[j] += ((lm[i].x - 0.5) * W - positions[j]) * k;
        positions[j + 1] += ((0.5 - lm[i].y) * H - positions[j + 1]) * k;
        positions[j + 2] = 0;
      }
      pos.needsUpdate = true;
      fresh = false;
      group.visible = true;
    },
    hide() {
      group.visible = false;
      fresh = true;
    },
  };
}
