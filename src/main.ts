import "./style.css";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import { OneEuro } from "./filter";
import { createFaceMesh } from "./mesh";

const BASE = import.meta.env.BASE_URL;
const $ = <T extends HTMLElement>(s: string) => document.querySelector(s) as T;
const video = $<HTMLVideoElement>("#cam");
const canvas = $<HTMLCanvasElement>("#gl");
const stage = $<HTMLElement>("#stage");
const splash = $<HTMLElement>("#splash");
const splashText = $<HTMLElement>("#splash-text");
const retry = $<HTMLButtonElement>("#retry");
const hint = $<HTMLElement>("#hint");
const DEBUG = location.search.includes("debug");

/** average distance between the centres of the two eyes, in metres (the model is in metres too) */
const EYE_SPAN = 0.062;

// ---------- camera ----------
async function openCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
}

// ---------- face tracking ----------
type Delegate = "GPU" | "CPU";
let files: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

function makeTracker(delegate: Delegate) {
  return FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: false,
  });
}

/** every browser on iPhone / iPad is Safari underneath, and its graphics-chip tracker can run yet never find a face */
const IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

async function loadTracker(): Promise<{ lm: FaceLandmarker; delegate: Delegate }> {
  files = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
  if (IOS) return { lm: await makeTracker("CPU"), delegate: "CPU" };
  try {
    return { lm: await makeTracker("GPU"), delegate: "GPU" };
  } catch {
    return { lm: await makeTracker("CPU"), delegate: "CPU" };
  }
}

// ---------- the glasses ----------
async function loadGlasses() {
  const gltf = await new GLTFLoader().loadAsync(`${BASE}models/sunglasses.glb`);
  const model = gltf.scene;
  model.updateMatrixWorld(true);
  // origin = the line between the eye corners: lens height, ~2 cm behind the front of the frame (where real glasses sit)
  const lensBox = new THREE.Box3().setFromObject(model.getObjectByName("LensesExterior") ?? model);
  const frameBox = new THREE.Box3().setFromObject(model.getObjectByName("Frames") ?? model);
  model.position.set(0, -(lensBox.min.y + lensBox.max.y) / 2, -(frameBox.max.z - 0.022));

  // prescription lenses: clear glass with a faint reflection and anti-reflective sheen.
  // The camera picture is drawn by the browser underneath, so the lenses are see-through, not refractive.
  const lens = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0,
    roughness: 0.06,
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    depthTest: false, // the lens is always in front of the face; never let the hidden face cover it
    specularIntensity: 0.8,
    iridescence: 0.3,
    iridescenceIOR: 1.8,
    iridescenceThicknessRange: [220, 420],
  });
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    // draw order is the whole trick: the front of the glasses first (nothing can hide it), then the hidden face,
    // head and ears (order 1), then the long arms (order 2), which disappear wherever they pass behind them
    m.renderOrder = m.name.startsWith("Earhook") ? 2 : 0;
    if (m.name.startsWith("LensesExterior")) m.material = lens;
    else if (m.name.startsWith("LensesInterior")) m.visible = false;
    else {
      // silicone nose pads etc: translucent instead of transmissive (nothing to refract on a transparent layer)
      const mat = m.material as THREE.MeshPhysicalMaterial;
      if (mat.transmission > 0) {
        mat.transmission = 0;
        mat.transparent = true;
        mat.opacity = 0.55;
      }
    }
  });

  // invisible back of the head and ears: they write depth only, so the temples vanish behind the real head and ears
  // (?occ shows them in red, to check how they sit on a real face)
  const ball = new THREE.SphereGeometry(1, 40, 28);
  const head = new THREE.Mesh(ball, HIDDEN);
  head.scale.set(0.062, 0.11, 0.095);
  head.position.set(0, -0.01, -0.1);
  // the ear covers the hook that bends down behind it, while the straight arm stays visible on top
  const earR = new THREE.Mesh(ball, HIDDEN);
  const earL = new THREE.Mesh(ball, HIDDEN);
  for (const [ear, side] of [
    [earR, -1],
    [earL, 1],
  ] as const) {
    ear.scale.set(0.017, 0.032, 0.03);
    ear.position.set(side * 0.074, -0.015, -0.1);
  }
  for (const o of [head, earR, earL]) o.renderOrder = 1;

  // the glasses themselves (frame + arms) scale together to fit the face; the hidden head and ears do not
  const glasses = new THREE.Group();
  glasses.add(model);
  const rig = new THREE.Group();
  rig.add(head, earR, earL, glasses);
  rig.visible = false;

  // each arm swings from its hinge, so it can open to the width of a real head and reach the ear
  rig.updateMatrixWorld(true);
  const arm = (name: string) => {
    const hook = model.getObjectByName(name);
    if (!hook) return new THREE.Group();
    const box = new THREE.Box3().setFromObject(hook);
    const pivot = new THREE.Group();
    pivot.position.set(box.min.x < 0 ? box.min.x : box.max.x, (box.min.y + box.max.y) / 2, box.max.z);
    glasses.add(pivot);
    pivot.updateMatrixWorld(true);
    pivot.attach(hook);
    return pivot;
  };
  const armR = arm("EarhookRight");
  const armL = arm("EarhookLeft");
  return { rig, head, earR, earL, lens, armR, armL, glasses };
}

const HIDDEN = location.search.includes("occ")
  ? new THREE.MeshBasicMaterial({ color: 0xff0044, wireframe: true, side: THREE.DoubleSide })
  : new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide });

/** the tracker gives the face as edges; its triangles are every three points that are all linked to each other */
function faceTriangles() {
  const near = new Map<number, Set<number>>();
  const link = (a: number, b: number) => (near.get(a) ?? near.set(a, new Set()).get(a)!).add(b);
  for (const { start, end } of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
    link(start, end);
    link(end, start);
  }
  const tris: number[] = [];
  for (const [a, na] of near)
    for (const b of na) {
      if (b <= a) continue;
      for (const c of near.get(b)!) if (c > b && na.has(c)) tris.push(a, b, c);
    }
  return tris;
}
type Glasses = Awaited<ReturnType<typeof loadGlasses>>;

// ---------- start ----------
async function main() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    return fail("המצלמה עובדת רק בכתובת מאובטחת (https).", false);
  }
  const tracker = loadTracker();
  const glasses = loadGlasses();
  try {
    await openCamera();
  } catch (e) {
    const name = (e as DOMException)?.name;
    if (name === "NotAllowedError" || name === "SecurityError") return fail("כדי למדוד משקפיים צריך גישה למצלמה.", true);
    if (name === "NotFoundError" || name === "OverconstrainedError") return fail("לא נמצאה מצלמה במכשיר.", false);
    return fail("לא הצלחנו לפתוח את המצלמה.", true);
  }
  // the live picture is visible right away; the glasses join when ready
  layoutStage();
  splashText.textContent = "מכינים את המשקפיים…";
  let t: Awaited<typeof tracker>;
  let parts: Glasses;
  try {
    [t, parts] = await Promise.all([tracker, glasses]);
  } catch {
    return fail("הטעינה נכשלה. נסו לרענן.", false);
  }
  run(t.lm, t.delegate, parts);
  splash.classList.add("gone");
}

function fail(msg: string, canRetry: boolean) {
  splash.classList.add("error");
  splashText.textContent = msg;
  retry.hidden = !canRetry;
}
retry.addEventListener("click", () => location.reload());

/** size the mirrored stage to cover the screen with the camera's aspect */
let onLayout: ((w: number, h: number) => void) | null = null;
function layoutStage() {
  const va = (video.videoWidth || 16) / (video.videoHeight || 9);
  let w = innerWidth;
  let h = w / va;
  if (h < innerHeight) {
    h = innerHeight;
    w = h * va;
  }
  stage.style.width = `${w}px`;
  stage.style.height = `${h}px`;
  onLayout?.(w, h);
}
addEventListener("resize", layoutStage);
video.addEventListener("resize", layoutStage); // the camera can change orientation

// ---------- render loop ----------
/** phone selfie cameras and webcams see about 75° across the diagonal; the vertical angle follows the picture's shape */
const DIAG_FOV = 75;
const dotsBtn = $<HTMLButtonElement>("#dots");

function run(first: FaceLandmarker, firstDelegate: Delegate, { rig, head, earR, earL, lens, armR, armL, glasses }: Glasses) {
  let landmarker = first;
  let delegate = firstDelegate;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.autoClear = false;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  // lenses get their own tilted reflection, so a light never sits right over the eyes
  lens.envMap = scene.environment;
  lens.envMapIntensity = 1;
  lens.envMapRotation.set(0.55, 0.45, 0);

  // the real face as an invisible surface: anything behind it (the far arm when you look sideways) disappears
  const N = 478;
  const facePos = new Float32Array(N * 3);
  const faceAttr = new THREE.BufferAttribute(facePos, 3).setUsage(THREE.DynamicDrawUsage);
  const faceGeo = new THREE.BufferGeometry();
  faceGeo.setAttribute("position", faceAttr);
  faceGeo.setIndex(faceTriangles());
  faceGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e3);
  const faceOcc = new THREE.Mesh(faceGeo, HIDDEN);
  faceOcc.renderOrder = 1;
  faceOcc.visible = false;
  scene.add(faceOcc, rig);

  // the face scan is its own flat layer in screen pixels, drawn first, so the glasses always sit on top of it
  const faceMesh = createFaceMesh("#e8ff5a");
  const meshScene = new THREE.Scene();
  meshScene.add(faceMesh.group);
  let showDots = true;
  try {
    showDots = localStorage.getItem("ar-dots") !== "off";
  } catch {
    /* private mode */
  }
  const syncDots = () => {
    dotsBtn.setAttribute("aria-pressed", String(showDots));
    dotsBtn.textContent = showDots ? "נקודות: פועל" : "נקודות: כבוי";
  };
  dotsBtn.hidden = false;
  syncDots();
  dotsBtn.addEventListener("click", () => {
    showDots = !showDots;
    syncDots();
    try {
      localStorage.setItem("ar-dots", showDots ? "on" : "off");
    } catch {
      /* private mode */
    }
  });

  // a real perspective camera, like the phone's lens: the arms run back toward the ears instead of hiding behind the hinge
  let W = video.videoWidth;
  let H = video.videoHeight;
  let f = 1;
  const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 10);
  const flat = new THREE.OrthographicCamera(-W / 2, W / 2, H / 2, -H / 2, -1000, 1000);
  const fitCamera = () => {
    W = video.videoWidth || W;
    H = video.videoHeight || H;
    f = Math.hypot(W, H) / 2 / Math.tan(THREE.MathUtils.degToRad(DIAG_FOV / 2));
    camera.aspect = W / H;
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(H / 2 / f));
    camera.updateProjectionMatrix();
    flat.left = -W / 2;
    flat.right = W / 2;
    flat.top = H / 2;
    flat.bottom = -H / 2;
    flat.updateProjectionMatrix();
  };
  onLayout = (w, h) => {
    fitCamera();
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
  };
  layoutStage();

  // smoothing in screen pixels: steady when still, quick when moving
  const fx = new OneEuro(1.4, 0.006);
  const fy = new OneEuro(1.4, 0.006);
  const fs = new OneEuro(1.0, 0.002);
  const target = new THREE.Quaternion();
  const m4 = new THREE.Matrix4();
  const tmpP = new THREE.Vector3();
  const tmpS = new THREE.Vector3();
  const eyeR = new THREE.Vector3();
  const eyeL = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const v = new THREE.Vector3();
  const across = new THREE.Vector3();
  const fwd = new THREE.Vector3();
  const core = new THREE.Vector3();
  type L = { x: number; y: number; z: number };
  /** image pixels, with depth in the same pixel scale (the tracker's z is relative to the head) */
  const px = (l: L, out: THREE.Vector3) => out.set(l.x * W, l.y * H, l.z * W);

  let depth = 0.4; // metres from the lens to the eyes
  let midZ = 0; // the eye line's z, in the tracker's pixel scale
  /** a landmark to camera space in metres, at its own depth */
  const toCam = (l: L, out: THREE.Vector3, push = 0) => {
    const z = depth + ((l.z * W - midZ) * depth) / f + push;
    return out.set(((l.x * W - W / 2) * z) / f, ((H / 2 - l.y * H) * z) / f, -z);
  };

  const ear = new THREE.Vector3();
  let halfFace = 0.07;
  let fit = 1;
  let seenAt = 0;
  let tracking = false;

  const lose = () => {
    tracking = false;
    rig.visible = false;
    faceOcc.visible = false;
    faceMesh.hide();
    hint.hidden = false;
  };

  const apply = (res: FaceLandmarkerResult, t: number) => {
    const lm = res.faceLandmarks?.[0];
    const mat = res.facialTransformationMatrixes?.[0];
    if (!lm || !mat) {
      if (tracking && t - seenAt > 300) lose();
      return;
    }
    if (showDots) faceMesh.update(lm, W, H);
    else faceMesh.hide();

    // eye centres from the corners (stable, unlike the irises which follow the gaze)
    eyeR.copy(px(lm[33], a)).add(px(lm[133], b)).multiplyScalar(0.5);
    eyeL.copy(px(lm[362], a)).add(px(lm[263], b)).multiplyScalar(0.5);
    const gap = Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y);
    const mid = eyeR.add(eyeL).multiplyScalar(0.5);
    m4.fromArray(mat.data);
    m4.decompose(tmpP, target, tmpS);

    const sec = t / 1000;
    if (!tracking) {
      fx.reset();
      fy.reset();
      fs.reset();
      rig.quaternion.copy(target);
    } else {
      rig.quaternion.slerp(target, Math.min(1, 0.35 + rig.quaternion.angleTo(target) * 5));
    }
    const mx = fx.next(mid.x, sec);
    const my = fy.next(mid.y, sec);
    // the real distance between the eyes tells how far the face is from the lens
    // distance from the lens: the eyes' on-screen gap, corrected for how much the head is turned (from the head pose,
    // not the tracker's depth guess, which flattens when you look sideways)
    across.set(EYE_SPAN, 0, 0).applyQuaternion(target);
    const seen = Math.max(0.02, Math.hypot(across.x, across.y));
    depth = (f * EYE_SPAN) / fs.next((gap * EYE_SPAN) / seen, sec);
    midZ = mid.z;
    rig.position.set(((mx - W / 2) * depth) / f, ((H / 2 - my) * depth) / f, -depth);
    rig.updateMatrixWorld();
    rig.localToWorld(core.set(0, -0.01, -0.085));

    // the face surface, drawn 13 mm smaller toward the middle of the head: the frame and arms that rest on the skin
    // stay visible, while anything truly behind the face (the far arm) is hidden
    for (let i = 0; i < N && i < lm.length; i++) {
      toCam(lm[i], v);
      v.sub(core);
      v.addScaledVector(v, -0.013 / Math.max(0.03, v.length())).add(core);
      facePos[i * 3] = v.x;
      facePos[i * 3 + 1] = v.y;
      facePos[i * 3 + 2] = v.z;
    }
    faceAttr.needsUpdate = true;

    // back of the head and ears, fitted to this face: width from the cheek edges by the ears (234 / 454)
    const eR = rig.worldToLocal(toCam(lm[234], ear)).clone();
    const eL = rig.worldToLocal(toCam(lm[454], ear)).clone();
    const w = THREE.MathUtils.clamp((Math.abs(eR.x) + Math.abs(eL.x)) / 2, 0.055, 0.085);
    // head width is measured only while the face looks at the camera; when it turns, the far side is a guess, so we hold it
    fwd.set(0, 0, 1).applyQuaternion(target);
    const frontal = Math.abs(Math.atan2(fwd.x, fwd.z)) < 0.26 && Math.abs(fwd.y) < 0.35;
    const k = frontal ? (tracking ? 0.15 : 1) : 0;
    halfFace += (w - halfFace) * k;
    head.scale.x = halfFace * 0.9;
    // open the arms just enough to run along the side of this head to the ear (real hinges flex the same way)
    // size the frame so its hinges land on the temples (face edge at eye level: 127 / 356)
    const tR = rig.worldToLocal(toCam(lm[127], ear)).x;
    const tL = rig.worldToLocal(toCam(lm[356], ear)).x;
    const fitTo = THREE.MathUtils.clamp((Math.abs(tR) + Math.abs(tL)) / 2 / 0.072, 0.92, 1.2);
    fit += (fitTo - fit) * k;
    glasses.scale.setScalar(fit);
    const open = THREE.MathUtils.clamp(Math.atan2(halfFace - 0.066 * fit, 0.1), -0.04, 0.22);
    armR.rotation.y = open;
    armL.rotation.y = -open;
    // the ear starts just behind that cheek point; clamp to a human range in case the depth guess is off
    for (const [o, e, s] of [
      [earR, eR, -1],
      [earL, eL, 1],
    ] as const) {
      const ex = s * (halfFace + 0.006);
      const ey = THREE.MathUtils.clamp(e.y - 0.008, -0.03, 0);
      const ez = THREE.MathUtils.clamp(e.z - 0.028, -0.12, -0.085);
      o.position.x += (ex - o.position.x) * k;
      o.position.y += (ey - o.position.y) * k;
      o.position.z += (ez - o.position.z) * k;
    }

    seenAt = t;
    if (!tracking) {
      tracking = true;
      rig.visible = true;
      faceOcc.visible = true;
      hint.hidden = true;
    }
  };

  // watchdog: if the graphics-chip tracker is slow, throws, or never finds a face, move to the processor once
  let avgMs = 0;
  let samples = 0;
  let switching = false;
  let errors = 0;
  let lastErr = "";
  let everSeen = false;
  const startedAt = performance.now();
  const toCPU = () => {
    if (delegate !== "GPU" || switching) return;
    switching = true;
    makeTracker("CPU")
      .then((cpu) => {
        const old = landmarker;
        landmarker = cpu;
        delegate = "CPU";
        samples = 0;
        try {
          old.close();
        } catch {
          /* already gone */
        }
      })
      .catch((e) => (lastErr = String(e)));
  };
  const watch = (ms: number, t: number) => {
    avgMs = samples ? avgMs * 0.9 + ms * 0.1 : ms;
    samples++;
    if (samples > 20 && avgMs > 60) toCPU();
    if (!everSeen && t - startedAt > 2500) toCPU();
  };
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    lastErr = "webgl context lost";
  });

  const stats = DEBUG ? Object.assign(document.createElement("pre"), { id: "stats" }) : null;
  if (stats) document.body.append(stats);
  let frames = 0;
  let fpsAt = performance.now();

  // one detection + one draw per new camera frame, paced by the screen (never by the video element)
  let lastVideoTime = -1;
  const loop = () => {
    requestAnimationFrame(loop);
    if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    const t = performance.now();
    let res: FaceLandmarkerResult;
    try {
      res = landmarker.detectForVideo(video, t);
    } catch (e) {
      lastErr = String(e).slice(0, 80);
      if (++errors > 3) toCPU();
      return;
    }
    apply(res, t);
    if (tracking) everSeen = true;
    watch(performance.now() - t, t);
    renderer.clear();
    renderer.render(meshScene, flat);
    renderer.render(scene, camera);
    if (stats) {
      frames++;
      if (t - fpsAt > 1000) {
        stats.textContent = `${frames} fps · detect ${avgMs.toFixed(0)}ms · ${delegate} · ${W}x${H} · face ${tracking ? "yes" : "no"} · glasses ${rig.visible ? "on" : "off"} · ${(depth * 100).toFixed(0)}cm · fit ${fit.toFixed(2)} · ${IOS ? "iOS" : ""}${lastErr ? " · " + lastErr : ""}`;
        frames = 0;
        fpsAt = t;
      }
    }
  };
  hint.hidden = false;
  requestAnimationFrame(loop);
}

main();
