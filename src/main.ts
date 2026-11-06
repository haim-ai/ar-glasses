import "./style.css";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import { OneEuro } from "./filter";

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

async function loadTracker(): Promise<{ lm: FaceLandmarker; delegate: Delegate }> {
  files = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
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
  // origin = the line between the eyes: lens height, a little behind the front of the frame
  const lensBox = new THREE.Box3().setFromObject(model.getObjectByName("LensesExterior") ?? model);
  const frameBox = new THREE.Box3().setFromObject(model.getObjectByName("Frames") ?? model);
  model.position.set(0, -(lensBox.min.y + lensBox.max.y) / 2, -(frameBox.max.z - 0.014));

  // prescription lenses: clear glass with a faint reflection and anti-reflective sheen.
  // The camera picture is drawn by the browser underneath, so the lenses are see-through, not refractive.
  const lens = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0,
    roughness: 0.06,
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    specularIntensity: 0.8,
    iridescence: 0.3,
    iridescenceIOR: 1.8,
    iridescenceThicknessRange: [220, 420],
  });
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    m.renderOrder = 2;
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

  // invisible head and ears: they write depth only, so the temples vanish behind the real head and ears
  const hidden = DEBUG ? new THREE.MeshBasicMaterial({ color: 0xff0044, wireframe: true }) : new THREE.MeshBasicMaterial({ colorWrite: false });
  const ball = new THREE.SphereGeometry(1, 40, 28);
  const head = new THREE.Mesh(ball, hidden);
  head.scale.set(0.066, 0.1, 0.1);
  head.position.set(0, -0.018, -0.1);
  // the ear covers the hook that bends down behind it, while the straight arm stays visible on top
  const earR = new THREE.Mesh(ball, hidden);
  const earL = new THREE.Mesh(ball, hidden);
  for (const [ear, side] of [
    [earR, -1],
    [earL, 1],
  ] as const) {
    ear.scale.set(0.016, 0.032, 0.032);
    ear.position.set(side * 0.074, -0.018, -0.112);
  }
  for (const o of [head, earR, earL]) o.renderOrder = 1;

  const rig = new THREE.Group();
  rig.add(head, earR, earL, model);
  rig.visible = false;
  return { rig, head, earR, earL, lens };
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
function run(first: FaceLandmarker, firstDelegate: Delegate, { rig, head, earR, earL, lens }: Glasses) {
  let landmarker = first;
  let delegate = firstDelegate;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  // lenses get their own tilted reflection, so a light never sits right over the eyes
  lens.envMap = scene.environment;
  lens.envMapIntensity = 1;
  lens.envMapRotation.set(0.55, 0.45, 0);
  scene.add(rig);

  // one unit = one camera pixel, origin in the middle of the picture
  let W = video.videoWidth;
  let H = video.videoHeight;
  const camera = new THREE.OrthographicCamera(-W / 2, W / 2, H / 2, -H / 2, 1, 10000);
  camera.position.z = 5000;
  const fitCamera = () => {
    W = video.videoWidth || W;
    H = video.videoHeight || H;
    camera.left = -W / 2;
    camera.right = W / 2;
    camera.top = H / 2;
    camera.bottom = -H / 2;
    camera.updateProjectionMatrix();
  };
  onLayout = (w, h) => {
    fitCamera();
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
  };
  layoutStage();

  // smoothing: steady when still, quick when moving
  const fx = new OneEuro(1.4, 0.006);
  const fy = new OneEuro(1.4, 0.006);
  const fz = new OneEuro(1.0, 0.004);
  const fs = new OneEuro(1.0, 0.002);
  const target = new THREE.Quaternion();
  const m4 = new THREE.Matrix4();
  const tmpP = new THREE.Vector3();
  const tmpS = new THREE.Vector3();
  const eyeR = new THREE.Vector3();
  const eyeL = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const side = new THREE.Vector3();
  const pt = (lm: { x: number; y: number; z: number }, out: THREE.Vector3) => out.set((lm.x - 0.5) * W, (0.5 - lm.y) * H, -lm.z * W);

  let halfFace = 0.07;
  let seenAt = 0;
  let tracking = false;

  const apply = (res: FaceLandmarkerResult, t: number) => {
    const lm = res.faceLandmarks?.[0];
    const mat = res.facialTransformationMatrixes?.[0];
    if (!lm || !mat) {
      if (tracking && t - seenAt > 300) {
        tracking = false;
        rig.visible = false;
        hint.hidden = false;
      }
      return;
    }
    // eye centres from the corners (stable, unlike the irises which follow the gaze)
    eyeR.copy(pt(lm[33], a)).add(pt(lm[133], b)).multiplyScalar(0.5);
    eyeL.copy(pt(lm[362], a)).add(pt(lm[263], b)).multiplyScalar(0.5);
    const span = eyeR.distanceTo(eyeL);
    const mid = eyeR.add(eyeL).multiplyScalar(0.5);
    m4.fromArray(mat.data);
    m4.decompose(tmpP, target, tmpS);

    const sec = t / 1000;
    if (!tracking) {
      fx.reset();
      fy.reset();
      fz.reset();
      fs.reset();
      rig.quaternion.copy(target);
    } else {
      const ang = rig.quaternion.angleTo(target);
      rig.quaternion.slerp(target, Math.min(1, 0.35 + ang * 5));
    }
    rig.position.set(fx.next(mid.x, sec), fy.next(mid.y, sec), fz.next(mid.z, sec));
    rig.scale.setScalar(fs.next(span / EYE_SPAN, sec));

    // fit the hidden head and ears to this face: its width, measured at the cheek edges by the ears
    rig.updateMatrixWorld();
    const wR = Math.abs(rig.worldToLocal(pt(lm[234], side)).x);
    const wL = Math.abs(rig.worldToLocal(pt(lm[454], side)).x);
    const w = THREE.MathUtils.clamp((wR + wL) / 2, 0.055, 0.085);
    halfFace += (w - halfFace) * (tracking ? 0.15 : 1);
    head.scale.x = halfFace * 0.94;
    earR.position.x = -(halfFace + 0.004);
    earL.position.x = halfFace + 0.004;

    seenAt = t;
    if (!tracking) {
      tracking = true;
      rig.visible = true;
      hint.hidden = true;
    }
  };

  // watchdog: if the graphics-chip tracker is slow on this phone, move to the processor once
  let avgMs = 0;
  let samples = 0;
  let switching = false;
  const watch = (ms: number) => {
    avgMs = samples ? avgMs * 0.9 + ms * 0.1 : ms;
    samples++;
    if (delegate === "GPU" && !switching && samples > 20 && avgMs > 60) {
      switching = true;
      makeTracker("CPU")
        .then((cpu) => {
          const old = landmarker;
          landmarker = cpu;
          delegate = "CPU";
          samples = 0;
          old.close();
        })
        .catch(() => {});
    }
  };

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
    apply(landmarker.detectForVideo(video, t), t);
    const ms = performance.now() - t;
    watch(ms);
    renderer.render(scene, camera);
    if (stats) {
      frames++;
      if (t - fpsAt > 1000) {
        stats.textContent = `${frames} fps · detect ${avgMs.toFixed(0)}ms · ${delegate} · ${W}x${H}`;
        frames = 0;
        fpsAt = t;
      }
    }
  };
  hint.hidden = false;
  requestAnimationFrame(loop);
}

main();
