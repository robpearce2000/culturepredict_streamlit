'use strict';
/* =========================================================
   OUTPACE
   A 3D race quiz. The whole class is one runner; 2 to 6 teams answer
   every question on whiteboards.
   Deal Round: the class votes for a deal and races the Hunter
   home; at least half the teams right moves the runner, otherwise the
   Hunter gains.
   Final Sprint: 60 seconds; every correct team is a step towards a target
   set by the pot.
   ========================================================= */
(function () {
let game = null;

function createOutpace(root) {
const $ = id => document.getElementById('op-' + id);
const escapeHtml = CGB.escapeHtml;
const bank = CGB.bank;
const picker = bank.createPicker();
const GAME_NAME = 'Outpace';
const reduced = () => CGB.settings.reduced();

/* ============ SOUND (synthesised, original) ============ */
const SFX = {
  correct: CGB.sfx.correct,
  wrong: CGB.sfx.wrong,
  pick() { CGB.sfx.tone(520, 0.08, 'triangle', 0.07); CGB.sfx.tone(780, 0.1, 'triangle', 0.07, 0.06); },
  caught() { [440, 349, 262, 196].forEach((f, i) => CGB.sfx.tone(f, 0.22, 'sawtooth', 0.045, i * 0.12)); },
  escaped() { [392, 523, 659, 784, 1047].forEach((f, i) => CGB.sfx.tone(f, 0.2, 'triangle', 0.08, i * 0.08)); },
  timeUp() { CGB.sfx.tone(300, 0.5, 'square', 0.05, 0, 0.6); }
};

/* ============ GAME STATE ============ */
const DEAL_ROUNDS = 1;   // one Deal Round then the Final Sprint: under 10 minutes with a class
const state = {
  groups: CGB.store.getJSON('op.groups', 4),
  className: 'Our class', dealRound: 0, dealRounds: DEAL_ROUNDS,
  players: [{ name: 'Team 1' }, { name: 'Team 2' }],
  phase: 'home',               // home | deal | dealEnd | sprint | finish | summary
  pot: 0,
  dealOutcomes: [],
  wrongAnswers: [[], []],
  tierConfig: {
    low: { reward: 300, gap: 5 },
    mid: { reward: 600, gap: 3 },
    high: { reward: 1000, gap: 1 }
  },
  dealReward: 0,
  currentQuestion: null,

  sprintNetScore: 0,
  sprintTarget: 0,
  sprintTimeLeft: 60,
  sprintFrozen: false,
  sprintTimerHandle: null
};
const timers = [];
const later = (fn, ms) => { const t = setTimeout(fn, ms); timers.push(t); return t; };
function clearTimers() { timers.forEach(clearTimeout); timers.length = 0; clearInterval(state.sprintTimerHandle); }

/* ============ THREE.JS SCENE ============ */
const COL = { bg: 0x0B1026, runner: 0xFFC93C, runnerLight: 0xFFF1C2, runnerNeutron: 0xFFE08A, hunter: 0xE879F9, hunterLight: 0xF9D2FF, hunterNeutron: 0xC084FC, tether: 0x8C93C8, spark: 0xFFF4D6 };
const wrap = $('stage');
const scene = new THREE.Scene();
scene.background = new THREE.Color(COL.bg);
scene.fog = new THREE.Fog(COL.bg, 10, 26);

const camera = new THREE.PerspectiveCamera(52, 16 / 9, 0.1, 100);
camera.position.set(0, 2.4, 6.5);
camera.lookAt(0, 0.6, 0);

const renderer = CGB.createRenderer();
if (!renderer) { $('nogl').hidden = false; $('nogl').innerHTML = CGB.noWebGLMessage; return null; }
/* Performance: Outpace caps its resolution, draws no shadows (the racers float above the
   track, so shadows add little) and steps its quality down by itself if a laptop cannot keep
   up (see watchPerformance) */
const perf = { level: 0, ema: 16, slowFor: 0, last: 0 };
const opPixelRatio = () => perf.level >= 2 ? 1 : Math.min(CGB.settings.pixelRatio(), 1.25);
renderer.setPixelRatio(opPixelRatio());
renderer.shadowMap.enabled = false;
wrap.insertBefore(renderer.domElement, wrap.firstChild);

const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));
const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(512, 512), 0.32, 0.4, 0.4);
composer.addPass(bloomPass);
const bloomAllowed = () => CGB.settings.get('quality') !== 'low' && perf.level < 1;
let useBloom = bloomAllowed();

scene.add(new THREE.AmbientLight(0x3A3F7E, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(4, 8, 5);
scene.add(keyLight);
const runnerGlow = new THREE.PointLight(COL.runner, 5, 8);
runnerGlow.position.set(-2.5, 1.2, 0);
scene.add(runnerGlow);
const hunterGlow = new THREE.PointLight(COL.hunter, 6, 8);
hunterGlow.position.set(2.5, 1.2, 0);
scene.add(hunterGlow);

function makeHexGridTexture() {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0D1230';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = 'rgba(150,160,255,0.16)';
  ctx.lineWidth = 1.4;
  const r = 26;
  const hexH = r * Math.sqrt(3);
  for (let row = -1; row < size / hexH + 2; row++) {
    for (let col = -1; col < size / (r * 1.5) + 2; col++) {
      const x = col * r * 1.5;
      const y = row * hexH + (col % 2 === 0 ? 0 : hexH / 2);
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 3 * i;
        const px = x + r * Math.cos(angle);
        const py = y + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  return tex;
}

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(30, 30),
  new THREE.MeshStandardMaterial({ color: 0x0D1230, roughness: 0.95, map: makeHexGridTexture() })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.3;
scene.add(floor);

const AMBIENT_PARTICLE_COUNT = 140;
const ambientParticleGeo = new THREE.BufferGeometry();
const ambientParticlePositions = new Float32Array(AMBIENT_PARTICLE_COUNT * 3);
for (let i = 0; i < AMBIENT_PARTICLE_COUNT; i++) {
  ambientParticlePositions[i * 3] = (Math.random() - 0.5) * 18;
  ambientParticlePositions[i * 3 + 1] = Math.random() * 5 + 0.2;
  ambientParticlePositions[i * 3 + 2] = (Math.random() - 0.5) * 18;
}
ambientParticleGeo.setAttribute('position', new THREE.BufferAttribute(ambientParticlePositions, 3));
const ambientParticles = new THREE.Points(ambientParticleGeo, new THREE.PointsMaterial({ color: 0xD8DCFF, size: 0.045, transparent: true, opacity: 0.55 }));
scene.add(ambientParticles);

function makeNucleus(protonColor, neutronColor, count, spread) {
  const group = new THREE.Group();
  const particlesArr = [];
  const protonMat = new THREE.MeshStandardMaterial({ color: protonColor, emissive: protonColor, emissiveIntensity: 0.6, roughness: 0.3, metalness: 0.2 });
  const neutronMat = new THREE.MeshStandardMaterial({ color: neutronColor, emissive: neutronColor, emissiveIntensity: 0.3, roughness: 0.5, metalness: 0.1 });
  const partGeo = new THREE.SphereGeometry(0.1, 14, 14);
  for (let i = 0; i < count; i++) {
    const mesh = new THREE.Mesh(partGeo, i % 2 === 0 ? protonMat : neutronMat);
    const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
    mesh.position.copy(dir.multiplyScalar(Math.random() * spread));
    group.add(mesh);
    particlesArr.push({ mesh, base: mesh.position.clone(), phase: Math.random() * Math.PI * 2 });
  }
  const shell = new THREE.Mesh(new THREE.SphereGeometry(spread + 0.14, 20, 20), new THREE.MeshStandardMaterial({ color: protonColor, transparent: true, opacity: 0.12, roughness: 1 }));
  group.add(shell);
  return { group, particles: particlesArr };
}

function makeElectronOrbits(config) {
  const orbits = [];
  const group = new THREE.Group();
  config.orbits.forEach((o) => {
    const orbitGroup = new THREE.Group();
    orbitGroup.rotation.x = o.tiltX;
    orbitGroup.rotation.z = o.tiltZ;
    orbitGroup.add(new THREE.Mesh(new THREE.RingGeometry(o.radius - 0.015, o.radius, 48), new THREE.MeshBasicMaterial({ color: config.color, transparent: true, opacity: 0.3, side: THREE.DoubleSide })));
    const electronMat = new THREE.MeshStandardMaterial({ color: config.electronColor, emissive: config.color, emissiveIntensity: 1 });
    const trailMat = new THREE.MeshBasicMaterial({ color: config.electronColor, transparent: true, opacity: 0.18 });
    const trailSteps = 5, trailSpacing = 0.12;
    for (let e = 0; e < o.electrons; e++) {
      const baseAngle = (Math.PI * 2 / o.electrons) * e;
      const electron = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 12), electronMat);
      electron.position.set(Math.cos(baseAngle) * o.radius, 0, Math.sin(baseAngle) * o.radius);
      orbitGroup.add(electron);
      const dir = o.speed >= 0 ? 1 : -1;
      for (let s = 1; s <= trailSteps; s++) {
        const trailAngle = baseAngle + dir * s * trailSpacing;
        const trailDot = new THREE.Mesh(new THREE.SphereGeometry(0.045 * (1 - s / (trailSteps + 1.5)), 8, 8), trailMat.clone());
        trailDot.material.opacity = 0.22 * (1 - s / (trailSteps + 1));
        trailDot.position.set(Math.cos(trailAngle) * o.radius, 0, Math.sin(trailAngle) * o.radius);
        orbitGroup.add(trailDot);
      }
    }
    group.add(orbitGroup);
    orbits.push({ group: orbitGroup, speed: o.speed });
  });
  return { group, orbits };
}

const runnerGroup = new THREE.Group();
scene.add(runnerGroup);
const hunterGroup = new THREE.Group();
scene.add(hunterGroup);

/* ============ SUBJECT LOOKS ============
   The look follows the subject chosen on the main screen. The runner is always gold and the
   Hunter always magenta, so the race reads the same in every subject; the characters, the
   symbols orbiting them, the track tiles and the floor change with the subject. Biology,
   Chemistry, Physics and Combined Science share the science look (atoms). */
function glyphTexture(text, color) {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  g.font = (text.length > 2 ? '64px' : '84px') + ' "Lilita One", "Arial Black", sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineWidth = 10; g.strokeStyle = 'rgba(10,12,40,0.9)'; g.strokeText(text, 64, 68);
  g.fillStyle = color; g.fillText(text, 64, 68);
  return new THREE.CanvasTexture(c);
}
function makeGlyphOrbit(glyphs, color, radius, speed, tiltX, tiltZ, size) {
  const group = new THREE.Group(); group.rotation.x = tiltX; group.rotation.z = tiltZ;
  group.add(new THREE.Mesh(new THREE.RingGeometry(radius - 0.012, radius, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.25, side: THREE.DoubleSide })));
  glyphs.forEach((gl, i) => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyphTexture(gl, color), transparent: true, depthWrite: false }));
    const a = Math.PI * 2 * i / glyphs.length;
    sp.position.set(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    sp.scale.setScalar(size || 0.26);
    group.add(sp);
  });
  return { group, speed };
}
const glowMat = (hex, k) => new THREE.MeshStandardMaterial({ color: hex, emissive: hex, emissiveIntensity: k == null ? 0.55 : k, roughness: 0.35, metalness: 0.25 });
const runnerHex = '#' + new THREE.Color(COL.runnerLight).getHexString(), hunterHex = '#' + new THREE.Color(COL.hunterLight).getHexString();

/* Each builder returns { group, update(t), boost(k) } */
const CHARACTERS = {
  science(isHunter) {
    const nuc = isHunter ? makeNucleus(COL.hunter, COL.hunterNeutron, 7, 0.16) : makeNucleus(COL.runner, COL.runnerNeutron, 5, 0.13);
    const el = isHunter
      ? makeElectronOrbits({ color: COL.hunter, electronColor: COL.hunterLight, orbits: [{ radius: 0.55, tiltX: Math.PI / 3, tiltZ: 0.3, electrons: 2, speed: 2.4 }, { radius: 0.7, tiltX: Math.PI / 1.8, tiltZ: -0.5, electrons: 1, speed: -2.0 }] })
      : makeElectronOrbits({ color: COL.runner, electronColor: COL.runnerLight, orbits: [{ radius: 0.5, tiltX: Math.PI / 2.4, tiltZ: 0, electrons: 1, speed: 1.4 }, { radius: 0.62, tiltX: Math.PI / 6, tiltZ: 0.9, electrons: 2, speed: -1.0 }] });
    const group = new THREE.Group(); group.add(nuc.group, el.group);
    const f = isHunter ? [6, 6.5, 5.5, 0.03, 0.9] : [2, 2.3, 1.8, 0.012, 0.4];
    return {
      group,
      update(t) {
        nuc.group.rotation.y = t * f[4];
        nuc.particles.forEach(p => p.mesh.position.set(p.base.x + Math.sin(t * f[0] + p.phase) * f[3], p.base.y + Math.cos(t * f[1] + p.phase) * f[3], p.base.z + Math.sin(t * f[2] + p.phase) * f[3]));
        el.orbits.forEach(o => o.group.rotation.y = t * o.speed);
      },
      boost(k) { el.orbits.forEach(o => { o.group.rotation.y += (o.speed >= 0 ? 1 : -1) * 0.05 * k; }); }
    };
  },
  maths(isHunter) {
    const group = new THREE.Group();
    const col = isHunter ? COL.hunter : COL.runner;
    const core = new THREE.Group();
    if (isHunter) {
      // a spiky "infinity engine": two crossed octahedra
      const a = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), glowMat(col, 0.6));
      const b = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), glowMat(COL.hunterNeutron, 0.4)); b.rotation.set(Math.PI / 4, Math.PI / 4, 0);
      core.add(a, b);
    } else {
      const d = new THREE.Mesh(new THREE.DodecahedronGeometry(0.27), glowMat(col, 0.5));
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(d.geometry), new THREE.LineBasicMaterial({ color: COL.runnerLight }));
      core.add(d, e);
    }
    const orbit = isHunter ? makeGlyphOrbit(['∞', '×', '÷'], hunterHex, 0.62, 2.2, Math.PI / 3, 0.3)
                           : makeGlyphOrbit(['π', '√', '∑', 'x²', '%'], runnerHex, 0.6, 1.2, Math.PI / 2.5, 0);
    group.add(core, orbit.group);
    return {
      group,
      update(t) { core.rotation.set(t * (isHunter ? 1.6 : 0.5), t * (isHunter ? 1.1 : 0.7), 0); orbit.group.rotation.y = t * orbit.speed; },
      boost(k) { orbit.group.rotation.y += 0.08 * k; }
    };
  },
  english(isHunter) {
    const group = new THREE.Group();
    const core = new THREE.Group();
    if (isHunter) {
      // an ink blot that wobbles
      const m = glowMat(COL.hunter, 0.45);
      [[0, 0, 0, 0.24], [0.17, 0.08, 0.05, 0.13], [-0.16, -0.06, 0.04, 0.14], [0.05, -0.18, -0.06, 0.11], [-0.04, 0.17, -0.08, 0.1], [0.2, -0.12, 0.1, 0.07]]
        .forEach(([x, y, z, r]) => { const b = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), m); b.position.set(x, y, z); b.userData.base = b.position.clone(); core.add(b); });
    } else {
      // an open book: gold covers with white pages
      const cover = glowMat(COL.runner, 0.35), page = new THREE.MeshStandardMaterial({ color: 0xFFFDF4, emissive: 0x6B5A2A, emissiveIntensity: 0.25, roughness: 0.8 });
      [-1, 1].forEach(sd => {
        const half = new THREE.Group(); half.rotation.z = sd * 0.42;
        const c = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.4), cover); c.position.x = sd * 0.15;
        const p = new THREE.Mesh(new THREE.BoxGeometry(0.27, 0.035, 0.36), page); p.position.set(sd * 0.145, 0.03, 0);
        half.add(c, p); core.add(half);
      });
      core.rotation.x = 0.5;
    }
    const orbit = isHunter ? makeGlyphOrbit(['?', '!', '…'], hunterHex, 0.6, 2.3, Math.PI / 3, 0.3)
                           : makeGlyphOrbit(['A', 'b', 'c', '“', '&'], runnerHex, 0.6, 1.2, Math.PI / 2.5, 0);
    group.add(core, orbit.group);
    return {
      group,
      update(t) {
        if (isHunter) core.children.forEach((b, i) => { b.position.copy(b.userData.base).multiplyScalar(1 + Math.sin(t * 5 + i) * 0.12); });
        else { core.rotation.y = Math.sin(t * 0.8) * 0.6; core.position.y = Math.sin(t * 2) * 0.03; }
        orbit.group.rotation.y = t * orbit.speed;
      },
      boost(k) { orbit.group.rotation.y += 0.08 * k; }
    };
  },
  history(isHunter) {
    const group = new THREE.Group();
    const core = new THREE.Group();
    let hands = null;
    if (isHunter) {
      // a ticking clock: time is running out
      const face = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.06, 32), new THREE.MeshStandardMaterial({ color: 0x2A1036, emissive: 0x3B0F4A, emissiveIntensity: 0.5, roughness: 0.6 }));
      face.rotation.x = Math.PI / 2;
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.04, 10, 32), glowMat(COL.hunter, 0.7));
      hands = new THREE.Group(); hands.position.z = 0.04;
      const hm = new THREE.MeshBasicMaterial({ color: COL.hunterLight });
      const h1 = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.22, 0.02), hm); h1.position.y = 0.1;
      const h2 = new THREE.Group(); const h2m = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.15, 0.02), hm); h2m.position.y = 0.07; h2.add(h2m);
      hands.add(h1, h2); hands.userData.h2 = h2;
      core.add(face, rim, hands);
    } else {
      // an hourglass in a gold frame
      const frame = glowMat(COL.runner, 0.4), sand = new THREE.MeshStandardMaterial({ color: 0xFFF1C2, emissive: 0xC9A040, emissiveIntensity: 0.4, roughness: 0.7, transparent: true, opacity: 0.9 });
      const top = new THREE.Mesh(new THREE.ConeGeometry(0.19, 0.26, 20), sand); top.rotation.x = Math.PI; top.position.y = 0.14;
      const bot = new THREE.Mesh(new THREE.ConeGeometry(0.19, 0.26, 20), sand); bot.position.y = -0.14;
      [0.29, -0.29].forEach(y => { const p = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.05, 24), frame); p.position.y = y; core.add(p); });
      [0, 1, 2].forEach(i => { const a = i * Math.PI * 2 / 3; const r = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.58, 8), frame); r.position.set(Math.cos(a) * 0.21, 0, Math.sin(a) * 0.21); core.add(r); });
      core.add(top, bot);
    }
    const orbit = isHunter ? makeGlyphOrbit(['XII', 'IX', 'III'], hunterHex, 0.62, 2.0, Math.PI / 3, 0.3, 0.3)
                           : makeGlyphOrbit(['I', 'V', 'X', 'L', 'C'], runnerHex, 0.6, 1.1, Math.PI / 2.5, 0);
    group.add(core, orbit.group);
    return {
      group,
      update(t) {
        if (hands) { hands.rotation.z = -t * 6; hands.userData.h2.rotation.z = t * 5.5; core.rotation.y = Math.sin(t * 1.5) * 0.4; }
        else { core.rotation.z = Math.sin(t * 0.9) * 0.25; core.rotation.y = t * 0.5; }
        orbit.group.rotation.y = t * orbit.speed;
      },
      boost(k) { orbit.group.rotation.y += 0.08 * k; }
    };
  },
  geography(isHunter) {
    const group = new THREE.Group();
    const core = new THREE.Group();
    if (isHunter) {
      // a storm cloud with lightning
      const cm = new THREE.MeshStandardMaterial({ color: 0x7A3F8C, emissive: COL.hunter, emissiveIntensity: 0.3, roughness: 0.9 });
      [[0, 0.04, 0, 0.2], [0.2, 0, 0, 0.15], [-0.2, -0.01, 0, 0.16], [0.08, 0.15, 0.02, 0.14], [-0.1, 0.12, -0.03, 0.13]].forEach(([x, y, z, r]) => { const b = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), cm); b.position.set(x, y, z); core.add(b); });
      const bolt = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyphTexture('⚡', '#FFF1C2'), transparent: true, depthWrite: false }));
      bolt.position.set(0, -0.26, 0.05); bolt.scale.setScalar(0.3); core.add(bolt); core.userData.bolt = bolt;
    } else {
      // a little globe with a gold equator ring
      const c = document.createElement('canvas'); c.width = 256; c.height = 128;
      const g = c.getContext('2d'); g.fillStyle = '#1E88C8'; g.fillRect(0, 0, 256, 128);
      g.fillStyle = '#5CC96B';
      [[40, 40, 30, 22], [70, 80, 18, 26], [130, 45, 34, 20], [150, 85, 20, 18], [205, 60, 26, 30], [230, 100, 14, 10]].forEach(([x, y, rx, ry]) => { g.beginPath(); g.ellipse(x, y, rx, ry, 0.4, 0, Math.PI * 2); g.fill(); });
      g.fillStyle = '#EEF6FF'; g.fillRect(0, 0, 256, 8); g.fillRect(0, 120, 256, 8);
      const globe = new THREE.Mesh(new THREE.SphereGeometry(0.26, 32, 20), new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(c), emissive: 0x153a52, emissiveIntensity: 0.6, roughness: 0.6 }));
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.018, 8, 40), glowMat(COL.runner, 0.7)); ring.rotation.x = Math.PI / 2;
      core.add(globe, ring); core.userData.globe = globe;
    }
    const orbit = isHunter ? makeGlyphOrbit(['☂', '❄', '~'], hunterHex, 0.62, 2.2, Math.PI / 3, 0.3)
                           : makeGlyphOrbit(['N', 'E', 'S', 'W'], runnerHex, 0.6, 1.1, Math.PI / 2.5, 0);
    group.add(core, orbit.group);
    return {
      group,
      update(t) {
        if (core.userData.globe) core.userData.globe.rotation.y = t * 0.9;
        if (core.userData.bolt) core.userData.bolt.material.opacity = (Math.sin(t * 9) > 0.6) ? 1 : 0.25;
        if (isHunter) core.rotation.z = Math.sin(t * 3) * 0.08;
        orbit.group.rotation.y = t * orbit.speed;
      },
      boost(k) { orbit.group.rotation.y += 0.08 * k; }
    };
  },
  general(isHunter) {
    const group = new THREE.Group();
    const core = new THREE.Group();
    if (isHunter) {
      // a spiky comet ball
      const m = glowMat(COL.hunter, 0.55);
      core.add(new THREE.Mesh(new THREE.IcosahedronGeometry(0.2, 0), m));
      const spikeGeo = new THREE.ConeGeometry(0.05, 0.2, 8);
      const dirs = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[0.7,0.7,0],[-0.7,0.7,0],[0.7,-0.7,0],[-0.7,-0.7,0],[0,0.7,0.7],[0,-0.7,-0.7]];
      dirs.forEach(d => { const v = new THREE.Vector3(...d).normalize(); const sp = new THREE.Mesh(spikeGeo, m); sp.position.copy(v.clone().multiplyScalar(0.25)); sp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v); core.add(sp); });
    } else {
      // a gold star
      const sh = new THREE.Shape();
      for (let i = 0; i < 10; i++) { const r = i % 2 ? 0.13 : 0.3, a = Math.PI / 2 + i * Math.PI / 5; const x = Math.cos(a) * r, y = Math.sin(a) * r; i ? sh.lineTo(x, y) : sh.moveTo(x, y); }
      sh.closePath();
      const star = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.08, bevelEnabled: true, bevelSize: 0.02, bevelThickness: 0.02, bevelSegments: 2 }), glowMat(COL.runner, 0.5));
      star.geometry.center();
      core.add(star);
    }
    const orbit = isHunter ? makeGlyphOrbit(['?', '?', '?'], hunterHex, 0.62, 2.3, Math.PI / 3, 0.3)
                           : makeGlyphOrbit(['?', '!', '★', '?'], runnerHex, 0.6, 1.2, Math.PI / 2.5, 0);
    group.add(core, orbit.group);
    return {
      group,
      update(t) { core.rotation.y = t * (isHunter ? 1.8 : 0.9); if (isHunter) core.rotation.x = t * 1.2; orbit.group.rotation.y = t * orbit.speed; },
      boost(k) { orbit.group.rotation.y += 0.08 * k; }
    };
  }
};
const THEMES = {
  science:   { label: 'Science', cell: 0x252C6B,   bg: 0x0B1026, floor: 'hex',     floorBg: '#0D1230', line: 'rgba(150,160,255,0.16)' },
  maths:     { label: 'Maths', cell: 0x1B3E70,     bg: 0x071A2E, floor: 'graph',   floorBg: '#0A1F38', line: 'rgba(120,200,255,0.22)' },
  english:   { label: 'English', cell: 0x3E2468,   bg: 0x1A0F2E, floor: 'lines',   floorBg: '#1D1336', line: 'rgba(255,190,230,0.2)' },
  history:   { label: 'History', cell: 0x5B3A1E,   bg: 0x1A1108, floor: 'stone',   floorBg: '#2A1D10', line: 'rgba(255,214,150,0.16)' },
  geography: { label: 'Geography', cell: 0x125452, bg: 0x061D1E, floor: 'contour', floorBg: '#0A2628', line: 'rgba(140,255,220,0.2)' },
  general:   { label: 'General', cell: 0x30246E,   bg: 0x120A26, floor: 'stars',   floorBg: '#170E30', line: 'rgba(220,200,255,0.18)' }
};
function themeFloorTexture(th) {
  if (th.floor === 'hex') return makeHexGridTexture();
  const size = 512, c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'); g.fillStyle = th.floorBg; g.fillRect(0, 0, size, size);
  g.strokeStyle = th.line; g.lineWidth = 1.4;
  if (th.floor === 'graph') {
    for (let i = 0; i <= size; i += 16) { g.lineWidth = i % 64 ? 1 : 2.2; g.beginPath(); g.moveTo(i, 0); g.lineTo(i, size); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(size, i); g.stroke(); }
  } else if (th.floor === 'lines') {
    for (let y = 24; y < size; y += 32) { g.beginPath(); g.moveTo(0, y); g.lineTo(size, y); g.stroke(); }
    g.strokeStyle = 'rgba(255,120,150,0.3)'; g.lineWidth = 2.5; g.beginPath(); g.moveTo(70, 0); g.lineTo(70, size); g.stroke();
  } else if (th.floor === 'stone') {
    for (let row = 0; row < size / 64; row++) for (let col = -1; col < size / 128 + 1; col++) g.strokeRect(col * 128 + (row % 2) * 64, row * 64, 128, 64);
  } else if (th.floor === 'contour') {
    [[150, 160], [380, 330], [120, 420]].forEach(([cx, cy]) => { for (let r = 20; r < 220; r += 26) { g.beginPath(); g.ellipse(cx, cy, r * 1.3, r, 0.5, 0, Math.PI * 2); g.stroke(); } });
  } else {
    g.fillStyle = th.line;
    for (let i = 0; i < 120; i++) { const x = (i * 97.3) % size, y = (i * 61.7 + (i % 7) * 37) % size, r = (i % 5) * 0.4 + 0.6; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill(); }   // fixed pattern: the same floor every time
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(4, 4);
  return tex;
}
const LOOK_FOR_SUBJECT = { biology: 'science', chemistry: 'science', physics: 'science', combined: 'science', maths: 'maths', english: 'english', history: 'history', geography: 'geography' };
const lookForSubject = sj => LOOK_FOR_SUBJECT[sj] || 'general';
let runnerChar = null, hunterChar = null, currentTheme = '';
function disposeTree(obj) {
  obj.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    [].concat(o.material || []).forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
  });
}
function applyTheme(force) {
  const id = lookForSubject(bank.subject());
  if (id === currentTheme && !force) return;
  currentTheme = id;
  const th = THEMES[id];
  [runnerGroup, hunterGroup].forEach(g => { g.children.slice().forEach(c => { g.remove(c); disposeTree(c); }); });
  runnerChar = CHARACTERS[id](false); hunterChar = CHARACTERS[id](true);
  runnerGroup.add(runnerChar.group); hunterGroup.add(hunterChar.group);
  scene.background.setHex(th.bg); scene.fog.color.setHex(th.bg);
  cellMat.color.setHex(th.cell);
  const old = floor.material.map; floor.material.map = themeFloorTexture(th); floor.material.color.set(th.floorBg); floor.material.needsUpdate = true; if (old) old.dispose();
}

const auraMat = new THREE.MeshBasicMaterial({ color: COL.hunter, transparent: true, opacity: 0.3 });
const auraMesh = new THREE.Mesh(new THREE.SphereGeometry(0.62, 20, 20), auraMat);
scene.add(auraMesh);

const escapeStreakMat = new THREE.MeshBasicMaterial({ color: COL.runner, transparent: true, opacity: 0 });
const escapeStreak = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.18, 2.2, 12), escapeStreakMat);
escapeStreak.rotation.z = Math.PI / 2;
scene.add(escapeStreak);

const tetherMat = new THREE.MeshBasicMaterial({ color: COL.tether, transparent: true, opacity: 0.35 });
const tether = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1, 8), tetherMat);
scene.add(tether);
const tetherColor = new THREE.Color(COL.tether);
const tetherTargetColor = new THREE.Color(COL.tether);
const tetherRest = new THREE.Color(COL.tether);
let tetherPulse = 0;

const pulses = [];
const pulseMatGood = new THREE.MeshBasicMaterial({ color: COL.runner });
const pulseMatBad = new THREE.MeshBasicMaterial({ color: COL.hunter });
for (let i = 0; i < 6; i++) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 10), pulseMatGood);
  mesh.visible = false;
  scene.add(mesh);
  pulses.push({ mesh, active: false, progress: 0 });
}
function spawnPulse(good) {
  const p = pulses.find(x => !x.active) || pulses[0];
  p.active = true; p.progress = 0;
  p.mesh.material = good ? pulseMatGood : pulseMatBad;
  p.mesh.visible = true;
}

const explosionParts = [];
const explosionMats = [new THREE.MeshBasicMaterial({ color: COL.hunter }), new THREE.MeshBasicMaterial({ color: COL.runner }), new THREE.MeshBasicMaterial({ color: COL.spark })];
for (let i = 0; i < 90; i++) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.055, 8, 8), explosionMats[i % 3]);
  mesh.visible = false;
  scene.add(mesh);
  explosionParts.push({ mesh, vel: new THREE.Vector3() });
}
const shockwaveMat = new THREE.MeshBasicMaterial({ color: COL.spark, transparent: true, opacity: 0, side: THREE.DoubleSide });
const shockwave = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.2, 48), shockwaveMat);
shockwave.rotation.x = -Math.PI / 2;
shockwave.visible = false;
scene.add(shockwave);
const flareLight = new THREE.PointLight(COL.spark, 0, 12);
scene.add(flareLight);

const victoryRingMat = new THREE.MeshBasicMaterial({ color: COL.runner, transparent: true, opacity: 0, side: THREE.DoubleSide });
const victoryRing = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.2, 48), victoryRingMat);
victoryRing.rotation.x = -Math.PI / 2;
victoryRing.visible = false;
scene.add(victoryRing);

const HUNTER_X = 1.6;
const HOME_BASE_X = -1.6;
let runnerVelX = 0;
let hunterVelX = 0;
const SPRING_K = 0.05;
const SPRING_DAMPING = 0.82;
let shakeAmt = 0;
const shake = v => { if (!reduced()) shakeAmt = Math.max(shakeAmt, v); };

/* Deal Round track: discrete cells, the Hunter and the runner hop between them */
const CELL_SPACING = 1.55;
const trackGroup = new THREE.Group();
scene.add(trackGroup);
function cellX(i) { return HOME_BASE_X + i * CELL_SPACING; }
const trimGeo = new THREE.BoxGeometry(CELL_SPACING - 0.14 - 0.04, 0.02, 0.06);
const cellGeo = new THREE.BoxGeometry(CELL_SPACING - 0.14, 0.16, 0.9);
const cellMat = new THREE.MeshStandardMaterial({ color: 0x252C6B, roughness: 0.55, metalness: 0.15 });
const homeMat = new THREE.MeshStandardMaterial({ color: 0x12A4A0, emissive: 0x0A4F4D, roughness: 0.4, metalness: 0.2 });
const trimMat = new THREE.MeshStandardMaterial({ color: 0x9AA2F0, emissive: 0x5A63C8, emissiveIntensity: 0.5, roughness: 0.3 });
const homeTrimMat = new THREE.MeshStandardMaterial({ color: 0x9FFCF0, emissive: 0x4FF0D8, emissiveIntensity: 0.8, roughness: 0.3 });
/* Painted tile tops: chevrons pointing home and the number of steps left */
const tileTex = {};
function tileTexture(i) {
  if (tileTex[i]) return tileTex[i];
  const c = document.createElement('canvas'); c.width = 256; c.height = 160;
  const g = c.getContext('2d');
  const home = i === 0;
  g.lineJoin = g.lineCap = 'round';
  // chevrons point left, towards home
  g.strokeStyle = home ? 'rgba(255,255,255,0.35)' : 'rgba(180,190,255,0.55)'; g.lineWidth = 12;
  [[40, 80], [216, 80]].forEach(([x, y]) => { g.beginPath(); g.moveTo(x + 18, y - 30); g.lineTo(x - 12, y); g.lineTo(x + 18, y + 30); g.stroke(); });
  g.fillStyle = '#fff'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.font = home ? '64px "Lilita One", sans-serif' : '104px "Lilita One", sans-serif';
  g.strokeStyle = 'rgba(10,12,40,0.9)'; g.lineWidth = 10;
  g.strokeText(home ? 'HOME' : String(i), 128, home ? 84 : 88);
  g.fillText(home ? 'HOME' : String(i), 128, home ? 84 : 88);
  const t = new THREE.CanvasTexture(c);
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return (tileTex[i] = t);
}
const tileTopGeo = new THREE.PlaneGeometry(CELL_SPACING - 0.2, 0.86);
/* Finish post at home: two glowing posts and a HOME sign facing the class */
const finishGroup = new THREE.Group();
(function buildFinish() {
  const postMat = new THREE.MeshStandardMaterial({ color: 0x9FFCF0, emissive: 0x2BD9C2, emissiveIntensity: 0.9, roughness: 0.3 });
  const postGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.9, 12);
  [-0.5, 0.5].forEach(z => { const m = new THREE.Mesh(postGeo, postMat); m.position.set(0, 0.95, z); finishGroup.add(m); });
  const c = document.createElement('canvas'); c.width = 512; c.height = 160;
  const g = c.getContext('2d');
  g.fillStyle = '#0A6663'; g.fillRect(0, 0, 512, 160);
  g.fillStyle = '#9FFCF0'; g.fillRect(0, 0, 512, 12); g.fillRect(0, 148, 512, 12);
  g.fillStyle = '#fff'; g.font = '108px "Lilita One", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('HOME', 256, 86);
  const tex = new THREE.CanvasTexture(c);
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.53), new THREE.MeshBasicMaterial({ map: tex }));
  sign.position.set(0, 1.95, 0.5);
  finishGroup.add(sign);
})();
function buildTrack(totalCells) {
  trackGroup.clear();
  finishGroup.position.set(cellX(0) - CELL_SPACING / 2 + 0.02, 0, 0);
  trackGroup.add(finishGroup);
  for (let i = 0; i < totalCells; i++) {
    const mesh = new THREE.Mesh(cellGeo, i === 0 ? homeMat : cellMat);
    mesh.position.set(cellX(i), 0.08, 0);
    trackGroup.add(mesh);
    const trim = new THREE.Mesh(trimGeo, i === 0 ? homeTrimMat : trimMat);
    trim.position.set(cellX(i), 0.17, (CELL_SPACING - 0.14) / 2 - 0.05);
    trackGroup.add(trim);
    const top = new THREE.Mesh(tileTopGeo, new THREE.MeshBasicMaterial({ map: tileTexture(i), transparent: true, depthWrite: false }));
    top.rotation.x = -Math.PI / 2;
    top.position.set(cellX(i), 0.165, 0);
    trackGroup.add(top);
  }
}
let currentRoundType = 'deal'; // 'deal' | 'sprint'
let runnerCellIndex = 0;
let hunterCellIndex = 0;

const BUILDUP_DURATION = 1.7;
const yankTimes = [0.15, 0.38, 0.58, 0.76, 0.92, 1.06, 1.2, 1.32, 1.43, 1.52, 1.6];
let buildupBaseX = 0;
let buildupYankIndex = 0;

const ESCAPE_BREAKFREE_DURATION = 1.0;
const escapeYankTimes = [0.15, 0.38, 0.6, 0.8];
let escapeBaseX = 0;
let escapeYankIndex = 0;
let snapFlashDone = false;

let sceneMode = 'idle'; // idle | buildup | explode | escape | deal-caught | deal-escape | idle-done
let sequenceStart = 0;
let onSequenceComplete = null; // fired once when a caught or escape sequence finishes

function runnerTargetX(netScore, target) {
  return HOME_BASE_X - (netScore / target) * 4.2;
}

const flashEl = $('flash');
function resetAtomsForRound(roundType) {
  currentRoundType = roundType;
  root.classList.toggle('deal-view', roundType === 'deal');   // smaller tags while the whole track is in view
  measureTags();
  sceneMode = 'idle';
  onSequenceComplete = null;
  runnerVelX = 0;
  hunterVelX = 0;
  runnerGroup.position.set(HOME_BASE_X, 0.62, 0);
  hunterGroup.position.set(HUNTER_X, 0.62, 0);
  runnerGroup.userData.targetX = HOME_BASE_X;
  hunterGroup.userData.targetX = HUNTER_X;
  runnerGroup.visible = true;
  hunterGroup.visible = true;
  auraMesh.visible = true;
  tether.visible = (roundType === 'sprint');
  trackGroup.visible = (roundType === 'deal');
  tetherColor.copy(tetherRest);
  tetherTargetColor.copy(tetherRest);
  tetherPulse = 0;
  shakeAmt = 0;
  escapeStreakMat.opacity = 0;
  shockwave.visible = false;
  shockwaveMat.opacity = 0;
  victoryRing.visible = false;
  victoryRingMat.opacity = 0;
  flareLight.intensity = 0;
  flareLight.color.setHex(COL.spark);
  flashEl.style.opacity = 0;
  camera.position.set(0, 2.4, 6.5);
  explosionParts.forEach(p => p.mesh.visible = false);
  pulses.forEach(p => { p.active = false; p.mesh.visible = false; });
}

function pulseCorrect() {
  runnerVelX -= 0.3;
  shake(0.06);
  if (currentRoundType === 'sprint') {
    tetherPulse = 1;
    tetherTargetColor.setHex(COL.runner);
    spawnPulse(true);
  }
}
function pulseWrong() {
  shake(0.1);
  if (currentRoundType === 'sprint') {
    runnerVelX += 0.3;
    tetherPulse = 1;
    tetherTargetColor.setHex(COL.hunter);
    spawnPulse(false);
  }
}

function triggerExplosionAt(collideX) {
  sceneMode = 'explode';
  sequenceStart = performance.now();
  runnerGroup.visible = false;
  hunterGroup.visible = false;
  auraMesh.visible = false;
  tether.visible = false;
  if (!reduced()) {
    // a soft flash, kept well below full brightness
    flashEl.style.transition = 'opacity 0.08s ease';
    flashEl.style.opacity = 0.45;
    later(() => { flashEl.style.transition = 'opacity 0.6s ease'; flashEl.style.opacity = 0; }, 90);
  }

  flareLight.position.set(collideX, 0.9, 0);
  flareLight.intensity = 22;
  shockwave.visible = true;
  shockwave.position.set(collideX, 0.05, 0);
  shockwave.scale.setScalar(1);
  shockwaveMat.opacity = 0.9;

  if (reduced()) camera.position.set(0, 1.4, 3.2);
  else {
    const shakeStart = performance.now();
    (function shakeCam() {
      const el = performance.now() - shakeStart;
      if (el > 550 || !active) { camera.position.set(0, 1.4, 3.2); return; }
      const decay = 1 - el / 550;
      camera.position.set((Math.random() - 0.5) * 0.5 * decay, 1.4 + (Math.random() - 0.5) * 0.3 * decay, 3.2 + (Math.random() - 0.5) * 0.2 * decay);
      requestAnimationFrame(shakeCam);
    })();
  }

  explosionParts.forEach(p => {
    p.mesh.visible = true;
    p.mesh.position.set(collideX, 0.6, 0);
    p.mesh.scale.setScalar(1);
    const dir = new THREE.Vector3((Math.random() - 0.5), (Math.random() - 0.2) * 1.6, (Math.random() - 0.5)).normalize();
    p.vel.copy(dir.multiplyScalar(0.035 + Math.random() * 0.09));
  });
}

function triggerCaughtSequence(callback) {
  sequenceStart = performance.now();
  onSequenceComplete = callback;
  SFX.caught();
  if (currentRoundType === 'deal') {
    sceneMode = 'deal-caught';
    hunterCellIndex = runnerCellIndex;
    hunterGroup.userData.targetX = cellX(hunterCellIndex);
  } else {
    sceneMode = 'buildup';
    buildupBaseX = runnerGroup.position.x;
    buildupYankIndex = 0;
  }
}
function triggerEscapeSequence(callback) {
  sequenceStart = performance.now();
  onSequenceComplete = callback;
  snapFlashDone = false;
  SFX.escaped();
  if (currentRoundType === 'deal') {
    sceneMode = 'deal-escape';
  } else {
    sceneMode = 'escape';
    escapeBaseX = runnerGroup.position.x;
    escapeYankIndex = 0;
  }
}
function finishSequence() {
  sceneMode = 'idle-done';
  const cb = onSequenceComplete; onSequenceComplete = null;
  if (cb) cb();
}

/* Frame the two atoms: keep both on screen at any aspect ratio (portrait tablets too) */
function framingZ(spread) {
  const vHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const fitZ = (spread / 2 + 1.3) / (vHalf * camera.aspect * viewFracX);   // only the part of the picture the race has
  return Math.max(5.0 + spread * 0.85, fitZ);
}

function dealFraming() {
  const minX = cellX(0) - CELL_SPACING / 2 - 1.6, maxX = cellX(TRACK_STEPS + MAX_GAP) + CELL_SPACING / 2 + 0.7;
  const hHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
  const dist = (maxX - minX) / 2 / hHalf + 1.2;
  const tilt = THREE.MathUtils.degToRad(24);
  return { x: (minX + maxX) / 2, y: 0.35 + dist * Math.sin(tilt), z: dist * Math.cos(tilt) };
}

const clock = new THREE.Clock();
let active = false, rafId = 0;
function animate() {
  if (!active) return;
  rafId = requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  watchPerformance(performance.now());
  if (runnerChar) runnerChar.update(t);
  if (hunterChar) hunterChar.update(t);

  if (sceneMode !== 'explode') {
    runnerGroup.position.y = 0.62 + Math.sin(t * 1.6) * 0.05;
    hunterGroup.position.y = 0.62 + Math.sin(t * 1.6 + Math.PI) * 0.05;
    auraMesh.position.copy(hunterGroup.position);
    auraMesh.scale.setScalar(1 + Math.sin(t * 4) * 0.12);
    auraMat.opacity = 0.26 + Math.sin(t * 4) * 0.08;
  }

  // the fog starts a little beyond the action wherever the camera is, so wide framings
  // (the whole Deal Round track on a portrait screen) are not lost in it
  const camDist = Math.hypot(camera.position.y - 0.4, camera.position.z);
  scene.fog.near = camDist + 3.5; scene.fog.far = camDist + 19.5;
  ambientParticles.rotation.y = t * 0.02;
  ambientParticles.position.y = Math.sin(t * 0.3) * 0.15;

  runnerGlow.intensity = 5 + Math.sin(t * 2) * 1.2;
  hunterGlow.intensity = 6 + Math.sin(t * 3) * 1.8;

  if (sceneMode === 'idle') {
    const force = (runnerGroup.userData.targetX - runnerGroup.position.x) * SPRING_K;
    runnerVelX += force;
    runnerVelX *= SPRING_DAMPING;
    runnerGroup.position.x += runnerVelX;

    if (currentRoundType === 'deal') {
      const hforce = (hunterGroup.userData.targetX - hunterGroup.position.x) * SPRING_K;
      hunterVelX += hforce;
      hunterVelX *= SPRING_DAMPING;
      hunterGroup.position.x += hunterVelX;
    }

    if (currentRoundType === 'deal') {
      // the whole track, from the finish post to the Hunter's furthest start, fills the width
      const f = dealFraming();
      camera.position.x += (f.x - camera.position.x) * 0.06;
      camera.position.y += (f.y - camera.position.y) * 0.06;
      camera.position.z += (f.z - camera.position.z) * 0.06;
      camera.lookAt(camera.position.x, 0.35, 0);
    } else {
      const midX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
      const spread = Math.abs(hunterGroup.position.x - runnerGroup.position.x);
      const desiredZ = framingZ(spread);
      camera.position.x += (midX - camera.position.x) * 0.05;
      camera.position.y += (2.4 - camera.position.y) * 0.05;
      camera.position.z += (desiredZ - camera.position.z) * 0.05;
      camera.lookAt(midX, 0.6, 0);
    }

    if (shakeAmt > 0.001) {
      camera.position.x += (Math.random() - 0.5) * shakeAmt;
      camera.position.y += (Math.random() - 0.5) * shakeAmt * 0.6;
      shakeAmt *= 0.82;
    } else { shakeAmt = 0; }
  }

  if (currentRoundType === 'sprint' && (sceneMode === 'idle' || sceneMode === 'buildup')) {
    const from = hunterGroup.position, to = runnerGroup.position;
    const dist = Math.abs(to.x - from.x);
    tether.position.set((from.x + to.x) / 2, 0.6, 0);
    tether.scale.set(1, Math.max(0.001, dist), 1);
    tether.rotation.z = Math.PI / 2;
    tetherPulse *= 0.94;
    tetherColor.lerp(tetherTargetColor, 0.06);
    tetherColor.lerp(tetherRest, 0.01);
    tetherMat.color.copy(tetherColor);
    tetherMat.opacity = 0.35 + tetherPulse * 0.5;
    tether.scale.x = tether.scale.z = 1 + tetherPulse * 2.5;

    pulses.forEach(p => {
      if (!p.active) return;
      p.progress += 0.045;
      const x = from.x + (to.x - from.x) * p.progress;
      p.mesh.position.set(x, 0.6, 0);
      p.mesh.scale.setScalar(1 - p.progress * 0.3);
      if (p.progress >= 1) { p.active = false; p.mesh.visible = false; }
    });
  }

  if (sceneMode === 'deal-caught') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    const HOP_DURATION = 0.4;
    const progress = Math.min(1, elapsed / HOP_DURATION);
    hunterGroup.position.x += (hunterGroup.userData.targetX - hunterGroup.position.x) * 0.25;
    shake(progress * 0.05);

    const midX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
    camera.position.x += (midX - camera.position.x) * 0.12;
    camera.position.z += ((3.4) - camera.position.z) * 0.06;
    camera.lookAt(midX, 0.6, 0);
    if (shakeAmt > 0.001) {
      camera.position.x += (Math.random() - 0.5) * shakeAmt;
      shakeAmt *= 0.85;
    }

    if (progress >= 1 && Math.abs(hunterGroup.position.x - hunterGroup.userData.targetX) < 0.02) {
      const collideX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
      triggerExplosionAt(collideX);
    }
  }

  if (sceneMode === 'buildup') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    const progress = Math.min(1, elapsed / BUILDUP_DURATION);
    const easedProgress = progress * progress;
    const midX = (buildupBaseX + HUNTER_X) / 2;
    buildupBaseX += (midX - buildupBaseX) * easedProgress * 0.1;
    hunterGroup.position.x += (midX - hunterGroup.position.x) * easedProgress * 0.14;

    while (buildupYankIndex < yankTimes.length && elapsed >= yankTimes[buildupYankIndex]) {
      const strength = 0.14 + buildupYankIndex * 0.045;
      const dir = (HUNTER_X - buildupBaseX) >= 0 ? 1 : -1;
      buildupBaseX += dir * strength;
      tetherPulse = 1;
      tetherTargetColor.setHex(COL.hunter);
      shake(0.05 + buildupYankIndex * 0.022);
      buildupYankIndex++;
    }

    const strainFreq = 5 + progress * 18;
    const strainAmp = (1 - progress) * 0.32;
    runnerGroup.position.x = buildupBaseX + Math.sin(elapsed * strainFreq) * strainAmp;

    shake(progress * 0.06);
    camera.position.lerp(new THREE.Vector3(0, 1.4, 3.2), 0.02);
    if (shakeAmt > 0.001) {
      camera.position.x += (Math.random() - 0.5) * shakeAmt;
      camera.position.y += (Math.random() - 0.5) * shakeAmt * 0.7;
      shakeAmt *= 0.88;
    }

    if (progress >= 1) {
      const collideX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
      triggerExplosionAt(collideX);
    }
  }

  if (sceneMode === 'explode') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    explosionParts.forEach(p => {
      p.mesh.position.add(p.vel);
      p.vel.y -= 0.002;
      p.vel.multiplyScalar(0.985);
      p.mesh.scale.multiplyScalar(0.965);
    });
    shockwave.scale.setScalar(1 + elapsed * 14);
    shockwaveMat.opacity = Math.max(0, 0.9 - elapsed * 1.3);
    flareLight.intensity = Math.max(0, 22 - elapsed * 40);
    if (elapsed > 1.6 && onSequenceComplete) {
      shockwave.visible = false;
      finishSequence();
    }
  }

  if (sceneMode === 'deal-escape') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    const ZOOM_DURATION = 0.6;
    const BURST_AT = 0.55;
    const TOTAL_DURATION = 1.3;

    const zoomProgress = Math.min(1, elapsed / ZOOM_DURATION);
    const px = runnerGroup.position.x;
    camera.position.x += (px - camera.position.x) * 0.06;
    camera.position.z += (1.8 - camera.position.z) * 0.05;
    camera.position.y += (0.9 - camera.position.y) * 0.05;
    camera.lookAt(px, 0.62, 0);

    if (runnerChar) runnerChar.boost(zoomProgress);

    if (elapsed >= BURST_AT && !snapFlashDone) {
      snapFlashDone = true;
      victoryRing.visible = true;
      victoryRing.position.set(px, 0.05, 0);
      victoryRing.scale.setScalar(1);
      victoryRingMat.opacity = 0.9;
      flareLight.position.set(px, 0.9, 0);
      flareLight.color.setHex(COL.runner);
      flareLight.intensity = 16;
    }
    if (snapFlashDone) {
      const burstElapsed = elapsed - BURST_AT;
      victoryRing.scale.setScalar(1 + burstElapsed * 10);
      victoryRingMat.opacity = Math.max(0, 0.9 - burstElapsed * 1.1);
      flareLight.intensity = Math.max(0, 16 - burstElapsed * 26);
    }

    if (elapsed > TOTAL_DURATION && onSequenceComplete) {
      victoryRing.visible = false;
      flareLight.color.setHex(COL.spark);
      finishSequence();
    }
  }

  if (sceneMode === 'escape') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    if (elapsed < ESCAPE_BREAKFREE_DURATION) {
      while (escapeYankIndex < escapeYankTimes.length && elapsed >= escapeYankTimes[escapeYankIndex]) {
        const strength = 0.16 + escapeYankIndex * 0.07;
        escapeBaseX -= strength;
        tetherPulse = 1;
        tetherTargetColor.setHex(COL.runner);
        shake(0.03 + escapeYankIndex * 0.015);
        escapeYankIndex++;
      }
      const wobble = Math.sin(elapsed * 22) * 0.06 * (1 - elapsed / ESCAPE_BREAKFREE_DURATION);
      runnerGroup.position.x = escapeBaseX + wobble;

      const from = hunterGroup.position, to = runnerGroup.position;
      const dist = Math.abs(to.x - from.x);
      tether.visible = true;
      tether.position.set((from.x + to.x) / 2, 0.6, 0);
      tether.rotation.z = Math.PI / 2;
      tetherPulse *= 0.94;
      tetherColor.lerp(tetherTargetColor, 0.08);
      tetherMat.color.copy(tetherColor);
      tetherMat.opacity = 0.4 + tetherPulse * 0.5;
      tether.scale.set(1 + tetherPulse * 2, dist, 1 + tetherPulse * 2);

      camera.position.lerp(new THREE.Vector3((from.x + to.x) / 2 * 0.3, 1.8, 4.4), 0.03);
      if (shakeAmt > 0.001) { camera.position.x += (Math.random() - 0.5) * shakeAmt; shakeAmt *= 0.85; }

    } else if (elapsed < ESCAPE_BREAKFREE_DURATION + 0.18) {
      const snapProgress = (elapsed - ESCAPE_BREAKFREE_DURATION) / 0.18;
      if (!snapFlashDone) {
        snapFlashDone = true;
        flareLight.position.set(hunterGroup.position.x, 0.9, 0);
        flareLight.intensity = 12;
      }
      const from = hunterGroup.position;
      tether.visible = true;
      tether.position.lerp(new THREE.Vector3(from.x, 0.6, 0), 0.55);
      tether.scale.y *= 0.82;
      tetherMat.opacity = Math.max(0, 0.85 * (1 - snapProgress));
      flareLight.intensity *= 0.8;
      if (snapProgress >= 1) tether.visible = false;

    } else {
      tether.visible = false;
      const launchElapsed = elapsed - ESCAPE_BREAKFREE_DURATION - 0.18;
      if (launchElapsed >= 0.1) {
        const progress = Math.min(1, (launchElapsed - 0.1) / 0.6);
        runnerGroup.position.x = escapeBaseX - progress * 6;
        escapeStreak.position.set(runnerGroup.position.x + 1, 0.6, 0);
        escapeStreakMat.opacity = 0.5 * (1 - progress);
        camera.position.lerp(new THREE.Vector3(-2.5, 1.6, 3.6), 0.05);
        if (progress >= 1 && launchElapsed > 0.85 && onSequenceComplete) {
          runnerGroup.visible = false;
          escapeStreakMat.opacity = 0;
          finishSequence();
        }
      }
    }
  }

  if (++shiftFrame % 6 === 0) updateViewShift();
  placeTags();
  if (useBloom) composer.render(); else renderer.render(scene, camera);
}
runnerGroup.userData.targetX = HOME_BASE_X;
hunterGroup.userData.targetX = HUNTER_X;

/* Name tags over the atoms: the runner's above, the Hunter's below, so they never collide */
const tagV = new THREE.Vector3();
/* Tag and stage sizes are measured only when they change (text, round, resize), never every
   frame: reading offsetWidth each frame forced the browser to lay out the page 60 times a second */
const stageSize = { w: 0, h: 0 };
function measureTags() {
  stageSize.w = wrap.clientWidth; stageSize.h = wrap.clientHeight;
  ['tagYou', 'tagHunter', 'tagHome'].forEach(id => { const el = $(id); el._w = el.offsetWidth; el._h = el.offsetHeight; });
}
function setTagText(id, text) { $(id).textContent = text; measureTags(); }
function placeTag(el, obj, dy, show) {
  if (!show) { if (el._shown) { el.classList.remove('show'); el._shown = false; } return; }
  if (!el._w) measureTags();
  tagV.set(obj.x, obj.y + dy, obj.z).project(camera);
  const w = stageSize.w, h = stageSize.h;
  const half = el._w / 2 + 8;
  const left = band.left || 0, right = Math.min(w, band.right || w);
  const sx = (tagV.x + 1) / 2 * w - viewShiftX;
  const x = Math.round(Math.min(right - half, Math.max(left + half, sx)));
  // keep tags inside the clear band so they never sit on the HUD panels
  const below = el !== $('tagYou');
  let y = (1 - tagV.y) / 2 * h;
  if (below) y = Math.max(band.top + 4, Math.min(band.bottom - el._h - 4, y));
  else y = Math.max(band.top + el._h + 4, Math.min(band.bottom - 4, y));
  y = Math.round(y);
  // moved with a transform (no page layout), and only when the position changes
  if (x !== el._x || y !== el._y) { el._x = x; el._y = y; el.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, ${below ? '0' : '-100%'})`; }
  const vis = tagV.z < 1 && Math.abs(tagV.x) < 1.1;
  if (vis !== el._shown) { el.classList.toggle('show', vis); el._shown = vis; }
}
const homePos = new THREE.Vector3(HOME_BASE_X, 0.2, 0);
function placeTags() {
  const playing = state.phase === 'deal' || state.phase === 'sprint';
  const visible = playing && (sceneMode === 'idle' || sceneMode === 'buildup');
  placeTag($('tagYou'), runnerGroup.position, 0.85, visible && runnerGroup.visible);
  placeTag($('tagHunter'), hunterGroup.position, -0.9, visible && hunterGroup.visible);
  placeTag($('tagHome'), homePos, -0.35, visible && currentRoundType === 'deal');
}

/* Shift the picture so the race sits in the clear space the HUD panels leave: between the
   top HUD and the bottom dock, or (Final Sprint on a wide screen) to the left of the question column */
let viewShift = 0, viewShiftX = 0, shiftFrame = 0, viewFracX = 1;
const band = { top: 0, bottom: 10000, left: 0, right: 0 };
function updateViewShift() {
  const h = wrap.clientHeight, w = wrap.clientWidth;
  if (!w || !h) return;
  let want = 0, wantX = 0;
  band.left = 0; band.right = w; viewFracX = 1;
  const hud = Object.values(huds).find(el => el.classList.contains('active'));
  if (hud) {
    const wr = wrap.getBoundingClientRect();
    const top = hud.querySelector('.op-top').getBoundingClientRect().bottom - wr.top;
    const dr = hud.querySelector('.op-dock').getBoundingClientRect();
    const sideDock = dr.left - wr.left > w * 0.45 && dr.height > h * 0.35;
    if (sideDock) {
      const avail = dr.left - wr.left - 12;
      band.top = top; band.bottom = h; band.right = avail;
      want = Math.round(h / 2 - (top + h) / 2);
      wantX = Math.round(w / 2 - avail / 2);
      viewFracX = avail / w;
    } else {
      const dock = dr.top - wr.top;
      if (dock > top) want = Math.round(h / 2 - (top + dock) / 2);
      band.top = top; band.bottom = dock;
    }
  } else { band.top = 0; band.bottom = h; }
  if (Math.abs(want - viewShift) < 2 && Math.abs(wantX - viewShiftX) < 2) return;
  const k = reduced() ? 1 : 0.25;
  viewShift += (want - viewShift) * k; viewShiftX += (wantX - viewShiftX) * k;
  if (Math.abs(viewShift) < 1 && Math.abs(viewShiftX) < 1) { camera.clearViewOffset(); viewShift = viewShiftX = 0; }
  else camera.setViewOffset(w, h, viewShiftX, viewShift, w, h);
}
function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  if (viewShift || viewShiftX) camera.setViewOffset(w, h, viewShiftX, viewShift, w, h);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  measureTags();
}
function applyQuality() {
  useBloom = bloomAllowed();
  renderer.setPixelRatio(opPixelRatio());
  composer.setPixelRatio(opPixelRatio());
  resize();
}
/* Step quality down, one level at a time, if frames stay slow (over 26 ms on average) for two
   seconds: first the glow goes, then the resolution drops to 1:1. The Graphics setting is untouched. */
function watchPerformance(now) {
  if (!perf.last) { perf.last = now; return; }
  const dt = Math.min(200, now - perf.last); perf.last = now;
  perf.ema += (dt - perf.ema) * 0.05;
  if (perf.level >= 2 || clock.getElapsedTime() < 3) return;
  perf.slowFor = perf.ema > 26 ? perf.slowFor + dt : 0;
  if (perf.slowFor > 2000) { perf.level++; perf.slowFor = 0; perf.ema = 16; applyQuality(); }
}
window.addEventListener('resize', resize);
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
// the question panels grow when the answer and marks appear: move the tags out of their way at once
if (window.ResizeObserver) { const ro = new ResizeObserver(() => updateViewShift()); root.querySelectorAll('.op-dock, .op-top').forEach(el => ro.observe(el)); }

/* ============ SCREEN CONTROL ============ */
const screens = { home: $('home'), summary: $('summary') };
const huds = { deal: $('hud-deal'), sprint: $('hud-sprint') };
function hideAll() {
  Object.values(screens).forEach(el => el.classList.remove('active'));
  Object.values(huds).forEach(el => el.classList.remove('active'));
}
function showScreen(name) { hideAll(); if (screens[name]) screens[name].classList.add('active'); }
function showHud(name) { hideAll(); if (huds[name]) huds[name].classList.add('active'); }

/* ============ SETUP SCREEN ============ */
$('logo').innerHTML = CGB.brand.opLogo();
function renderPack() { $('startBtn').disabled = !CGB.renderPackLine($('pack')); }
bank.onChange(() => { picker.reset(); applyTheme(); if (state.phase === 'home') renderPack(); });
applyTheme();
// the orbiting symbols are drawn with the display font: draw them again once it has loaded
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => applyTheme(true));
if (![2, 3, 4, 5, 6].includes(state.groups)) state.groups = 4;
$('className').value = CGB.store.get('op.className') || 'Our class';
function renderGroupNames(fresh) {
  const prev = [0, 1, 2, 3, 4, 5].map(i => { const el = $('gname' + i); return el && !fresh ? el.value : null; });
  $('gnames').innerHTML = [0, 1, 2, 3, 4, 5].slice(0, state.groups).map(i => `<label style="--tc:${CGB.TEAMS[i].css}"><span>${CGB.TEAMS[i].mark} Team ${i + 1}</span><input id="op-gname${i}" type="text" maxlength="16" autocomplete="off" value="${escapeHtml(prev[i] || CGB.teamNames(6)[i])}"></label>`).join('');
}
function paintGroups() { $('segGroups').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.v === state.groups))); }
$('segGroups').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  state.groups = +b.dataset.v; CGB.store.setJSON('op.groups', state.groups);
  paintGroups(); renderGroupNames(); CGB.fitSetups();
});
paintGroups();
renderGroupNames();
$('startBtn').addEventListener('click', startGame);

/* The host: one corner host that moves to whichever panel is showing; captions only */
const hostC = CGB.createHostCorner(document.createElement('div'));
function hostSay(slot, text, gesture, ms) { const el = $(slot); if (hostC.el.parentElement !== el) el.appendChild(hostC.el); hostC.say(text, gesture, ms); }
/* ---------- Every team answers, the class moves as one runner ---------- */
const misc = CGB.createMisconceptions();
const dealBoard = CGB.createTeamBoard($('dealBoard'));
const sprintBoard = CGB.createTeamBoard($('sprintBoard'));
function paintBoards(earned) {
  const teams = state.players.map(p => ({ name: p.name, score: '' }));
  dealBoard.set({ teams, earned: earned || [] }); sprintBoard.set({ teams, earned: earned || [] });
}
let dealSnap = null, sprintSnap = null, sprintNextT = 0;
function logClassWrongs(res) {
  const q = state.currentQuestion;
  res.forEach((ok, i) => { if (!ok) { state.wrongAnswers[i].push({ subject: q.subject, topic: q.topic, q: q.q, a: q.a }); bank.logWrong(state.players[i].name, q, GAME_NAME); } });
  const c = res.filter(Boolean).length, n = res.length;
  misc.add(q, (n - c) / n, `${n - c} of ${n} teams wrong`);
}
function unlogClassWrongs(res) {
  const q = state.currentQuestion;
  res.forEach((ok, i) => { if (!ok) { state.wrongAnswers[i].pop(); bank.unlogWrong(state.players[i].name, q); } });
  misc.remove(q);
}
/* Deal Round: at least half the teams right moves the runner; otherwise the Hunter gains.
   The step shows at once; a step that ends the round waits for Enter, so it can still be undone. */
const roundDeal = CGB.createClassRound({
  root: root, board: dealBoard, countEl: $('dealCount'), btnEl: $('dealBtnRow'),
  seconds: () => CGB.COUNTDOWN, teams: () => state.players.length,
  doneHtml: () => `<button class="btn go" type="button" data-op="next">${runnerCellIndex <= 0 ? 'Home!' : hunterCellIndex <= runnerCellIndex ? 'Caught!' : 'Next question'} <span class="kbd">Enter</span></button>`,
  onConfirm(res) {
    const c = res.filter(Boolean).length, n = res.length, ok = c * 2 >= n;
    dealSnap = { res, runner: runnerCellIndex, hunter: hunterCellIndex };
    logClassWrongs(res);
    $('dealQAnswer').classList.add('shown');
    $('dealVerdict').innerHTML = `<b>${c} of ${n} teams correct.</b> ${ok ? 'One step closer to home!' : 'Fewer than half, so the Hunter gains a step.'}`;
    $('dealVerdict').className = 'op-cmline ' + (ok ? 'ok' : 'no');
    if (ok) { SFX.correct(); runnerCellIndex--; runnerGroup.userData.targetX = cellX(runnerCellIndex); pulseCorrect(); }
    else { SFX.wrong(); hunterCellIndex--; hunterGroup.userData.targetX = cellX(hunterCellIndex); pulseWrong(); }
    dealStatus();
    hostSay('hostDeal', CGB.classLine(c, n), CGB.classGesture(c, n), 1500);
  },
  onUndo() {
    const u = dealSnap; if (!u) return;
    unlogClassWrongs(u.res);
    runnerCellIndex = u.runner; hunterCellIndex = u.hunter;
    runnerGroup.userData.targetX = cellX(runnerCellIndex); hunterGroup.userData.targetX = cellX(hunterCellIndex);
    $('dealQAnswer').classList.remove('shown'); $('dealVerdict').textContent = 'Marking undone. Mark each team again.'; $('dealVerdict').className = 'op-cmline';
    dealStatus(); dealSnap = null;
  }
});
$('dealBtnRow').addEventListener('click', e => { if (e.target.closest('button[data-op="next"]')) classDealNext(); });
function classDealNext() {
  if (roundDeal.phase !== 'done' || state.phase !== 'deal') return;
  roundDeal.stop();
  if (runnerCellIndex <= 0) { state.phase = 'dealEnd'; $('qcard').hidden = true; later(() => triggerEscapeSequence(() => finishDealRound(true)), 300); }
  else if (hunterCellIndex <= runnerCellIndex) { state.phase = 'dealEnd'; $('qcard').hidden = true; later(() => triggerCaughtSequence(() => finishDealRound(false)), 300); }
  else askDealQuestion();
}
/* Final Sprint: every correct team is one step. In 60 seconds a class that thinks, talks and
   writes gets through 3 or 4 questions; with about 60% of boards right that is about 2 steps
   per team. The target per team is 2, 2.25 or 2.5 for a pot of up to 300, up to 600 or more
   (a bolder deal makes the sprint harder), which a class makes about 60%, 45% and 35% of the
   time in simulation: about half the time overall. */
function classTarget() {
  const pot = state.pot / Math.max(1, state.dealRounds);
  const per = pot <= 300 ? 2 : pot <= 600 ? 2.25 : 2.5;
  return Math.max(3, Math.ceil(state.players.length * per));
}
const WIN_UNDO_MS = 1600;   // after the target is reached: time to undo a marking slip before the escape plays
const roundSprint = CGB.createClassRound({
  root: root, board: sprintBoard, btnEl: $('sprintBtnRow'), fast: true,
  seconds: () => 0, teams: () => state.players.length,
  onConfirm(res) {
    const c = res.filter(Boolean).length, n = res.length;
    sprintSnap = { res, net: state.sprintNetScore };
    logClassWrongs(res);
    state.sprintNetScore += c;
    runnerGroup.userData.targetX = runnerTargetX(Math.min(state.sprintNetScore, state.sprintTarget), state.sprintTarget);
    paintSprintTarget();
    $('sprintQAnswer').classList.add('shown');
    $('sprintVerdict').innerHTML = `<b>${c} of ${n} teams correct: +${c}</b>`;
    $('sprintVerdict').className = 'op-cmline ' + (c ? 'ok' : 'no');
    paintBoards(res.map(ok => ok ? '+1' : ''));
    sprintBoard.set({ marks: res });
    if (c) { SFX.correct(); pulseCorrect(); } else { SFX.wrong(); pulseWrong(); }
    clearTimeout(sprintNextT);
    if (state.sprintNetScore >= state.sprintTarget) {
      // target reached: the clock stops now and the escape plays after a short pause, in which
      // the marking can still be undone
      state.sprintFrozen = true;
      hostSay('hostSprint', 'Target reached! You did it!', 'cheer', 1400);
      sprintNextT = later(() => { roundSprint.stop(); if (state.phase === 'sprint') resolveSprint(true); }, WIN_UNDO_MS);
      return;
    }
    hostSay('hostSprint', CGB.classLine(c, n), CGB.classGesture(c, n), 1200);
    sprintNextT = later(() => { roundSprint.stop(); if (state.phase === 'sprint' && state.sprintTimeLeft > 0) askSprintQuestion(); }, 1400);
  },
  onUndo() {
    const u = sprintSnap; if (!u) return;
    clearTimeout(sprintNextT);
    state.sprintFrozen = false;          // undoing the winning answer starts the clock again
    unlogClassWrongs(u.res);
    state.sprintNetScore = u.net;
    runnerGroup.userData.targetX = runnerTargetX(Math.min(state.sprintNetScore, state.sprintTarget), state.sprintTarget);
    paintSprintTarget();
    $('sprintQAnswer').classList.remove('shown'); $('sprintVerdict').textContent = '';
    sprintSnap = null;
  }
});
function startGame() {
  if (!bank.active()) { renderPack(); return; }
  CGB.leaveField();
  clearTimers(); roundDeal.stop(); roundSprint.stop();
  const names = CGB.saveTeamNames([0, 1, 2, 3, 4, 5].slice(0, state.groups).map(i => $('gname' + i).value)).map(n => n.slice(0, 16));
  state.players = names.map(name => ({ name }));
  state.className = ($('className').value.trim() || 'Our class').slice(0, 16);
  CGB.store.set('op.className', state.className);
  state.dealRound = 0;
  misc.reset();
  state.pot = 0;
  state.dealOutcomes = [];
  state.wrongAnswers = state.players.map(() => []);
  picker.reset();
  $('roundEnd').classList.remove('show');
  // the track is shown with nothing ticking until the teacher presses Start game
  state.phase = 'ready';
  hideAll();
  gate.show(startDealRound);
}
const gate = CGB.createStartGate(document.getElementById('game-outpace'), 'The class votes for a deal, then every team answers each question on a whiteboard. Nothing starts until you press Start.');
const pickQuestion = () => picker.pick(state.players.map(p => p.name));

/* ============ DEAL ROUND ============ */
const TRACK_STEPS = 6; // fixed distance from start to home, whatever the deal
const MAX_GAP = Math.max(...Object.values(state.tierConfig).map(c => c.gap));
const TIERS = ['low', 'mid', 'high'];

function startDealRound() {
  state.phase = 'deal';
  state.dealReward = 0;
  state.currentQuestion = null;
  resetAtomsForRound('deal');

  // build the full track straight away so it's visible while choosing a deal
  runnerCellIndex = TRACK_STEPS;
  hunterCellIndex = TRACK_STEPS + MAX_GAP;
  buildTrack(hunterCellIndex + 1);
  trackGroup.visible = true;
  runnerGroup.position.x = cellX(runnerCellIndex);
  hunterGroup.position.x = cellX(hunterCellIndex);
  runnerGroup.userData.targetX = cellX(runnerCellIndex);
  hunterGroup.userData.targetX = cellX(hunterCellIndex);
  setTagText('tagYou', '▲ ' + state.className);
  const f = dealFraming();                 // start already framed on the whole track
  camera.position.set(f.x, f.y, f.z);
  camera.lookAt(f.x, 0.35, 0);

  showHud('deal');
  $('dealLabel').textContent = `${state.className}: vote for a deal${state.dealRounds > 1 ? ` (Deal Round ${state.dealRound + 1} of ${state.dealRounds})` : ''}. Hold up 1, 2 or 3 fingers!`;
  $('dealRow').hidden = false;
  $('qcard').hidden = true;
  $('dealStatus').textContent = `${TRACK_STEPS} steps to home. Class pot so far: ${state.pot} points.`;
  hostSay('hostDeal', state.dealRound === 0 ? 'Welcome to Outpace! Vote for your deal: 1, 2 or 3 fingers. Majority wins.' : 'Another Deal Round! Vote again: 1, 2 or 3 fingers.', 'wave', 2200);
  const first = $('dealRow').querySelector('button'); if (first) first.focus({ preventScroll: true });
}
function chooseDeal(tier) {
  if (state.phase !== 'deal' || state.dealReward) return;
  const cfg = state.tierConfig[tier];
  if (!cfg) return;
  SFX.pick();
  state.dealReward = cfg.reward;

  // don't rebuild or snap the track: glide the Hunter to its new starting cell
  hunterCellIndex = TRACK_STEPS + cfg.gap;
  hunterGroup.userData.targetX = cellX(hunterCellIndex);

  $('dealRow').hidden = true;
  $('dealLabel').textContent = `${state.className}: playing for ${cfg.reward} points`;
  $('qcard').hidden = false;
  hostSay('hostDeal', tier === 'high' ? 'A bold deal! The Hunter is right behind you.' : tier === 'low' ? 'A cautious start. Off you go!' : 'Standard deal. Let us race!', 'present', 1600);
  askDealQuestion();
}
$('dealRow').addEventListener('click', e => {
  const b = e.target.closest('button[data-tier]');
  if (b) chooseDeal(b.dataset.tier);
});
function dealStatus() {
  const gap = hunterCellIndex - runnerCellIndex;
  $('dealStatus').textContent = `${runnerCellIndex} step${runnerCellIndex === 1 ? '' : 's'} to home · Hunter ${gap} step${gap === 1 ? '' : 's'} behind`;
}

function askDealQuestion() {
  const q = pickQuestion();
  state.currentQuestion = q;
  $('dealQTag').textContent = q.subject + ' · ' + q.topic;
  $('dealQText').textContent = q.q;
  $('dealQAnswer').textContent = q.a;
  $('dealQAnswer').classList.remove('shown'); $('dealVerdict').textContent = '';
  paintBoards(); dealStatus(); roundDeal.think();
}

function finishDealRound(escaped) {
  const reward = escaped ? state.dealReward : Math.round(state.dealReward / 2);
  state.pot += reward;
  state.dealOutcomes.push({ label: `Deal Round ${state.dealRound + 1}`, escaped, reward });
  $('dealStatus').textContent = '';
  hostSay('hostDeal', escaped ? `Home safe! ${reward} points banked.` : `Caught! Still, ${reward} points go in the pot.`, escaped ? 'cheer' : 'shrug', 2000);
  showRoundEnd(escaped ? 'Escaped!' : 'Caught!', escaped, `${state.className} bank ${reward} points. Class pot: ${state.pot}`, () => {
    state.dealRound++;
    if (state.dealRound < state.dealRounds) startDealRound(); else startSprint();
  });
}
let roundEndNext = null;
function showRoundEnd(verdict, good, detail, next) {
  $('roundEndVerdict').textContent = verdict;
  $('roundEndVerdict').className = 'op-verdict ' + (good ? 'escaped' : 'caught');
  $('roundEndDetail').textContent = detail;
  $('roundEnd').classList.add('show');
  roundEndNext = next;
  $('roundEndContinue').focus({ preventScroll: true });
}
function continueRoundEnd() {
  if (!roundEndNext) return;
  const next = roundEndNext; roundEndNext = null;
  $('roundEnd').classList.remove('show');
  next();
}
$('roundEndContinue').addEventListener('click', continueRoundEnd);

/* ============ FINAL SPRINT ============ */
function startSprint() {
  state.phase = 'sprint';
  resetAtomsForRound('sprint');
  runnerGroup.userData.targetX = HOME_BASE_X;
  hunterGroup.userData.targetX = HUNTER_X;
  setTagText('tagYou', '▲ ' + state.className);
  state.sprintNetScore = 0;
  state.sprintTarget = classTarget();
  state.sprintTimeLeft = 60;
  hostSay('hostSprint', `Final Sprint! Sixty seconds. Every correct team is a step. You need ${state.sprintTarget}!`, 'point', 1800);

  paintSprintTarget();

  $('sprintCard').hidden = false;
  showHud('sprint');
  updateSprintTimer();

  clearInterval(state.sprintTimerHandle);
  state.sprintFrozen = false;
  state.sprintTimerHandle = setInterval(() => {
    if (state.sprintFrozen) return;      // target reached: the clock stops
    state.sprintTimeLeft -= 0.1;
    if (state.sprintTimeLeft <= 0) {
      state.sprintTimeLeft = 0;
      clearInterval(state.sprintTimerHandle);
      resolveSprint();
    }
    updateSprintTimer();
  }, 100);

  askSprintQuestion();
}

function paintSprintTarget() {
  $('sprintTarget').textContent = `${state.sprintNetScore} of ${state.sprintTarget} steps · one for each correct team (pot ${state.pot} points)`;
}
function updateSprintTimer() {
  const secs = Math.max(0, Math.ceil(state.sprintTimeLeft));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  $('sprintTimer').textContent = m + ':' + String(s).padStart(2, '0');
  $('sprintTimer').classList.toggle('low', state.sprintTimeLeft <= 10);
  const fill = $('sprintFill');
  fill.style.width = Math.max(0, state.sprintTimeLeft / 60 * 100) + '%';
  fill.classList.toggle('low', state.sprintTimeLeft <= 10);
}
function askSprintQuestion() {
  if (state.sprintTimeLeft <= 0) return;
  const q = pickQuestion();
  state.currentQuestion = q;
  $('sprintQTag').textContent = q.subject + ' · every team answers';
  $('sprintQText').textContent = q.q;
  $('sprintQAnswer').textContent = q.a;
  $('sprintQAnswer').classList.remove('shown'); $('sprintVerdict').textContent = '';
  paintBoards(); roundSprint.think();
}

/* early: the class reached the target before the clock ran out */
function resolveSprint(early) {
  if (state.phase !== 'sprint') return;
  state.phase = 'finish';
  clearInterval(state.sprintTimerHandle);
  roundSprint.stop(); clearTimeout(sprintNextT);
  if (!early) SFX.timeUp();
  $('sprintCard').hidden = true;
  const won = state.sprintNetScore >= state.sprintTarget;
  if (won) triggerEscapeSequence(() => showSummary(true));
  else triggerCaughtSequence(() => showSummary(false));
}

/* ============ SUMMARY ============ */
function showSummary(escaped) {
  state.phase = 'summary';
  $('finalVerdict').textContent = escaped ? `Escaped! The class keeps all ${state.pot} points.` : 'Caught! The pot is wiped.';
  $('finalVerdict').className = 'op-verdict ' + (escaped ? 'escaped' : 'caught');
  hostSay('hostSum', escaped ? `${state.className} outpaced the Hunter! Brilliant teamwork.` : 'The Hunter got you this time. Great effort, everyone!', escaped ? 'cheer' : 'shrug', 2200);
  const deals = state.dealOutcomes;
  $('summaryPlayers').innerHTML = `<div class="op-sum-p op-sum-class"><h3>${escapeHtml(state.className)}</h3>${deals.map(d => `<div class="hint">${escapeHtml(d.label)}: ${d.escaped ? 'escaped' : 'caught'}, banked ${d.reward} points.</div>`).join('')}<div class="hint">Final Sprint: ${state.sprintNetScore} of ${state.sprintTarget} steps.</div></div>${misc.html(5)}`;
  showScreen('summary');
  $('playAgainBtn').focus({ preventScroll: true });
}
$('playAgainBtn').addEventListener('click', startGame);
$('setupBtn').addEventListener('click', goHome);
$('menuBtn2').addEventListener('click', () => CGB.app.requestLauncher());

function goHome() {
  if (CGB.fitSetups) CGB.fitSetups();
  clearTimers(); roundDeal.stop(); roundSprint.stop(); clearTimeout(sprintNextT); gate.hide();
  state.phase = 'home';
  roundEndNext = null;
  $('roundEnd').classList.remove('show');
  resetAtomsForRound('deal');
  runnerCellIndex = TRACK_STEPS; hunterCellIndex = TRACK_STEPS + 3;
  buildTrack(hunterCellIndex + 1);
  runnerGroup.position.x = runnerGroup.userData.targetX = cellX(runnerCellIndex);
  hunterGroup.position.x = hunterGroup.userData.targetX = cellX(hunterCellIndex);
  showScreen('home');
  renderPack();
}

/* ============ TOP BAR, SETTINGS, KEYBOARD ============ */
function paintMute() { $('muteBtn').textContent = CGB.settings.get('sound') ? 'Sound on' : 'Sound off'; $('muteBtn').setAttribute('aria-pressed', String(!CGB.settings.get('sound'))); }
$('muteBtn').addEventListener('click', () => { CGB.settings.set('sound', !CGB.settings.get('sound')); CGB.sfx.unlock(); });
$('menuBtn').addEventListener('click', () => CGB.app.requestLauncher());
$('settingsBtn').addEventListener('click', () => CGB.modal.open('settingsModal'));
CGB.settings.onChange(k => { if (k === 'quality') applyQuality(); if (k === 'sound') paintMute(); });
paintMute();

document.addEventListener('keydown', e => {
  if (!active || CGB.modal.isOpen() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { e.preventDefault(); CGB.app.requestLauncher(); return; }
  const isButton = e.target.matches && e.target.matches('button');
  const inField = e.target.matches && e.target.matches('input, textarea, select');
  const k = e.key.toLowerCase();
  if (roundEndNext) {
    if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); continueRoundEnd(); }
    return;
  }
  if (state.phase === 'home') {
    if (k === 'enter' && !(e.target.matches && e.target.matches('button, select, textarea, summary'))) { e.preventDefault(); startGame(); }
    return;
  }
  if (state.phase === 'summary') {
    if (k === 'enter' && !isButton) { e.preventDefault(); startGame(); }
    return;
  }
  if (inField) return;
  if (state.phase === 'deal' && !state.dealReward) {
    if (['1', '2', '3'].includes(k)) chooseDeal(TIERS[+k - 1]);
    return;
  }
  if (state.phase === 'deal' || state.phase === 'sprint') {
    const r = state.phase === 'deal' ? roundDeal : roundSprint;
    if (r.handleKey(k)) { e.preventDefault(); return; }
    if ((k === 'enter' || k === ' ') && !isButton && state.phase === 'deal' && roundDeal.phase === 'done') { e.preventDefault(); classDealNext(); }
  }
});

goHome();
resize();

/* @test-only: shortcuts for tests, removed from the shipped file by build.js */
CGB.test.outpace = {
  setTime(sec) { state.sprintTimeLeft = sec; },
  // skip the Deal Round: start the Final Sprint with this pot
  toSprint(pot) { clearTimers(); roundEndNext = null; $('roundEnd').classList.remove('show'); state.pot = pot == null ? 600 : pot; startSprint(); }
};
/* @end-test-only */

return {
  enter() {
    active = true;
    applyQuality();
    clock.start();
    rafId = requestAnimationFrame(animate);
    if (state.phase === 'home') { renderGroupNames(true); renderPack(); $('startBtn').focus({ preventScroll: true }); }
  },
  exit() {
    active = false;
    cancelAnimationFrame(rafId);
    if (state.phase !== 'home') goHome();
  },
  inProgress: () => ['deal', 'dealEnd', 'sprint', 'finish'].includes(state.phase),
  _state: () => ({ round: state.phase === 'sprint' ? roundSprint.phase : roundDeal.phase, undoable: (state.phase === 'sprint' ? roundSprint : roundDeal).undoable, times: (state.phase === 'sprint' ? roundSprint : roundDeal).times(), runner: runnerCellIndex, hunter: hunterCellIndex, net: state.sprintNetScore, target: state.sprintTarget, dealRound: state.dealRound, misconceptions: misc.top(5).map(x => x.q.q), look: currentTheme, perfLevel: perf.level, phase: state.phase, teams: state.players.map(p => ({ name: p.name })), pot: state.pot, roundEnd: !!roundEndNext, timeLeft: state.sprintTimeLeft, frozen: state.sprintFrozen, q: state.currentQuestion, dealReward: state.dealReward })
};
}

CGB.registerGame('outpace', {
  title: 'Outpace',
  init(root) { game = createOutpace(root); },
  enter() { if (game) game.enter(); },
  exit() { if (game) game.exit(); },
  inProgress() { return !!(game && game.inProgress()); },
  state() { return game && game._state(); }
});
})();
