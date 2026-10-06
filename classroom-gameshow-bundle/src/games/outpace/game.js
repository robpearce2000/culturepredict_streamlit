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
/* A browser drawing 3D without the graphics chip (hardware acceleration off, common on managed
   school laptops) starts lean rather than waiting to stutter first */
try {
  const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info');
  const name = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  if (/swiftshader|llvmpipe|softpipe|software|basic render/i.test(name)) perf.level = 1;
} catch (e) { /* unknown: start at full quality */ }
const opPixelRatio = () => 1;   // drawn at 1:1 (with antialiasing): the biggest saving on high-resolution laptop screens
renderer.setPixelRatio(opPixelRatio());
renderer.shadowMap.enabled = false;
wrap.insertBefore(renderer.domElement, wrap.firstChild);

/* No full-screen glow (bloom) pass: it was the costliest thing drawn each frame on school laptops.
   The glow comes from cheap see-through sprites instead (see glowSprite). */
const useBloom = false;

scene.add(new THREE.AmbientLight(0x3A3F7E, 1.35));   // a little brighter: there are no point lights now
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(4, 8, 5);
scene.add(keyLight);
/* Only two lights (ambient and one directional), and simple Lambert shading: each extra light and
   physically based shading cost every pixel of every surface. Glows are additive sprites. */
function litMat(o) { const c = Object.assign({}, o || {}); delete c.roughness; delete c.metalness; delete c.envMapIntensity; return new THREE.MeshLambertMaterial(c); }
const glowTex = (() => {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'), gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
})();
function glowSprite(color, size) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.5 }));
  sp.scale.setScalar(size); scene.add(sp); return sp;
}
// a stand-in for a light: something that glows, with an "intensity" the effects can set
function glowLight(color, size, per) {
  const sp = glowSprite(color, size); let v = 0;
  Object.defineProperty(sp, 'intensity', { get: () => v, set: x => { v = x; sp.visible = x > 0.05; sp.material.opacity = Math.min(1, x / per); sp.scale.setScalar(size * (0.6 + Math.min(1, x / per) * 0.8)); } });
  sp.color = sp.material.color;
  return sp;
}
const runnerGlow = glowSprite(COL.runner, 2.2), hunterGlow = glowSprite(COL.hunter, 2.4);

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
  litMat({ color: 0x0D1230, roughness: 0.95, map: makeHexGridTexture() })
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
  const protonMat = litMat({ color: protonColor, emissive: protonColor, emissiveIntensity: 0.6, roughness: 0.3, metalness: 0.2 });
  const neutronMat = litMat({ color: neutronColor, emissive: neutronColor, emissiveIntensity: 0.3, roughness: 0.5, metalness: 0.1 });
  const partGeo = new THREE.SphereGeometry(0.1, 14, 14);
  for (let i = 0; i < count; i++) {
    const mesh = new THREE.Mesh(partGeo, i % 2 === 0 ? protonMat : neutronMat);
    const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
    mesh.position.copy(dir.multiplyScalar(Math.random() * spread));
    group.add(mesh);
    particlesArr.push({ mesh, base: mesh.position.clone(), phase: Math.random() * Math.PI * 2 });
  }
  const shell = new THREE.Mesh(new THREE.SphereGeometry(spread + 0.14, 20, 20), litMat({ color: protonColor, transparent: true, opacity: 0.12, roughness: 1 }));
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
    const electronMat = litMat({ color: config.electronColor, emissive: config.color, emissiveIntensity: 1 });
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
   Chemistry and Physics share the science look (atoms); Other has the general look. */
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
const glowMat = (hex, k) => litMat({ color: hex, emissive: hex, emissiveIntensity: k == null ? 0.55 : k, roughness: 0.35, metalness: 0.25 });
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
  history(isHunter) {
    const group = new THREE.Group();
    const core = new THREE.Group();
    let hands = null;
    if (isHunter) {
      // a ticking clock: time is running out
      const face = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.06, 32), litMat({ color: 0x2A1036, emissive: 0x3B0F4A, emissiveIntensity: 0.5, roughness: 0.6 }));
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
      const frame = glowMat(COL.runner, 0.4), sand = litMat({ color: 0xFFF1C2, emissive: 0xC9A040, emissiveIntensity: 0.4, roughness: 0.7, transparent: true, opacity: 0.9 });
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
      const cm = litMat({ color: 0x7A3F8C, emissive: COL.hunter, emissiveIntensity: 0.3, roughness: 0.9 });
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
      const globe = new THREE.Mesh(new THREE.SphereGeometry(0.26, 32, 20), litMat({ map: new THREE.CanvasTexture(c), emissive: 0x153a52, emissiveIntensity: 0.6, roughness: 0.6 }));
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
const LOOK_FOR_SUBJECT = { biology: 'science', chemistry: 'science', physics: 'science', maths: 'maths', history: 'history', geography: 'geography' };   // Other: the general look
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
  trimBevelMat.color.setHex(th.cell).offsetHSL(0, 0, 0.1);   // the tiles' raised tops follow the look
  const old = floor.material.map; floor.material.map = themeFloorTexture(th); floor.material.color.set(th.floorBg); floor.material.needsUpdate = true; if (old) old.dispose();
  warmUp();
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
const flareLight = glowLight(COL.spark, 4.5, 16);
flareLight.intensity = 0;

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
const SPRING_DAMPING = 0.72;   // a little bounce on arrival, without sliding back on screen
/* Critically damped follow (as game engines do it): eases in and out, never overshoots, and
   gives the same path at any frame rate. v is a small object holding the velocity. */
function smoothDamp(cur, target, v, key, time, dt) {
  const o = 2 / time, x = o * dt, e = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const ch = cur - target, tmp = ((v[key] || 0) + o * ch) * dt;
  v[key] = ((v[key] || 0) - o * tmp) * e;
  return target + (ch + tmp) * e;
}
const camVel = {};
const CAM_TIME = 0.45;   // seconds for the camera to settle: unhurried but never lagging behind a move
/* The racers' spring, in steps of a quarter of a 60 Hz frame whatever the screen's frame rate
   (velocity is in scene units per 60 Hz frame, as the surge and lunge kicks are) */
function springStep(grp, vel, f) {
  const n = Math.max(1, Math.ceil(f * 4)), s = f / n, damp = Math.pow(SPRING_DAMPING, s);
  for (let i = 0; i < n; i++) {
    vel += (grp.userData.targetX - grp.position.x) * SPRING_K * s;
    vel *= damp;
    grp.position.x += vel * s;
  }
  return vel;
}
let shakeAmt = 0;
const shake = v => { if (!reduced()) shakeAmt = Math.max(shakeAmt, v); };

/* Deal Round track: discrete cells, the Hunter and the runner hop between them */
const CELL_SPACING = 1.55;
const trackGroup = new THREE.Group();
scene.add(trackGroup);
function cellX(i) { return HOME_BASE_X + i * CELL_SPACING; }
const trimGeo = new THREE.BoxGeometry(CELL_SPACING - 0.14 - 0.04, 0.05, 0.06);
const cellGeo = new THREE.BoxGeometry(CELL_SPACING - 0.14, 0.3, 0.96);   // chunky step tiles
const bevelGeo = new THREE.BoxGeometry(CELL_SPACING - 0.24, 0.04, 0.86);   // a raised, lighter top: reads as a bevelled edge
const cellMat = litMat({ color: 0x252C6B, roughness: 0.55, metalness: 0.15 });
const homeMat = litMat({ color: 0x12A4A0, emissive: 0x0A4F4D, roughness: 0.4, metalness: 0.2 });
const trimMat = litMat({ color: 0x9AA2F0, emissive: 0x5A63C8, emissiveIntensity: 0.5, roughness: 0.3 });
const homeTrimMat = litMat({ color: 0x9FFCF0, emissive: 0x4FF0D8, emissiveIntensity: 0.8, roughness: 0.3 });
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
const tileTopGeo = new THREE.PlaneGeometry(CELL_SPACING - 0.26, 0.84);
const trimBevelMat = litMat({ color: 0x3A4396, roughness: 0.4, metalness: 0.2 });
const railMat = litMat({ color: 0x9AA2F0, emissive: 0x5A63C8, emissiveIntensity: 0.9, roughness: 0.3 });
const plinthMat = litMat({ color: 0x0A0D22, roughness: 0.8 });
/* The finish arch at home: two chunky pillars and a beam across the track, studded with bulbs,
   with a HOME sign facing the class. The bulbs run a light show when the class gets home. */
const finishGroup = new THREE.Group();
const archBulbSets = [];     // one instanced mesh of bulbs per arch: a single draw call each
const BULB_REST = [0xFFC93C, 0x9FFCF0, 0xFFFFFF].map(c => new THREE.Color(c));
function buildArch(group, label) {
  const pillarMat = litMat({ color: 0x0A6663, emissive: 0x0A4F4D, emissiveIntensity: 0.6, roughness: 0.35, metalness: 0.3 });
  const glow = litMat({ color: 0x9FFCF0, emissive: 0x2BD9C2, emissiveIntensity: 1.1, roughness: 0.3 });
  const pillarGeo = new THREE.BoxGeometry(0.26, 2.3, 0.26);
  [-0.66, 0.66].forEach(z => {
    const m = new THREE.Mesh(pillarGeo, pillarMat); m.position.set(0, 1.0, z); group.add(m);
    const strip = new THREE.Mesh(new THREE.BoxGeometry(0.05, 2.1, 0.05), glow); strip.position.set(0, 1.0, z + (z > 0 ? 0.14 : -0.14)); group.add(strip);
  });
  const beam = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 1.66), pillarMat); beam.position.set(0, 2.25, 0); group.add(beam);
  const spots = [];
  for (let k = 0; k < 7; k++) spots.push([0, 2.44, -0.66 + k * 0.22]);
  for (let k = 0; k < 5; k++) [-0.66, 0.66].forEach(z => spots.push([0.15, 0.25 + k * 0.42, z]));
  const bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.055, 8, 6), new THREE.MeshBasicMaterial(), spots.length);
  const d = new THREE.Object3D();
  spots.forEach((p, i) => { d.position.set(p[0], p[1], p[2]); d.updateMatrix(); bulbs.setMatrixAt(i, d.matrix); bulbs.setColorAt(i, BULB_REST[i % 3]); });
  group.add(bulbs); archBulbSets.push(bulbs);
  const c = document.createElement('canvas'); c.width = 512; c.height = 160;
  const g = c.getContext('2d');
  g.fillStyle = '#0A6663'; g.fillRect(0, 0, 512, 160);
  g.fillStyle = '#9FFCF0'; g.fillRect(0, 0, 512, 12); g.fillRect(0, 148, 512, 12);
  g.fillStyle = '#fff'; g.font = '108px "Lilita One", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(label, 256, 86);
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.53), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }));
  sign.position.set(0, 2.72, 0.2);
  group.add(sign);
}
buildArch(finishGroup, 'HOME');
function buildTrack(totalCells) {
  trackGroup.clear();
  finishGroup.position.set(cellX(0) - CELL_SPACING / 2 + 0.02, 0, 0);
  trackGroup.add(finishGroup);
  for (let i = 0; i < totalCells; i++) {
    const mesh = new THREE.Mesh(cellGeo, i === 0 ? homeMat : cellMat);
    mesh.position.set(cellX(i), 0.0, 0);
    trackGroup.add(mesh);
    const bevel = new THREE.Mesh(bevelGeo, i === 0 ? homeTrimMat : trimBevelMat);
    bevel.position.set(cellX(i), 0.17, 0);
    trackGroup.add(bevel);
    const top = new THREE.Mesh(tileTopGeo, new THREE.MeshBasicMaterial({ map: tileTexture(i), transparent: true, depthWrite: false }));
    top.rotation.x = -Math.PI / 2;
    top.position.set(cellX(i), 0.196, 0);
    trackGroup.add(top);
  }
  // side rails and a plinth along the whole track
  const len = totalCells * CELL_SPACING, midX = cellX(0) + (totalCells - 1) * CELL_SPACING / 2;
  [-0.56, 0.56].forEach(z => { const r = new THREE.Mesh(new THREE.BoxGeometry(len, 0.1, 0.08), railMat); r.position.set(midX, 0.2, z); trackGroup.add(r); });
  const plinth = new THREE.Mesh(new THREE.BoxGeometry(len + 0.4, 0.12, 1.4), plinthMat); plinth.position.set(midX, -0.2, 0); trackGroup.add(plinth);
}
/* ============ THE STUDIO SET ============
   A backdrop with light strips, two lighting rigs with soft beams, a Final Sprint track with
   its own finish arch, and a countdown clock built into the set. All of it is unlit or uses the
   existing lights (no new lights, no shadows), so it costs little to draw. */
const setGroup = new THREE.Group();
scene.add(setGroup);
const stripMat = new THREE.MeshBasicMaterial({ color: 0x5A63C8 });
const stripRest = new THREE.Color(0x5A63C8), stripRed = new THREE.Color(0xFF3B4E), stripGold = new THREE.Color(0xFFC93C);
(function buildSet() {
  // a neon line along the foot of the backdrop (it turns red for the last ten seconds of the sprint)
  const line = new THREE.Mesh(new THREE.BoxGeometry(44, 0.08, 0.08), stripMat); line.position.set(0, 0.2, -6.8); setGroup.add(line);
  // two lighting rigs overhead with soft beams falling on the track
  const truss = litMat({ color: 0x2A2F55, roughness: 0.6, metalness: 0.6 });
  const beamMat = new THREE.MeshBasicMaterial({ color: 0xBFC6FF, transparent: true, opacity: 0.07, depthWrite: false, blending: THREE.AdditiveBlending });
  const coneGeo = new THREE.ConeGeometry(1.2, 4.6, 20, 1, true);
  [-1.4, 1.6].forEach(z => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(26, 0.14, 0.14), truss); bar.position.set(0, 5.2, z - 2.2); setGroup.add(bar);
  });
  [-5.5, -1.5, 2.5, 6.5].forEach((x, i) => {
    const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.22, 0.3, 12), truss); lamp.position.set(x, 4.95, i % 2 ? -0.6 : -3.6); setGroup.add(lamp);
    const cone = new THREE.Mesh(coneGeo, beamMat); cone.position.set(x, 2.75, (i % 2 ? -0.6 : -3.6) * 0.6); cone.userData.beam = true; setGroup.add(cone);
  });
})();
const beams = setGroup.children.filter(o => o.userData.beam);
// the soft light beams are see-through, which is the costly kind of drawing: High graphics only,
// and off as soon as the automatic step-down starts
const showBeams = () => beams.forEach(b => { b.visible = CGB.settings.get('quality') !== 'low' && perf.level < 1; });
showBeams();

/* Final Sprint track: from the start line to the target, marked in steps, with its own arch */
const sprintGroup = new THREE.Group();
scene.add(sprintGroup);
const sprintArch = new THREE.Group();
buildArch(sprintArch, 'HOME');
// the Final Sprint has no track: just the two racers (the arch is not shown; its position marks home for the escape)
function buildSprintTrack(target) {
  const startX = HOME_BASE_X, endX = runnerTargetX(target, target);
  sprintArch.position.set(endX - 0.45, 0, 0);
  setClock.position.set((startX + endX) / 2 + 1.0, 2.35, -2.0);   // over the track, behind the race
}

/* The countdown clock built into the set for the Final Sprint; it turns red in the last ten seconds */
const clockCanvas = document.createElement('canvas'); clockCanvas.width = 256; clockCanvas.height = 112;
const clockTex = new THREE.CanvasTexture(clockCanvas);
const setClock = new THREE.Group();
(function buildClock() {
  const frame = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.92, 0.12), litMat({ color: 0x14183A, roughness: 0.5, metalness: 0.4 }));
  setClock.add(frame);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(1.76, 0.78), new THREE.MeshBasicMaterial({ map: clockTex }));
  face.position.z = 0.065; setClock.add(face);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2.2, 8), litMat({ color: 0x2A2F55, metalness: 0.6, roughness: 0.5 }));
  pole.position.y = -1.5; setClock.add(pole);
})();
setClock.visible = false;
scene.add(setClock);
let clockShown = '';
function paintSetClock(secs, low) {
  const txt = Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
  if (txt + low === clockShown) return;
  clockShown = txt + low;
  const g = clockCanvas.getContext('2d');
  g.fillStyle = low ? '#3A0710' : '#070A1E'; g.fillRect(0, 0, 256, 112);
  g.fillStyle = low ? '#FF4D5E' : '#FFC93C'; g.font = '88px "Lilita One", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(txt, 128, 60);
  clockTex.needsUpdate = true;
}

/* Lighting mood: normal, red for the last ten seconds of the sprint, gold for an escape */
const ambientLight = scene.children.find(o => o.isAmbientLight);
const ambientRest = ambientLight ? ambientLight.color.clone() : null, keyRest = keyLight.color.clone();
let mood = 'normal';
function setMood(m) {
  if (m === mood) return;
  mood = m;
  const red = m === 'red', gold = m === 'gold';
  stripMat.color.copy(red ? stripRed : gold ? stripGold : stripRest);
  keyLight.color.copy(red ? new THREE.Color(0xFF8A8A) : keyRest);
  if (ambientLight) ambientLight.color.copy(red ? new THREE.Color(0x6A2A3E) : ambientRest);
  beams.forEach(b => b.material.color.setHex(red ? 0xFF6B7A : gold ? 0xFFE08A : 0xBFC6FF));
}

/* Confetti for an escape: one instanced mesh, so it is a single draw call */
const CONFETTI = 140;
const confettiMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.07, 0.12), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), CONFETTI);
confettiMesh.visible = false;
confettiMesh.frustumCulled = false;
scene.add(confettiMesh);
const confetti = [], confettiDummy = new THREE.Object3D();
const confettiCols = [0xFFC93C, 0x2BD9C2, 0xFF7A59, 0xFFFFFF, 0xE879F9].map(h => new THREE.Color(h));
for (let i = 0; i < CONFETTI; i++) { confetti.push({ p: new THREE.Vector3(), v: new THREE.Vector3(), r: 0, rv: 0 }); confettiMesh.setColorAt(i, confettiCols[i % confettiCols.length]); }
let confettiT = 0;
function burstConfetti(x) {
  if (reduced()) return;
  const n = CGB.settings.get('quality') === 'low' ? 60 : CONFETTI;
  confettiMesh.count = n;
  confetti.forEach((c, i) => {
    c.p.set(x + (Math.random() - 0.5) * 0.4, 2.4, (Math.random() - 0.5) * 1.2);
    c.v.set((Math.random() - 0.7) * 0.09, 0.03 + Math.random() * 0.08, (Math.random() - 0.5) * 0.06);
    c.r = Math.random() * 6; c.rv = (Math.random() - 0.5) * 0.4;
  });
  confettiMesh.visible = true; confettiT = 2.6;
}
function updateConfetti(dt) {
  if (!confettiMesh.visible) return;
  confettiT -= dt;
  const f = dt * 60;
  if (confettiT <= 0) { confettiMesh.visible = false; return; }
  for (let i = 0; i < confettiMesh.count; i++) {
    const c = confetti[i];
    c.v.y -= 0.0022 * f; c.v.multiplyScalar(Math.pow(0.985, f)); c.p.addScaledVector(c.v, f); c.r += c.rv * f;
    confettiDummy.position.copy(c.p); confettiDummy.rotation.set(c.r, c.r * 0.7, 0);
    confettiDummy.updateMatrix(); confettiMesh.setMatrixAt(i, confettiDummy.matrix);
  }
  confettiMesh.instanceMatrix.needsUpdate = true;
}
let lightShow = 0;
const showCol = new THREE.Color();
function updateArchLights(t, dt) {
  if (lightShow <= 0) return;
  lightShow -= dt;
  const done = lightShow <= 0;
  archBulbSets.forEach(m => {
    for (let i = 0; i < m.count; i++) m.setColorAt(i, done ? BULB_REST[i % 3] : showCol.setHSL((t * 0.9 + i * 0.07) % 1, 1, 0.62));
    m.instanceColor.needsUpdate = true;
  });
}

/* Surge and lunge: a streak behind whoever moves, for half a second */
function makeTrail(color) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.2, 1.6, 10), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false }));
  m.rotation.z = Math.PI / 2; scene.add(m); m.userData.life = 0; return m;
}
const runnerTrail = makeTrail(COL.runner), hunterTrail = makeTrail(COL.hunter);
let hunterLunge = 0;   // a short jump towards the runner, then back (sprint), in scene units
function updateTrails(dt) {
  [[runnerTrail, runnerGroup, 1], [hunterTrail, hunterGroup, 1]].forEach(([m, grp, side]) => {
    if (m.userData.life <= 0) { m.visible = false; return; }
    m.userData.life -= dt;
    m.visible = grp.visible;
    const dir = m.userData.dir || 1;
    m.position.set(grp.position.x + dir * 0.95 * side, grp.position.y, 0);
    m.material.opacity = Math.max(0, m.userData.life / 0.5) * 0.55;
  });
}
function streak(m, dir) { if (reduced()) return; m.userData.life = 0.5; m.userData.dir = dir; m.rotation.z = dir > 0 ? Math.PI / 2 : -Math.PI / 2; }

/* The gap meter in the panel: home, the class, the Hunter and the steps between them */
function paintGap() {
  if (currentRoundType === 'deal') {
    const n = TRACK_STEPS + MAX_GAP + 1, gap = Math.max(0, hunterCellIndex - runnerCellIndex);
    let cells = '';
    for (let i = 0; i < n; i++) {
      const cls = i === 0 ? 'home' : i === runnerCellIndex ? 'you' : i === hunterCellIndex ? 'hunter' : i > runnerCellIndex && i < hunterCellIndex ? 'gap' : '';
      cells += `<i class="${cls}"></i>`;
    }
    $('dealGap').innerHTML = `<span class="op-gapnum"><b>${gap}</b> step${gap === 1 ? '' : 's'} ahead of the Hunter</span><span class="op-gapbar" aria-hidden="true">${cells}</span><span class="op-gapnum"><b>${runnerCellIndex}</b> to home</span>`;
  } else {
    const left = Math.max(0, state.sprintTarget - state.sprintNetScore);
    $('sprintGap').innerHTML = `<span class="op-gapnum"><b>${left}</b> step${left === 1 ? '' : 's'} to home</span>`;
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
  sprintGroup.visible = (roundType === 'sprint');
  setClock.visible = false;   // the Final Sprint shows one timer, above the question (no clock or pole in the set)
  setMood('normal'); confettiMesh.visible = false; lightShow = 0; hunterLunge = 0; slowMo = 1;
  runnerTrail.userData.life = hunterTrail.userData.life = 0;
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
  streak(runnerTrail, 1);                         // the class surges forward a step
  if (currentRoundType === 'sprint') {
    tetherPulse = 1;
    tetherTargetColor.setHex(COL.runner);
    spawnPulse(true);
  }
}
function pulseWrong() {
  shake(0.1);
  streak(hunterTrail, 1);                         // the Hunter lunges, with a burst of speed
  if (currentRoundType === 'deal') hunterVelX -= reduced() ? 0 : 0.22;
  else if (!reduced()) hunterLunge = 1;
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

  // the camera swings round the catch in slow motion (still, with reduced motion)
  explodeX = collideX;
  slowMo = reduced() ? 1 : 0.28;

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
    escapeConfetti = false;
    escapeBaseX = runnerGroup.position.x;
    escapeYankIndex = 0;
  }
}
/* Space or Enter skips a catch or escape sequence: everything settles at once */
function skipSequence() {
  if (!onSequenceComplete || !/^(buildup|explode|deal-caught|deal-escape|escape)$/.test(sceneMode)) return false;
  explosionParts.forEach(p => { p.mesh.visible = false; });
  shockwave.visible = false; victoryRing.visible = false; tether.visible = false;
  flareLight.intensity = 0; escapeStreakMat.opacity = 0; confettiMesh.visible = false; lightShow = 0.01;
  flashEl.style.opacity = 0;
  finishSequence();
  return true;
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
let lastFrame = performance.now(), frameT = 0, ease = k => k, slowMo = 1, lookX = 0, explodeX = 0, escapeConfetti = false;
const V3 = new THREE.Vector3();
/* Warm-up: the first time anything is drawn, the browser uploads its geometry and pictures to
   the graphics chip and waits for it, which showed as a jolt whenever the camera swung part of
   the set into view (or an effect appeared) for the first time. So once after each change of
   look, and when the sprint track is built, everything is drawn once with nothing hidden or
   culled; the real frame is drawn straight after, in the same frame, so it is never seen. */
let warmPending = true;
const warmUp = () => { warmPending = true; };
function warmScene() {
  warmPending = false;
  const saved = [];
  scene.traverse(o => { saved.push([o, o.visible, o.frustumCulled]); o.visible = true; o.frustumCulled = false; });
  try { renderer.compile(scene, camera); renderer.render(scene, camera); }
  finally { saved.forEach(([o, v, c]) => { o.visible = v; o.frustumCulled = c; }); }
}
function animate(ts) {
  if (!active) return;
  rafId = requestAnimationFrame(animate);
  // every movement runs on the frame's own timestamp, scaled to a 60 Hz frame (f = 1 at 60 Hz),
  // so the camera and racers glide at the same speed on 60, 120 and 144 Hz screens and an
  // uneven frame does not make them jump
  const nowMs = performance.now(), frameMs = typeof ts === 'number' ? ts : nowMs;
  const dt = Math.min(0.05, Math.max(0, (frameMs - lastFrame) / 1000));
  lastFrame = frameMs; frameT += dt;
  const t = frameT, f = dt * 60;
  ease = k => 1 - Math.pow(1 - k, f);
  if (warmPending) warmScene();
  updateConfetti(dt);
  updateArchLights(t, dt);
  updateTrails(dt);

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

  runnerGlow.position.copy(runnerGroup.position); runnerGlow.visible = runnerGroup.visible;
  hunterGlow.position.copy(hunterGroup.position); hunterGlow.visible = hunterGroup.visible;
  runnerGlow.material.opacity = 0.42 + Math.sin(t * 2) * 0.08;
  hunterGlow.material.opacity = 0.45 + Math.sin(t * 3) * 0.1;

  if (sceneMode !== 'idle') { camVel.x = camVel.y = camVel.z = camVel.l = 0; }
  if (sceneMode === 'idle') {
    runnerVelX = springStep(runnerGroup, runnerVelX, f);
    if (currentRoundType === 'deal') hunterVelX = springStep(hunterGroup, hunterVelX, f);

    if (currentRoundType === 'deal') {
      let fx, fy, fz, lx, ly = 0.35;
      if (state.dealReward && !reduced()) {
        // during the questions: beside and a little behind the class, with the Hunter in view
        // behind it and room ahead towards home, so the gap is plain to see
        // framed on where the racers are heading, so the camera makes one glide while they
        // bounce into place, instead of wobbling with them
        const rx = runnerGroup.userData.targetX, hx = hunterGroup.userData.targetX;
        const lo = Math.min(rx, hx) - 1.3, hi = Math.max(rx, hx) + 0.9, mid = (lo + hi) / 2;
        fz = framingZ(hi - lo - 1.2) * 0.95; fx = mid + 0.8; fy = 1.5 + fz * 0.14; lx = mid - 0.15; ly = 0.5;
      } else {
        // choosing the deal (and always with reduced motion): the whole track, from the finish
        // arch to the Hunter's furthest start, fills the width
        const f = dealFraming(); fx = f.x; fy = f.y; fz = f.z; lx = f.x;
      }
      camera.position.x = smoothDamp(camera.position.x, fx, camVel, 'x', CAM_TIME, dt);
      camera.position.y = smoothDamp(camera.position.y, fy, camVel, 'y', CAM_TIME, dt);
      camera.position.z = smoothDamp(camera.position.z, fz, camVel, 'z', CAM_TIME, dt);
      lookX = smoothDamp(lookX, lx, camVel, 'l', CAM_TIME, dt);
      camera.lookAt(lookX, ly, 0);
    } else {
      // Final Sprint: the class and the Hunter, from slightly behind the class; with reduced
      // motion one still view of the whole sprint track
      const wide = reduced();
      const endX = runnerTargetX(1, 1) - 1.2;
      const rT = runnerGroup.userData.targetX;   // where the class is heading, so the Hunter's lunges don't jolt the camera
      const lo = wide ? endX : Math.min(rT, HUNTER_X), hi = wide ? HUNTER_X + 0.8 : Math.max(rT, HUNTER_X);
      const midX = (lo + hi) / 2;
      const desiredZ = framingZ(hi - lo);
      camera.position.x = smoothDamp(camera.position.x, midX + (wide ? 0 : 0.5), camVel, 'x', CAM_TIME, dt);
      camera.position.y = smoothDamp(camera.position.y, 2.4, camVel, 'y', CAM_TIME, dt);
      camera.position.z = smoothDamp(camera.position.z, desiredZ, camVel, 'z', CAM_TIME, dt);
      lookX = smoothDamp(lookX, midX, camVel, 'l', CAM_TIME, dt);
      camera.lookAt(lookX, 0.6, 0);
      if (sceneMode === 'idle') {      // the Hunter's lunge on a missed question
        hunterLunge = Math.max(0, hunterLunge - dt * 2.4);
        hunterGroup.position.x = HUNTER_X - Math.sin(hunterLunge * Math.PI) * 0.7;
      }
    }

    if (shakeAmt > 0.001) shakeAmt *= Math.pow(0.82, f); else shakeAmt = 0;
  }

  if (currentRoundType === 'sprint' && (sceneMode === 'idle' || sceneMode === 'buildup')) {
    const from = hunterGroup.position, to = runnerGroup.position;
    const dist = Math.abs(to.x - from.x);
    tether.position.set((from.x + to.x) / 2, 0.6, 0);
    tether.scale.set(1, Math.max(0.001, dist), 1);
    tether.rotation.z = Math.PI / 2;
    tetherPulse *= Math.pow(0.94, f);
    tetherColor.lerp(tetherTargetColor, ease(0.06));
    tetherColor.lerp(tetherRest, ease(0.01));
    tetherMat.color.copy(tetherColor);
    tetherMat.opacity = 0.35 + tetherPulse * 0.5;
    tether.scale.x = tether.scale.z = 1 + tetherPulse * 2.5;

    pulses.forEach(p => {
      if (!p.active) return;
      p.progress += 0.045 * f;
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
    hunterGroup.position.x += (hunterGroup.userData.targetX - hunterGroup.position.x) * ease(0.25);
    shake(progress * 0.05);

    const midX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
    camera.position.x += (midX - camera.position.x) * ease(0.12);
    camera.position.z += ((3.4) - camera.position.z) * ease(0.06);
    camera.lookAt(midX, 0.6, 0);
    if (shakeAmt > 0.001) shakeAmt *= Math.pow(0.85, f);

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
    buildupBaseX += (midX - buildupBaseX) * ease(easedProgress * 0.1);
    hunterGroup.position.x += (midX - hunterGroup.position.x) * ease(easedProgress * 0.14);

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
    camera.position.lerp(V3.set(0, 1.4, 3.2), ease(0.02));
    if (shakeAmt > 0.001) shakeAmt *= Math.pow(0.88, f);

    if (progress >= 1) {
      const collideX = (runnerGroup.position.x + hunterGroup.position.x) / 2;
      triggerExplosionAt(collideX);
    }
  }

  if (sceneMode === 'explode') {
    const elapsed = (performance.now() - sequenceStart) / 1000;
    // slow motion for the first moments, easing back to full speed
    const sm = reduced() ? 1 : Math.min(1, slowMo + elapsed * 0.75);
    const simT = reduced() ? elapsed : Math.max(0, elapsed - (1 - slowMo) * Math.min(elapsed, 0.95) * 0.6);
    explosionParts.forEach(p => {
      p.mesh.position.addScaledVector(p.vel, sm * f);
      p.vel.y -= 0.002 * sm * f;
      p.vel.multiplyScalar(Math.pow(1 - 0.015 * sm, f));
      p.mesh.scale.multiplyScalar(Math.pow(1 - 0.035 * sm, f));
    });
    shockwave.scale.setScalar(1 + simT * 14);
    shockwaveMat.opacity = Math.max(0, 0.9 - simT * 1.3);
    flareLight.intensity = Math.max(0, 22 - simT * 40);
    if (reduced()) { camera.position.set(explodeX, 1.4, 3.4); camera.lookAt(explodeX, 0.6, 0); }
    else {
      const a = -0.5 + Math.min(1, elapsed / 2.0) * 0.9, r = 4.4, shk = Math.max(0, 0.55 - elapsed) * 0.5;
      camera.position.set(explodeX + Math.sin(a) * r + Math.sin(elapsed * 47) * shk * 0.5, 2.1 + elapsed * 0.15, Math.cos(a) * r);
      camera.lookAt(explodeX, 0.6, 0);
    }
    if (elapsed > (reduced() ? 1.6 : 2.1) && onSequenceComplete) {
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
    const px = runnerGroup.position.x, archX = finishGroup.position.x;
    if (!reduced()) {
      camera.position.x += (archX + 1.0 - camera.position.x) * ease(0.06);
      camera.position.z += (3.1 - camera.position.z) * ease(0.05);
      camera.position.y += (1.4 - camera.position.y) * ease(0.05);
      camera.lookAt(archX + 0.2, 1.0, 0);
    }
    // the class bursts through the finish arch
    if (elapsed >= BURST_AT) runnerGroup.position.x += (archX - 1.5 - runnerGroup.position.x) * ease(0.12);

    if (runnerChar) runnerChar.boost(zoomProgress * f);

    if (elapsed >= BURST_AT && !snapFlashDone) {
      snapFlashDone = true;
      burstConfetti(archX); lightShow = reduced() ? 0 : 2.4; setMood('gold');
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

    if (elapsed > (reduced() ? TOTAL_DURATION : 2.3) && onSequenceComplete) {
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
      tetherPulse *= Math.pow(0.94, f);
      tetherColor.lerp(tetherTargetColor, ease(0.08));
      tetherMat.color.copy(tetherColor);
      tetherMat.opacity = 0.4 + tetherPulse * 0.5;
      tether.scale.set(1 + tetherPulse * 2, dist, 1 + tetherPulse * 2);

      camera.position.lerp(V3.set((from.x + to.x) / 2 * 0.3, 1.8, 4.4), ease(0.03));
      if (shakeAmt > 0.001) shakeAmt *= Math.pow(0.85, f);

    } else if (elapsed < ESCAPE_BREAKFREE_DURATION + 0.18) {
      const snapProgress = (elapsed - ESCAPE_BREAKFREE_DURATION) / 0.18;
      if (!snapFlashDone) {
        snapFlashDone = true;
        flareLight.position.set(hunterGroup.position.x, 0.9, 0);
        flareLight.intensity = 12;
      }
      const from = hunterGroup.position;
      tether.visible = true;
      tether.position.lerp(V3.set(from.x, 0.6, 0), ease(0.55));
      tether.scale.y *= Math.pow(0.82, f);
      tetherMat.opacity = Math.max(0, 0.85 * (1 - snapProgress));
      flareLight.intensity *= Math.pow(0.8, f);
      if (snapProgress >= 1) tether.visible = false;

    } else {
      tether.visible = false;
      const launchElapsed = elapsed - ESCAPE_BREAKFREE_DURATION - 0.18;
      if (launchElapsed >= 0.1) {
        const progress = Math.min(1, (launchElapsed - 0.1) / 0.6);
        runnerGroup.position.x = escapeBaseX - progress * 6;
        escapeStreak.position.set(runnerGroup.position.x + 1, 0.6, 0);
        escapeStreakMat.opacity = 0.5 * (1 - progress);
        const archX = sprintArch.position.x;
        if (!escapeConfetti && runnerGroup.position.x < archX) { escapeConfetti = true; burstConfetti(archX); lightShow = reduced() ? 0 : 2.4; setMood('gold'); }
        if (!reduced()) { camera.position.lerp(V3.set(archX + 2.6, 1.7, 4.4), ease(0.05)); lookX += (archX + 0.8 - lookX) * ease(0.06); camera.lookAt(lookX, 0.9, 0); }
        if (progress >= 1 && launchElapsed > (reduced() ? 0.85 : 1.9) && onSequenceComplete) {
          runnerGroup.visible = false;
          escapeStreakMat.opacity = 0;
          finishSequence();
        }
      }
    }
  }

  if (++shiftFrame % 6 === 0) updateViewShift();
  slideView(dt);
  // the shake is a smooth sway added for this frame only (it used to jump to a new random spot
  // every frame, which read as stutter, and stayed baked into the camera's glide)
  const sx = shakeAmt * 0.5 * Math.sin(t * 31), sy = shakeAmt * 0.35 * Math.sin(t * 23 + 1.3);
  camera.position.x += sx; camera.position.y += sy;
  placeTags();
  renderer.render(scene, camera);
  camera.position.x -= sx; camera.position.y -= sy;
  /* @test-only */ frameStats.js += performance.now() - nowMs; frameStats.n++; /* @end-test-only */
}
/* @test-only */ const frameStats = { js: 0, n: 0 }; /* @end-test-only */
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
  const sx = (tagV.x + 1) / 2 * w;   // the camera's view offset is already in the projection
  const x = +Math.min(right - half, Math.max(left + half, sx)).toFixed(1);   // sub-pixel, so tags glide with the racers
  // keep tags inside the clear band so they never sit on the HUD panels
  const below = el !== $('tagYou');
  let y = (1 - tagV.y) / 2 * h;
  if (below) y = Math.max(band.top + 4, Math.min(band.bottom - el._h - 4, y));
  else y = Math.max(band.top + el._h + 4, Math.min(band.bottom - 4, y));
  y = +y.toFixed(1);
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
  viewWant.y = want; viewWant.x = wantX; viewWant.w = w; viewWant.h = h;
  if (reduced()) slideView(1);
}
/* ...and slide the picture there smoothly, every frame (it used to jump a quarter of the way
   every sixth frame, which shook the whole picture whenever the question panel changed size) */
const viewWant = { x: 0, y: 0, w: 0, h: 0 }, viewVel = {};
function slideView(dt) {
  const { w, h } = viewWant;
  if (!w || !h) return;
  if (reduced()) { viewShift = viewWant.y; viewShiftX = viewWant.x; viewVel.x = viewVel.y = 0; }
  else if (Math.abs(viewWant.y - viewShift) > 0.05 || Math.abs(viewWant.x - viewShiftX) > 0.05 || Math.abs(viewVel.y || 0) > 0.05 || Math.abs(viewVel.x || 0) > 0.05) {
    viewShift = smoothDamp(viewShift, viewWant.y, viewVel, 'y', 0.3, dt);
    viewShiftX = smoothDamp(viewShiftX, viewWant.x, viewVel, 'x', 0.3, dt);
  } else if (viewShift === viewWant.y && viewShiftX === viewWant.x && camera.view && camera.view.fullWidth === w) return;
  else { viewShift = viewWant.y; viewShiftX = viewWant.x; }
  if (Math.abs(viewShift) < 0.05 && Math.abs(viewShiftX) < 0.05) { if (camera.view) camera.clearViewOffset(); }
  else camera.setViewOffset(w, h, viewShiftX, viewShift, w, h);
}
let lastSize = '';
function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  if (viewShift || viewShiftX) camera.setViewOffset(w, h, viewShiftX, viewShift, w, h);
  camera.updateProjectionMatrix();
  // resizing the drawing buffer (even to the same size) makes the browser rebuild it and wait,
  // so only when the size or resolution has really changed
  const pr = renderer.getPixelRatio(), key = w + 'x' + h + '@' + pr;
  if (key !== lastSize) { lastSize = key; renderer.setSize(w, h, false); }
  measureTags();
}
function applyQuality() {
  showBeams();
  // level 1 (and Low graphics): no floating dust, set dressing or light beams; level 2: three-quarter resolution
  const lean = CGB.settings.get('quality') === 'low' || perf.level >= 1;
  ambientParticles.visible = !lean;
  setGroup.children.forEach(o => { if (!o.userData.beam) o.visible = !lean; });
  const pr = perf.level >= 2 ? 0.75 : opPixelRatio();
  if (pr !== renderer.getPixelRatio()) {
    renderer.setPixelRatio(pr);   // three.js resizes the buffer itself here, so resize() need not again
    const v = renderer.getSize(new THREE.Vector2()); lastSize = v.x + 'x' + v.y + '@' + pr;
  }
  resize();
}
/* Step quality down, one level at a time, as soon as frames stay slow (under about 45 a second on
   average) for a second and a half. The Graphics setting itself is untouched. */
function watchPerformance(now) {
  if (!perf.last) { perf.last = now; return; }
  const dt = Math.min(200, now - perf.last); perf.last = now;
  perf.ema += (dt - perf.ema) * 0.05;
  if (perf.level >= 2 || clock.getElapsedTime() < 3) return;
  perf.slowFor = perf.ema > 22 ? perf.slowFor + dt : 0;
  if (perf.slowFor > 1500) { perf.level++; perf.slowFor = 0; perf.ema = 16; applyQuality(); }
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
    if (!ok && hunterCellIndex - runnerCellIndex === 1) hostSay('hostDeal', 'The Hunter is right behind you!', 'gasp', 1500);
    else hostSay('hostDeal', CGB.classLine(c, n), CGB.classGesture(c, n), 1500);
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
  paintGap();
}

function askDealQuestion() {
  const q = pickQuestion();
  state.currentQuestion = q;
  $('dealQTag').textContent = q.subject + ' · ' + q.topic;
  $('dealQText').textContent = q.q;
  $('dealQAnswer').innerHTML = CGB.answerHTML(q);
  $('dealQAnswer').classList.remove('shown'); $('dealVerdict').textContent = '';
  paintBoards(); dealStatus(); roundDeal.think();
}

function finishDealRound(escaped) {
  const reward = escaped ? state.dealReward : Math.round(state.dealReward / 2);
  state.pot += reward;
  state.dealOutcomes.push({ label: `Deal Round ${state.dealRound + 1}`, escaped, reward });
  $('dealStatus').textContent = '';
  hostSay('hostDeal', escaped ? `Home safe! ${reward} points banked.` : `Caught! Still, ${reward} points go in the pot.`, escaped ? 'cheer' : 'groan', 2000);
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
  buildSprintTrack(state.sprintTarget); warmUp();
  paintSprintTarget();
  showHud('sprint');
  updateSprintTimer();
  $('sprintCard').hidden = true;
  // nothing ticks until the teacher presses Start: a short explanation first
  state.sprintReady = true;
  sprintGate.show(beginSprint, `You have <b>60 seconds</b>. Every team answers each question on its whiteboard, and each correct team moves the class <b>one step</b> towards home. Reach <b>${state.sprintTarget} steps</b> before the clock runs out to bank the pot of ${state.pot} points. If time runs out first, the Hunter catches you.`);
}
const sprintGate = CGB.createStartGate(document.getElementById('game-outpace'), '', { title: 'Final Sprint', label: 'Start the Final Sprint', className: 'sprint-gate' });
function beginSprint() {
  if (state.phase !== 'sprint' || !state.sprintReady) return;
  state.sprintReady = false;
  hostSay('hostSprint', `Final Sprint! Sixty seconds. Every correct team is a step. You need ${state.sprintTarget}!`, 'point', 1800);
  $('sprintCard').hidden = false;
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
  paintGap();
}
function updateSprintTimer() {
  const secs = Math.max(0, Math.ceil(state.sprintTimeLeft));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  $('sprintTimer').textContent = m + ':' + String(s).padStart(2, '0');
  $('sprintTimer').classList.toggle('low', state.sprintTimeLeft <= 10);
  paintSetClock(secs, state.sprintTimeLeft <= 10);
  if (state.phase === 'sprint') setMood(state.sprintTimeLeft <= 10 ? 'red' : 'normal');
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
  $('sprintQAnswer').innerHTML = CGB.answerHTML(q);
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
  clearTimers(); roundDeal.stop(); roundSprint.stop(); clearTimeout(sprintNextT); gate.hide(); sprintGate.hide(); state.sprintReady = false;
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
  if ((k === 'enter' || k === ' ') && skipSequence()) { e.preventDefault(); return; }
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
  frameStats() { const r = renderer.info.render, o = { calls: r.calls, triangles: r.triangles, programs: renderer.info.programs.length, lights: scene.children.filter(c => c.isLight).length, msPerFrame: frameStats.n ? +(frameStats.js / frameStats.n).toFixed(2) : 0, bloom: useBloom, pixelRatio: renderer.getPixelRatio(), perfLevel: perf.level }; frameStats.js = 0; frameStats.n = 0; return o; },
  motion() { const v = runnerGroup.position.clone().project(camera), hv = hunterGroup.position.clone().project(camera); return { cam: camera.position.toArray().map(n => +n.toFixed(4)), lookX: +lookX.toFixed(4), shift: [viewShift, viewShiftX], runner: [+runnerGroup.position.x.toFixed(4), +runnerGroup.position.y.toFixed(4)], rs: [+((v.x + 1) / 2 * stageSize.w).toFixed(2), +((1 - v.y) / 2 * stageSize.h).toFixed(2)], hs: +((hv.x + 1) / 2 * stageSize.w).toFixed(2), tag: [$('tagYou')._x, $('tagYou')._y], shake: +shakeAmt.toFixed(4), mode: sceneMode, fov: camera.fov, aspect: camera.aspect }; },
  scene3d() { return { mode: sceneMode, mood, clock: setClock.visible, arch: !!finishGroup.parent, sprintTrack: sprintGroup.visible && sprintGroup.children.length > 0, confetti: confettiMesh.visible, camera: camera.position.toArray() }; },
  // skip the Deal Round: start the Final Sprint with this pot
  toSprint(pot) { clearTimers(); roundEndNext = null; $('roundEnd').classList.remove('show'); state.pot = pot == null ? 600 : pot; startSprint(); }
};
/* @end-test-only */

return {
  enter() {
    active = true;
    applyQuality();
    warmUp();
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
