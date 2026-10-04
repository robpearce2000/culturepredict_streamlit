'use strict';
/* =========================================================
   OVER THE EDGE
   A 3D counter-pusher quiz game for two teams.
   Round 1: answer to win counters, drop them down lanes 1 to 4.
   Final: both teams join forces to push the jackpot counter over the edge.
   The physics and difficulty numbers are tuned; change with care.
   ========================================================= */
(function () {
let game = null;

function createOverTheEdge(root) {
const $ = id => document.getElementById('ote-' + id);
const escapeHtml = CGB.escapeHtml;
const bank = CGB.bank;
const picker = bank.createPicker();
const GAME_NAME = 'Over the Edge';

function pickQuestion(playerName) { return picker.pick(playerName, G.focusWeak); }
function loadWrongLog(n) { return bank.wrongLog(n); }

/* =========================================================
   PHYSICS: two shelves seen from above, plus the peg board
   The top shelf moves back and forth; counters on it are held
   by the back wall and spill off its front onto the bottom shelf.
   The top shelf's front face pushes the bottom shelf's counters.
   ========================================================= */
const PHY = { W: 480, D: 400, R: 18, JR: 34, PF_MIN: 40, PF_MAX: 128, PERIOD: 3.0, LOSTW: 64, FRIC: 900, E: 0.15, WALL: -40 };
function pusherFront(t) { const ph = (1 - Math.cos(2 * Math.PI * t / PHY.PERIOD)) / 2; return PHY.PF_MIN + (PHY.PF_MAX - PHY.PF_MIN) * ph; }
function makeTray() { return { lower: [], upper: [], t: 0, pf: PHY.PF_MIN, nextId: 1, onTransfer: null }; }
function mkBody(tray, x, y, o) {
  o = o || {}; const r = o.r || PHY.R;
  return { id: tray.nextId++, x, y, vx: 0, vy: 0, r, m: (r * r) / (PHY.R * PHY.R) * (o.heavy || 1), kind: o.kind || 'std', owner: o.owner == null ? -1 : o.owner, wildcard: o.wildcard || null, rot: CGB.random() * 6.28, mesh: null, anim: 0, base: null, rider: null, ox: 0, oy: 0, h: 0, hv: 0 };
}
function addLower(tray, x, y, o) { const b = mkBody(tray, x, y, o); tray.lower.push(b); return b; }
function addUpper(tray, x, y, o) { const b = mkBody(tray, x, y, o); tray.upper.push(b); return b; }
function solveTier(B, h, front) {
  for (const b of B) {
    if (b.base) continue;
    const sp = Math.hypot(b.vx, b.vy), dec = PHY.FRIC * h;
    if (sp <= dec) { b.vx = 0; b.vy = 0; } else { const k = (sp - dec) / sp; b.vx *= k; b.vy *= k; }
    b.x += b.vx * h; b.y += b.vy * h; b.rot += b.vx * h * 0.02;
  }
  for (let it = 0; it < 5; it++) {
    for (let i = 0; i < B.length; i++) {
      const a = B[i]; if (a.base) continue;
      for (let j = i + 1; j < B.length; j++) {
        const c = B[j]; if (c.base) continue;
        const dx = c.x - a.x, dy = c.y - a.y, min = a.r + c.r;
        if (Math.abs(dx) >= min || Math.abs(dy) >= min) continue;
        const d2 = dx * dx + dy * dy; if (d2 >= min * min) continue;
        const d = Math.sqrt(d2) || 0.001, nx = dx / d, ny = dy / d, pen = min - d;
        const wa = 1 / a.m, wc = 1 / c.m, cor = pen / (wa + wc);
        a.x -= nx * cor * wa; a.y -= ny * cor * wa; c.x += nx * cor * wc; c.y += ny * cor * wc;
        const vn = (c.vx - a.vx) * nx + (c.vy - a.vy) * ny;
        if (vn < 0) { const imp = -(1 + PHY.E) * vn / (wa + wc); a.vx -= imp * wa * nx; a.vy -= imp * wa * ny; c.vx += imp * wc * nx; c.vy += imp * wc * ny; }
      }
    }
    for (const b of B) {
      if (b.base) continue;
      if (b.x < b.r) { b.x = b.r; b.vx = Math.abs(b.vx) * 0.2; }
      if (b.x > PHY.W - b.r) { b.x = PHY.W - b.r; b.vx = -Math.abs(b.vx) * 0.2; }
      front(b);
    }
  }
}
function overlapInfo(B, x, y, r, skip) {
  let worst = 0, who = null;
  for (const b of B) { if (b.base || b === skip) continue; const p = b.r + r - Math.hypot(b.x - x, b.y - y); if (p > worst) { worst = p; who = b; } }
  return { worst, who };
}
function stepTray(tray, dt, onFall) {
  const SUB = 3, h = dt / SUB;
  for (let s = 0; s < SUB; s++) {
    tray.t += h;
    const prev = tray.pf, pf = pusherFront(tray.t), pv = (pf - prev) / h;
    tray.pf = pf;
    for (const b of tray.upper) if (!b.base) b.y += pf - prev;
    solveTier(tray.upper, h, b => { if (b.y - b.r < PHY.WALL) { b.y = PHY.WALL + b.r; if (b.vy < 0) b.vy = 0; } });
    solveTier(tray.lower, h, b => { if (b.y - b.r < pf) { b.y = pf + b.r; if (b.vy < pv) b.vy = Math.max(pv, 0); } });
    for (const T of [tray.upper, tray.lower]) for (const b of T) if (b.base) { b.x = b.base.x + b.ox; b.y = b.base.y + b.oy; }
    for (let i = tray.upper.length - 1; i >= 0; i--) {
      const b = tray.upper[i];
      if (b.base || b.y <= pf + b.r * 0.15) continue;
      tray.upper.splice(i, 1);
      b.y = pf + b.r + 1; b.vy = Math.max(b.vy, 40);
      tray.lower.push(b);
      const moved = [b];
      if (b.rider) { const r = b.rider, j = tray.upper.indexOf(r); if (j >= 0) { tray.upper.splice(j, 1); if (j < i) i--; } tray.lower.push(r); moved.push(r); }
      if (tray.onTransfer) moved.forEach(m => tray.onTransfer(m));
    }
    for (const T of [tray.lower, tray.upper]) for (const b of T) {
      if (!b.base || CGB.random() > 0.001) continue;
      const tx = b.base.x + (b.ox >= 0 ? 1 : -1) * (b.r * 2 + 1), ty = b.base.y + 4;
      if (tx > b.r && tx < PHY.W - b.r && overlapInfo(T, tx, ty, b.r, b).worst < 2) { b.base.rider = null; b.base = null; b.x = tx; b.y = ty; b.h = Math.max(b.h, 0.16); }
    }
  }
  const L = tray.lower;
  for (let i = L.length - 1; i >= 0; i--) {
    const b = L[i];
    if (b.base || b.y <= PHY.D + b.r * 0.1) continue;
    L.splice(i, 1);
    const ok = b.x >= PHY.LOSTW && b.x <= PHY.W - PHY.LOSTW;
    if (onFall) onFall(b, ok);
    if (b.rider) {
      const r = b.rider; r.base = null; b.rider = null;
      const j = L.indexOf(r); if (j >= 0) { L.splice(j, 1); if (j < i) i--; }
      if (onFall) onFall(r, ok);
    }
  }
}
function upperLanding(tray, x) {
  const r = PHY.R; x = Math.min(PHY.W - r, Math.max(r, x));
  if (CGB.random() < 0.25) {
    let best = null, bd = 1e9;
    for (const b of tray.upper) {
      if (b.base || b.rider || Math.abs(b.x - x) > r * 2) continue;
      const d = Math.abs(b.x - x) + Math.abs(b.y - (PHY.WALL + 40)) * 0.5;
      if (d < bd) { bd = d; best = b; }
    }
    if (best) return { x: best.x, y: best.y, onto: best };
  }
  for (let y = PHY.WALL + r + 2; y < tray.pf - r; y += 5) for (const off of [0, -9, 9, -18, 18]) {
    const xx = Math.min(PHY.W - r, Math.max(r, x + off));
    if (overlapInfo(tray.upper, xx, y, r).worst < r * 0.25) return { x: xx, y };
  }
  return { x, y: PHY.WALL + r + 2 };
}
function fillTray(tray) {
  const r = PHY.R, dx = r * 2.02, dy = r * 1.76;
  let row = 0;
  for (let y = PHY.PF_MAX + r + 1; y < PHY.D - r * 0.3; y += dy, row++)
    for (let x = r + (row % 2 ? dx / 2 : 1); x < PHY.W - r; x += dx) if (CGB.random() < 0.93) addLower(tray, x + (CGB.random() - 0.5) * 3, y + (CGB.random() - 0.5) * 3);
  row = 0;
  for (let y = PHY.WALL + r + 1; y < PHY.PF_MAX - r * 0.2; y += dy, row++)
    for (let x = r + (row % 2 ? dx / 2 : 1); x < PHY.W - r; x += dx) if (CGB.random() < 0.97) addUpper(tray, x, y);
  for (let i = 0; i < 240; i++) stepTray(tray, 1 / 60, null);
  // a few ready-made stacks
  for (let k = 0; k < 6; k++) {
    const T = k < 3 ? tray.upper : tray.lower;
    const free = T.filter(b => !b.base && !b.rider && b.kind === 'std');
    const pair = free.slice(0, -1)[Math.floor(CGB.random() * (free.length - 1))];
    if (!pair) continue;
    const top = (k < 3 ? addUpper : addLower)(tray, pair.x, pair.y);
    top.base = pair; pair.rider = top; top.ox = (CGB.random() - 0.5) * 12; top.oy = (CGB.random() - 0.5) * 12;
  }
}

const PEG = { W: 480, H: 360, R: 16, PR: 5, G: 2300, E: 0.36, CHUTES: [85, 190, 290, 395] };
const PEGS = (() => {
  const pegs = []; let row = 0;
  for (let y = 70; y < PEG.H - 30; y += 38, row++) {
    const off = row % 2 ? 25 : 0;
    for (let x = 15 + off; x <= PEG.W - 10; x += 50) pegs.push({ x, y, flash: 0 });
  }
  return pegs;
})();
function stepPegCoin(c, dt) {
  const SUB = 4, h = dt / SUB;
  for (let s = 0; s < SUB; s++) {
    c.vy += PEG.G * h; c.vx *= 0.999; c.x += c.vx * h; c.y += c.vy * h;
    for (const p of PEGS) {
      const dx = c.x - p.x, dy = c.y - p.y, min = PEG.R + PEG.PR;
      if (Math.abs(dx) > min || Math.abs(dy) > min) continue;
      const d = Math.hypot(dx, dy);
      if (d < min && d > 0) {
        const nx = dx / d, ny = dy / d;
        c.x = p.x + nx * min; c.y = p.y + ny * min;
        const vn = c.vx * nx + c.vy * ny;
        if (vn < 0) {
          c.vx -= (1 + PEG.E) * vn * nx; c.vy -= (1 + PEG.E) * vn * ny;
          c.vx += (CGB.random() - 0.5) * 22; c.spin += c.vx * 0.02;
          if (-vn > 60) { SFX.peg(); p.flash = 1; }
        }
        if (Math.abs(nx) < 0.08) c.vx += (CGB.random() < 0.5 ? -1 : 1) * 25;
      }
    }
    if (c.x < PEG.R) { c.x = PEG.R; c.vx = Math.abs(c.vx) * 0.5; }
    if (c.x > PEG.W - PEG.R) { c.x = PEG.W - PEG.R; c.vx = -Math.abs(c.vx) * 0.5; }
    const sp = Math.hypot(c.vx, c.vy);
    if (sp > 640) { c.vx *= 640 / sp; c.vy *= 640 / sp; }
  }
  c.ang += c.spin * dt;
  if (c.y > PEG.H + PEG.R) c.done = true;
}

/* =========================================================
   SOUND (synthesised with the shared engine, no files)
   ========================================================= */
const SFX = (() => {
  const tone = CGB.sfx.tone;
  let lastPeg = 0, lastClink = 0;
  return {
    peg() { const n = performance.now(); if (n - lastPeg < 45) return; lastPeg = n; tone(1500 + Math.random() * 900, 0.05, 'triangle', 0.04); },
    clink() { const n = performance.now(); if (n - lastClink < 60) return; lastClink = n; tone(2200 + Math.random() * 600, 0.06, 'triangle', 0.035); },
    land() { tone(170, 0.14, 'sine', 0.2, 0, 0.5); tone(1100, 0.04, 'triangle', 0.04); },
    win() { tone(880, 0.12, 'triangle', 0.1); tone(1320, 0.2, 'triangle', 0.1, 0.08); },
    lost() { tone(150, 0.32, 'sawtooth', 0.05, 0, 0.6); },
    correct: CGB.sfx.correct,
    wrong: CGB.sfx.wrong,
    wildcard() { [988, 1175, 1480, 1976].forEach((f, i) => tone(f, 0.14, 'sine', 0.08, i * 0.06)); },
    jackpot() { [523, 659, 784, 1047, 784, 1047, 1319, 1568].forEach((f, i) => tone(f, 0.3, 'triangle', 0.11, i * 0.11)); }
  };
})();

/* =========================================================
   GAME STATE
   ========================================================= */
const T2 = CGB.TEAMS.slice(0, 2);
const COLORS = { hex: T2.map(t => t.hex), css: T2.map(t => t.css), text: T2.map(t => t.text), mark: T2.map(t => t.mark) };
const VALUE = 100, JACKPOT_VALUE = 5000;
const WILDCARDS = [
  { id: 'bonus', title: 'Bonus counter', text: n => `${n} gets an extra drop` },
  { id: 'cash', title: 'Cash bonus', text: n => `+£250 for ${n}` },
  { id: 'steal', title: 'Steal', text: (n, o) => `${n} takes up to £150 from ${o}` }
];
const G = {
  phase: 'home', step: null, attract: true,
  players: [], team: 0, finalWinnings: 0,
  r1Each: 6, finalN: 12, finalDiff: 'normal', focusWeak: false,
  qIndex: 0, qTotal: 0, q: null, turn: 0, answerShown: false,
  dropper: -1, dropCount: 1, lastDropper: -1, pending: 0, landed: 0, settleAt: 0,
  dropWon: 0, dropLost: 0, dropNotes: [], pendingExtra: [], bonusDrop: false, cascadeSaid: false,
  jackpot: null, jackpotY0: 0, jackpotWon: false, jackpotFell: false, lastMsg: null
};
let tray = makeTray();

/* =========================================================
   THREE.JS SET
   ========================================================= */
const S = 0.025;
const UH = 0.85;                       // height of the moving top shelf
const UPPER_DEPTH = 260;               // depth of the top shelf, in physics units
const TX = x => (x - PHY.W / 2) * S, TZ = y => (y - 180) * S;
const BOARD_BOTTOM = UH + 0.6, BOARD_Z = TZ(PHY.WALL) - 0.3;
const BX = x => (x - PEG.W / 2) * S, BY = y => BOARD_BOTTOM + (PEG.H - y) * S;
const COIN_T = 0.16, JACK_T = 0.26;

/* Set palette (original to this game) */
const SET = { bg: 0x0B2A3A, aqua: 0x4FF0D8, coral: 0xFF6B4A, sun: 0xFFC93C, teal: 0x127C82, navy: 0x16324A };

const wrap = $('stageWrap');
const renderer = CGB.createRenderer();
if (!renderer) { $('nogl').hidden = false; $('nogl').innerHTML = CGB.noWebGLMessage; return null; }
renderer.setPixelRatio(CGB.settings.pixelRatio());
renderer.shadowMap.enabled = CGB.settings.get('quality') !== 'low';
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
wrap.insertBefore(renderer.domElement, wrap.firstChild);

const scene = new THREE.Scene();
scene.background = new THREE.Color(SET.bg);
scene.fog = new THREE.Fog(SET.bg, 40, 80);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 150);
const cam = { x: 2.2, y: 8.0, z: 21.5, lx: 2.2, ly: 3.0 };

const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));
const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(256, 256), 0.18, 0.35, 0.97);
composer.addPass(bloom);
let useBloom = CGB.settings.get('quality') !== 'low';

/* Studio reflections */
(function makeEnv() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 256;
  const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, '#e9fffb'); grd.addColorStop(0.4, '#5fb8b4'); grd.addColorStop(0.6, '#1d4f5e'); grd.addColorStop(1, '#0b2030');
  g.fillStyle = grd; g.fillRect(0, 0, 512, 256);
  g.fillStyle = '#ffffff'; g.fillRect(180, 20, 150, 30); g.fillRect(20, 60, 40, 80); g.fillRect(450, 60, 40, 80);
  g.fillStyle = '#ffb199'; g.fillRect(100, 70, 18, 110); g.fillRect(390, 70, 18, 110);
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromEquirectangular(tex).texture;
  tex.dispose(); pm.dispose();
})();

scene.add(new THREE.HemisphereLight(0xeafffb, 0x0b2a3a, 0.45));
scene.add(new THREE.AmbientLight(0xffffff, 0.08));
const key = new THREE.SpotLight(0xffffff, 0.78, 70, Math.PI / 4, 0.55, 1);
key.position.set(4, 18, 12); key.target.position.set(1.5, 0, 0);
key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004;
scene.add(key); scene.add(key.target);
const coralBack = new THREE.PointLight(0xff8a66, 0.55, 40); coralBack.position.set(0, 9, -10); scene.add(coralBack);
const aquaL = new THREE.PointLight(0x4ff0d8, 0.65, 26); aquaL.position.set(-10, 6, 4); scene.add(aquaL);

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.userData = { draw };
  return t;
}
function redrawTex(t) { const c = t.image; t.userData.draw(c.getContext('2d'), c.width, c.height); t.needsUpdate = true; }
const glow = (hex, k) => new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(k || 1.4) });
function box(w, h, d, mat, x, y, z, parent) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z); (parent || scene).add(m); return m;
}
function hazard(g, w, h, stripe) {
  g.fillStyle = '#FFC93C'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#1B1F3B';
  for (let x = -h; x < w + h; x += stripe * 2) { g.beginPath(); g.moveTo(x, h); g.lineTo(x + stripe, h); g.lineTo(x + stripe + h, 0); g.lineTo(x + h, 0); g.closePath(); g.fill(); }
}

/* Backdrop: deep teal studio wall with cliff shapes, LED dots and light strips */
const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(90, 40), new THREE.MeshBasicMaterial({ map: canvasTex(1024, 456, (g, w, h) => {
  const bg = g.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#06141d'); bg.addColorStop(0.4, '#0d4352'); bg.addColorStop(0.68, '#16707a'); bg.addColorStop(0.78, '#e2734f'); bg.addColorStop(0.86, '#0b2a3a'); bg.addColorStop(1, '#071a24');
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(79,240,216,0.22)';
  for (let y = 18; y < h * 0.55; y += 16) for (let x = 8 + ((y / 16) % 2) * 8; x < w; x += 16) { g.beginPath(); g.arc(x, y, 2, 0, Math.PI * 2); g.fill(); }
  const cliffs = [['#0a3240', [[0, 0.82], [0.08, 0.6], [0.16, 0.66], [0.26, 0.5], [0.34, 0.62], [0.4, 0.82]]], ['#0a3240', [[0.6, 0.82], [0.68, 0.56], [0.78, 0.64], [0.86, 0.48], [0.94, 0.6], [1, 0.55], [1, 0.82]]], ['#072530', [[0, 0.86], [0.2, 0.7], [0.5, 0.78], [0.8, 0.68], [1, 0.74], [1, 0.9], [0, 0.9]]]];
  cliffs.forEach(([col, pts]) => { g.fillStyle = col; g.beginPath(); pts.forEach(([px, py], i) => i ? g.lineTo(px * w, py * h) : g.moveTo(px * w, py * h)); g.closePath(); g.fill(); });
  for (let i = 0; i < 18; i++) {
    const x = (i + 0.5) * w / 18, cw = 6;
    const gr = g.createLinearGradient(0, h * 0.05, 0, h * 0.55);
    const col = i % 3 === 1 ? '255,138,102' : '79,240,216';
    gr.addColorStop(0, `rgba(${col},0)`); gr.addColorStop(0.5, `rgba(${col},0.75)`); gr.addColorStop(1, `rgba(${col},0)`);
    g.fillStyle = gr; g.fillRect(x - cw / 2, h * 0.05, cw, h * 0.5);
  }
}) }));
backdrop.position.set(2, 7, -22); scene.add(backdrop);
const towerMat = new THREE.MeshStandardMaterial({ color: SET.navy, metalness: 0.6, roughness: 0.35 });
const aquaGlow = glow(0x9ffcf0, 1.2), coralGlow = glow(0xff9a7a, 1.15);
[[-17, -12, 0.3], [-13.5, -9, -0.2], [17.5, -12, -0.3], [21.5, -9, 0.2], [14.5, -15, 0.1], [-21, -15, -0.1]].forEach(([x, z, ry], i) => {
  const p = box(1.3, 30, 1.3, towerMat, x, 8, z); p.rotation.y = ry;
  const s = box(0.22, 30, 0.22, i % 2 ? coralGlow : aquaGlow, x, 8, z + 0.72); s.rotation.y = ry;
});

/* Glossy studio floor with an aqua light ring */
const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 64), new THREE.MeshStandardMaterial({ color: 0x0a1f2b, metalness: 0.55, roughness: 0.24 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = -4.6; floor.receiveShadow = true; scene.add(floor);
const ring = new THREE.Mesh(new THREE.RingGeometry(11.5, 11.8, 96), glow(0x9ffcf0, 1.2));
ring.rotation.x = -Math.PI / 2; ring.position.set(1.5, -4.58, 0); scene.add(ring);

/* Machine materials */
const chrome = new THREE.MeshStandardMaterial({ color: 0xe8ecf2, metalness: 1.0, roughness: 0.18 });
const chromeDark = new THREE.MeshStandardMaterial({ color: 0x8c97a3, metalness: 1.0, roughness: 0.3 });
const sunTop = new THREE.MeshStandardMaterial({ color: SET.sun, metalness: 0.15, roughness: 0.4 });
const coralFace = new THREE.MeshStandardMaterial({ color: SET.coral, metalness: 0.2, roughness: 0.35 });
const tealPanel = new THREE.MeshStandardMaterial({ color: SET.teal, emissive: 0x08343a, emissiveIntensity: 0.5, metalness: 0.3, roughness: 0.35 });
const aquaNeon = glow(0xc9fff8, 1.3);
const acrylic = new THREE.MeshStandardMaterial({ color: 0xbff7f0, transparent: true, opacity: 0.2, metalness: 0.4, roughness: 0.05, depthWrite: false });
const hazardTex = canvasTex(1024, 32, (g, w, h) => hazard(g, w, h, 22));
const hazardMat = new THREE.MeshStandardMaterial({ map: hazardTex, metalness: 0.1, roughness: 0.5 });

/* Bottom shelf and cabinet */
const lowerShelf = box(12.3, 0.3, 11.4, [chrome, chrome, sunTop, chrome, hazardMat, chrome], 0, -0.15, -0.2);
lowerShelf.receiveShadow = true;
const cabinet = box(13.4, 3.9, 12.6, [tealPanel, tealPanel, chromeDark, chromeDark, tealPanel, tealPanel], 0, -2.65, -0.6);
cabinet.receiveShadow = true;
box(13.0, 0.12, 0.25, aquaNeon, 0, -0.62, 5.7);
box(9.0, 0.08, 1.2, new THREE.MeshStandardMaterial({ color: 0x08141c, roughness: 0.9 }), 0, -0.55, 6.15);
const winW = (PHY.W - 2 * PHY.LOSTW) * S, lostW = PHY.LOSTW * S;
box(winW, 0.06, 0.12, chrome, 0, 0.0, TZ(PHY.D) + 0.03);
[-1, 1].forEach(s => {
  const cx = s * (winW / 2 + lostW / 2);
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(lostW, 1.5), new THREE.MeshStandardMaterial({ color: 0x3a4148, metalness: 0.6, roughness: 0.35 }));
  plate.rotation.x = -Math.PI / 2; plate.position.set(cx, 0.003, TZ(PHY.D) - 0.75); plate.receiveShadow = true; scene.add(plate);
  box(lostW, 0.06, 0.12, chromeDark, cx, 0.0, TZ(PHY.D) + 0.03);
  // side walls: clear acrylic with chrome caps
  box(0.12, 1.5, 11.4, acrylic, s * 6.24, 0.75, -0.2);
  box(0.2, 0.12, 11.4, chrome, s * 6.24, 1.52, -0.2);
  box(0.18, 0.08, 11.4, aquaNeon, s * 6.36, 0.05, -0.2);
});

/* Moving top shelf (it is also the pusher for the bottom shelf) */
const pusher = new THREE.Group();
const upperDepth = UPPER_DEPTH * S;
const upperBody = box(12.3, UH, upperDepth, [chrome, chrome, sunTop, chrome, coralFace, chrome], 0, UH / 2, -upperDepth / 2, pusher);
upperBody.castShadow = true; upperBody.receiveShadow = true;
box(12.32, 0.08, 0.06, aquaNeon, 0, UH - 0.02, 0.01, pusher);
scene.add(pusher);

/* Back wall and peg board: dark slate with aqua lane lines */
box(12.9, 0.62, 0.6, chrome, 0, UH + 0.31, TZ(PHY.WALL) - 0.3).castShadow = true;
const boardBack = box(12.6, 9.3, 0.25, new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.0, roughness: 0.7, map: canvasTex(512, 380, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#24414d'); gr.addColorStop(1, '#152a33');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(255,255,255,0.04)';
  for (let y = 0; y < h; y += 12) for (let x = (y / 12) % 2 ? 6 : 0; x < w; x += 12) g.fillRect(x, y, 2, 2);
  g.strokeStyle = 'rgba(79,240,216,0.55)'; g.lineWidth = 3; g.setLineDash([10, 8]);
  for (let i = 1; i < 4; i++) { const x = i * w / 4; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
}) }), 0, BOARD_BOTTOM + 4.6, BOARD_Z - 0.38);
boardBack.receiveShadow = true;
[-1, 1].forEach(s => {
  box(0.55, 10.6, 1.2, tealPanel, s * 6.6, BOARD_BOTTOM + 5.0, BOARD_Z - 0.1);
  box(0.08, 10.6, 0.08, aquaNeon, s * 6.32, BOARD_BOTTOM + 5.0, BOARD_Z + 0.52);
  box(0.08, 10.6, 0.08, glow(0xff9a7a, 1.15), s * 6.88, BOARD_BOTTOM + 5.0, BOARD_Z + 0.52);
});

/* Header sign */
box(14.2, 1.9, 1.2, towerMat, 0, BY(0) + 1.75, BOARD_Z - 0.1);
function drawSign(g, w, h) {
  const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#16324a'); gr.addColorStop(1, '#0b1d2b');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
  g.save(); g.beginPath(); g.rect(0, 0, 150, h); g.rect(w - 150, 0, 150, h); g.clip(); hazard(g, w, h, 26); g.restore();
  g.font = '92px "Lilita One", "Arial Rounded MT Bold", Impact, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineJoin = 'round'; g.lineWidth = 14; g.strokeStyle = '#1B1F3B'; g.strokeText('OVER THE EDGE', w / 2, h / 2 + 6);
  g.fillStyle = '#FFC93C'; g.fillText('OVER THE EDGE', w / 2, h / 2 + 6);
}
const signTex = canvasTex(1024, 128, drawSign);
const sign = new THREE.Mesh(new THREE.PlaneGeometry(12.4, 1.5), new THREE.MeshBasicMaterial({ map: signTex }));
sign.position.set(0, BY(0) + 1.75, BOARD_Z + 0.52); scene.add(sign);
box(14.2, 0.08, 0.08, aquaNeon, 0, BY(0) + 0.82, BOARD_Z + 0.52);
box(14.2, 0.08, 0.08, aquaNeon, 0, BY(0) + 2.68, BOARD_Z + 0.52);
const glass = new THREE.Mesh(new THREE.PlaneGeometry(12.6, 9.3), new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.06, metalness: 1, roughness: 0.05, depthWrite: false }));
glass.position.set(0, BOARD_BOTTOM + 4.6, BOARD_Z + 0.42); scene.add(glass);

const brass = new THREE.MeshStandardMaterial({ color: 0xd9a441, metalness: 1, roughness: 0.25 });
const pegGeo = new THREE.CylinderGeometry(PEG.PR * S, PEG.PR * S, 0.75, 12); pegGeo.rotateX(Math.PI / 2);
const pegMesh = new THREE.InstancedMesh(pegGeo, brass, PEGS.length);
const tipGeo = new THREE.SphereGeometry(PEG.PR * S * 1.3, 12, 8);
const tipMats = [];
const dummy = new THREE.Object3D();
PEGS.forEach((p, i) => {
  dummy.position.set(BX(p.x), BY(p.y), BOARD_Z); dummy.updateMatrix(); pegMesh.setMatrixAt(i, dummy.matrix);
  const mat = new THREE.MeshStandardMaterial({ color: 0xf0d48a, metalness: 1, roughness: 0.2, emissive: SET.aqua, emissiveIntensity: 0 });
  const tip = new THREE.Mesh(tipGeo, mat); tip.position.set(BX(p.x), BY(p.y), BOARD_Z + 0.38); scene.add(tip);
  tipMats.push(mat);
});
pegMesh.castShadow = true; scene.add(pegMesh);

/* Lanes 1 to 4 */
const chuteMeshes = [], chuteMats = [];
const CHUTE_BASE = 0x0a5a60;
function numberSprite(n) {
  const t = canvasTex(128, 128, g => {
    g.font = '100px "Lilita One", "Arial Rounded MT Bold", Impact, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.lineJoin = 'round'; g.lineWidth = 14; g.strokeStyle = '#1B1F3B'; g.strokeText(String(n), 64, 70);
    g.fillStyle = '#ffffff'; g.fillText(String(n), 64, 70);
  });
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  sp.scale.set(0.95, 0.95, 1); return sp;
}
PEG.CHUTES.forEach((cx, i) => {
  const mat = new THREE.MeshStandardMaterial({ color: 0x7fe9df, emissive: CHUTE_BASE, emissiveIntensity: 0.3, metalness: 0.3, roughness: 0.3 });
  const m = box(1.6, 0.22, 0.75, mat, BX(cx), BY(0) + 0.2, BOARD_Z + 0.05);
  m.userData.chute = i; chuteMeshes.push(m); chuteMats.push(mat);
  const sp = numberSprite(i + 1); sp.position.set(BX(cx), BY(0) - 0.35, BOARD_Z + 0.5); scene.add(sp);
  m.userData.sprite = sp;
});
/* Canvas text needs the embedded font loaded first */
if (document.fonts && document.fonts.load) document.fonts.load('100px "Lilita One"').then(() => {
  chuteMeshes.forEach(m => redrawTex(m.userData.sprite.material.map));
  redrawTex(signTex);
}).catch(() => { /* fallback font stays */ });

/* Sparkles in the air */
const dustGeo = new THREE.BufferGeometry();
const dustPos = new Float32Array(200 * 3);
for (let i = 0; i < 200; i++) { dustPos[i * 3] = (Math.random() - 0.5) * 44; dustPos[i * 3 + 1] = Math.random() * 18 - 3; dustPos[i * 3 + 2] = (Math.random() - 0.5) * 20 - 8; }
dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xc9fff8, size: 0.07, transparent: true, opacity: 0.55 }));
scene.add(dust);

/* Counters */
const coinGeo = new THREE.CylinderGeometry(PHY.R * S, PHY.R * S, COIN_T, 44);
const jackGeo = new THREE.CylinderGeometry(PHY.JR * S, PHY.JR * S, JACK_T, 64);
function drawAtom(g, cx, cy, s, col) {
  g.strokeStyle = col; g.lineWidth = s * 0.07;
  for (let k = 0; k < 3; k++) { g.save(); g.translate(cx, cy); g.rotate(k * Math.PI / 3); g.beginPath(); g.ellipse(0, 0, s, s * 0.36, 0, 0, Math.PI * 2); g.stroke(); g.restore(); }
  g.fillStyle = col; g.beginPath(); g.arc(cx, cy, s * 0.15, 0, Math.PI * 2); g.fill();
}
function drawStar(g, cx, cy, r) {
  g.beginPath();
  for (let i = 0; i < 10; i++) { const a = i * Math.PI / 5 - Math.PI / 2, rr = i % 2 ? r * 0.45 : r; i ? g.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr) : g.moveTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr); }
  g.closePath();
}
function faceTexture(kind, owner) {
  return canvasTex(256, 256, (g, w) => {
    const c = w / 2;
    let base = ['#ffffff', '#c3cad3', '#7d8792'];
    if (kind === 'wildcard') base = ['#d9fffa', '#2fc9bf', '#0a6663'];
    if (kind === 'jackpot') base = ['#fff6cf', '#f4c24a', '#9a6a10'];
    const gr = g.createRadialGradient(c * 0.7, c * 0.6, 10, c, c, c);
    gr.addColorStop(0, base[0]); gr.addColorStop(0.55, base[1]); gr.addColorStop(1, base[2]);
    g.fillStyle = gr; g.fillRect(0, 0, w, w);
    g.lineWidth = 6; g.strokeStyle = 'rgba(0,0,0,0.22)'; g.beginPath(); g.arc(c, c, c * 0.86, 0, Math.PI * 2); g.stroke();
    g.lineWidth = 3; g.strokeStyle = 'rgba(255,255,255,0.6)'; g.beginPath(); g.arc(c, c, c * 0.8, 0, Math.PI * 2); g.stroke();
    if (owner >= 0) { g.lineWidth = 18; g.strokeStyle = COLORS.css[owner]; g.beginPath(); g.arc(c, c, c * 0.93, 0, Math.PI * 2); g.stroke(); }
    if (kind === 'wildcard') {
      drawStar(g, c, c + 4, 82); g.fillStyle = '#ffffff'; g.fill(); g.lineWidth = 8; g.strokeStyle = '#1B1F3B'; g.stroke();
    } else if (kind === 'jackpot') {
      drawAtom(g, c, c - 20, 62, 'rgba(110,70,5,0.75)');
      g.fillStyle = 'rgba(110,70,5,0.9)'; g.font = 'bold 34px Impact, sans-serif'; g.textAlign = 'center'; g.fillText('JACKPOT', c, c + 78);
    } else {
      drawAtom(g, c, c, 66, 'rgba(60,70,85,0.45)');
    }
  });
}
const MAT = {};
function coinMats(kind, owner) {
  const k = kind + ':' + owner;
  if (MAT[k]) return MAT[k];
  const sideColor = kind === 'jackpot' ? 0xf2c14e : kind === 'wildcard' ? 0x12a4a0 : owner >= 0 ? COLORS.hex[owner] : 0xd5dbe2;
  const side = new THREE.MeshStandardMaterial({ color: sideColor, metalness: 0.95, roughness: 0.22 });
  const face = new THREE.MeshStandardMaterial({ map: faceTexture(kind, owner), metalness: 0.55, roughness: 0.35 });
  if (kind === 'wildcard') { side.emissive = new THREE.Color(0x064442); face.emissive = new THREE.Color(0x05302e); }
  if (kind === 'jackpot') { side.emissive = new THREE.Color(0x4a3200); face.emissive = new THREE.Color(0x2a1c00); }
  MAT[k] = [side, face, face];
  return MAT[k];
}
function makeCoinMesh(kind, owner) {
  const m = new THREE.Mesh(kind === 'jackpot' ? jackGeo : coinGeo, coinMats(kind, owner));
  m.castShadow = true; m.receiveShadow = true;
  scene.add(m);
  return m;
}

/* =========================================================
   THE HOST (shared Showtime mascot) with speech captions
   ========================================================= */
const host = CGB.createHost2D($('host'), { className: 'host-ote' });
/* The machine's outline on screen: bottom shelf (with its front edge), lanes, peg board and
   header sign. The host is placed and sized so he never covers any of it. */
const machineParts = [lowerShelf, pusher, glass, sign];
const machineBox = new THREE.Box3(), corner = new THREE.Vector3();
/* The drawn figure (head, body and arms), in page coordinates */
function hostRect() {
  const r = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
  $('host').querySelectorAll('.host-head, .host-body, .host-arm-a, .host-arm-b').forEach(el => {
    const b = el.getBoundingClientRect(); if (!b.width) return;
    r.left = Math.min(r.left, b.left); r.right = Math.max(r.right, b.right); r.top = Math.min(r.top, b.top); r.bottom = Math.max(r.bottom, b.bottom);
  });
  return r;
}
/* Camera shots (the camera eases between them) */
const CAM = {
  play: { x: 2.2, y: 8.0, z: 21.5, lx: 2.2, ly: 3.0 },
  board: { x: 1.2, y: 8.8, z: 20.4, lx: 1.2, ly: 4.0 },
  jackpot: { x: 1.2, y: 7.0, z: 17.8, lx: 1.2, ly: 1.6 },
  home: { x: 3.0, y: 7.6, z: 21.5, lx: 3.6, ly: 3.2 }
};
const SWAY = 0.35;
/* Size and place the host in the space to the right of the machine. The machine's
   right edge is taken as the furthest it reaches in any camera shot (with the sway),
   and the host's widest gestures span from 0.16 of his height left of the art's left edge
   (pointing) to 0.76 right of it (shrugging), measured with every pose applied. */
const FIG_L = 0.16, FIG_R = 0.76, ART_W = 300 / 440;
let hostFit = { h: 0, hidden: false, machineRight: 0, shot: 'home', shotRight: null };
function rightEdgeFor(c) {
  const keep = camera.position.clone(), keepQ = camera.quaternion.clone();
  let right = 0;
  [-SWAY, SWAY].forEach(sw => {
    camera.position.set(c.x + sw, c.y, c.z); camera.lookAt(c.lx, c.ly, 0); camera.updateMatrixWorld();
    right = Math.max(right, machineRect().right);
  });
  camera.position.copy(keep); camera.quaternion.copy(keepQ); camera.updateMatrixWorld();
  return right;
}
/* Fit the host to the camera. He is sized for the further-reaching of the shot the camera is
   heading for and where the machine is on screen right now, checked every 150 ms, so he
   shrinks as soon as the machine comes his way and only grows once it has moved away. */
function fitHost(shot) {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  if (shot || hostFit.shotRight == null) { if (shot) hostFit.shot = shot; hostFit.shotRight = rightEdgeFor(CAM[hostFit.shot]); }
  const right = Math.max(hostFit.shotRight, machineRect().right + 4);
  const gapPx = 12, edge = 6;
  const H = Math.floor(Math.min(h * 0.66, 560, (w - edge - gapPx - right) / (FIG_L + FIG_R)));
  if (H === hostFit.h) return;
  const el = $('host');
  el.classList.toggle('growing', H > hostFit.h);   // shrink instantly, grow smoothly
  hostFit.h = H; hostFit.hidden = H < 140; hostFit.machineRight = right;
  el.classList.toggle('nofit', hostFit.hidden);
  if (hostFit.hidden) return;
  el.style.height = H + 'px';
  el.style.right = Math.ceil(edge + (FIG_R - ART_W) * H) + 'px';
}
function machineRect() {
  // each part is projected on its own: one box round them all would pull the top of the
  // board forward to the shelf's front edge and claim far more of the screen than it uses
  const w = wrap.clientWidth, h = wrap.clientHeight, r = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
  machineParts.forEach(o => {
    machineBox.setFromObject(o);
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? machineBox.max.x : machineBox.min.x, i & 2 ? machineBox.max.y : machineBox.min.y, i & 4 ? machineBox.max.z : machineBox.min.z).project(camera);
      const x = (corner.x + 1) / 2 * w, y = (1 - corner.y) / 2 * h;
      r.left = Math.min(r.left, x); r.right = Math.max(r.right, x); r.top = Math.min(r.top, y); r.bottom = Math.max(r.bottom, y);
    }
  });
  return r;
}
function hostGesture(name, ms) { host.gesture(name, ms); }
let hostCheck = 0;
function updateHost(force) {
  host.setRest((G.step === 'chute' || G.step === 'dropping') ? 'present' : 'idle');
  host.setLook(drops.length || transits.length || G.step === 'dropping' ? 'left' : '');
  // step aside when the setup or results card would cover him
  const now = performance.now();
  if (!force && now - hostCheck < 150) return;
  hostCheck = now;
  fitHost();
  const hr = $('host').getBoundingClientRect();
  let covered = false;
  ['home', 'summary'].forEach(id => {
    const ov = $(id); if (ov.classList.contains('hidden')) return;
    const c = ov.querySelector('.ote-card').getBoundingClientRect();
    if (c.right > hr.left + 8 && c.left < hr.right - 8 && c.bottom > hr.top && c.top < hr.bottom) covered = true;
  });
  $('host').classList.toggle('away', covered);
}

/* Speech bubble */
const bubble = $('bubble');
let typeTimer = null, hideTimer = null;
function hostSay(text, gesture, ms) {
  if (!text) return;
  if (gesture) hostGesture(gesture, ms || 1800);
  $('bubbleWho').textContent = CGB.hostCfg.name || 'Host';
  const el = $('bubbleText');
  clearInterval(typeTimer); clearTimeout(hideTimer);
  el.textContent = '';
  bubble.classList.add('show');
  let i = 0;
  if (CGB.settings.reduced()) { el.textContent = text; i = text.length; }
  host.talk(text.length * 32 + 150);
  typeTimer = setInterval(() => {
    i += 2; el.textContent = text.slice(0, i);
    if (i >= text.length) { clearInterval(typeTimer); el.textContent = text; }
  }, 64);
  hideTimer = setTimeout(() => bubble.classList.remove('show'), Math.max(3800, text.length * 75));
}
function hideBubble() { clearInterval(typeTimer); clearTimeout(hideTimer); bubble.classList.remove('show'); }
function placeBubble() {
  if (!bubble.classList.contains('show')) return;
  const hp = host.headPoint(); if (!hp) return;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const wr = wrap.getBoundingClientRect();
  const hx = hp.x - wr.left;
  let x = hx, y = hp.y - wr.top - 6;
  const bw = bubble.offsetWidth, bh = bubble.offsetHeight + 14;   // include the tail
  x = Math.min(w - bw / 2 - 10, Math.max(bw / 2 + 10, x));
  y = Math.max(bh + 60, Math.min(h - 10, y));   // keep clear of the menu bar
  // never sit on top of the round banner or behind a setup or results card
  const base = wrap.getBoundingClientRect();
  const rel = el => { const r = el.getBoundingClientRect(); return { l: r.left - base.left, t: r.top - base.top, r: r.right - base.left, b: r.bottom - base.top }; };
  const hits = (o, yy) => !(x + bw / 2 < o.l || x - bw / 2 > o.r || yy < o.t || yy - bh > o.b);
  let hidden = false;
  // things the bubble must not cover: the round banner and Marty's own head
  const avoid = [];
  const banner = $('banner');
  if (banner.classList.contains('show')) {
    const parts = [banner.querySelector('.verdict'), banner.querySelector('.detail')].filter(el => el.offsetParent).map(rel);
    if (parts.length) avoid.push({ l: Math.min(...parts.map(p => p.l)) - 8, t: Math.min(...parts.map(p => p.t)) - 8, r: Math.max(...parts.map(p => p.r)) + 8, b: Math.max(...parts.map(p => p.b)) + 8 });
  }
  const pipEl = $('host').querySelector('.host-body');   // his whole figure, head included
  if (pipEl && !$('host').classList.contains('away')) { const r = rel(pipEl); avoid.push({ l: r.l, t: r.t + 4, r: r.r, b: r.b }); }
  const clear = yy => yy - bh >= 56 && yy <= h - 10 && !avoid.some(o => hits(o, yy));
  if (!clear(y)) {
    // try just above or below each obstacle, nearest first
    const options = avoid.flatMap(o => [o.t - 2, o.b + bh + 2]).filter(clear).sort((p, q) => Math.abs(p - y) - Math.abs(q - y));
    if (options.length) y = options[0]; else hidden = true;
  }
  ['home', 'summary'].forEach(id => {
    const ov = $(id);
    if (ov.classList.contains('hidden')) return;
    const card = ov.querySelector('.ote-card');
    if (card && hits(rel(card), y)) hidden = true;
  });
  bubble.classList.toggle('muted', hidden);
  bubble.style.left = x + 'px'; bubble.style.top = y + 'px';
  // point the tail at the host even when the bubble is pushed in from the edge
  const tail = Math.max(18, Math.min(bw - 18, bw / 2 + (hx - x)));
  bubble.style.setProperty('--tail', tail + 'px');
}
const pick = a => a[Math.floor(CGB.random() * a.length)];
const LINES = {
  intro: ["Welcome to Over the Edge! Let's see who's been revising.", "Hello and welcome! Two teams, one machine, and a lot of revision."],
  ask: ["{n}, here's your question.", "This one's for you, {n}.", "Over to you, {n}.", "{n}, have a think about this one."],
  correct: ["That's right! Have a counter.", "Spot on, {n}!", "Correct! Nicely done.", "Yes! Textbook answer."],
  wrong: ["Ooh, not quite.", "Sorry {n}, that's not the one.", "Close, but no counter."],
  steal: ["{o}, can you steal it?", "Over to you, {o}. Steal it for a counter!"],
  chute: ["Which lane, {n}?", "Pick your lane, {n}.", "Lane 1, 2, 3 or 4, {n}?"],
  dropping: ["Here it comes...", "Come on, come on...", "Let's see where it lands..."],
  cascade: ["Look at that! They're pouring off!", "The machine's paying out!"],
  one: ["One over the edge! That's £100.", "There's one!"],
  some: ["{k} counters over the edge! Lovely.", "{k} of them! Brilliant drop."],
  none: ["Nothing that time.", "The machine's being stubborn.", "Not this time. It'll come."],
  edge: ["Ooh, it's right on the edge!", "So close! It's hanging there!", "That one's teetering..."],
  lost: ["Straight down the side, unlucky."],
  wildcard: ["A wildcard counter!", "Ooh, a wildcard!"],
  final: ["It's the final! Work together and push that jackpot over the edge.", "Team up, you two. There's £5,000 on that gold counter."],
  close: ["The jackpot's teetering!", "It's so close to the edge!"],
  jackpot: ["JACKPOT! Incredible! £5,000!"],
  noJackpot: ["So close! The jackpot lives to see another day.", "Not quite, but what a game."],
  outro: ["Great game, both of you. Well played!"]
};
function line(key, vars) {
  let t = pick(LINES[key] || ['']);
  Object.entries(vars || {}).forEach(([k, v]) => { t = t.split('{' + k + '}').join(v); });
  return t;
}

/* Floating labels */
const labelsEl = $('labels');
const labels = [];
let combo = null;
function popWin(amount, pos) {
  if (combo && combo.L.t < combo.L.life * 0.6 && labels.includes(combo.L)) {
    combo.total += amount; combo.count++;
    combo.L.el.textContent = '+' + fmt(combo.total) + (combo.count > 1 ? '  ×' + combo.count : '');
    combo.L.t = 0.12; combo.L.life = 2.0;
    combo.L.pos.x += (pos.x - combo.L.pos.x) / combo.count;
    combo.L.el.style.fontSize = Math.min(3.4, 2.1 + combo.count * 0.18) + 'rem';
    return;
  }
  popLabel('+' + fmt(amount), pos, '', 1.8);
  combo = { L: labels[labels.length - 1], total: amount, count: 1 };
}
let lostStack = 0, lostAt = 0;
function popLost(pos) {
  const now = performance.now();
  lostStack = now - lostAt < 900 ? lostStack + 1 : 0; lostAt = now;
  const p = pos.clone(); p.y += lostStack * 0.55;
  popLabel('Lost', p, 'small', 1.3);
}
function popLabel(text, pos, cls, life) {
  const el = document.createElement('div');
  el.className = 'flabel' + (cls ? ' ' + cls : '');
  el.textContent = text;
  labelsEl.appendChild(el);
  labels.push({ el, pos: pos.clone(), t: 0, life: life || 1.6 });
}
function clearLabels() { labels.forEach(L => L.el.remove()); labels.length = 0; combo = null; }
const tmpV = new THREE.Vector3();
function updateLabels(dt) {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  // keep labels clear of Marty: anything that would land on him moves to his left
  const body = labels.length && !$('host').classList.contains('away') ? $('host').querySelector('.host-body') : null;
  const base = body ? wrap.getBoundingClientRect() : null;
  const hb = body ? body.getBoundingClientRect() : null;
  const placed = [];
  for (let i = labels.length - 1; i >= 0; i--) {
    const L = labels[i]; L.t += dt;
    const k = L.t / L.life;
    if (k >= 1) { L.el.remove(); labels.splice(i, 1); continue; }
    tmpV.copy(L.pos); tmpV.y += k * 1.4; tmpV.project(camera);
    let lx = (tmpV.x + 1) / 2 * w;
    const ly = (1 - tmpV.y) / 2 * h;
    if (hb) {
      const half = L.el.offsetWidth / 2 + 6, hh = L.el.offsetHeight / 2;
      const left = hb.left - base.left, top = hb.top - base.top, bottom = hb.bottom - base.top;
      if (lx + half > left && ly + hh > top && ly - hh < bottom) lx = left - half;
    }
    // stack labels that would land on one another
    const lw = L.el.offsetWidth, lh = L.el.offsetHeight;
    let top = ly;
    for (let tries = 0; tries < 6; tries++) {
      const hit = placed.find(r => lx - lw / 2 < r.r + 4 && lx + lw / 2 > r.l - 4 && top - lh / 2 < r.b + 2 && top + lh / 2 > r.t - 2);
      if (!hit) break;
      top = hit.t - lh / 2 - 4;
    }
    placed.push({ l: lx - lw / 2, r: lx + lw / 2, t: top - lh / 2, b: top + lh / 2 });
    L.el.style.left = lx + 'px';
    L.el.style.top = top + 'px';
    L.el.style.opacity = k < 0.15 ? k / 0.15 : k > 0.7 ? (1 - k) / 0.3 : 1;
  }
}

/* Moving coins: peg board, flight to the top shelf, falling off */
const drops = [], transits = [], falling = [];
function launchDrop(chute, owner, onLand, rnd) {
  rnd = rnd || CGB.random;
  const c = { x: PEG.CHUTES[chute] + (rnd() - 0.5) * 10, y: 4, vx: (rnd() - 0.5) * 30, vy: 0, spin: (rnd() - 0.5) * 4, ang: 0, done: false, owner, onLand };
  c.mesh = makeCoinMesh('std', owner);
  drops.push(c);
  chuteMats[chute].emissive.setHex(owner >= 0 ? COLORS.hex[owner] : SET.aqua); chuteMats[chute].emissiveIntensity = 1.2;
  setTimeout(() => { if (G.step !== 'chute') resetChuteGlow(); }, 400);
}
function updateDrops(dt) {
  for (let i = drops.length - 1; i >= 0; i--) {
    const c = drops[i];
    stepPegCoin(c, dt);
    c.mesh.position.set(BX(c.x), BY(c.y), BOARD_Z + 0.02);
    c.mesh.rotation.set(Math.PI / 2, c.ang, 0);
    if (c.done) {
      drops.splice(i, 1);
      const L = upperLanding(tray, c.x);
      const onto = L.onto && tray.upper.includes(L.onto) && !L.onto.rider ? L.onto : null;
      transits.push({ mesh: c.mesh, from: c.mesh.position.clone(), to: new THREE.Vector3(TX(L.x), UH + COIN_T / 2 + (onto ? COIN_T : 0), TZ(L.y)), L, onto, t: 0, dur: 0.38, owner: c.owner, ang: c.ang, onLand: c.onLand });
    }
  }
  PEGS.forEach((p, i) => {
    if (p.flash > 0) { p.flash = Math.max(0, p.flash - dt * 3); tipMats[i].emissiveIntensity = p.flash * 2.2; }
  });
}
function updateTransits(dt) {
  for (let i = transits.length - 1; i >= 0; i--) {
    const T = transits[i]; T.t += dt;
    const k = Math.min(1, T.t / T.dur);
    // the target may have moved with the shelf while we were in the air
    const L = T.onto && tray.upper.includes(T.onto) ? { x: T.onto.x, y: T.onto.y } : T.L;
    T.to.x = TX(L.x); T.to.z = TZ(L.y);
    T.mesh.position.lerpVectors(T.from, T.to, k);
    T.mesh.position.y = T.from.y + (T.to.y - T.from.y) * k * k;
    T.mesh.rotation.set(Math.PI / 2 * (1 - k), T.ang, 0);
    if (k >= 1) {
      transits.splice(i, 1);
      let b;
      if (T.onto && tray.upper.includes(T.onto) && !T.onto.rider) {
        b = addUpper(tray, T.onto.x, T.onto.y, { owner: T.owner });
        b.base = T.onto; T.onto.rider = b; b.ox = (CGB.random() - 0.5) * 12; b.oy = (CGB.random() - 0.5) * 12;
      } else {
        const LL = T.onto ? upperLanding(tray, T.L.x) : T.L;
        b = addUpper(tray, LL.x, LL.y, { owner: T.owner });
        b.vy = 30;
      }
      b.mesh = T.mesh; b.rot = T.ang;
      SFX.land();
      if (T.onLand) T.onLand(b);
    }
  }
}
function updateFalling(dt) {
  for (let i = falling.length - 1; i >= 0; i--) {
    const f = falling[i];
    f.vy -= 24 * dt;
    f.mesh.position.x += f.vx * dt; f.mesh.position.y += f.vy * dt; f.mesh.position.z += f.vz * dt;
    f.mesh.rotation.x += f.spin * dt;
    if (f.mesh.position.y < f.floorY) { scene.remove(f.mesh); falling.splice(i, 1); }
  }
}
tray.onTransfer = b => { b.h = UH; b.hv = 0; };
function syncTray(dt) {
  const pf = tray.pf;
  for (const b of tray.upper) {
    if (!b.mesh) continue;
    const t = COIN_T;
    const edge = pf - b.r * 0.55;
    const tilt = !b.base && b.y > edge ? Math.min(1, (b.y - edge) / (b.r * 0.7)) : 0;
    const y = UH + t / 2 + (b.base ? t : 0) - tilt * 0.05;
    b.mesh.position.set(TX(b.x), y, TZ(b.y));
    b.mesh.rotation.set(tilt * 0.3 + (b.base ? 0.08 : 0), b.rot, b.base ? 0.06 : 0);
  }
  for (const b of tray.lower) {
    if (!b.mesh) continue;
    if (b.h > 0) { b.hv -= 22 * dt; b.h += b.hv * dt; if (b.h <= 0) { b.h = 0; b.hv = 0; SFX.clink(); } }
    const t = b.kind === 'jackpot' ? JACK_T : COIN_T;
    const edge = PHY.D - b.r * 0.6;
    const tilt = !b.base && b.y > edge ? Math.min(1, (b.y - edge) / (b.r * 0.7)) : 0;
    let y = t / 2 - tilt * 0.06 + b.h;
    if (b.base) y = (b.base.kind === 'jackpot' ? JACK_T : COIN_T) + t / 2 + Math.max(b.h, b.base.h);
    if (b.anim > 0) y += b.anim * b.anim * 7;
    b.mesh.position.set(TX(b.x), y, TZ(b.y));
    const baseTilt = b.base && b.base.y > PHY.D - b.base.r * 0.6 ? Math.min(1, (b.base.y - (PHY.D - b.base.r * 0.6)) / (b.base.r * 0.7)) : 0;
    b.mesh.rotation.set((b.base ? baseTilt : tilt) * 0.3 + (b.base ? 0.08 : 0), b.rot, b.base ? 0.06 : 0);
  }
}

/* =========================================================
   TRAY LIFECYCLE
   ========================================================= */
function clearTray() {
  tray.upper.concat(tray.lower).forEach(b => { if (b.mesh) scene.remove(b.mesh); });
  drops.forEach(c => scene.remove(c.mesh)); drops.length = 0;
  transits.forEach(T => scene.remove(T.mesh)); transits.length = 0;
  falling.forEach(f => scene.remove(f.mesh)); falling.length = 0;
  tray = makeTray();
  tray.onTransfer = b => { b.h = UH; b.hv = 0; };
}
function newTray(withWildcards) {
  clearTray();
  fillTray(tray);
  if (withWildcards) {
    const cand = tray.lower.filter(b => !b.base && !b.rider && b.y > 220 && b.y < 360 && b.x > 90 && b.x < 390);
    const ids = WILDCARDS.map(m => m.id).sort(() => CGB.random() - 0.5);
    for (let i = 0; i < 3 && cand.length; i++) {
      const b = cand.splice(Math.floor(CGB.random() * cand.length), 1)[0];
      b.kind = 'wildcard'; b.wildcard = ids[i];
    }
  }
  tray.upper.concat(tray.lower).forEach(b => { b.mesh = makeCoinMesh(b.kind, b.owner); });
  syncTray(0);
}

function onTrayFall(b, inWin) {
  if (b.mesh) {
    falling.push({ mesh: b.mesh, vx: b.vx * S * 0.5, vy: 0.5, vz: 1.6 + Math.max(0, b.vy) * S + Math.random() * 0.6, spin: 5 + Math.random() * 4, floorY: inWin ? -0.6 : -7 });
  }
  if (G.attract || G.phase === 'summary') return;   // nothing scores once the results are showing
  const pos = new THREE.Vector3(TX(b.x), 0.5, TZ(PHY.D) + 0.4);
  if (b.kind === 'jackpot') { onJackpotFall(inWin, pos); return; }
  if (!inWin) { SFX.lost(); popLost(pos); G.dropLost++; renderScores(); return; }
  SFX.win();
  G.dropWon++;
  const who = G.lastDropper;
  if (G.phase === 'final') { G.team += VALUE; G.finalWinnings += VALUE; popWin(VALUE, pos); }
  else if (who >= 0) { G.players[who].money += VALUE; G.players[who].won++; popWin(VALUE, pos); }
  if (G.dropWon >= 3 && !G.cascadeSaid && G.step === 'dropping') { G.cascadeSaid = true; hostSay(line('cascade'), 'cheer', 2200); }
  if (b.wildcard) triggerWildcard(b.wildcard, who, pos);
  renderScores();
}
function triggerWildcard(id, who, pos) {
  SFX.wildcard();
  const def = WILDCARDS.find(m => m.id === id);
  if (G.phase === 'final' || who < 0) {
    G.team += 250; G.finalWinnings += 250;
    G.dropNotes.push('Wildcard counter: +£250 for the team');
    popLabel('★ +£250', pos.clone().add(new THREE.Vector3(0, 1.3, 0)), 'wild', 2.2);
    return;
  }
  const P = G.players[who], O = G.players[1 - who];
  if (id === 'bonus') G.pendingExtra.push(who);
  else if (id === 'cash') P.money += 250;
  else if (id === 'steal') { const amt = Math.min(150, O.money); O.money -= amt; P.money += amt; }
  const ln = def.title + ': ' + def.text(P.name, O.name);
  G.dropNotes.push(ln);
  popLabel('★ ' + def.title + '!', pos.clone().add(new THREE.Vector3(0, 1.3, 0)), 'wild', 2.4);
  showBanner('★ Wildcard', ln, 2600);
  hostSay(line('wildcard'), 'shrug', 2000);
}
function onJackpotFall(inWin, pos) {
  G.jackpotFell = true;
  if (inWin) {
    G.jackpotWon = true; G.team += JACKPOT_VALUE; G.finalWinnings += JACKPOT_VALUE;
    SFX.jackpot();
    popLabel('+£5,000', pos.clone().add(new THREE.Vector3(0, 1.6, 0)), 'huge', 3);
    showBanner('Jackpot!', 'The team pushed the jackpot counter over the edge', 4200);
    hostSay(line('jackpot'), 'cheer', 4000);
    celebrate();
  } else {
    SFX.lost();
    showBanner('Into the gutter', 'The jackpot counter fell off the side', 3200);
    hostSay('Oh no! Down the side!', 'groan', 2500);
  }
  renderScores();
}
function celebrate() {
  const n = CGB.settings.reduced() ? 12 : 44;
  for (let i = 0; i < n; i++) {
    setTimeout(() => {
      const m = makeCoinMesh(Math.random() < 0.3 ? 'jackpot' : 'std', -1);
      if (m.geometry === jackGeo) m.scale.setScalar(0.5);
      m.position.set((Math.random() - 0.5) * 14 + 1, 13 + Math.random() * 3, (Math.random() - 0.5) * 8 + 1);
      falling.push({ mesh: m, vx: (Math.random() - 0.5) * 2, vy: -2, vz: Math.random() * 1.5, spin: 4 + Math.random() * 8, floorY: -6 });
    }, i * 45);
  }
}

/* =========================================================
   CAMERA, RESIZE, QUALITY, PICKING
   ========================================================= */
function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); composer.setSize(w, h); bloom.setSize(w, h);
  camera.aspect = w / h;
  const needH = Math.tan(THREE.MathUtils.degToRad(31));
  const vfov = Math.max(THREE.MathUtils.degToRad(46), 2 * Math.atan(needH / camera.aspect));
  camera.fov = THREE.MathUtils.radToDeg(vfov);
  // on narrow stages (portrait tablets) drop the picture a little so the header sign clears the menu buttons
  if (w < 900) camera.setViewOffset(w, h, 0, -Math.round(Math.min(64, h * 0.11)), w, h); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
  hostFit.shotRight = null;
  fitHost();
}
window.addEventListener('resize', resize);
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
function applyQuality() {
  const low = CGB.settings.get('quality') === 'low';
  useBloom = !low;
  renderer.setPixelRatio(CGB.settings.pixelRatio());
  if (renderer.shadowMap.enabled === low) {
    renderer.shadowMap.enabled = !low;
    scene.traverse(o => { if (o.material) [].concat(o.material).forEach(m => { m.needsUpdate = true; }); });
  }
  resize();
}
CGB.settings.onChange(k => {
  if (k === 'quality') applyQuality();
  if (k === 'sound') paintMute();
});

function updateCamera(dt, now) {
  const inBoard = drops.length > 0;
  let t = CAM.play;
  if (inBoard) t = CAM.board;
  else if (G.phase === 'final' && G.jackpot && !G.jackpotFell && jackpotProgress() > 0.7) t = CAM.jackpot;
  else if (G.phase === 'home') t = CAM.home;
  const shot = Object.keys(CAM).find(k => CAM[k] === t);
  if (shot !== hostFit.shot) fitHost(shot);
  const k = 1 - Math.exp(-dt * 1.5);
  Object.keys(cam).forEach(key => { cam[key] += (t[key] - cam[key]) * k; });
  const sway = CGB.settings.reduced() ? 0 : Math.sin(now * 0.00022) * SWAY;
  camera.position.set(cam.x + sway, cam.y, cam.z);
  camera.lookAt(cam.lx, cam.ly, 0);
}

const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', e => {
  if (G.step !== 'chute') return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObjects(chuteMeshes.concat([boardBack, glass]))[0];
  if (!hit) return;
  if (hit.object.userData.chute != null) { chooseChute(hit.object.userData.chute); return; }
  const px = hit.point.x / S + PEG.W / 2;
  let best = 0; PEG.CHUTES.forEach((cx, i) => { if (Math.abs(cx - px) < Math.abs(PEG.CHUTES[best] - px)) best = i; });
  chooseChute(best);
});
function updateChuteGlow(now) {
  if (G.step !== 'chute') return;
  const pulse = CGB.settings.reduced() ? 0.9 : 0.7 + Math.sin(now * 0.008) * 0.5;
  chuteMats.forEach(m => { m.emissive.setHex(G.dropper >= 0 ? COLORS.hex[G.dropper] : SET.aqua); m.emissiveIntensity = pulse; });
}
function resetChuteGlow() { chuteMats.forEach(m => { m.emissive.setHex(CHUTE_BASE); m.emissiveIntensity = 0.3; }); }

/* =========================================================
   UI HELPERS
   ========================================================= */
const fmt = n => '£' + n.toLocaleString('en-GB');
let bannerTimer = null;
function showBanner(title, detail, ms) {
  const b = $('banner');
  b.querySelector('.verdict').textContent = title;
  const d = b.querySelector('.detail'); d.textContent = detail || ''; d.style.display = detail ? 'inline-block' : 'none';
  b.classList.add('show');
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => b.classList.remove('show'), ms || 2200);
}
function hideBanner() { clearTimeout(bannerTimer); $('banner').classList.remove('show'); }
function feed(text) {
  const f = $('feed');
  const d = document.createElement('div'); d.textContent = text;
  f.insertBefore(d, f.firstChild);
  while (f.children.length > 5) f.removeChild(f.lastChild);
}
function jackpotProgress() {
  if (!G.jackpot) return 0;
  if (G.jackpotFell) return 1;
  const end = PHY.D + PHY.JR * 0.1;
  return Math.max(0, Math.min(1, (G.jackpot.y - G.jackpotY0) / (end - G.jackpotY0)));
}
function renderScores() {
  if (G.phase === 'home') {
    [0, 1].forEach(i => {
      const c = $('pc' + i);
      c.querySelector('.name').textContent = $('name' + i).value.trim() || CGB.teamFallback(i);
      c.querySelector('.num').textContent = fmt(0);
      c.querySelector('.meta').textContent = 'Ready to play';
      c.classList.remove('active');
    });
  } else G.players.forEach((p, i) => {
    const c = $('pc' + i);
    c.querySelector('.name').textContent = p.name;
    c.querySelector('.num').textContent = fmt(p.money);
    c.querySelector('.meta').textContent = `${p.correct} correct` + (p.steals ? `, ${p.steals} stolen` : '');
    const active = (G.step === 'chute' || G.step === 'dropping') ? G.dropper === i : (G.step === 'steal' ? i === 1 - G.turn : G.turn === i && G.step === 'ask');
    c.classList.toggle('active', active && G.phase !== 'home');
  });
  $('teamCard').style.display = G.phase === 'final' ? 'block' : 'none';
  if (G.phase === 'final') {
    $('teamVal').textContent = fmt(G.team);
    const pr = jackpotProgress();
    $('jackMeter').style.width = (pr * 100).toFixed(0) + '%';
    $('jackBar').setAttribute('aria-valuenow', String(Math.round(pr * 100)));
    $('jackCap').textContent = G.jackpotWon ? 'Jackpot counter: won!' : G.jackpotFell ? 'Jackpot counter: lost down the side' : `Jackpot counter: ${Math.round(pr * 100)}% of the way to the edge`;
  }
}
let lastMeterUpdate = 0, closeSaid = false;
function updateJackpotMeter(now) {
  if (now - lastMeterUpdate < 250) return;
  lastMeterUpdate = now;
  renderScores();
  if (!closeSaid && !G.jackpotFell && jackpotProgress() > 0.75) { closeSaid = true; hostSay(line('close'), 'groan', 2200); }
}
function renderQuestion() {
  const q = G.q;
  const who = G.step === 'steal' ? 1 - G.turn : G.turn;
  if (!q) { $('qWho').textContent = ''; $('qTag').textContent = ''; $('qText').textContent = ''; $('qAnswer').classList.remove('show'); return; }
  $('qcard').style.setProperty('--pc', COLORS.css[who]);
  $('qcard').style.setProperty('--pct', COLORS.text[who]);
  const label = G.phase === 'final'
    ? (G.step === 'steal' ? `${G.players[who].name} to the rescue` : `${G.players[who].name} answers for the team`)
    : (G.step === 'steal' ? `${G.players[who].name} can steal` : `${G.players[who].name}'s question`);
  $('qWho').textContent = COLORS.mark[who] + ' ' + label;
  $('qTag').textContent = `${q.subject}: ${q.topic}`;
  $('qText').textContent = q.q;
  $('qAnswer').textContent = q.a;
  $('qAnswer').classList.toggle('show', G.answerShown);
}
function renderActions() {
  const a = $('actions');
  a.innerHTML = '';
  const add = html => { const d = document.createElement('div'); d.innerHTML = html; while (d.firstChild) a.appendChild(d.firstChild); };
  const ansBtn = `<button class="linkish" type="button" data-act="answer">${G.answerShown ? 'Hide answer' : 'Show answer'} <span class="kbd">A</span></button>`;
  if (G.step === 'ask') {
    add(`<div class="act-row"><button class="btn ok" type="button" data-act="correct">✓ Correct <kbd>C</kbd></button><button class="btn no" type="button" data-act="wrong">✗ Wrong <kbd>W</kbd></button></div>${ansBtn}`);
  } else if (G.step === 'steal') {
    const o = G.players[1 - G.turn].name;
    const verb = G.phase === 'final' ? 'rescue' : 'steal';
    add(`<div class="hint">${escapeHtml(G.players[G.turn].name)} got it wrong. ${escapeHtml(o)} can ${verb} it for one counter.</div>
      <div class="act-row"><button class="btn ok" type="button" data-act="scorrect">✓ ${escapeHtml(o)} correct <kbd>C</kbd></button><button class="btn no" type="button" data-act="swrong">✗ Wrong <kbd>W</kbd></button></div>
      <button class="btn plain" type="button" data-act="snone">No attempt <kbd>N</kbd></button>${ansBtn}`);
  } else if (G.step === 'chute') {
    const p = G.players[G.dropper], n = G.dropCount;
    const lead = G.bonusDrop ? `Bonus counter for ${p.name}.` : `${p.name} wins ${n === 1 ? 'a counter' : n + ' counters'}.`;
    add(`<div class="ote-dropmsg" style="--pct:${COLORS.text[G.dropper]}">${COLORS.mark[G.dropper]} ${escapeHtml(lead)}<small>Pick a lane with the buttons or keys 1 to 4, or tap one on the machine</small></div>
      <div class="ote-chutes" style="--pc:${COLORS.css[G.dropper]}">${[0, 1, 2, 3].map(i => `<button class="ote-chute-btn" type="button" data-act="chute" data-i="${i}" aria-label="Lane ${i + 1}">${i + 1}<small>lane</small></button>`).join('')}</div>`);
  } else if (G.step === 'dropping') {
    add(`<div class="ote-dropmsg">Counters dropping<small>Waiting for the shelves to settle</small></div>`);
  } else if (G.step === 'next') {
    if (G.lastMsg) add(`<div class="ote-dropmsg">${escapeHtml(G.lastMsg.title)}<small>${escapeHtml(G.lastMsg.detail)}</small></div>`);
    const last = G.jackpotWon || G.qIndex >= G.qTotal;
    const label = !last ? 'Next question' : G.phase === 'r1' ? 'Go to the final' : 'See the results';
    add(`<button class="btn go" type="button" data-act="next">${label} <kbd>Space</kbd></button>`);
  }
}
$('actions').addEventListener('click', e => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const act = btn.dataset.act;
  if (act === 'correct') markCorrect();
  else if (act === 'wrong') markWrong();
  else if (act === 'answer') toggleAnswer();
  else if (act === 'scorrect') stealResult('correct');
  else if (act === 'swrong') stealResult('wrong');
  else if (act === 'snone') stealResult('none');
  else if (act === 'chute') chooseChute(+btn.dataset.i);
  else if (act === 'next') nextQuestion();
});
function render() { renderScores(); renderQuestion(); renderActions(); }

/* =========================================================
   GAME FLOW
   ========================================================= */
function startGame() {
  const names = CGB.saveTeamNames([0, 1].map(i => $('name' + i).value)).map(n => n.slice(0, 16));
  G.players = names.map(n => ({ name: n, money: 0, correct: 0, asked: 0, won: 0, steals: 0, wrong: [] }));
  G.focusWeak = $('focusWeak').checked;
  G.team = 0; G.finalWinnings = 0; G.jackpot = null; G.jackpotWon = false; G.jackpotFell = false;
  G.pendingExtra = []; G.lastDropper = -1; G.q = null; G.lastMsg = null; closeSaid = false;
  picker.reset();
  $('feed').innerHTML = '';
  clearLabels();
  G.attract = false;
  newTray(true);
  G.phase = 'r1'; G.qIndex = 0; G.qTotal = G.r1Each * 2;
  $('home').classList.add('hidden'); $('summary').classList.add('hidden');
  $('roundName').textContent = 'Round 1: Counter Drop';
  showBanner('Round 1', 'Answer right to win counters. Look out for the star wildcard counters.', 2800);
  hostSay(line('intro'), 'cheer', 2200);
  nextQuestion(true);
}
function nextQuestion(quiet) {
  resetChuteGlow();
  if (G.jackpotWon || G.qIndex >= G.qTotal) { endRound(); return; }
  G.turn = G.qIndex % 2;
  G.q = pickQuestion(G.players[G.turn].name);
  G.players[G.turn].asked++;
  G.qIndex++;
  G.answerShown = false; G.bonusDrop = false; G.lastMsg = null;
  G.step = 'ask';
  $('qCount').textContent = `Question ${G.qIndex} of ${G.qTotal}`;
  if (!quiet) hostSay(line('ask', { n: G.players[G.turn].name }), 'present', 1400);
  render();
}
function toggleAnswer() { G.answerShown = !G.answerShown; renderQuestion(); renderActions(); }
function logWrong(i) { const p = G.players[i]; p.wrong.push(G.q); bank.logWrong(p.name, G.q, GAME_NAME); }
function markCorrect() {
  if (G.step !== 'ask') return;
  SFX.correct();
  G.players[G.turn].correct++;
  G.answerShown = true;
  G.dropper = G.turn;
  G.dropCount = G.phase === 'final' ? 2 : 1;
  G.step = 'chute';
  hostSay(line('correct', { n: G.players[G.turn].name }) + ' ' + line('chute', { n: G.players[G.turn].name }), 'clap', 1300);
  render();
}
function markWrong() {
  if (G.step !== 'ask') return;
  SFX.wrong();
  logWrong(G.turn);
  G.step = 'steal';
  hostSay(line('wrong', { n: G.players[G.turn].name }) + ' ' + line('steal', { o: G.players[1 - G.turn].name }), 'shrug', 1600);
  render();
}
function stealResult(res) {
  if (G.step !== 'steal') return;
  const o = 1 - G.turn;
  G.answerShown = true;
  if (res === 'correct') {
    SFX.correct();
    G.players[o].steals++;
    G.dropper = o; G.dropCount = 1; G.step = 'chute';
    hostSay(line('correct', { n: G.players[o].name }) + ' ' + line('chute', { n: G.players[o].name }), 'clap', 1300);
  } else {
    if (res === 'wrong') { SFX.wrong(); logWrong(o); }
    G.lastMsg = { title: 'No counter this time', detail: res === 'wrong' ? 'Both answers were wrong.' : 'Nobody took the counter.' };
    hostSay(res === 'wrong' ? "Nobody got that one. Here's the answer." : "No steal. Here's the answer.", 'shrug', 1500);
    G.step = 'next';
  }
  render();
}
let dropTimers = [];
function chooseChute(i) {
  if (G.step !== 'chute' || i < 0 || i > 3) return;
  G.step = 'dropping';
  G.dropWon = 0; G.dropLost = 0; G.dropNotes = []; G.cascadeSaid = false;
  G.pending = G.dropCount; G.landed = 0; G.settleAt = Infinity;
  const owner = G.dropper;
  hostSay(line('dropping'), 'point', 2200);
  // counters are released 0.65 s apart in simulation time, so a run plays out the same at any frame rate
  for (let k = 0; k < G.dropCount; k++) {
    dropTimers.push({ at: simT + k * 0.65, go: () => launchDrop(i, owner, () => {
      G.lastDropper = owner;
      G.landed++;
      if (G.landed >= G.pending) G.settleAt = simT + PHY.PERIOD * 1.4;
    }) });
  }
  render();
}
function edgeTeeter() { return tray.lower.some(b => !b.base && b.y > PHY.D - b.r * 0.35); }
function finishDrop() {
  const p = G.players[G.dropper];
  const won = G.dropWon * VALUE;
  let title;
  if (G.phase === 'final') title = G.dropWon ? `${G.dropWon} over the edge: +${fmt(won)} for the team` : 'Nothing fell this time';
  else title = G.dropWon ? `${p.name} pushed ${G.dropWon} over the edge: +${fmt(won)}` : `Nothing fell for ${p.name}`;
  const detail = [G.dropLost ? `${G.dropLost} lost down the sides.` : '', ...G.dropNotes].filter(Boolean).join(' ') || 'The shelves have settled.';
  G.lastMsg = { title, detail };
  feed(title);
  if (!G.jackpotWon && !G.cascadeSaid) {
    if (G.dropWon === 1) hostSay(line('one'), 'clap', 1400);
    else if (G.dropWon > 1) hostSay(line('some', { k: G.dropWon }), 'cheer', 1800);
    else if (edgeTeeter()) hostSay(line('edge'), 'groan', 2000);
    else hostSay(line('none'), 'shrug', 1500);
  }
  if (G.jackpotWon) { G.step = 'next'; render(); return; }
  if (G.pendingExtra.length && G.phase === 'r1') {
    G.dropper = G.pendingExtra.shift(); G.dropCount = 1; G.bonusDrop = true; G.step = 'chute';
    render(); return;
  }
  G.step = 'next';
  render();
}
function endRound() {
  if (G.phase === 'r1') {
    G.phase = 'final';
    G.team = G.players[0].money + G.players[1].money;
    setupFinal();
    G.qIndex = 0; G.qTotal = G.finalN;
    $('roundName').textContent = 'The Final: Jackpot';
    showBanner('The Final', 'Team up to push the jackpot counter over the edge. Each correct answer wins two counters.', 3400);
    hostSay(line('final'), 'cheer', 2200);
    nextQuestion(true);
  } else {
    G.phase = 'summary'; G.step = null;
    hostSay(G.jackpotWon ? line('outro') : line('noJackpot'), G.jackpotWon ? 'cheer' : 'shrug', 2400);
    showSummary();
  }
}
function setupFinal() {
  const diff = { easy: [4, 0.5], normal: [12, 0.6], hard: [24, 0.8] }[G.finalDiff] || [12, 0.6];
  const jx = PHY.W / 2, jy = PHY.D - PHY.JR - diff[0];
  clearSpace(jx, jy);
  const J = addLower(tray, jx, jy, { r: PHY.JR, kind: 'jackpot', heavy: diff[1] });
  J.mesh = makeCoinMesh('jackpot', -1);
  J.anim = 1;
  G.jackpot = J; G.jackpotY0 = jy;
}
/* Remove the counters where the jackpot counter is about to go */
function clearSpace(jx, jy) {
  tray.lower = tray.lower.filter(b => {
    const keep = Math.hypot(b.x - jx, b.y - jy) > PHY.JR + b.r - 2 && !(b.base && Math.hypot(b.base.x - jx, b.base.y - jy) <= PHY.JR + b.r - 2);
    if (!keep) { if (b.mesh) scene.remove(b.mesh); if (b.rider) { b.rider.base = null; } if (b.base) b.base.rider = null; }
    return keep;
  });
  tray.lower.forEach(b => { if (b.base && !tray.lower.includes(b.base)) b.base = null; if (b.rider && !tray.lower.includes(b.rider)) b.rider = null; });
}

/* @test-only: shortcuts for tests, removed from the shipped file by build.js */
CGB.test.ote = {
  // manual clock: the game only moves on when a test calls advance(), so physics is exactly repeatable
  manual(on) { manualClock = !!on; simAcc = 0; },
  advance(seconds) { const n = Math.round(seconds / STEP); for (let i = 0; i < n; i++) simulate(STEP); },
  toFinal() { if (G.phase === 'r1' && G.step !== 'dropping') { G.step = null; endRound(); } },
  // put the jackpot counter's centre this fraction of its radius short of where it falls
  jackpotNear(frac) {
    const J = G.jackpot; if (!J || G.jackpotFell) return;
    const y = PHY.D + PHY.JR * 0.1 - PHY.JR * (frac == null ? 0.4 : frac);
    tray.lower = tray.lower.filter(b => b !== J); clearSpace(J.x, y); tray.lower.push(J); J.y = y; J.vx = J.vy = 0;
  },
  simTime: () => simT,
  info: () => ({ jackpotWon: !!G.jackpotWon, jackpotFell: !!G.jackpotFell, progress: jackpotProgress(), team: G.team,
    tray: tray.lower.map(b => `${b.kind}:${b.x.toFixed(2)},${b.y.toFixed(2)}`).join(';') })
};
/* @end-test-only */

/* =========================================================
   SUMMARY
   ========================================================= */
function showSummary() {
  const total = G.team;
  const head = G.jackpotWon
    ? `<div class="big">Jackpot won: ${fmt(total)}</div><div class="small">Round 1 ${fmt(G.players[0].money + G.players[1].money)}, final ${fmt(G.finalWinnings)} including the £5,000 jackpot</div>`
    : `<div class="big">Team total: ${fmt(total)}</div><div class="small">The jackpot counter finished ${Math.round(jackpotProgress() * 100)}% of the way to the edge. Final winnings ${fmt(G.finalWinnings)}.</div>`;
  const cols = G.players.map((p, i) => {
    const missed = p.wrong.length ? p.wrong.map(q => `<div class="missed"><div>${escapeHtml(q.q)}</div><div class="a">${escapeHtml(q.a)}</div><div class="t">${escapeHtml(q.subject)}: ${escapeHtml(q.topic)}</div></div>`).join('') : '<div class="none">No wrong answers this game.</div>';
    const hist = bank.weakTopics(p.name, 5);
    const histHtml = hist.length ? hist.map(([t, n]) => `<div class="trow"><span>${escapeHtml(t)}</span><span>${n} wrong</span></div>`).join('') : '<div class="none">No history yet.</div>';
    return `<div class="sum-p" style="--pc:${COLORS.css[i]}">
      <h3>${COLORS.mark[i]} ${escapeHtml(p.name)}</h3>
      <div class="sum-stats">Round 1: ${fmt(p.money)}. ${p.correct} correct from ${p.asked} questions${p.steals ? `, plus ${p.steals} steal${p.steals > 1 ? 's' : ''}` : ''}.</div>
      <div class="sum-sub">Missed this game</div>${missed}
      <div class="sum-sub">Weakest topics across all games</div>${histHtml}
      <button class="linkish" type="button" data-clear="${i}">Clear ${escapeHtml(p.name)}'s history</button>
    </div>`;
  }).join('');
  $('sumCard').innerHTML = `<div class="sum-head">${head}</div><div class="sum-grid">${cols}</div>
    <div class="sum-btns"><button class="btn go" type="button" id="ote-againBtn">Play again <kbd>Enter</kbd></button><button class="btn plain" type="button" id="ote-homeBtn">Change settings</button><button class="btn plain" type="button" id="ote-menuBtn2">Back to menu</button></div>${CGB.REVIEW_NOTE}`;
  $('summary').classList.remove('hidden');
  clearLabels();
  updateHost(true);
  $('againBtn').onclick = () => startGame();
  $('homeBtn').onclick = () => goHome();
  $('menuBtn2').onclick = () => CGB.app.requestLauncher();
  $('sumCard').querySelectorAll('[data-clear]').forEach(btn => {
    btn.onclick = () => CGB.armButton(btn, 'Tap again to clear', () => { bank.clearHistory(G.players[+btn.dataset.clear].name); showSummary(); });
  });
  render();
  $('summary').scrollTop = 0; $('sumCard').scrollTop = 0;
  $('againBtn').focus({ preventScroll: true });
}
function goHome() {
  if (CGB.fitSetups) CGB.fitSetups();
  dropTimers = [];
  $('summary').classList.add('hidden');
  $('home').classList.remove('hidden');
  updateHost(true);
  G.phase = 'home'; G.step = null; G.q = null; G.attract = true;
  $('roundName').textContent = 'Ready'; $('qCount').textContent = '';
  hideBanner(); clearLabels();
  newTray(false);
  render();
  $('qText').textContent = 'Set up the game to begin.';
}

/* =========================================================
   SETUP SCREEN
   ========================================================= */
$('logo').innerHTML = CGB.brand.oteLogo();
function renderSetSelect() {
  bank.fillSelect($('setSelect'));
}
$('setSelect').addEventListener('change', e => { bank.setActive(e.target.value); picker.reset(); });
bank.onChange(renderSetSelect);
$('openBank').addEventListener('click', () => CGB.bankUI.open());
function wireToggle(btnId, panelId, openText, closedText) {
  $(btnId).addEventListener('click', () => {
    const p = $(panelId); p.classList.toggle('show');
    const open = p.classList.contains('show');
    $(btnId).textContent = open ? openText : closedText;
    $(btnId).setAttribute('aria-expanded', String(open));
  });
}
wireToggle('toggleHost', 'hostEditor', 'Hide host options', 'Customise the host');
function wireSeg(id, key) {
  const seg = $(id);
  const paint = () => seg.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.v === G[key])));
  seg.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; G[key] = +b.dataset.v; CGB.store.set('ote.' + key, String(G[key])); paint(); });
  const saved = +CGB.store.get('ote.' + key); if ([4, 6, 8, 12, 16].includes(saved)) G[key] = saved;
  paint();
}
wireSeg('segR1', 'r1Each');
wireSeg('segF', 'finalN');
(function wireDiff() {
  const seg = $('segDiff');
  const saved = CGB.store.get('ote.finalDiff'); if (['easy', 'normal', 'hard'].includes(saved)) G.finalDiff = saved;
  const paint = () => seg.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === G.finalDiff)));
  seg.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; G.finalDiff = b.dataset.v; CGB.store.set('ote.finalDiff', G.finalDiff); paint(); });
  paint();
})();

/* Host editor (changes the Showtime mascot everywhere) */
function saveHost() { CGB.saveHost(); hostSay('How do I look?', 'present', 1200); }
function swatchRow(id, list, key, label) {
  const el = $(id);
  el.innerHTML = list.map((c, i) => `<button type="button" class="sw" data-i="${i}" style="background:${c}" aria-label="${label} option ${i + 1}"></button>`).join('');
  const paint = () => el.querySelectorAll('.sw').forEach(b => { const on = +b.dataset.i === CGB.hostCfg[key]; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
  el.addEventListener('click', e => { const b = e.target.closest('.sw'); if (!b) return; CGB.hostCfg[key] = +b.dataset.i; paint(); saveHost(); });
  paint();
}
function segRow(id, list, key) {
  const el = $(id);
  el.innerHTML = list.map(([v, label]) => `<button type="button" data-v="${v}">${label}</button>`).join('');
  const paint = () => el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === CGB.hostCfg[key])));
  el.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; CGB.hostCfg[key] = b.dataset.v; paint(); saveHost(); });
  paint();
}
swatchRow('swSkin', CGB.hostOpts.skin, 'skin', 'Skin');
swatchRow('swHair', CGB.hostOpts.hair, 'hair', 'Hair colour');
swatchRow('swSuit', CGB.hostOpts.suit, 'suit', 'Suit colour');
segRow('segStyle', CGB.hostOpts.style, 'style');
segRow('segBeard', CGB.hostOpts.beard, 'beard');
segRow('segGlasses', CGB.hostOpts.glasses, 'glasses');
$('hostName').value = CGB.hostCfg.name;
$('hostName').addEventListener('input', e => { CGB.hostCfg.name = e.target.value.trim() || 'Host'; CGB.store.setJSON('host', CGB.hostCfg); });
$('hostName').addEventListener('change', () => CGB.saveHost());

const fillNames = () => CGB.teamNames(2).forEach((n, i) => { $('name' + i).value = n.slice(0, 16); });
fillNames();
[0, 1].forEach(i => $('name' + i).addEventListener('input', renderScores));
$('focusWeak').checked = CGB.store.get('ote.focusWeak') === '1';
$('focusWeak').addEventListener('change', e => CGB.store.set('ote.focusWeak', e.target.checked ? '1' : '0'));
$('startBtn').addEventListener('click', startGame);
function paintMute() { $('muteBtn').textContent = CGB.settings.get('sound') ? 'Sound on' : 'Sound off'; $('muteBtn').setAttribute('aria-pressed', String(!CGB.settings.get('sound'))); }
$('muteBtn').addEventListener('click', () => { CGB.settings.set('sound', !CGB.settings.get('sound')); CGB.sfx.unlock(); });
$('menuBtn').addEventListener('click', () => CGB.app.requestLauncher());
$('settingsBtn').addEventListener('click', () => CGB.modal.open('settingsModal'));
paintMute();
renderSetSelect();

let active = false;
document.addEventListener('keydown', e => {
  if (!active || CGB.modal.isOpen() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { e.preventDefault(); CGB.app.requestLauncher(); return; }
  const inField = e.target.matches && e.target.matches('input, textarea, select');
  if (G.phase === 'home') {
    if (e.key === 'Enter' && !(e.target.matches && e.target.matches('button, select, textarea'))) { e.preventDefault(); startGame(); }
    return;
  }
  if (G.phase === 'summary') {
    if (e.key === 'Enter' && !(e.target.matches && e.target.matches('button'))) { e.preventDefault(); startGame(); }
    return;
  }
  if (inField) return;
  const k = e.key.toLowerCase();
  if (G.step === 'ask') { if (k === 'c') markCorrect(); else if (k === 'w') markWrong(); else if (k === 'a') toggleAnswer(); }
  else if (G.step === 'steal') { if (k === 'c') stealResult('correct'); else if (k === 'w') stealResult('wrong'); else if (k === 'n') stealResult('none'); else if (k === 'a') toggleAnswer(); }
  else if (G.step === 'chute') { if (['1', '2', '3', '4'].includes(k)) chooseChute(+k - 1); }
  else if (G.step === 'next') { if (k === ' ' || k === 'enter') { e.preventDefault(); nextQuestion(); } }
});

/* =========================================================
   MAIN LOOP (runs only while this game is on screen)
   ========================================================= */
let lastAttract = 0;
function attractTick(now) {
  if (now - lastAttract < 3400 || drops.length || transits.length) return;
  lastAttract = now;
  launchDrop(Math.floor(Math.random() * 4), -1, null, Math.random);   // decoration on the setup screen: not part of play
}
newTray(false);
resize();
render();
$('qText').textContent = 'Set up the game to begin.';
let lastT = performance.now(), rafId = 0;
/* The game advances in fixed 1/60 s steps of simulation time (up to 4 per frame), so
   the physics gives the same result whatever the frame rate. */
const STEP = 1 / 60, MAX_STEPS = 4;
let simT = 0, simAcc = 0, manualClock = false;
/* @test-only */ if (window.__SHOWTIME_MANUAL__) manualClock = true; /* @end-test-only */
function simulate(h) {
  simT += h;
  for (let k = dropTimers.length - 1; k >= 0; k--) if (dropTimers[k].at <= simT) dropTimers.splice(k, 1)[0].go();
  stepTray(tray, h, onTrayFall);
  tray.lower.forEach(b => { if (b.anim > 0) b.anim = Math.max(0, b.anim - h * 1.1); });
  updateDrops(h);
  updateTransits(h);
  if (G.step === 'dropping' && G.landed >= G.pending && simT >= G.settleAt) finishDrop();
}
function loop(now) {
  if (!active) return;
  rafId = requestAnimationFrame(loop);
  const raw = Math.max(0, (now - lastT) / 1000), dt = Math.min(0.033, raw);
  lastT = now;
  if (!manualClock) {
    simAcc = Math.min(simAcc + raw, STEP * MAX_STEPS);
    let n = 0;
    while (simAcc >= STEP && n < MAX_STEPS) { simulate(STEP); simAcc -= STEP; n++; }
  }
  updateFalling(dt);
  syncTray(dt);
  pusher.position.z = TZ(tray.pf);
  updateChuteGlow(now);
  updateCamera(dt, now);
  updateHost();
  updateLabels(dt);
  placeBubble();
  dust.rotation.y += dt * 0.01;
  if (G.attract) attractTick(now);
  if (G.phase === 'final') updateJackpotMeter(now);
  if (useBloom) composer.render(); else renderer.render(scene, camera);
}

return {
  enter() {
    active = true;
    applyQuality();
    lastT = performance.now();
    rafId = requestAnimationFrame(loop);
    updateHost(true);
    if (G.phase === 'home') {
      fillNames(); renderScores();
      setTimeout(() => { if (active && G.phase === 'home') hostSay(`Hello! I'm ${CGB.hostCfg.name || 'your host'}. Set up the game and let's play.`, 'present', 1800); }, 700);
      $('startBtn').focus();
    }
  },
  exit() {
    active = false;
    cancelAnimationFrame(rafId);
    hideBubble();
    if (G.phase !== 'home') goHome();
  },
  inProgress: () => G.phase === 'r1' || G.phase === 'final',
  /* used by the automated tests */
  _layout: () => {
    const sr = wrap.getBoundingClientRect(), hr = hostRect();
    return { stage: { w: sr.width, h: sr.height }, machine: machineRect(), machineRight: hostFit.machineRight, hidden: hostFit.hidden, host: { left: hr.left - sr.left, right: hr.right - sr.left, top: hr.top - sr.top, bottom: hr.bottom - sr.top }, away: $('host').classList.contains('away') };
  },
  _state: () => ({ phase: G.phase, step: G.step, qIndex: G.qIndex, qTotal: G.qTotal, q: G.q, players: G.players })
};
}

CGB.registerGame('over-the-edge', {
  title: 'Over the Edge',
  init(root) { game = createOverTheEdge(root); },
  enter() { if (game) game.enter(); },
  exit() { if (game) game.exit(); },
  inProgress() { return !!(game && game.inProgress()); },
  state() { return game && game._state(); },
  layout() { return game && game._layout(); }
});
})();
