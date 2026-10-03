'use strict';
/* =========================================================
   THE HOST: the bundle mascot, a stylised cartoon presenter.
   Built from simple shapes so it can be customised. The same
   settings are used on the launcher and in Over the Edge.
   ========================================================= */
CGB.hostOpts = {
  skin: ['#F6D3B8', '#EBBE9B', '#D29A6E', '#A06A45', '#6E4429'],
  hair: ['#1d1a18', '#3b2a1e', '#6b4a2b', '#9a7448', '#d8b46a', '#c4622d', '#8f8f94'],
  suit: ['#1f3b73', '#3a3d46', '#0f7c80', '#7a2e8c', '#6e2233'],
  style: [['short', 'Short'], ['swept', 'Swept up'], ['curly', 'Curly'], ['buzz', 'Buzz'], ['bald', 'None']],
  beard: [['none', 'None'], ['stubble', 'Stubble'], ['beard', 'Beard']],
  glasses: [['no', 'No'], ['yes', 'Yes']]
};
CGB.hostCfg = (() => {
  const defaults = { name: 'Professor Pip', skin: 1, hair: 5, suit: 2, style: 'curly', beard: 'none', glasses: 'yes' };
  const saved = CGB.store.getJSON('host', {});
  const cfg = Object.assign({}, defaults, saved && typeof saved === 'object' ? saved : {});
  // guard against out-of-range values from older saves
  ['skin', 'hair', 'suit'].forEach(k => { if (!(cfg[k] >= 0 && cfg[k] < CGB.hostOpts[k].length)) cfg[k] = defaults[k]; });
  ['style', 'beard', 'glasses'].forEach(k => { if (!CGB.hostOpts[k].some(o => o[0] === cfg[k])) cfg[k] = defaults[k]; });
  return cfg;
})();
CGB.hostListeners = [];
CGB.saveHost = function () {
  CGB.store.setJSON('host', CGB.hostCfg);
  CGB.hostListeners.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } });
};

CGB.HOST_POSES = {
  // [shoulder x (forward), shoulder z (sideways), elbow bend, shoulder y (swing in/out)]
  // R = host's right = towards the machine in Over the Edge
  idle:  { L: [-0.32, 0.05, -1.45, -0.95], R: [-0.32, -0.05, -1.45, 0.95] },
  point: { L: [-0.32, 0.05, -1.45, -0.95], R: [-1.45, -0.1, -0.1, -0.75] },
  cheer: { L: [-2.75, 0.35, -0.35, 0], R: [-2.75, -0.35, -0.35, 0] },
  groan: { L: [-2.5, 0.2, -2.3, -0.35], R: [-2.5, -0.2, -2.3, 0.35] },
  shrug: { L: [-0.25, 0.55, -1.5, 0.45], R: [-0.25, -0.55, -1.5, -0.45] },
  clap:  { L: [-0.85, 0.05, -1.25, -0.95], R: [-0.85, -0.05, -1.25, 0.95] },
  present: { L: [-0.32, 0.05, -1.45, -0.95], R: [-0.95, -0.25, -0.55, -0.55] },
  wave:  { L: [-0.32, 0.05, -1.45, -0.95], R: [-2.6, -0.45, -0.5, 0] }
};

/* Create a host in a scene. Returns a controller with update(dt, now, ctx). */
CGB.createHost = function (scene, opts) {
  opts = opts || {};
  const host = { group: null, parts: {}, pose: null, target: 'idle', gestureUntil: 0, talkUntil: 0, blinkAt: 0, look: 0 };
  function build() {
    const cfg = CGB.hostCfg, O = CGB.hostOpts;
    if (host.group) scene.remove(host.group);
    const g = new THREE.Group();
    const P = {};
    const skin = new THREE.MeshStandardMaterial({ color: O.skin[cfg.skin], roughness: 0.8, metalness: 0, envMapIntensity: 0.3 });
    const hairM = new THREE.MeshStandardMaterial({ color: O.hair[cfg.hair], roughness: 0.75 });
    const suit = new THREE.MeshStandardMaterial({ color: O.suit[cfg.suit], roughness: 0.55, metalness: 0.08 });
    const shirt = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x16121f, roughness: 0.4 });
    const accent = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness: 0.45 });
    const add = (geo, mat, x, y, z, parent) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; (parent || g).add(m); return m; };
    // legs and shoes
    [-0.52, 0.52].forEach(x => {
      add(new THREE.CylinderGeometry(0.5, 0.44, 3.6, 18), suit, x, 1.8, 0);
      const shoe = add(new THREE.SphereGeometry(0.45, 16, 10), dark, x, 0.18, 0.18); shoe.scale.set(1, 0.45, 1.5);
    });
    // torso
    const torso = new THREE.Group(); torso.position.y = 3.6; g.add(torso); P.torso = torso;
    add(new THREE.CylinderGeometry(1.28, 1.05, 3.4, 26), suit, 0, 1.7, 0, torso);
    const yoke = add(new THREE.SphereGeometry(1.28, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2), suit, 0, 3.32, 0, torso); yoke.scale.set(1, 0.3, 0.9);
    const vee = new THREE.Mesh(new THREE.CircleGeometry(0.7, 3), shirt); vee.rotation.z = -Math.PI / 2; vee.scale.set(1.5, 0.7, 1); vee.position.set(0, 2.95, 1.17); vee.rotation.x = -0.15; torso.add(vee);
    [-1, 1].forEach(s => { const lap = add(new THREE.BoxGeometry(0.16, 1.5, 0.08), suit, s * 0.33, 2.75, 1.2, torso); lap.rotation.z = s * 0.38; lap.material = new THREE.MeshStandardMaterial({ color: new THREE.Color(O.suit[cfg.suit]).multiplyScalar(0.8), roughness: 0.5 }); });
    const pocket = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.16, 0.05), accent); pocket.position.set(-0.7, 2.35, 1.13); pocket.rotation.y = -0.4; torso.add(pocket);
    add(new THREE.SphereGeometry(0.06, 8, 6), dark, 0, 1.5, 1.12, torso);
    // bow tie: the mascot's signature
    [-1, 1].forEach(s => { const w = add(new THREE.ConeGeometry(0.2, 0.36, 4), accent, s * 0.17, 3.28, 1.22, torso); w.rotation.z = s * Math.PI / 2; w.scale.set(1, 1, 0.45); });
    add(new THREE.SphereGeometry(0.09, 10, 8), accent, 0, 3.28, 1.26, torso);
    // arms (shoulder pivot groups)
    const mkArm = (side) => {
      const sh = new THREE.Group(); sh.rotation.order = 'YXZ'; sh.position.set(side * 1.25, 3.05, 0); torso.add(sh);
      add(new THREE.SphereGeometry(0.42, 16, 12), suit, 0, 0, 0, sh);
      add(new THREE.CylinderGeometry(0.32, 0.29, 1.5, 14), suit, 0, -0.75, 0, sh);
      const el = new THREE.Group(); el.position.set(0, -1.5, 0); sh.add(el);
      add(new THREE.CylinderGeometry(0.29, 0.26, 1.4, 14), suit, 0, -0.7, 0, el);
      add(new THREE.CylinderGeometry(0.25, 0.25, 0.14, 14), shirt, 0, -1.4, 0, el);
      add(new THREE.SphereGeometry(0.3, 14, 10), skin, 0, -1.65, 0, el);
      return { sh, el };
    };
    P.armL = mkArm(1);   // host's left = screen right
    P.armR = mkArm(-1);  // host's right = screen left
    // neck and head
    add(new THREE.CylinderGeometry(0.36, 0.4, 0.6, 14), skin, 0, 3.6, 0, torso);
    const collarL = add(new THREE.BoxGeometry(0.5, 0.2, 0.3), shirt, -0.28, 3.4, 0.35, torso); collarL.rotation.z = 0.5;
    const collarR = add(new THREE.BoxGeometry(0.5, 0.2, 0.3), shirt, 0.28, 3.4, 0.35, torso); collarR.rotation.z = -0.5;
    const head = new THREE.Group(); head.position.set(0, 4.95, 0); torso.add(head); P.head = head;
    const skull = add(new THREE.SphereGeometry(1.15, 32, 24), skin, 0, 0, 0, head); skull.scale.set(1, 1.1, 1.02);
    [-1, 1].forEach(s => add(new THREE.SphereGeometry(0.24, 12, 10), skin, s * 1.13, 0.0, 0, head));
    add(new THREE.SphereGeometry(0.15, 12, 10), skin, 0, -0.1, 1.18, head);
    P.eyes = [-1, 1].map(s => {
      const eg = new THREE.Group(); eg.position.set(s * 0.4, 0.18, 1.0); head.add(eg);
      const white = add(new THREE.SphereGeometry(0.2, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 }), 0, 0, 0, eg); white.scale.z = 0.6;
      add(new THREE.SphereGeometry(0.1, 12, 10), dark, 0, 0, 0.1, eg);
      return eg;
    });
    P.brows = [-1, 1].map(s => { const b = add(new THREE.BoxGeometry(0.42, 0.09, 0.08), hairM, s * 0.42, 0.52, 1.07, head); b.rotation.z = s * -0.08; return b; });
    // mouth: a smile line and an open mouth that scales while talking
    const smile = add(new THREE.TorusGeometry(0.28, 0.045, 8, 24, Math.PI), dark, 0, -0.42, 1.12, head); smile.rotation.z = Math.PI; P.smile = smile;
    const mouth = add(new THREE.SphereGeometry(0.22, 16, 12), new THREE.MeshStandardMaterial({ color: 0x5a1420, roughness: 0.6 }), 0, -0.5, 1.08, head); mouth.scale.set(1.15, 0.05, 0.5); P.mouth = mouth;
    // hair
    const st = cfg.style;
    if (st !== 'bald') {
      const capR = st === 'buzz' ? 1.17 : 1.22;
      const cap = add(new THREE.SphereGeometry(capR, 32, 16, 0, Math.PI * 2, 0, Math.PI * (st === 'buzz' ? 0.42 : 0.47)), hairM, 0, 0.12, -0.06, head);
      cap.rotation.x = -0.32; cap.scale.set(1, 1.08, 1.04);
      const back = add(new THREE.SphereGeometry(capR * 0.99, 24, 12, Math.PI, Math.PI, Math.PI * 0.35, Math.PI * 0.3), hairM, 0, 0.05, -0.04, head);
      back.scale.set(1, 1.08, 1.02);
      if (st === 'swept') { const q = add(new THREE.SphereGeometry(0.75, 20, 12), hairM, 0.05, 1.05, 0.55, head); q.scale.set(1.25, 0.42, 0.75); q.rotation.x = -0.35; }
      if (st === 'short') { const f = add(new THREE.SphereGeometry(0.6, 16, 10), hairM, 0.25, 0.92, 0.75, head); f.scale.set(1.5, 0.3, 0.6); f.rotation.z = -0.15; }
      if (st === 'curly') {
        // seeded so the curls look the same every time
        let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
        for (let i = 0; i < 34; i++) { const a = rnd() * Math.PI * 2, t = rnd() * 1.1; add(new THREE.SphereGeometry(0.28 + rnd() * 0.1, 10, 8), hairM, Math.cos(a) * Math.sin(t) * 1.15, 0.25 + Math.cos(t) * 1.12, Math.sin(a) * Math.sin(t) * 1.15 - 0.1, head); }
      }
    }
    if (cfg.beard !== 'none') {
      const bm = new THREE.MeshStandardMaterial({ color: O.hair[cfg.hair], roughness: 0.9, transparent: cfg.beard === 'stubble', opacity: cfg.beard === 'stubble' ? 0.45 : 1, depthWrite: cfg.beard !== 'stubble' });
      const br = cfg.beard === 'beard' ? 1.21 : 1.19;
      const beard = add(new THREE.SphereGeometry(br, 32, 16, Math.PI * 0.12, Math.PI * 0.76, Math.PI * 0.5, Math.PI * 0.42), bm, 0, 0, 0.02, head);
      beard.scale.set(1, 1.1, 1.02);
      if (cfg.beard === 'beard') { P.smile.position.z = 1.24; P.mouth.position.z = 1.2; }
    }
    if (cfg.glasses === 'yes') {
      const gm = new THREE.MeshStandardMaterial({ color: 0x1a1a22, metalness: 0.4, roughness: 0.3 });
      [-1, 1].forEach(s => add(new THREE.TorusGeometry(0.27, 0.045, 8, 24), gm, s * 0.42, 0.18, 1.2, head));
      add(new THREE.BoxGeometry(0.22, 0.05, 0.05), gm, 0, 0.22, 1.24, head);
    }
    if (opts.position) g.position.copy(opts.position);
    g.rotation.y = opts.rotationY == null ? -0.42 : opts.rotationY;
    scene.add(g);
    host.group = g; host.parts = P;
    host.pose = JSON.parse(JSON.stringify(CGB.HOST_POSES.idle));
  }
  function gesture(name, ms) { host.target = name; host.gestureUntil = performance.now() + (ms || 1800); }
  /* ctx: { rest: pose name when no gesture, watching: bool, watchLook: number } */
  function update(dt, now, ctx) {
    if (!host.group) return;
    ctx = ctx || {};
    const P = host.parts;
    if (now > host.gestureUntil) host.target = ctx.rest || 'idle';
    const T = CGB.HOST_POSES[host.target] || CGB.HOST_POSES.idle;
    const k = 1 - Math.exp(-dt * 7);
    ['L', 'R'].forEach(side => { for (let i = 0; i < 4; i++) host.pose[side][i] += (T[side][i] - host.pose[side][i]) * k; });
    const still = CGB.settings.reduced();
    const clapWob = host.target === 'clap' && !still ? Math.sin(now * 0.03) * 0.18 : 0;
    const waveWob = host.target === 'wave' && !still ? Math.sin(now * 0.012) * 0.35 : 0;
    P.armL.sh.rotation.set(host.pose.L[0], host.pose.L[3] + clapWob, host.pose.L[1]);
    P.armL.el.rotation.set(host.pose.L[2], 0, 0);
    P.armR.sh.rotation.set(host.pose.R[0], host.pose.R[3] - clapWob, host.pose.R[1] + waveWob);
    P.armR.el.rotation.set(host.pose.R[2], 0, 0);
    // breathing and a little bounce on cheers
    const bounce = host.target === 'cheer' && !still ? Math.abs(Math.sin(now * 0.012)) * 0.25 : 0;
    P.torso.position.y = 3.6 + Math.sin(now * 0.002) * 0.03 + bounce;
    P.torso.rotation.z = Math.sin(now * 0.0011) * 0.02;
    const lookTarget = ctx.watching ? (ctx.watchLook == null ? -0.55 : ctx.watchLook) : (now % 9000 < 4500 ? 0.1 : 0.25);
    host.look += (lookTarget - host.look) * (1 - Math.exp(-dt * 4));
    P.head.rotation.y = host.look;
    P.head.rotation.x = host.target === 'groan' ? 0.25 : host.target === 'cheer' ? -0.15 : Math.sin(now * 0.0017) * 0.03;
    P.head.rotation.z = Math.sin(now * 0.0013) * 0.04;
    if (now > host.blinkAt) host.blinkAt = now + 2200 + Math.random() * 2600;
    const blink = host.blinkAt - now < 130 ? 0.1 : 1;
    P.eyes.forEach(e => e.scale.y = blink);
    const browUp = host.target === 'cheer' || host.target === 'shrug' ? 0.1 : host.target === 'groan' ? -0.05 : 0;
    P.brows.forEach(b => b.position.y = 0.52 + browUp);
    const talking = now < host.talkUntil;
    const open = talking ? 0.25 + Math.abs(Math.sin(now * 0.028)) * 0.75 : (host.target === 'cheer' ? 0.9 : host.target === 'groan' ? 0.5 : 0);
    P.mouth.scale.y += ((open > 0 ? open : 0.05) - P.mouth.scale.y) * (1 - Math.exp(-dt * 18));
    P.smile.visible = P.mouth.scale.y < 0.2;
  }
  build();
  return {
    get group() { return host.group; },
    get parts() { return host.parts; },
    rebuild: build, gesture, update,
    talk(ms) { host.talkUntil = performance.now() + ms; }
  };
};
