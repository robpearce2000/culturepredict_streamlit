'use strict';
/* =========================================================
   THE HOST: the Showtime mascot. He has no name unless the teacher gives him one.
   A 2D cartoon drawn as SVG in the same ink-outlined sticker
   style as the Showtime logos. Arms pose with CSS transforms,
   he blinks, talks and pulls faces. The look is customisable
   (skin, hair, style, facial hair, suit, glasses) and shared by
   the launcher and Over the Edge.
   ========================================================= */
CGB.hostOpts = {
  skin: ['#F6D3B8', '#EBBE9B', '#D29A6E', '#A06A45', '#6E4429'],
  hair: ['#1d1a18', '#3b2a1e', '#6b4a2b', '#9a7448', '#d8b46a', '#c4622d', '#8f8f94'],
  suit: ['#1f3b73', '#3a3d46', '#0f7c80', '#7a2e8c', '#6e2233'],
  style: [['short', 'Short'], ['swept', 'Swept up'], ['curly', 'Curly'], ['bob', 'Bob'], ['ponytail', 'Ponytail'], ['buzz', 'Buzz'], ['bald', 'None']],
  gender: [['male', 'Male'], ['female', 'Female']],
  beard: [['none', 'None'], ['stubble', 'Stubble'], ['beard', 'Beard']],
  glasses: [['no', 'No'], ['yes', 'Yes']]
};
CGB.hostCfg = (() => {
  const defaults = { name: '', gender: 'male', skin: 1, hair: 5, suit: 2, style: 'curly', beard: 'none', glasses: 'yes' };
  const saved = CGB.store.getJSON('host', {});
  const cfg = Object.assign({}, defaults, saved && typeof saved === 'object' ? saved : {});
  // guard against out-of-range values from older saves
  // a name saved by an earlier version that is just one of its default names counts as no name
  cfg.name = String(cfg.name || '').trim();
  // (the old defaults are kept reversed, so no default name appears anywhere in the shipped file)
  if (['pip rosseforp', 'eeuqram ytram', 'tsoh'].includes(cfg.name.toLowerCase().split('').reverse().join(''))) cfg.name = '';
  ['skin', 'hair', 'suit'].forEach(k => { if (!(cfg[k] >= 0 && cfg[k] < CGB.hostOpts[k].length)) cfg[k] = defaults[k]; });
  ['style', 'beard', 'glasses', 'gender'].forEach(k => { if (!CGB.hostOpts[k].some(o => o[0] === cfg[k])) cfg[k] = defaults[k]; });
  return cfg;
})();
CGB.hostListeners = [];
/* The host's name, or '' when the teacher hasn't given him one (then no name label is shown) */
CGB.hostName = () => String(CGB.hostCfg.name || '').trim();
CGB.saveHost = function () {
  CGB.store.setJSON('host', CGB.hostCfg);
  CGB.hostListeners.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } });
};

/* Arm angles in degrees: [upper arm, forearm]. 0 = hanging straight down.
   A = the arm on the viewer's left (the host's right), B = the viewer's right. */
CGB.HOST_POSES = {
  idle:    { A: [14, 10], B: [-14, -10], face: 'smile' },
  present: { A: [52, -28], B: [-14, -10], face: 'smile' },
  point:   { A: [84, 6], B: [-14, -10], face: 'smile' },
  wave:    { A: [14, 10], B: [-150, -30], face: 'grin' },
  cheer:   { A: [158, -14], B: [-158, 14], face: 'grin' },
  clap:    { A: [6, -96], B: [-6, 96], face: 'grin' },
  shrug:   { A: [26, 118], B: [-26, -118], face: 'unsure' },
  groan:   { A: [152, -128], B: [-152, 128], face: 'groan' },
  gasp:    { A: [36, -132], B: [-36, 132], face: 'gasp' }
};

let hostSvgId = 0;
CGB.hostSVG = function (cfg) {
  const O = CGB.hostOpts, INK = '#1B1F3B', W = 5, female = cfg.gender === 'female', uid = 'hs' + (++hostSvgId);
  const skin = O.skin[cfg.skin], hair = O.hair[cfg.hair], suit = O.suit[cfg.suit];
  const shade = (hex, k) => { const n = parseInt(hex.slice(1), 16); const c = [16, 8, 0].map(s => Math.round(((n >> s) & 255) * k)); return '#' + c.map(v => Math.min(255, v).toString(16).padStart(2, '0')).join(''); };
  const suitDark = shade(suit, 0.72), skinDark = shade(skin, 0.88);
  // the woman wears a striped long-sleeved shirt (stripes in the chosen colour on white) and plain trousers;
  // her build is the man's, so the two hosts are the same figure in different clothes
  const top = female ? `url(#${uid}-stripe)` : suit, trousers = female ? shade(suit, 0.55) : suitDark;
  const stripes = female ? `<defs><pattern id="${uid}-stripe" patternUnits="userSpaceOnUse" width="30" height="30"><rect width="30" height="15" fill="#fff"/><rect y="15" width="30" height="15" fill="${suit}"/></pattern></defs>` : '';
  const st = `stroke="${INK}" stroke-width="${W}" stroke-linejoin="round" stroke-linecap="round"`;
  // one arm, drawn hanging down from its shoulder at (0,0); the forearm bends at the elbow (0,64)
  const arm = side => `
    <g class="host-arm host-arm-${side}">
      <circle cx="0" cy="2" r="18" fill="${top}" ${st}/>
      <path d="M-17 0 Q-19 34 -16 66 L16 66 Q19 34 17 0 Z" fill="${top}" ${st}/>
      <g transform="translate(0 64)"><g class="host-fore">
        <circle cx="0" cy="0" r="15.5" fill="${top}" ${st}/>
        <path d="M-15 -2 L-14 56 L14 56 L15 -2 Z" fill="${top}" ${st}/>
        <rect x="-15" y="50" width="30" height="11" rx="4" fill="#fff" ${st}/>
        <ellipse cx="0" cy="72" rx="15" ry="16" fill="${skin}" ${st}/>
        <ellipse cx="${side === 'a' ? 12 : -12}" cy="66" rx="6" ry="9" fill="${skin}" ${st} transform="rotate(${side === 'a' ? -25 : 25} ${side === 'a' ? 12 : -12} 66)"/>
      </g></g>
    </g>`;
  let hairBack = '', hairFront = '';
  if (cfg.style === 'curly') {
    const curls = [[96, 104, 19], [100, 78, 21], [114, 56, 22], [137, 44, 22], [163, 44, 22], [186, 56, 22], [200, 78, 21], [204, 104, 19], [126, 62, 18], [150, 56, 20], [174, 62, 18]];
    hairBack = curls.slice(0, 8).map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${hair}" ${st}/>`).join('');
    hairFront = `<path d="M100 98 Q104 56 150 52 Q196 56 200 98 Q186 76 150 74 Q114 76 100 98 Z" fill="${hair}"/>` +
      curls.slice(8).map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${hair}" ${st}/>`).join('') +
      `<path d="M106 92 Q118 72 140 70" fill="none" stroke="${shade(hair, 1.35)}" stroke-width="4" stroke-linecap="round" opacity="0.7"/>`;
  } else if (cfg.style === 'short') {
    hairFront = `<path d="M92 112 Q88 50 150 44 Q212 50 208 112 Q200 84 186 76 Q160 92 112 82 Q100 92 92 112 Z" fill="${hair}" ${st}/>`;
  } else if (cfg.style === 'swept') {
    hairFront = `<path d="M92 112 Q86 56 132 40 Q150 22 184 30 Q214 44 208 112 Q200 82 182 74 Q150 80 120 70 Q100 86 92 112 Z" fill="${hair}" ${st}/>
      <path d="M134 44 Q154 30 182 36" fill="none" stroke="${shade(hair, 1.35)}" stroke-width="4" stroke-linecap="round" opacity="0.7"/>`;
  } else if (cfg.style === 'bob') {
    // chin-length hair framing the face, with a side fringe
    hairBack = `<path d="M90 106 Q82 176 116 182 L184 182 Q218 176 210 106 Q212 48 150 42 Q88 48 90 106 Z" fill="${hair}" ${st}/>`;
    hairFront = `<path d="M94 110 Q90 52 150 46 Q210 52 206 110 Q200 80 176 74 Q146 90 112 80 Q98 90 94 110 Z" fill="${hair}" ${st}/>
      <path d="M118 58 Q142 48 170 54" fill="none" stroke="${shade(hair, 1.35)}" stroke-width="4" stroke-linecap="round" opacity="0.7"/>`;
  } else if (cfg.style === 'ponytail') {
    // hair pulled back and tied at the side, the tail swinging out behind
    hairBack = `<path d="M198 66 Q250 62 252 118 Q254 162 232 190 Q238 146 214 112 Z" fill="${hair}" ${st}/>`;
    hairFront = `<path d="M92 112 Q86 56 132 40 Q150 22 184 30 Q214 44 208 112 Q200 82 182 74 Q150 80 120 70 Q100 86 92 112 Z" fill="${hair}" ${st}/>
      <path d="M134 44 Q154 30 182 36" fill="none" stroke="${shade(hair, 1.35)}" stroke-width="4" stroke-linecap="round" opacity="0.7"/>
      <rect x="196" y="64" width="16" height="12" rx="5" fill="#FF7A1A" ${st} transform="rotate(24 204 70)"/>`;
  } else if (cfg.style === 'buzz') {
    hairFront = `<path d="M94 100 Q92 50 150 46 Q208 50 206 100 Q196 70 150 66 Q104 70 94 100 Z" fill="${hair}" opacity="0.85" ${st}/>`;
  } else {
    hairFront = `<path d="M118 62 Q134 52 152 54" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" opacity="0.55"/>`;
  }
  let beard = '';
  if (female) cfg = Object.assign({}, cfg, { beard: 'none' });
  if (cfg.beard === 'stubble') beard = `<path d="M100 118 Q104 170 150 172 Q196 170 200 118 Q186 150 150 152 Q114 150 100 118 Z" fill="${hair}" opacity="0.28"/>`;
  if (cfg.beard === 'beard') beard = `<path d="M96 112 Q96 182 150 186 Q204 182 204 112 Q194 140 176 146 Q150 136 124 146 Q106 140 96 112 Z" fill="${hair}" ${st}/>`;
  const glasses = cfg.glasses === 'yes' ? `
    <g class="host-glasses">
      <circle cx="128" cy="104" r="19" fill="rgba(210,235,255,0.22)" ${st}/>
      <circle cx="172" cy="104" r="19" fill="rgba(210,235,255,0.22)" ${st}/>
      <path d="M147 102 Q150 98 153 102" fill="none" ${st}/>
      <path d="M109 101 L96 98 M191 101 L204 98" fill="none" ${st}/>
    </g>` : '';
  const eye = x => `
    <g class="host-eye" style="transform-origin:${x}px 104px">
      <ellipse cx="${x}" cy="104" rx="12" ry="14" fill="#fff" stroke="${INK}" stroke-width="3.5"/>
      <g class="host-iris"><circle cx="${x}" cy="106" r="7" fill="#3b6e8f"/><circle cx="${x}" cy="106" r="3.8" fill="${INK}"/><circle cx="${x + 2.5}" cy="103" r="2" fill="#fff"/></g>
    </g>`;
  const browCol = shade(hair, cfg.style === 'bald' ? 1 : 0.8);
  return `<svg class="host-art" viewBox="0 0 300 440" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${CGB.escapeHtml(cfg.name || 'The host')}, the Showtime host">
  <g class="host-body">
    <ellipse cx="150" cy="430" rx="78" ry="9" fill="rgba(0,0,0,0.22)"/>
    ${stripes}
    <path d="M114 300 L110 404 L146 404 L150 320 L154 404 L190 404 L186 300 Z" fill="${trousers}" ${st}/>
    <ellipse cx="124" cy="412" rx="26" ry="12" fill="#6b4632" ${st}/>
    <ellipse cx="176" cy="412" rx="26" ry="12" fill="#6b4632" ${st}/>
    <path d="M96 182 Q150 160 204 182 Q218 236 210 312 Q150 326 90 312 Q82 236 96 182 Z" fill="${top}" ${st}/>
    ${female ? `<path d="M126 170 Q150 196 174 170 Q150 178 126 170 Z" fill="#fff" ${st}/>
    <path d="M150 196 L150 312" stroke="${INK}" stroke-width="3" opacity="0.35"/>
    <circle cx="150" cy="226" r="4" fill="${INK}"/><circle cx="150" cy="252" r="4" fill="${INK}"/><circle cx="150" cy="278" r="4" fill="${INK}"/>` : `<path d="M132 172 L150 236 L168 172 Z" fill="#fff" ${st}/>
    <path d="M128 172 L116 200 L132 214 L124 230 L150 270 L138 214 Z M172 172 L184 200 L168 214 L176 230 L150 270 L162 214 Z" fill="${suitDark}" ${st}/>
    <circle cx="150" cy="276" r="4.5" fill="${INK}"/><circle cx="150" cy="298" r="4.5" fill="${INK}"/>`}
    <path d="M102 222 L124 216 L126 226 L104 232 Z" fill="#FF7A1A" ${st}/>
    <path d="M150 184 L130 172 L130 196 Z M150 184 L170 172 L170 196 Z" fill="#FF7A1A" ${st}/>
    <circle cx="150" cy="184" r="6" fill="#FF7A1A" ${st}/>
    <g transform="translate(88 188)">${arm('a')}</g>
    <g transform="translate(212 188)">${arm('b')}</g>
    <g class="host-head">
      <rect x="136" y="148" width="28" height="28" rx="8" fill="${skinDark}" ${st}/>
      ${hairBack}
      <ellipse cx="94" cy="112" rx="11" ry="16" fill="${skin}" ${st}/>
      <ellipse cx="206" cy="112" rx="11" ry="16" fill="${skin}" ${st}/>
      <path class="host-face" d="M150 48 Q206 48 206 108 Q206 168 150 170 Q94 168 94 108 Q94 48 150 48 Z" fill="${skin}" ${st}/>
      ${beard}
      <circle cx="114" cy="132" r="10" fill="#FF8A80" opacity="0.35"/><circle cx="186" cy="132" r="10" fill="#FF8A80" opacity="0.35"/>
      ${eye(128)}${eye(172)}
      <path class="host-brow host-brow-a" d="M114 82 Q127 74 140 81" fill="none" stroke="${browCol}" stroke-width="6" stroke-linecap="round"/>
      <path class="host-brow host-brow-b" d="M160 81 Q173 74 186 82" fill="none" stroke="${browCol}" stroke-width="6" stroke-linecap="round"/>
      <path d="M146 112 Q140 126 150 128 Q156 128 156 124" fill="none" stroke="${INK}" stroke-width="3.5" stroke-linecap="round"/>
      <g class="host-mouth">
        <path class="m-smile" d="M130 140 Q150 156 170 140" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round"/>
        <path class="m-grin" d="M126 138 Q150 170 174 138 Z" fill="#7a1f2b" ${st}/>
        <path class="m-unsure" d="M134 146 Q150 140 166 146" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round"/>
        <ellipse class="m-groan" cx="150" cy="148" rx="9" ry="11" fill="#7a1f2b" ${st}/>
        <ellipse class="m-talk" cx="150" cy="146" rx="15" ry="9" fill="#7a1f2b" ${st}/>
      </g>
      ${glasses}
      ${hairFront}
    </g>
  </g>
</svg>`;
};

/* Mount the host in a container element. Returns a small controller. */
CGB.createHost2D = function (container, opts) {
  opts = opts || {};
  const wrap = document.createElement('div');
  wrap.className = 'host-wrap' + (opts.className ? ' ' + opts.className : '');
  container.appendChild(wrap);
  let rest = 'idle', target = 'idle', gestureTimer = null, talkTimer = null, blinkTimer = null, gesturing = false;
  function apply() {
    const P = CGB.HOST_POSES[target] || CGB.HOST_POSES.idle;
    const set = (sel, deg) => { const el = wrap.querySelector(sel); if (el) el.style.transform = `rotate(${deg}deg)`; };
    set('.host-arm-a', P.A[0]); set('.host-arm-a .host-fore', P.A[1]);
    set('.host-arm-b', P.B[0]); set('.host-arm-b .host-fore', P.B[1]);
    wrap.dataset.pose = target;
    wrap.dataset.face = P.face;
  }
  function render() { wrap.innerHTML = CGB.hostSVG(CGB.hostCfg); apply(); }
  function gesture(name, ms) {
    target = CGB.HOST_POSES[name] ? name : 'idle';
    gesturing = true;
    apply();
    clearTimeout(gestureTimer);
    gestureTimer = setTimeout(() => { gesturing = false; target = rest; apply(); }, ms || 1800);
  }
  function blinkLoop() {
    clearTimeout(blinkTimer);
    blinkTimer = setTimeout(() => {
      wrap.classList.add('blink');
      setTimeout(() => wrap.classList.remove('blink'), 140);
      blinkLoop();
    }, 2400 + Math.random() * 2600);
  }
  render();
  blinkLoop();
  CGB.hostListeners.push(render);
  return {
    el: wrap,
    rebuild: render,
    gesture,
    setRest(name) { if (name === rest) return; rest = name; if (!gesturing) { target = rest; apply(); } },
    setLook(dir) { const v = dir || ''; if (wrap.dataset.look !== v) wrap.dataset.look = v; },
    talk(ms) { wrap.classList.add('talking'); clearTimeout(talkTimer); talkTimer = setTimeout(() => wrap.classList.remove('talking'), ms); },
    /* top-centre of the head, in page coordinates, for placing a speech bubble */
    headPoint() {
      const f = wrap.querySelector('.host-head'); if (!f) return null;   // includes the hair
      const r = f.getBoundingClientRect();
      if (!r.width) return null;
      return { x: r.left + r.width / 2, y: r.top, w: r.width };
    }
  };
};

/* A small corner presence for the games without a stage: the host's head and shoulders
   with a caption bubble beside him. It sits in the page layout (never on top of the
   board, track, questions or controls). Captions only; no voice. */
CGB.createHostCorner = function (container, opts) {
  opts = opts || {};
  const box = document.createElement('div');
  box.className = 'host-corner' + (opts.className ? ' ' + opts.className : '');
  box.innerHTML = '<div class="hc-bust" aria-hidden="true"></div><div class="hc-bubble" role="status" aria-live="polite"><b class="hc-name"></b><span class="hc-text"></span></div>';
  container.appendChild(box);
  const host = CGB.createHost2D(box.querySelector('.hc-bust'), { className: 'host-mini' });
  const bubble = box.querySelector('.hc-bubble');
  let hideTimer = null;
  function say(text, gesture, ms) {
    if (!text) return;
    const nm = box.querySelector('.hc-name');
    nm.textContent = CGB.hostName(); nm.hidden = !CGB.hostName();
    box.querySelector('.hc-text').textContent = text;
    bubble.classList.add('show');
    if (gesture) host.gesture(gesture, ms || 1800);
    host.talk(Math.min(2400, text.length * 45 + 150));
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => bubble.classList.remove('show'), Math.max(3800, text.length * 80));
  }
  function quiet() { clearTimeout(hideTimer); bubble.classList.remove('show'); }
  return { el: box, host, say, quiet };
};
