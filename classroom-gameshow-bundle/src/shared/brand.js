'use strict';
/* =========================================================
   BRAND ARTWORK, drawn in code as SVG (no image files)
   - Showtime wordmark
   - Over the Edge and Outpace logos
   - Launcher card art for each game
   ========================================================= */
CGB.brand = (() => {
  const INK = '#1B1F3B', SUN = '#FFC93C', TAN = '#FF7A1A', TEAL = '#12A4A0', TEAL_D = '#0A6663';
  const T = (x, y, size, text, extra) => `<text x="${x}" y="${y}" font-family="Lilita One, Arial Rounded MT Bold, Trebuchet MS, sans-serif" font-size="${size}" text-anchor="middle" ${extra || ''}>${text}</text>`;
  function star(cx, cy, r, fill, rot) {
    let d = '';
    for (let i = 0; i < 10; i++) {
      const a = (i * Math.PI) / 5 - Math.PI / 2 + (rot || 0), rr = i % 2 ? r * 0.45 : r;
      d += (i ? 'L' : 'M') + (cx + Math.cos(a) * rr).toFixed(1) + ' ' + (cy + Math.sin(a) * rr).toFixed(1);
    }
    return `<path d="${d}Z" fill="${fill}"/>`;
  }

  /* The Showtime wordmark. Text lengths are fixed so it lays out the same with any fallback font. */
  function wordmark(opts) {
    opts = opts || {};
    const title = opts.title !== false ? '<title>Showtime: Classroom Gameshows</title>' : '';
    // marquee bulbs around the main word
    let bulbs = '';
    for (let i = 0; i < 17; i++) { const x = 80 + i * 52.5; bulbs += `<circle cx="${x.toFixed(1)}" cy="34" r="9" fill="${i % 2 ? SUN : '#fff'}" stroke="${INK}" stroke-width="3"/><circle cx="${x.toFixed(1)}" cy="266" r="9" fill="${i % 2 ? '#fff' : SUN}" stroke="${INK}" stroke-width="3"/>`; }
    return `<svg viewBox="0 0 1000 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Showtime: Classroom Gameshows">${title}
  <g>
    ${star(36, 150, 30, SUN, 0.2)}${star(966, 140, 26, TAN, 0.5)}${star(950, 330, 16, TEAL, 0.1)}${star(52, 340, 16, SUN, 0.7)}
  </g>
  <rect x="58" y="16" width="884" height="268" rx="34" fill="${INK}" stroke="#fff" stroke-width="6"/>
  ${bulbs}
  <g transform="rotate(-2.5 500 160)">
    ${T(508, 222, 196, 'SHOWTIME', `fill="${TAN}" stroke="${INK}" stroke-width="16" stroke-linejoin="round" paint-order="stroke" textLength="780" lengthAdjust="spacingAndGlyphs"`)}
    ${T(500, 210, 196, 'SHOWTIME', `fill="#fff" stroke="${INK}" stroke-width="16" stroke-linejoin="round" paint-order="stroke" textLength="780" lengthAdjust="spacingAndGlyphs"`)}
  </g>
  <g>
    <path d="M150 296 L850 296 L828 342 L850 388 L150 388 L172 342 Z" fill="${TEAL_D}" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    <path d="M150 296 L96 308 L114 342 L96 376 L150 388 Z" fill="#064442" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    <path d="M850 296 L904 308 L886 342 L904 376 L850 388 Z" fill="#064442" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    ${T(500, 365, 54, 'CLASSROOM GAMESHOWS', `fill="#fff" textLength="600" lengthAdjust="spacingAndGlyphs"`)}
  </g>
</svg>`;
  }

  /* Over the Edge logo: hazard-stripe edge under the name */
  function oteLogo() {
    let stripes = '';
    for (let x = -20; x < 640; x += 40) stripes += `<path d="M${x} 150 l20 0 l-26 34 l-20 0 Z" fill="${INK}"/>`;
    return `<svg viewBox="0 0 620 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Over the Edge">
  ${T(310, 52, 52, 'OVER THE', `fill="${SUN}" stroke="${INK}" stroke-width="10" paint-order="stroke" textLength="260" lengthAdjust="spacingAndGlyphs"`)}
  ${T(310, 142, 110, 'EDGE', `fill="#fff" stroke="${INK}" stroke-width="14" paint-order="stroke" textLength="300" lengthAdjust="spacingAndGlyphs"`)}
  <clipPath id="oteStripe"><rect x="150" y="150" width="320" height="34" rx="6"/></clipPath>
  <rect x="150" y="150" width="320" height="34" rx="6" fill="${SUN}"/>
  <g clip-path="url(#oteStripe)">${stripes}</g>
  <rect x="150" y="150" width="320" height="34" rx="6" fill="none" stroke="${INK}" stroke-width="5"/>
</svg>`;
  }

  /* Outpace logo: slanted with speed lines */
  function opLogo() {
    return `<svg viewBox="0 0 620 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Outpace">
  <g stroke="${SUN}" stroke-width="10" stroke-linecap="round"><path d="M20 70 H110"/><path d="M40 100 H120"/><path d="M10 130 H100"/></g>
  <g transform="skewX(-12)">
    ${T(370, 135, 130, 'OUTPACE', `fill="${SUN}" stroke="${INK}" stroke-width="14" paint-order="stroke" textLength="440" lengthAdjust="spacingAndGlyphs"`)}
  </g>
</svg>`;
  }

  /* Launcher card art for Over the Edge */
  function oteArt() {
    let stripes = '';
    for (let x = -40; x < 700; x += 36) stripes += `<path d="M${x} 232 l18 0 l-20 22 l-18 0 Z" fill="${INK}"/>`;
    const coins = [];
    const pos = [[150, 205], [205, 212], [262, 206], [318, 214], [372, 207], [428, 213], [484, 206], [180, 186], [240, 190], [300, 186], [360, 191], [420, 186], [470, 190]];
    pos.forEach(([x, y], i) => coins.push(`<ellipse cx="${x}" cy="${y}" rx="27" ry="10" fill="${i === 9 ? SUN : '#D7DEE6'}" stroke="${INK}" stroke-width="4"/>`));
    return `<svg viewBox="0 0 640 328" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs><linearGradient id="oteBg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0E4D5C"/><stop offset="1" stop-color="#082430"/></linearGradient></defs>
  <rect width="640" height="328" fill="url(#oteBg)"/>
  <g opacity="0.35" stroke="#4FF0D8" stroke-width="3">${[80, 180, 280, 380, 480, 580].map(x => `<path d="M${x} 0 V150"/>`).join('')}</g>
  <g fill="#E8F1F4">${[[60, 40], [110, 78], [160, 40], [210, 78], [260, 40], [310, 78], [360, 40], [410, 78], [460, 40], [510, 78], [560, 40]].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="7"/>`).join('')}</g>
  <path d="M100 160 H540 L560 232 H80 Z" fill="${SUN}" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/>
  ${coins.join('')}
  <clipPath id="oteArtStripe"><path d="M80 232 H560 V254 H80 Z"/></clipPath>
  <rect x="80" y="232" width="480" height="22" fill="${SUN}"/>
  <g clip-path="url(#oteArtStripe)">${stripes}</g>
  <rect x="80" y="232" width="480" height="22" fill="none" stroke="${INK}" stroke-width="5"/>
  <g transform="translate(300 286) rotate(28)"><ellipse rx="27" ry="10" fill="#D7DEE6" stroke="${INK}" stroke-width="4"/></g>
  <g transform="translate(372 300) rotate(-20)"><ellipse rx="27" ry="10" fill="#D7DEE6" stroke="${INK}" stroke-width="4"/></g>
  <g stroke="#fff" stroke-width="5" stroke-linecap="round" opacity="0.9"><path d="M262 276 l-14 10"/><path d="M405 290 l14 8"/></g>
  <g transform="translate(560 120)">${star(0, 0, 30, TAN, 0.3)}</g>
  <path d="M530 120 h-0" />
</svg>`;
  }

  /* Launcher card art for Outpace */
  function opArt() {
    function atom(cx, cy, col, glow) {
      return `<g transform="translate(${cx} ${cy})">
        <circle r="46" fill="${glow}" opacity="0.25"/>
        <ellipse rx="44" ry="16" fill="none" stroke="${col}" stroke-width="4"/>
        <ellipse rx="44" ry="16" fill="none" stroke="${col}" stroke-width="4" transform="rotate(60)"/>
        <ellipse rx="44" ry="16" fill="none" stroke="${col}" stroke-width="4" transform="rotate(-60)"/>
        <circle r="13" fill="${col}" stroke="${INK}" stroke-width="3"/>
      </g>`;
    }
    let cells = '';
    for (let i = 0; i < 7; i++) cells += `<rect x="${40 + i * 82}" y="226" width="74" height="30" rx="6" fill="${i === 0 ? TEAL_D : '#2A3170'}" stroke="#0B1026" stroke-width="3"/>`;
    let starsBg = '';
    [[40, 30], [120, 70], [210, 24], [300, 60], [420, 30], [520, 66], [600, 28], [580, 140], [30, 150]].forEach(([x, y]) => { starsBg += `<circle cx="${x}" cy="${y}" r="2.5" fill="#fff" opacity="0.8"/>`; });
    return `<svg viewBox="0 0 640 328" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs><radialGradient id="opBg" cx="0.5" cy="0.2" r="0.9"><stop offset="0" stop-color="#2B2F7A"/><stop offset="1" stop-color="#0B1026"/></radialGradient></defs>
  <rect width="640" height="328" fill="url(#opBg)"/>
  ${starsBg}
  ${cells}
  <g transform="translate(77 214)"><path d="M0 0 V-62" stroke="#fff" stroke-width="5"/><path d="M0 -62 h34 l-8 12 l8 12 h-34 Z" fill="#fff" stroke="${INK}" stroke-width="3"/></g>
  <g stroke="${SUN}" stroke-width="6" stroke-linecap="round" opacity="0.8"><path d="M300 170 H380"/><path d="M310 196 H370"/><path d="M296 144 H356"/></g>
  ${atom(250, 170, SUN, SUN)}
  ${atom(480, 170, '#E879F9', '#C026D3')}
  <g font-family="Lilita One, Trebuchet MS, sans-serif" font-size="22" text-anchor="middle">
    <text x="250" y="110" fill="#fff" stroke="${INK}" stroke-width="5" paint-order="stroke">YOU</text>
    <text x="480" y="110" fill="#fff" stroke="${INK}" stroke-width="5" paint-order="stroke">HUNTER</text>
  </g>
</svg>`;
  }

  /* Category Clash logo */
  function ccLogo() {
    return `<svg viewBox="0 0 620 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Category Clash">
  <rect x="150" y="10" width="320" height="64" rx="32" fill="${INK}" stroke="#fff" stroke-width="5"/>
  ${T(310, 58, 46, 'CATEGORY', `fill="${SUN}" textLength="250" lengthAdjust="spacingAndGlyphs"`)}
  <g transform="rotate(-3 310 150)">
    ${T(316, 190, 118, 'CLASH', `fill="${TAN}" stroke="${INK}" stroke-width="14" stroke-linejoin="round" paint-order="stroke" textLength="380" lengthAdjust="spacingAndGlyphs"`)}
    ${T(310, 182, 118, 'CLASH', `fill="#fff" stroke="${INK}" stroke-width="14" stroke-linejoin="round" paint-order="stroke" textLength="380" lengthAdjust="spacingAndGlyphs"`)}
  </g>
  <path d="M86 92 L112 92 L98 124 L120 124 L80 178 L92 138 L72 138 Z" fill="${SUN}" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/>
  <path d="M534 92 L560 92 L546 124 L568 124 L528 178 L540 138 L520 138 Z" fill="${SUN}" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/>
</svg>`;
  }

  /* Hex Hunt logo */
  function hexPath(cx, cy, r) {
    let d = '';
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); d += (i ? 'L' : 'M') + (cx + r * Math.cos(a)).toFixed(1) + ' ' + (cy + r * Math.sin(a)).toFixed(1); }
    return d + 'Z';
  }
  function hhLogo() {
    return `<svg viewBox="0 0 620 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Hex Hunt">
  <path d="${hexPath(118, 100, 86)}" fill="${TEAL_D}" stroke="${INK}" stroke-width="8" stroke-linejoin="round"/>
  <path d="${hexPath(118, 100, 66)}" fill="none" stroke="${SUN}" stroke-width="4" stroke-dasharray="10 8"/>
  ${T(118, 124, 70, 'HEX', `fill="#fff" stroke="${INK}" stroke-width="10" paint-order="stroke" textLength="112" lengthAdjust="spacingAndGlyphs"`)}
  ${T(400, 150, 128, 'HUNT', `fill="${TAN}" stroke="${INK}" stroke-width="14" stroke-linejoin="round" paint-order="stroke" textLength="360" lengthAdjust="spacingAndGlyphs"`)}
  ${T(394, 142, 128, 'HUNT', `fill="#fff" stroke="${INK}" stroke-width="14" stroke-linejoin="round" paint-order="stroke" textLength="360" lengthAdjust="spacingAndGlyphs"`)}
</svg>`;
  }

  /* Launcher art: a category board */
  function ccArt() {
    let tiles = '';
    const cols = 5, w = 104, h = 52, x0 = 44, y0 = 82;
    for (let c = 0; c < cols; c++) {
      tiles += `<rect x="${x0 + c * (w + 8)}" y="22" width="${w}" height="50" rx="9" fill="${INK}" stroke="#000" stroke-width="3"/><rect x="${x0 + c * (w + 8) + 18}" y="40" width="${w - 36}" height="12" rx="6" fill="${SUN}"/>`;
      for (let r = 0; r < 4; r++) {
        const x = x0 + c * (w + 8), y = y0 + r * (h + 8);
        const used = (c === 1 && r === 0) || (c === 3 && r === 1) || (c === 0 && r === 2);
        tiles += used
          ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="#2B2045" stroke="#120A20" stroke-width="3"/><text x="${x + w / 2}" y="${y + 36}" font-size="26" text-anchor="middle" fill="${c === 3 ? SUN : TEAL}">${c === 3 ? '■' : '●'}</text>`
          : `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="#F06A12" stroke="${INK}" stroke-width="4"/>` + T(x + w / 2, y + 38, 30, String((r + 1) * 100), `fill="#fff" stroke="${INK}" stroke-width="6" paint-order="stroke"`);
      }
    }
    return `<svg viewBox="0 0 640 328" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs><radialGradient id="ccBg" cx="0.5" cy="0" r="1"><stop offset="0" stop-color="#3A1650"/><stop offset="1" stop-color="#1A0F2E"/></radialGradient></defs>
  <rect width="640" height="328" fill="url(#ccBg)"/>${tiles}
  <g transform="translate(560 286)">${star(0, 0, 26, SUN, 0.2)}</g>
</svg>`;
  }

  /* Launcher art: a hexagon board with two paths */
  function hhArt() {
    const r = 30, dx = r * Math.sqrt(3), dy = r * 1.5;
    const A = new Set(['0,2', '1,2', '2,1', '3,1', '4,2']), B = new Set(['2,0', '2,3', '3,4']);
    const letters = 'PCRGAMHDEOSTNBLKVIFW';
    let hexes = '', k = 0;
    for (let row = 0; row < 5; row++) for (let col = 0; col < 7; col++) {
      const cx = 92 + col * dx + (row % 2 ? dx / 2 : 0), cy = 58 + row * dy * 1.12;
      const key = col + ',' + row;
      const fill = A.has(key) ? TAN : B.has(key) ? TEAL : '#262C66';
      hexes += `<path d="${hexPath(cx, cy, r)}" fill="${fill}" stroke="${INK}" stroke-width="4" stroke-linejoin="round"/>`;
      if (!A.has(key) && !B.has(key)) hexes += T(cx, cy + 9, 26, letters[k++ % letters.length], `fill="#fff"`);
      else hexes += `<text x="${cx}" y="${cy + 8}" font-size="22" text-anchor="middle" fill="${INK}">${A.has(key) ? '●' : '■'}</text>`;
    }
    return `<svg viewBox="0 0 640 328" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs><radialGradient id="hhBg" cx="0.5" cy="0.3" r="0.9"><stop offset="0" stop-color="#14425A"/><stop offset="1" stop-color="#0A1A2C"/></radialGradient></defs>
  <rect width="640" height="328" fill="url(#hhBg)"/>
  <rect x="18" y="20" width="14" height="290" rx="7" fill="${TAN}"/><rect x="608" y="20" width="14" height="290" rx="7" fill="${TAN}"/>
  ${hexes}
</svg>`;
  }
  return { wordmark, oteLogo, opLogo, oteArt, opArt, ccLogo, hhLogo, ccArt, hhArt, hexPath, star };
})();
