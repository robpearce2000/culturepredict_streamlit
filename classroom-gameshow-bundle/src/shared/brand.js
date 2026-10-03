'use strict';
/* =========================================================
   BRAND ARTWORK, drawn in code as SVG (no image files)
   - Bundle wordmark
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

  /* The bundle wordmark. Text lengths are fixed so it lays out the same with any fallback font. */
  function wordmark(opts) {
    opts = opts || {};
    const title = opts.title !== false ? '<title>The Classroom Gameshow Bundle</title>' : '';
    return `<svg viewBox="0 0 1000 450" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The Classroom Gameshow Bundle">${title}
  <g>
    ${star(70, 120, 34, SUN, 0.2)}${star(940, 95, 26, TAN, 0.5)}${star(905, 300, 18, TEAL, 0.1)}${star(110, 330, 16, SUN, 0.7)}
    <circle cx="160" cy="70" r="9" fill="${TEAL}"/><circle cx="860" cy="200" r="7" fill="${SUN}"/><circle cx="60" cy="230" r="6" fill="${TAN}"/>
  </g>
  <g>
    <rect x="245" y="22" width="510" height="76" rx="38" fill="${INK}" stroke="#fff" stroke-width="5"/>
    ${T(500, 79, 54, 'THE CLASSROOM', `fill="${SUN}" textLength="430" lengthAdjust="spacingAndGlyphs"`)}
  </g>
  <g transform="rotate(-3 500 205)">
    ${T(508, 278, 200, 'GAMESHOW', `fill="${TAN}" stroke="${INK}" stroke-width="16" stroke-linejoin="round" paint-order="stroke" textLength="880" lengthAdjust="spacingAndGlyphs"`)}
    ${T(500, 266, 200, 'GAMESHOW', `fill="#fff" stroke="${INK}" stroke-width="16" stroke-linejoin="round" paint-order="stroke" textLength="880" lengthAdjust="spacingAndGlyphs"`)}
  </g>
  <g>
    <path d="M250 318 L750 318 L728 362 L750 406 L250 406 L272 362 Z" fill="${TEAL_D}" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    <path d="M250 318 L196 330 L214 362 L196 394 L250 406 Z" fill="#064442" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    <path d="M750 318 L804 330 L786 362 L804 394 L750 406 Z" fill="#064442" stroke="${INK}" stroke-width="6" stroke-linejoin="round"/>
    ${T(500, 386, 66, 'BUNDLE', `fill="#fff" textLength="300" lengthAdjust="spacingAndGlyphs"`)}
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
  return { wordmark, oteLogo, opLogo, oteArt, opArt, star };
})();
