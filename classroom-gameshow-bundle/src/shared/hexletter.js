'use strict';
/* =========================================================
   HEX HUNT LETTERS
   A hexagon's letter is the first letter of the answer's first significant word, ignoring
   "the", "a" and "an" (and a leading "in", "on", "at", "from" or "into"). Answers that can't
   be given away by one letter return null and never go on a Hex Hunt board: numbers,
   equations and formulae, yes/no, long answers, and answers starting with words like
   "any", "both" or "because", and algebra or symbols ("x² + 3x", "y = 2x + 1").
   A single number (Maths and the sciences, not History or Geography, where numbers are years
   said differently) takes the first letter of the number said in words: 1236 is "one thousand…"
   (O), −5 "minus five" (M), 0.4 "zero point four" (Z), 3/4 "three quarters" (T), 6 × 10⁷
   "six times ten to the seven" (S), 20 cm "twenty" (T). Answers with more than one number
   (ratios, coordinates, "15 and 25") or with letters joined to the number (2x) stay off the board.
   Shared by the game (CGB.hexLetter) and the question-pack checker (tools/packs.js), so a
   pack's hexOk flags always match what the board does.
   ========================================================= */
(function (root) {
  const SKIP_LEAD = /^((in|into|on|at|from)\s+)?((the|a|an)\s+)?/i;
  const NOT_A_CLUE = /^(any|yes|no|true|false|it|its|it's|they|them|their|this|these|those|there|to|so|when|because|by|one|if|both|either|all|each|only|more|less|same|that|which|with|without|after|before|as|for|of|about|roughly|around|over|under|nearly)\b/i;
  const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
  const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
  /* the first word of a whole number said in English: 1236 "one" (thousand), 45 "forty", 0 "zero" */
  function firstWord(n) {
    for (const scale of [1e12, 1e9, 1e6, 1e3, 100]) if (n >= scale) return firstWord(Math.floor(n / scale));
    return n < 20 ? ONES[n] : TENS[Math.floor(n / 10)];
  }
  /* an answer that is one number (with its unit), said in words: its first word, or null */
  const NUM = /^([£$€]\s?)?([−-])?(\d{1,3}(?:[ ,\u2009]\d{3})+|\d+)?(\.\d+)?(?:\s?\/\s?\d+)?([½¼¾])?(?:\s?[×x]\s?10\s?(?:\^|\*\*)?\s?[⁻⁰¹²³⁴⁵⁶⁷⁸⁹-]+)?/;
  function spokenNumber(answer) {
    const a = String(answer).trim();
    if (!/^[£$€]?\s?[−-]?(\d|[½¼¾])/.test(a)) return null;
    const m = NUM.exec(a); if (!m || !m[0]) return null;
    const rest = a.slice(m[0].length);
    if (/^[A-Za-z(]/.test(rest)) return null;                      // 2x, 3n, 4(x + 1): algebra
    if (/\d/.test(rest) || /[=+×÷^<>≤≥:]|\s[−-]\s/.test(rest)) return null;   // a second number or an expression
    if (m[2]) return 'minus';
    const whole = m[3] ? +m[3].replace(/[ ,\u2009]/g, '') : null;
    if (whole === null) return m[5] ? (m[5] === '½' ? 'one' : m[5] === '¼' ? 'one' : 'three') : (m[4] ? 'point' : null);
    return firstWord(whole);
  }
  function hexLetter(answer, subject) {
    const a = String(answer).trim();
    if (!/^(history|geography)$/i.test(subject || '')) { const w = spokenNumber(a); if (w) return w[0].toUpperCase(); }
    if (NOT_A_CLUE.test(a) || a.split(/\s+/).length > 10 || /[→⇌₀-₉ₙ]/.test(a.split(/[,(]/)[0])) return null;
    const core = a.split(/[.,;:(]|\s[-–]\s/)[0].trim();
    if (core.split(/\s+/).length > 5) return null;
    const s = core.replace(SKIP_LEAD, '');
    if (NOT_A_CLUE.test(s)) return null;
    // algebra and symbols are not word clues: "x² + 3x", "y = 2x + 1", "C"
    if (/[=+×÷^²³√<>≤≥]|\s[−-]\s|\d\s*\//.test(s) || /^[A-Za-z]([\s\d(=+−×÷^²³√/]|$)/.test(s)) return null;
    const m = /^[A-Za-z]/.exec(s);
    return m ? m[0].toUpperCase() : null;
  }
  hexLetter.spoken = (answer, subject) => /^(history|geography)$/i.test(subject || '') ? null : spokenNumber(answer);
  if (typeof module !== 'undefined' && module.exports) module.exports = hexLetter;
  else root.CGB.hexLetter = hexLetter;
})(typeof window !== 'undefined' ? window : globalThis);
