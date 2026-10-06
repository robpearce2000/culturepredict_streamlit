'use strict';
/* =========================================================
   HEX HUNT LETTERS
   A hexagon's letter is the first letter of the answer's first significant word, ignoring
   "the", "a" and "an" (and a leading "in", "on", "at", "from" or "into"). Answers that can't
   be given away by one letter return null and never go on a Hex Hunt board: numbers,
   equations and formulae, yes/no, long answers, and answers starting with words like
   "any", "both" or "because", and algebra or symbols ("x² + 3x", "y = 2x + 1").
   Shared by the game (CGB.hexLetter) and the question-pack checker (tools/packs.js), so a
   pack's hexOk flags always match what the board does.
   ========================================================= */
(function (root) {
  const SKIP_LEAD = /^((in|into|on|at|from)\s+)?((the|a|an)\s+)?/i;
  const NOT_A_CLUE = /^(any|yes|no|true|false|it|its|it's|they|them|their|this|these|those|there|to|so|when|because|by|one|if|both|either|all|each|only|more|less|same|that|which|with|without|after|before|as|for|of|about|roughly|around|over|under|nearly)\b/i;
  function hexLetter(answer) {
    const a = String(answer).trim();
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
  if (typeof module !== 'undefined' && module.exports) module.exports = hexLetter;
  else root.CGB.hexLetter = hexLetter;
})(typeof window !== 'undefined' ? window : globalThis);
