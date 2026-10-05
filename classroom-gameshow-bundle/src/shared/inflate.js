'use strict';
/* =========================================================
   UNPACKING THE BUILT-IN QUESTIONS
   The build stores each topic pack's questions DEFLATE-compressed (as base64) to keep the
   single file small; CGB.unpackJSON(text) turns one back into its rows the first time the
   pack is used. A small, self-contained raw DEFLATE (RFC 1951) decoder, so it works offline
   in every browser and needs nothing asynchronous. Shared with the tests (module.exports).
   ========================================================= */
(function (root) {
  const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

  // a canonical Huffman code from code lengths: counts per length and symbols in code order
  function table(lengths) {
    const counts = new Uint16Array(16), symbols = new Uint16Array(lengths.length), offs = new Uint16Array(16);
    lengths.forEach(l => { counts[l]++; });
    counts[0] = 0;
    for (let i = 1; i < 16; i++) offs[i] = offs[i - 1] + counts[i - 1];
    lengths.forEach((l, s) => { if (l) symbols[offs[l]++] = s; });
    return { counts, symbols };
  }
  let FIXED_LIT = null, FIXED_DIST = null;

  function inflateRaw(src) {
    let pos = 0, bit = 0, out = new Uint8Array(src.length * 4 + 1024), n = 0;
    const need = k => { if (n + k > out.length) { const o = new Uint8Array(Math.max(out.length * 2, n + k)); o.set(out); out = o; } };
    const bits = k => {
      let v = 0;
      for (let i = 0; i < k; i++) {
        if (pos >= src.length) throw new Error('inflate: unexpected end of data');
        v |= ((src[pos] >> bit) & 1) << i;
        if (++bit === 8) { bit = 0; pos++; }
      }
      return v;
    };
    const decode = t => {
      let code = 0, first = 0, index = 0;
      for (let len = 1; len < 16; len++) {
        code |= bits(1);
        const c = t.counts[len];
        if (code - first < c) return t.symbols[index + code - first];
        index += c; first = (first + c) << 1; code <<= 1;
      }
      throw new Error('inflate: bad code');
    };
    let last = 0;
    while (!last) {
      last = bits(1);
      const type = bits(2);
      if (type === 0) {
        if (bit) { bit = 0; pos++; }
        const len = src[pos] | (src[pos + 1] << 8);
        pos += 4;
        need(len);
        out.set(src.subarray(pos, pos + len), n);
        n += len; pos += len;
        continue;
      }
      let lit, dist;
      if (type === 1) {
        if (!FIXED_LIT) {
          const l = new Array(288);
          for (let i = 0; i < 288; i++) l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
          FIXED_LIT = table(l);
          FIXED_DIST = table(new Array(30).fill(5));
        }
        lit = FIXED_LIT; dist = FIXED_DIST;
      } else if (type === 2) {
        const hlit = bits(5) + 257, hdist = bits(5) + 1, hclen = bits(4) + 4;
        const cl = new Array(19).fill(0);
        for (let i = 0; i < hclen; i++) cl[CL_ORDER[i]] = bits(3);
        const clt = table(cl), lengths = [];
        while (lengths.length < hlit + hdist) {
          const s = decode(clt);
          if (s < 16) lengths.push(s);
          else if (s === 16) { const p = lengths[lengths.length - 1], r = 3 + bits(2); for (let i = 0; i < r; i++) lengths.push(p); }
          else { const r = s === 17 ? 3 + bits(3) : 11 + bits(7); for (let i = 0; i < r; i++) lengths.push(0); }
        }
        lit = table(lengths.slice(0, hlit));
        dist = table(lengths.slice(hlit));
      } else throw new Error('inflate: bad block type');
      for (;;) {
        const s = decode(lit);
        if (s < 256) { need(1); out[n++] = s; continue; }
        if (s === 256) break;
        const li = s - 257, len = LEN_BASE[li] + bits(LEN_EXTRA[li]);
        const di = decode(dist), d = DIST_BASE[di] + bits(DIST_EXTRA[di]);
        need(len);
        for (let i = 0; i < len; i++, n++) out[n] = out[n - d];
      }
    }
    return out.subarray(0, n);
  }

  function base64Bytes(b64) {
    const bin = root.atob ? root.atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  function unpackJSON(b64) {
    return JSON.parse(new TextDecoder('utf-8').decode(inflateRaw(base64Bytes(b64))));
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { inflateRaw, unpackJSON };
  else { root.CGB = root.CGB || {}; root.CGB.unpackJSON = unpackJSON; }
})(typeof window !== 'undefined' ? window : globalThis);
