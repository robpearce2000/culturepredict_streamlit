/* Teams: one palette and one set of team names shared by every game.
   Team 1 is always blue, Team 2 orange, Team 3 purple, Team 4 teal, and each
   colour comes with a shape so nothing depends on colour alone.
     css   fills, borders and stripes
     text  the colour as text on white or paper
     light the colour as text or marks on the dark game backgrounds */
(function () {
  const CGB = window.CGB;
  CGB.TEAMS = [
    { hex: 0x2563EB, css: '#2563EB', text: '#1D4ED8', light: '#93C5FD', mark: '●', shape: 'circle' },
    { hex: 0xEA580C, css: '#EA580C', text: '#C2410C', light: '#FDBA74', mark: '■', shape: 'square' },
    { hex: 0x9333EA, css: '#9333EA', text: '#7E22CE', light: '#D8B4FE', mark: '▲', shape: 'triangle' },
    { hex: 0x0D9488, css: '#0D9488', text: '#0F766E', light: '#5EEAD4', mark: '◆', shape: 'diamond' }
  ];
  const fallback = i => 'Team ' + (i + 1);
  const tidy = (n, i) => {
    n = String(n == null ? '' : n).trim();
    if (!n || /^player\s*\d$/i.test(n)) return fallback(i);   // older versions defaulted to "Player 1"
    return n.slice(0, 18);
  };
  function load() {
    let names = CGB.store.getJSON('teams', null);
    if (!Array.isArray(names)) {
      // first run after the update: take whichever game last saved names
      names = ['cc.names', 'hh.names', 'names'].map(k => CGB.store.getJSON(k, null)).find(Array.isArray) || [];
    }
    return [0, 1, 2, 3].map(i => tidy(names[i], i));
  }
  /* The saved names for teams 1 to n */
  CGB.teamNames = n => load().slice(0, n || 4);
  CGB.teamFallback = fallback;
  /* Save names typed in any game: only the teams that game uses are replaced */
  CGB.saveTeamNames = list => {
    const all = load();
    list.forEach((n, i) => { all[i] = tidy(n, i); });
    CGB.store.setJSON('teams', all);
    return all.slice(0, list.length);
  };
})();
