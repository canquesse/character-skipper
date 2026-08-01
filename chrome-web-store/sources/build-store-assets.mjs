import fs from 'node:fs';
import path from 'node:path';

const sourceDir = path.dirname(decodeURIComponent(new URL(import.meta.url).pathname));
const castBase64 = fs.readFileSync(path.join(sourceDir, 'fictional-cast.png')).toString('base64');
const castHref = `data:image/png;base64,${castBase64}`;

const C = {
  cream: '#FBF6EE',
  paper: '#FFFFFF',
  warm: '#F4EDE1',
  line: '#DCD1BE',
  ink: '#2C2620',
  muted: '#6F6559',
  faint: '#A79C8E',
  coral: '#FF8A5C',
  coralDeep: '#E5643A',
  coralSoft: '#FFE8DC',
  green: '#3E8E5E',
  greenSoft: '#E1F1E4',
  navy: '#172238',
  navy2: '#0D1628',
};

function svg(w, h, body, extraDefs = '') {
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <defs>
    <linearGradient id="pageBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFF9F1"/>
      <stop offset="0.62" stop-color="#FBF6EE"/>
      <stop offset="1" stop-color="#F2E7D7"/>
    </linearGradient>
    <linearGradient id="coralBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FF9B72"/>
      <stop offset="1" stop-color="#E5643A"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#4B3322" flood-opacity="0.16"/>
    </filter>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#4B3322" flood-opacity="0.13"/>
    </filter>
    <style>
      text { font-family: 'Avenir Next', Avenir, Helvetica, Arial, sans-serif; }
      .h1 { font-weight: 700; fill: ${C.ink}; letter-spacing: -1.2px; }
      .h2 { font-weight: 700; fill: ${C.ink}; letter-spacing: -0.45px; }
      .body { font-weight: 500; fill: ${C.muted}; }
      .small { font-weight: 600; fill: ${C.muted}; }
      .label { font-weight: 700; fill: ${C.ink}; letter-spacing: 0.7px; }
      .white { fill: white; }
    </style>
    ${extraDefs}
  </defs>
  ${body}
</svg>`;
}

function logoMark(x, y, size = 44, dark = false) {
  const bg = dark ? C.ink : C.coralDeep;
  const fg = dark ? C.coral : '#FFFFFF';
  const s = size / 96;
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <rect width="96" height="96" rx="24" fill="${bg}"/>
    <circle cx="31" cy="31" r="11" fill="${fg}"/>
    <path d="M14 67c0-13 7.6-23 17-23s17 10 17 23v8H14z" fill="${fg}"/>
    <path d="M54 31l14 13-14 13zM70 31l14 13-14 13z" fill="${fg}"/>
  </g>`;
}

function topBrand(x = 64, y = 38) {
  return `${logoMark(x, y, 46)}
    <text x="${x + 60}" y="${y + 22}" class="h2" font-size="19">Character Skipper</text>
    <text x="${x + 60}" y="${y + 41}" class="small" font-size="11">FOR YOUTUBE · PRIVATE ON-DEVICE AI</text>`;
}

function pageHeading(title, subtitle) {
  return `<text x="64" y="145" class="h1" font-size="39">${title}</text>
    <text x="65" y="177" class="body" font-size="17">${subtitle}</text>`;
}

function switchControl(x, y, on = true) {
  return `<g transform="translate(${x} ${y})">
    <rect width="48" height="28" rx="14" fill="${on ? C.green : '#D9D5CE'}"/>
    <circle cx="${on ? 34 : 14}" cy="14" r="10" fill="#FFFFFF"/>
  </g>`;
}

function button(x, y, w, label, primary = true) {
  return `<g transform="translate(${x} ${y})">
    <rect width="${w}" height="48" rx="12" fill="${primary ? C.coral : C.paper}" stroke="${primary ? C.coralDeep : C.line}" stroke-width="1.5"/>
    <path d="M${w - 25} 16v14M${w - 31} 24l6 6 6-6" fill="none" stroke="${C.ink}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="${w / 2 - 8}" y="30" text-anchor="middle" class="h2" font-size="14">${label}</text>
  </g>`;
}

function panelHeader(x, y, w, active = true) {
  return `<g transform="translate(${x} ${y})">
    ${logoMark(0, 0, 38)}
    <text x="50" y="17" class="h2" font-size="16">Character Skipper</text>
    <text x="50" y="33" class="body" font-size="10">YouTube · AI Face Detection</text>
    <text x="${w - 91}" y="23" class="small" font-size="10">EN</text>
    ${switchControl(w - 50, 5, active)}
  </g>`;
}

function charRow(x, y, w, name, note, selected = false, avatar = 'M', color = '#67CDEB', on = true) {
  return `<g transform="translate(${x} ${y})">
    <rect width="${w}" height="58" rx="13" fill="${selected ? C.ink : C.paper}" stroke="${selected ? C.ink : '#E8DFD2'}"/>
    <circle cx="30" cy="29" r="18" fill="${color}"/>
    <text x="30" y="35" text-anchor="middle" font-size="16" font-weight="700" fill="#FFFFFF">${avatar}</text>
    <text x="58" y="25" font-size="14" font-weight="700" fill="${selected ? '#FFFFFF' : C.ink}">${name}</text>
    <text x="58" y="42" font-size="10.5" font-weight="500" fill="${selected ? '#B8B0A6' : C.muted}">${note}</text>
    ${switchControl(w - 56, 15, on)}
  </g>`;
}

function popupShell(x, y, w, h, body) {
  return `<g transform="translate(${x} ${y})" filter="url(#shadow)">
    <rect width="${w}" height="${h}" rx="24" fill="${C.cream}" stroke="#E8DDCC"/>
    ${panelHeader(22, 20, w - 44)}
    <path d="M22 76H${w - 22}" stroke="${C.ink}" stroke-width="2"/>
    ${body}
  </g>`;
}

function player(x, y, w, h, withBoxes = true, dim = false) {
  const sx = w / 1792;
  const sy = h / 1024;
  const boxes = withBoxes ? `<g fill="none" stroke="${C.coral}" stroke-width="3">
      <rect x="${x + 176 * sx}" y="${y + 132 * sy}" width="235" height="260" rx="10"/>
      <rect x="${x + 682 * sx}" y="${y + 220 * sy}" width="235" height="260" rx="10"/>
      <rect x="${x + 1260 * sx}" y="${y + 164 * sy}" width="235" height="260" rx="10"/>
    </g>
    <g font-size="12" font-weight="700" fill="${C.ink}">
      <rect x="${x + 176 * sx}" y="${y + 105 * sy}" width="74" height="24" rx="6" fill="${C.coral}"/><text x="${x + 186 * sx}" y="${y + 122 * sy}">MARCUS</text>
      <rect x="${x + 682 * sx}" y="${y + 193 * sy}" width="63" height="24" rx="6" fill="${C.coral}"/><text x="${x + 692 * sx}" y="${y + 210 * sy}">MAYA</text>
      <rect x="${x + 1260 * sx}" y="${y + 137 * sy}" width="66" height="24" rx="6" fill="${C.coral}"/><text x="${x + 1270 * sx}" y="${y + 154 * sy}">ETHAN</text>
    </g>` : '';
  return `<g>
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="22" fill="${C.navy2}"/>
    <clipPath id="playerClip${x}${y}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="22"/></clipPath>
    <image href="${castHref}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#playerClip${x}${y})"/>
    ${dim ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="22" fill="#08111F" opacity="0.66"/>` : ''}
    ${boxes}
    <rect x="${x + 20}" y="${y + h - 38}" width="${w - 40}" height="5" rx="3" fill="#FFFFFF" opacity="0.42"/>
    <rect x="${x + 20}" y="${y + h - 38}" width="${(w - 40) * 0.43}" height="5" rx="3" fill="${C.coral}"/>
    <circle cx="${x + 20 + (w - 40) * 0.43}" cy="${y + h - 35.5}" r="7" fill="${C.coral}"/>
  </g>`;
}

function faceThumb(x, y, size, align, selected = false) {
  const id = `face${x}${y}`;
  return `<g>
    <clipPath id="${id}"><circle cx="${x}" cy="${y}" r="${size / 2}"/></clipPath>
    <image href="${castHref}" x="${x - size / 2}" y="${y - size / 2}" width="${size}" height="${size}" preserveAspectRatio="${align}YMid slice" clip-path="url(#${id})"/>
    <circle cx="${x}" cy="${y}" r="${size / 2 + 5}" fill="none" stroke="${selected ? C.coralDeep : C.line}" stroke-width="${selected ? 5 : 2}"/>
  </g>`;
}

// Store icon: exactly 96x96 artwork with 16px transparent padding.
fs.writeFileSync(path.join(sourceDir, 'store-icon.svg'), svg(128, 128,
  `<rect width="128" height="128" fill="none"/>
   <g transform="translate(16 16)">
     <rect width="96" height="96" rx="24" fill="${C.coralDeep}"/>
     <rect x="3" y="3" width="90" height="90" rx="21" fill="none" stroke="#FFAA84" stroke-width="2" opacity="0.65"/>
     <circle cx="30" cy="31" r="11" fill="#FFFFFF"/>
     <path d="M13 68c0-13 7.6-23 17-23s17 10 17 23v8H13z" fill="#FFFFFF"/>
     <path d="M53 31l14 13-14 13zM69 31l14 13-14 13z" fill="#FFF4EC"/>
   </g>`));

const screenshot1 = svg(1280, 800, `
  <rect width="1280" height="800" fill="url(#pageBg)"/>
  ${topBrand()}
  ${pageHeading('Choose the character. We handle the rest.', 'Find characters in the video, select once, and keep watching without unwanted scenes.')}
  ${player(64, 216, 735, 500, true)}
  ${popupShell(836, 142, 380, 610, `
    ${button(22, 94, 336, 'Scan Entire Video')}
    <text x="24" y="174" class="label" font-size="11">LIBRARY</text>
    <text x="336" y="174" text-anchor="end" class="body" font-size="11">3 saved</text>
    ${charRow(22, 192, 336, 'Maya', 'Active · skip ready', true, 'M', '#FF8A5C', true)}
    ${charRow(22, 260, 336, 'Marcus', 'Saved across videos', false, 'M', '#54BFD9', false)}
    ${charRow(22, 328, 336, 'Ethan', 'Saved across videos', false, 'E', '#7BCB8E', false)}
    <rect x="22" y="408" width="336" height="118" rx="15" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="40" y="438" class="h2" font-size="14">Add Character</text>
    <text x="40" y="462" class="body" font-size="11">Scan current frame or upload a photo.</text>
    <rect x="40" y="480" width="300" height="30" rx="9" fill="${C.coralSoft}"/>
    <text x="190" y="500" text-anchor="middle" class="h2" font-size="11">Scan Current Frame</text>
  `)}
`);
fs.writeFileSync(path.join(sourceDir, 'screenshot-01.svg'), screenshot1);

const screenshot2 = svg(1280, 800, `
  <rect width="1280" height="800" fill="url(#pageBg)"/>
  ${topBrand()}
  ${pageHeading('Scan the full video in one click.', 'Character Skipper groups repeated appearances automatically—even across changing scenes.')}
  <g transform="translate(105 220)" filter="url(#shadow)">
    <rect width="1070" height="492" rx="28" fill="#FFFFFF" stroke="#E6D9C8"/>
    <g transform="translate(34 30)">
      <rect width="63" height="42" rx="13" fill="${C.coral}"/><text x="31.5" y="29" text-anchor="middle" class="h2" font-size="20">12</text>
      <text x="78" y="20" class="h2" font-size="18">Characters detected</text>
      <text x="78" y="40" class="body" font-size="12">Select one to skip · name and save for future videos</text>
      <text x="1002" y="23" text-anchor="end" class="small" font-size="12">Full video · On-device scan</text>
    </g>
    ${faceThumb(150, 155, 112, 'xMin', false)}
    ${faceThumb(342, 155, 112, 'xMid', true)}
    ${faceThumb(534, 155, 112, 'xMax', false)}
    <circle cx="726" cy="155" r="56" fill="#A8D9B1"/><text x="726" y="167" text-anchor="middle" class="white" font-size="32" font-weight="700">R</text>
    <circle cx="918" cy="155" r="56" fill="#BDA8DF"/><text x="918" y="167" text-anchor="middle" class="white" font-size="32" font-weight="700">N</text>
    <text x="150" y="235" text-anchor="middle" class="small" font-size="12">Marcus · 81</text>
    <text x="342" y="235" text-anchor="middle" font-size="12" font-weight="700" fill="${C.coralDeep}">Maya · 96 · Selected</text>
    <text x="534" y="235" text-anchor="middle" class="small" font-size="12">Ethan · 74</text>
    <text x="726" y="235" text-anchor="middle" class="small" font-size="12">Character 4 · 38</text>
    <text x="918" y="235" text-anchor="middle" class="small" font-size="12">Character 5 · 24</text>
    <rect x="56" y="284" width="700" height="56" rx="12" fill="#F7F2EA" stroke="${C.line}"/>
    <text x="80" y="319" class="body" font-size="15">Maya</text>
    <rect x="774" y="284" width="240" height="56" rx="12" fill="${C.coral}" stroke="${C.coralDeep}"/>
    <text x="894" y="319" text-anchor="middle" class="h2" font-size="15">Save Character</text>
    <rect x="56" y="374" width="304" height="72" rx="15" fill="${C.cream}"/>
    <rect x="382" y="374" width="304" height="72" rx="15" fill="${C.cream}"/>
    <rect x="708" y="374" width="306" height="72" rx="15" fill="${C.cream}"/>
    <text x="80" y="405" class="h2" font-size="13">Full video scanned</text><text x="80" y="425" class="body" font-size="11">Every scene checked automatically</text>
    <text x="406" y="405" class="h2" font-size="13">Smart grouping</text><text x="406" y="425" class="body" font-size="11">Repeated appearances merged</text>
    <text x="732" y="405" class="h2" font-size="13">Save once</text><text x="732" y="425" class="body" font-size="11">Ready for future videos</text>
  </g>
`);
fs.writeFileSync(path.join(sourceDir, 'screenshot-02.svg'), screenshot2);

const screenshot3 = svg(1280, 800, `
  <rect width="1280" height="800" fill="url(#pageBg)"/>
  ${topBrand()}
  ${pageHeading('Scenes skip automatically.', 'When your selected character appears, playback moves forward—no clicks required.')}
  ${player(64, 225, 760, 475, false, true)}
  <g transform="translate(238 377)">
    <rect width="412" height="136" rx="24" fill="#FFFFFF" opacity="0.96" filter="url(#softShadow)"/>
    <circle cx="58" cy="58" r="28" fill="${C.coral}"/>
    <path d="M49 45l15 13-15 13zM62 45l15 13-15 13z" fill="#FFFFFF"/>
    <text x="102" y="51" class="label" font-size="11">SCENE SKIPPED</text>
    <text x="102" y="80" class="h2" font-size="22">Maya detected</text>
    <text x="102" y="103" class="body" font-size="12">Jumped ahead 2m 18s</text>
  </g>
  ${popupShell(858, 162, 358, 570, `
    <text x="24" y="116" class="label" font-size="11">ACTIVE CHARACTER</text>
    ${charRow(20, 138, 318, 'Maya', '96 samples · Active', true, 'M', '#FF8A5C', true)}
    <rect x="20" y="222" width="318" height="88" rx="16" fill="${C.greenSoft}"/>
    <circle cx="49" cy="266" r="14" fill="${C.green}"/>
    <path d="M43 266l4 4 8-9" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
    <text x="72" y="260" class="h2" font-size="14">Watching for Maya</text>
    <text x="72" y="281" class="body" font-size="11">Recognition is running locally.</text>
    <text x="24" y="348" class="label" font-size="11">STATISTICS</text>
    <rect x="20" y="368" width="98" height="90" rx="14" fill="#FFFFFF" stroke="#E8DFD2"/>
    <rect x="130" y="368" width="98" height="90" rx="14" fill="#FFFFFF" stroke="#E8DFD2"/>
    <rect x="240" y="368" width="98" height="90" rx="14" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="69" y="406" text-anchor="middle" class="h2" font-size="22">247</text><text x="69" y="432" text-anchor="middle" class="body" font-size="10">Detections</text>
    <text x="179" y="406" text-anchor="middle" class="h2" font-size="22">38</text><text x="179" y="432" text-anchor="middle" class="body" font-size="10">Skipped</text>
    <text x="289" y="406" text-anchor="middle" font-size="20" font-weight="700" fill="${C.coralDeep}">1h 12m</text><text x="289" y="432" text-anchor="middle" class="body" font-size="10">Time saved</text>
    <rect x="20" y="482" width="318" height="44" rx="11" fill="#FFFFFF" stroke="${C.line}"/>
    <text x="42" y="509" class="small" font-size="11">Detection threshold</text><text x="316" y="509" text-anchor="end" class="h2" font-size="12">0.52</text>
  `)}
`);
fs.writeFileSync(path.join(sourceDir, 'screenshot-03.svg'), screenshot3);

const screenshot4 = svg(1280, 800, `
  <rect width="1280" height="800" fill="url(#pageBg)"/>
  ${topBrand()}
  ${pageHeading('Your character library stays ready.', 'Saved profiles work across different videos—activate, pause, refine, or rename anytime.')}
  <g transform="translate(120 222)" filter="url(#shadow)">
    <rect width="1040" height="500" rx="28" fill="${C.cream}" stroke="#E6D9C8"/>
    <g transform="translate(42 36)">${logoMark(0, 0, 44)}<text x="58" y="20" class="h2" font-size="18">Character Library</text><text x="58" y="40" class="body" font-size="11">4 saved profiles · stored only in your browser</text></g>
    <rect x="42" y="104" width="610" height="336" rx="20" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="68" y="139" class="label" font-size="11">SAVED CHARACTERS</text>
    ${charRow(68, 158, 558, 'Maya', '96 samples · Active', true, 'M', '#FF8A5C', true)}
    ${charRow(68, 226, 558, 'Marcus', '81 samples · Active', false, 'M', '#54BFD9', true)}
    ${charRow(68, 294, 558, 'Ethan', '74 samples · Paused', false, 'E', '#7BCB8E', false)}
    ${charRow(68, 362, 558, 'Rose', '38 samples · Active', false, 'R', '#BDA8DF', true)}
    <rect x="680" y="104" width="318" height="336" rx="20" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="706" y="139" class="label" font-size="11">PROFILE DETAILS</text>
    ${faceThumb(839, 213, 112, 'xMid', true)}
    <text x="839" y="296" text-anchor="middle" class="h2" font-size="21">Maya</text>
    <text x="839" y="318" text-anchor="middle" class="body" font-size="11">96 face samples</text>
    <rect x="706" y="344" width="132" height="48" rx="12" fill="${C.coralSoft}"/>
    <rect x="850" y="344" width="122" height="48" rx="12" fill="#F4EDE1"/>
    <text x="772" y="375" text-anchor="middle" class="h2" font-size="12">View photos</text>
    <text x="911" y="375" text-anchor="middle" class="h2" font-size="12">Rename</text>
    <text x="706" y="421" class="body" font-size="11">Auto learn</text>${switchControl(925, 403, true)}
  </g>
`);
fs.writeFileSync(path.join(sourceDir, 'screenshot-04.svg'), screenshot4);

const screenshot5 = svg(1280, 800, `
  <rect width="1280" height="800" fill="url(#pageBg)"/>
  ${topBrand()}
  ${pageHeading('Private by design. Fast by default.', 'Face recognition runs on your device. Your photos and viewing data never leave the browser.')}
  <g transform="translate(92 232)">
    <rect width="480" height="430" rx="30" fill="${C.navy}" filter="url(#shadow)"/>
    <g transform="translate(56 52)">
      <circle cx="72" cy="72" r="72" fill="${C.coral}"/>
      <path d="M72 30l38 14v31c0 34-21 56-38 66-17-10-38-32-38-66V44z" fill="#FFFFFF"/>
      <path d="M55 75l12 12 24-28" fill="none" stroke="${C.coralDeep}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="0" y="198" class="white" font-size="28" font-weight="700">100% on-device</text>
      <text x="0" y="231" font-size="15" font-weight="500" fill="#D9DDE6">No uploads. No account. No server.</text>
      <g transform="translate(0 268)">
        <rect width="150" height="48" rx="24" fill="#FFFFFF" opacity="0.11"/><text x="75" y="30" text-anchor="middle" class="white" font-size="12" font-weight="700">LOCAL AI</text>
        <rect x="164" width="166" height="48" rx="24" fill="#FFFFFF" opacity="0.11"/><text x="247" y="30" text-anchor="middle" class="white" font-size="12" font-weight="700">NO SERVER</text>
      </g>
    </g>
  </g>
  ${popupShell(618, 194, 568, 520, `
    <text x="26" y="116" class="label" font-size="11">CHARACTERS</text>
    <rect x="24" y="136" width="520" height="78" rx="15" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="46" y="169" class="h2" font-size="14">Auto learn</text>
    <text x="46" y="190" class="body" font-size="11">Refine characters automatically while watching</text>${switchControl(474, 155, true)}
    <rect x="24" y="230" width="520" height="112" rx="15" fill="#FFFFFF" stroke="#E8DFD2"/>
    <text x="46" y="263" class="h2" font-size="14">Detection Threshold</text>
    <text x="520" y="263" text-anchor="end" class="h2" font-size="13">0.52</text>
    <rect x="46" y="290" width="452" height="8" rx="4" fill="#E5DED2"/>
    <rect x="46" y="290" width="246" height="8" rx="4" fill="${C.coral}"/>
    <circle cx="292" cy="294" r="10" fill="${C.coralDeep}"/>
    <text x="46" y="322" class="body" font-size="10.5">Lower = more matches · Higher = stricter</text>
    <rect x="24" y="358" width="520" height="96" rx="15" fill="${C.greenSoft}"/>
    <circle cx="62" cy="406" r="20" fill="${C.green}"/>
    <path d="M53 406l6 6 13-15" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <text x="96" y="398" class="h2" font-size="14">Everything stays in your browser</text>
    <text x="96" y="421" class="body" font-size="11">Photos, profiles, and detections are never uploaded.</text>
  `)}
`);
fs.writeFileSync(path.join(sourceDir, 'screenshot-05.svg'), screenshot5);

const smallPromo = svg(440, 280, `
  <rect width="440" height="280" fill="url(#coralBg)"/>
  <circle cx="34" cy="34" r="84" fill="#FFFFFF" opacity="0.10"/>
  <circle cx="418" cy="258" r="112" fill="#8E361E" opacity="0.18"/>
  ${logoMark(36, 86, 108, true)}
  <g transform="translate(178 50)" filter="url(#shadow)">
    <rect width="226" height="180" rx="18" fill="${C.navy2}"/>
    <clipPath id="promoClip"><rect width="226" height="180" rx="18"/></clipPath>
    <image href="${castHref}" width="226" height="180" preserveAspectRatio="xMidYMid slice" clip-path="url(#promoClip)"/>
    <rect x="90" y="38" width="62" height="78" rx="8" fill="none" stroke="#FFFFFF" stroke-width="3"/>
    <rect x="90" y="17" width="55" height="22" rx="6" fill="#FFFFFF"/>
    <text x="117" y="32" text-anchor="middle" font-size="10" font-weight="700" fill="${C.ink}">SKIP</text>
    <rect x="20" y="145" width="186" height="5" rx="3" fill="#FFFFFF" opacity="0.45"/>
    <rect x="20" y="145" width="116" height="5" rx="3" fill="#FFFFFF"/>
  </g>
  <g transform="translate(135 177)">
    <circle cx="0" cy="0" r="30" fill="#FFFFFF"/>
    <path d="M-12 -12L2 0-12 12zM1 -12L15 0 1 12z" fill="${C.coralDeep}"/>
  </g>
`);
fs.writeFileSync(path.join(sourceDir, 'promo-small.svg'), smallPromo);

const marquee = svg(1400, 560, `
  <rect width="1400" height="560" fill="url(#coralBg)"/>
  <circle cx="110" cy="40" r="230" fill="#FFFFFF" opacity="0.08"/>
  <circle cx="1370" cy="540" r="310" fill="#8E361E" opacity="0.15"/>
  <g transform="translate(82 120)">
    ${logoMark(0, 0, 84, true)}
    <text x="0" y="142" font-size="58" font-weight="700" fill="#FFFFFF" letter-spacing="-2">Character Skipper</text>
    <text x="2" y="185" font-size="22" font-weight="600" fill="#FFF4EC">Skip scenes. Keep the story.</text>
    <g transform="translate(2 230)">
      <rect width="136" height="40" rx="20" fill="#FFFFFF" opacity="0.16"/><text x="68" y="26" text-anchor="middle" class="white" font-size="11" font-weight="700">ON-DEVICE AI</text>
      <rect x="148" width="128" height="40" rx="20" fill="#FFFFFF" opacity="0.16"/><text x="212" y="26" text-anchor="middle" class="white" font-size="11" font-weight="700">AUTO SKIP</text>
    </g>
  </g>
  <g transform="translate(650 74)" filter="url(#shadow)">
    <rect width="664" height="412" rx="28" fill="${C.navy2}"/>
    <clipPath id="marqueeClip"><rect width="664" height="412" rx="28"/></clipPath>
    <image href="${castHref}" width="664" height="412" preserveAspectRatio="xMidYMid slice" clip-path="url(#marqueeClip)"/>
    <rect width="664" height="412" rx="28" fill="#08111F" opacity="0.12"/>
    <rect x="238" y="76" width="164" height="204" rx="12" fill="none" stroke="#FFFFFF" stroke-width="4"/>
    <rect x="238" y="40" width="125" height="36" rx="9" fill="#FFFFFF"/>
    <text x="300" y="64" text-anchor="middle" font-size="14" font-weight="700" fill="${C.ink}">MAYA · SKIP</text>
    <rect x="36" y="350" width="592" height="7" rx="4" fill="#FFFFFF" opacity="0.45"/>
    <rect x="36" y="350" width="378" height="7" rx="4" fill="#FFFFFF"/>
    <circle cx="414" cy="353.5" r="10" fill="#FFFFFF"/>
    <g transform="translate(548 24)">
      <rect width="92" height="38" rx="19" fill="${C.ink}" opacity="0.88"/>
      <path d="M18 11l10 8-10 8zM28 11l10 8-10 8z" fill="${C.coral}"/>
      <text x="48" y="24" class="white" font-size="10" font-weight="700">SKIP</text>
    </g>
  </g>
`);
fs.writeFileSync(path.join(sourceDir, 'promo-marquee.svg'), marquee);

console.log('SVG sources created.');
