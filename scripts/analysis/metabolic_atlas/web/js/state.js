/* Screen state: which view/layer is shown, and the one place problems are reported.
 * Loaded right after 00-overview.js. cyHost / escherHost are created later, so they are
 * resolved lazily inside the functions below.
 *
 *   view  : 'map' | 'search' | 'source'   (which page section is visible)
 *   layer : 'cy' | 'escher' | 'sankey'    (which renderer fills the map pane)
 */
const AtlasUI = {view: 'map', layer: 'cy', failures: 0};
const VIEWS = ['map', 'search', 'source'];

function setLayer(layer) {
  AtlasUI.layer = layer;
  if (typeof cyHost !== 'undefined') cyHost.style.display = layer === 'cy' ? 'block' : 'none';
  if (typeof escherHost !== 'undefined') escherHost.style.display = layer === 'escher' ? 'block' : 'none';
  if (layer === 'sankey' && typeof cyHost !== 'undefined') cyHost.style.display = 'none';
  for (const [name, h] of Object.entries(AtlasUI.hosts || {})) h.hidden = name !== layer;
  document.body.dataset.layer = layer;
}

function setView(view) {
  if (!VIEWS.includes(view)) throw new Error('unknown view: ' + view);
  AtlasUI.view = view;
  document.body.dataset.view = view;
  $('mappane').style.display = view === 'map' ? 'grid' : 'none';
  $('searchpane').style.display = view === 'search' ? 'block' : 'none';
  $('provenance').style.display = view === 'source' ? 'block' : 'none';
  $('export').style.display = view === 'map' ? 'block' : 'none';
  // The map view has two tabs: the fixed overview (mapTab) and the medium-driven flow (flowTab).
  const inFlow = typeof flowMode !== 'undefined' && flowMode;
  const overview = AtlasUI.layer === 'fluxov';
  $('mapTab').classList.toggle('active', view === 'map' && !inFlow && !overview);
  $('flowTab')?.classList.toggle('active', view === 'map' && inFlow && !overview);
  $('fluxTab')?.classList.toggle('active', view === 'map' && overview);
  $('searchTab').classList.toggle('active', view === 'search');
  $('sourceTab').classList.toggle('active', view === 'source');
}

/* Problems that used to be visible only inside a collapsed side panel.
 * kind: 'error' (needs attention) | 'info'. A key makes repeated reports replace each other. */
function reportProblem(message, {kind = 'error', key = message} = {}) {
  let host = document.getElementById('atlasBanners');
  if (!host) {
    host = document.createElement('div');
    host.id = 'atlasBanners';
    host.setAttribute('role', 'status');
    host.setAttribute('aria-live', 'polite');
    document.querySelector('main').before(host);
  }
  let item = host.querySelector('[data-key="' + CSS.escape(key) + '"]');
  if (!item) {
    item = document.createElement('div');
    item.dataset.key = key;
    item.innerHTML = '<span></span><button type="button" aria-label="閉じる">×</button>';
    item.querySelector('button').onclick = () => item.remove();
    host.append(item);
  }
  item.className = 'banner banner-' + kind;
  item.firstChild.textContent = message;
}

function clearProblem(key) {
  document.querySelector('#atlasBanners [data-key="' + CSS.escape(key) + '"]')?.remove();
}

/* Calculation-server reachability: report once after repeated failures, clear on success. */
function noteServerResult(ok) {
  AtlasUI.failures = ok ? 0 : AtlasUI.failures + 1;
  document.body.dataset.server = ok ? 'up' : AtlasUI.failures >= 2 ? 'down' : document.body.dataset.server || 'unknown';
  if (ok) clearProblem('server');
  else if (AtlasUI.failures === 2)
    reportProblem('計算サーバーに接続できません。地図の閲覧は可能ですが、FBA・構造図・サーバー保存は使えません（structure_server.py を起動してください）。', {key: 'server'});
}
