/* Home ("概観"): how the viewer is organised, the modelling assumptions, and one-click entry points.
 * The screens follow the way a result is read: model inventory -> medium and FBA flow -> central
 * carbon detail -> single reactions. */
(() => {
  const host = document.createElement('div');
  host.id = 'homeHost';
  host.hidden = true;
  cyHost.parentNode.append(host);
  AtlasUI.hosts = AtlasUI.hosts || {};
  AtlasUI.hosts.home = host;

  const snap = JSON.parse(document.getElementById('fluxSnapshot')?.textContent || 'null');
  const STEPS = [
    {n: 1, tab: 'mapTab', title: '全体マップ', lead: 'モデルの規模と内訳',
     body: '3菌種のGEMに登録された反応が、どの分類にどれだけあるかを Sankey で見ます。分類の帯から反応一覧へ進めます。'},
    {n: 2, tab: 'fluxTab', title: 'FBAの流れ', lead: '培地成分 → 取込 → 代謝',
     body: '培地の成分を、どの菌がどれだけ取り込み、何を分泌し、どの代謝カテゴリにどれだけ流量があるかをFBAで概観します。培地ごとの成長も比べられます。'},
    {n: 3, tab: 'flowTab', title: '中心代謝マップ', lead: '中心炭素代謝への接続',
     body: '栄養を選んで、解糖系・PPP・ED・TCA への接続を経路として確認します。FBAの流量を線の太さや動きで重ねられます。'},
    {n: 4, tab: 'searchTab', title: '反応を調べる', lead: '個々の反応の詳細',
     body: '反応式・上下限・遺伝子・FBAでの成立判定を確認します。'}
  ];

  function growthChips() {
    if (!snap?.scenarios) return '';
    return snap.scenarios.map(sc => {
      const parts = D.species.map(s => {
        const e = sc.species[s.short], ok = e?.status === 'optimal' && e.objective_value > 1e-6;
        return `<span style="color:${colors[s.short]}">${esc(s.short)} ${ok ? e.objective_value.toFixed(2) : '−'}</span>`;
      }).join(' · ');
      return `<button type="button" class="hm-chip" data-scenario="${esc(sc.id)}"><strong>${esc(sc.label)}</strong><small>${parts}</small></button>`;
    }).join('');
  }

  function render() {
    const total = D.species.reduce((a, s) => a + s.reactions.length, 0);
    host.innerHTML = `
      <section class="hm-intro">
        <h2>3菌種共培養 代謝アトラス</h2>
        <p>ゴム分解から共有プール、各菌種の代謝までを、モデルに基づいて見るための画面です。左から右へ、全体 → 培地とFBA → 中心代謝 → 個々の反応と、粒度を上げて読み進められます。</p>
      </section>
      <ol class="hm-steps">${STEPS.map(x => `
        <li><button type="button" class="hm-card" data-open="${x.tab}">
          <span class="hm-n">${x.n}</span><strong>${x.title}</strong><em>${x.lead}</em><span>${x.body}</span>
        </button></li>`).join('')}
      </ol>
      <section class="hm-scen">
        <h3>培地を変えたときの成長（/h）　<small>クリックで「FBAの流れ」をその培地で開く</small></h3>
        <div class="hm-chips">${growthChips()}</div>
      </section>
      <section class="hm-facts">
        <div><strong>${D.species.length} 菌種 · ${total.toLocaleString()} 反応</strong><span>${D.species.map(s => `${s.short} ${s.reactions.length.toLocaleString()}`).join(' / ')}</span></div>
        <div><strong>前提</strong><span>静的なGEMと共通のpFBA。各菌種を単独で同じ培地条件で解き、菌種間の授受・動的な培養（dFBA）は含みません。参照培地は流加前の初期培地です。乳酸など菌種特有の栄養は流加で与える設計なので、「乳酸流加相当」の行で比べてください。</span></div>
        <div><strong>読み方</strong><span>帯や線の太さは、反応の数またはモデル上の流量。実培養の測定値ではありません。</span></div>
      </section>`;
  }

  host.addEventListener('click', e => {
    const chip = e.target.closest('[data-scenario]');
    if (chip) { openFluxScenario(chip.dataset.scenario); return; }
    const card = e.target.closest('[data-open]');
    if (card) $(card.dataset.open).click();
  });

  window.showHome = () => {
    tab('map');
    flowCanvasSuspended = true;
    flowMode = false;
    if (!host.firstChild) render();
    setLayer('home');
    setView('map');
  };

  const tabBtn = document.createElement('button');
  tabBtn.id = 'homeTab';
  tabBtn.textContent = '概観';
  document.querySelector('.toolbar').prepend(tabBtn);
  tabBtn.onclick = () => showHome();

  // Returning to the core map must not rebuild it (that would drop a restored session or the user's
  // additions): reuse the open map, and fit it once if it was drawn while hidden with no saved view.
  let hadSaved = new URLSearchParams(location.search).has('session');
  try { hadSaved = hadSaved || !!localStorage.getItem('metabolic-atlas-session'); } catch { /* storage unavailable */ }
  let revealed = false;
  flowTab.onclick = () => {
    if (flowState?.coreMode && mapCy && !mapCy.destroyed()) {
      tab('map');
      flowMode = true;
      flowCanvasSuspended = false;
      setLayer('cy');
      setView('map');
      mapCy.resize();
      if (!revealed && !hadSaved) mapCy.fit(mapCy.elements().filter(e => e.visible()), 32);
      revealed = true;
      flowRender();
    } else openFlow();
  };

  // The home page is the landing view; a shared ?session= link opens the map it was saved from.
  // A locally saved session is still restored underneath: open 「中心代謝マップ」 to continue it.
  const baseStart = startAtlas;
  startAtlas = async function (...args) {
    const shared = new URLSearchParams(location.search).has('session');
    await baseStart.apply(this, args);
    if (!shared) showHome();
  };
})();
