# 代謝アトラス（3種共培養ビューア）

```
metabolic_atlas/
├── build.py            GEMを読み出し、web/ を1枚の index.html に束ねる → outputs/metabolic_map_20260922/
├── web/
│   ├── index.template.html   マークアップのみ（CSS/JS/データはビルド時に差し込み）
│   ├── css/                  番号順に連結。00 base → 10-50 各機能 → 90 theme（最後に上書き）
│   ├── js/                   読み込み順は build.py の JS_MODULES。後段が前段の関数をラップする
│   │   ├── 00-overview.js    全体図・反応検索・出典タブ
│   │   ├── state.js          画面状態（setView/setLayer）と問題表示の帯
│   │   ├── sankey.js         「全体マップ」: 菌種→反応分類のSankey（軽量SVG、反応数の内訳）
│   │   ├── cytoscape.js / escher.js / complete.js   地図描画とEscher表示
│   │   ├── flow.js / flux.js 経路の展開・FBA流量表示
│   │   ├── runtime.js        状態・非同期ジョブ・セッション保存
│   │   ├── core.js           中心代謝の骨格と栄養接続
│   │   └── theme.js          Cytoscape側の見た目（ノード色・凡例・文字サイズ）
│   └── vendor/               cytoscape / escher とライセンス
├── server/             atlas_store.py（セッション・ジョブ）, fba_service.py, structure_server.py（FBA/構造API）, FBA.md
└── tests/              *.cjs（JS）, test_*.py（pytest）, validate_flow.cjs
```

## よく使うコマンド（リポジトリのルートから）

```bash
python3 scripts/analysis/metabolic_atlas/build.py                 # index.html を再生成
python3 scripts/analysis/metabolic_atlas/server/structure_server.py   # http://localhost:8768/
for t in scripts/analysis/metabolic_atlas/tests/*.cjs; do node $t; done
python3 -m pytest scripts/analysis/metabolic_atlas/tests
```

## デザインの方針

- 色・余白・角丸は `css/90-theme.css` の変数とルールに集約。機能別CSSにはレイアウトだけを置く。
- 経路色（解糖系/PPP/ED/TCA）は CSS の `--emp --ppp --ed --tca` と `js/theme.js` の `PATHWAY` に同じ値を持つ。変える時は両方。
- 見た目の変更で座標・反応・FBAの判定は変えない。
