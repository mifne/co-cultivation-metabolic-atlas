# 代謝アトラス：画面状態の整理（2026-09-30）

実装は `scripts/analysis/metabolic_atlas/web/js/state.js`。テストは `tests/test_screen_state.cjs`。

## 状態の軸

|軸|値|持ち主|
|---|---|---|
|view（ページ区画）|`map` / `search` / `source`|`AtlasUI.view`、`body[data-view]`、`setView()`|
|layer（地図の描画器）|`cy`（Cytoscape）/ `escher` / `sankey`|`AtlasUI.layer`、`body[data-layer]`、`setLayer()`|
|地図の種類|概要（mapTab）/ 培地からの流れ（flowTab）|`flowMode`。`setView` が2つのタブの active を排他にする|
|計算サーバー|`unknown` / `up` / `down`|`body[data-server]`、`noteServerResult()`（`atlasJSON` から呼ぶ）|
|復元中|`sessionRestoring`|runtime.js（従来どおり）|
|計算中|`atlasRequests` / `atlasJobStatus`|runtime.js（従来どおり）|

## ルール

- `style.display` を直接書かない。表示の切替は `setView` / `setLayer` のみ。
- ユーザーに知らせるべき問題は `reportProblem(message,{key})` で画面上部の帯に出す（同じ key は上書き、`clearProblem(key)` で消す）。
  - `server`: サーバー通信が2回連続で失敗したとき。成功すると自動で消える。
  - `session`: 保存された表示状態を読み込めなかったとき（初期表示で開始）。
- 地図のステータス欄（`#atlasStatus`）は、サーバー未接続のとき末尾に「サーバー未接続」を付ける。

## 未着手（次の段階）

- 保存内容の版管理と旧キー（`metabolic-atlas-before-core` など）の整理。
- 計算中・失敗・未適用の培地変更を1つのステータス帯にまとめる（現在は `#atlasJobStatus` / `#fluxStatus` が別々）。
- 接続不能な栄養・FBA失敗の共通の空表示。
