# 代謝ビューア：FBA供給判定

実装: metabolic_fba_service.py / metabolic_structure_server.py の /fba。

表示中の反応を方向別に最大化し、COBRApyのCycleFreeFluxで内部循環を除去した解を検証する。
循環除去時は対象反応の目的を0へ置換する。元の最大値を固定すると、対象反応自身の循環が残るため。
循環除去後も対象方向に1e-6 mmol/gDW/hを超える流量があり、物質収支・上下限誤差が1e-7未満なら供給可能とする。
最大化解が閾値以下なら指定条件で成立せず。循環除去で消失、時間切れ、例外、最適性未確認は判定保留。
対象反応を含む定常解で全基質が同時に供給されることを確認し、各基質の供給反応を記録。

## 条件
- model_data.jsonにエクスポートされた同じ反応・上下限を使用。
- 初期参照培地の全成分を使用。選択したグラフの起点だけを栄養源とする検証ではない。
- 既存compute_uptake_limitsを利用。単一細胞0.21 gDW/L、在庫時間幅0.025 h、Vmax 20 mmol/gDW/hの明示的な検証用スナップショット。
- 酸素は初期DOのMonod上限と在庫上限、再曝気なし。水・H+は既存交換規則。
- 未登録の培地成分、その他境界からの流入を閉鎖。
- 増殖下限なし。独自の維持代謝要求は追加しない。元GEM内の下限は維持。
- 共培養の資源競争・追加クロスフィーディング、培養時刻、実測の供給保証ではない。
- 複数の反応に個別に得た証明は、それらの同時実行の証明ではない。

## UI
黄色: 計算中・未確認。橙: 少なくとも関連する一反応の証明で供給可能。
赤枠: 関連する計算済み反応がすべて指定条件で成立せず。特定の化合物だけが不足原因とは断定しない。
反応詳細に反応ごとの結果、供給反応、条件、モデル指紋を表示。
モデルJSONのハッシュ・菌種・反応・方向をキーにサーバーで成功/不成立をキャッシュ。
条件は現在固定。培地を編集する機能を追加する際は条件をAPIとキャッシュキーに含めること。

## 起動・検証
WSLで `python3 scripts/analysis/metabolic_structure_server.py`（127.0.0.1:8769）。ビューアはlocalhost:8768。
`OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/test_metabolic_fba_service.py`
同時供給、必須共基質欠損、循環のみの偽陽性を検証。
GLUabc: 循環除去後1.818181818 mmol/gDW/h、収支残差1.78e-15。
ATP・水の橙色への変更、詳細オーバーレイの供給反応と条件表示を実画面確認。

## 共通pFBAによる候補順位
/ranking?species=NS21 はエクスポートされたGEM本来の線形目的関数を最適化し、最適値を維持したpFBAを返す。反応別最大化の供給判定とは独立。
候補は各表示状態内で max(0, sign * flux) * abs(stoichiometric coefficient) の降順。ゼロ流量も残す。未計算・失敗は反応ID順で、ゼロと区別する。
現在NS21の目的関数はGrowth最大化。初期参照培地の全成分、取り込み条件は供給判定と同じ。選択した起点だけが栄養源ではない。
条件・目的関数・解の値は候補ウィンドウで確認できる。共通解はモデルJSONハッシュと菌種を含むキーでキャッシュする。
pFBAは一意性や熱力学的実現を保証しない。供給判定のCycleFreeFlux証明と混同しない。
実画面でglu__L_c候補がGLUDxi 3.591、ASPTA逆0.9320、PSERT 0.5495 mmol/gDW/hの順となることを確認。


Flux visualization: metabolic_map_flux.js uses a single shared pFBA solution for signed chemical-endpoint arrows and log-scaled particle screen speed (not molecular velocity or time-resolved culture dynamics). Particle paths follow Cytoscape endpoints/control points; animation is limited to 30 fps and 600 visible edges, with pause, reduced-motion default, and hidden-pane/document suspension. Line widths retain main/auxiliary meaning. The explicit medium editor sends nonnegative finite mmol/L concentrations through the existing inventory/Monod uptake conversion; water and proton rules are retained and disclosed. Medium is part of ranking and feasibility cache keys; stale feasibility replies from previous medium settings are discarded. Display-root additions alone do not change FBA conditions.
