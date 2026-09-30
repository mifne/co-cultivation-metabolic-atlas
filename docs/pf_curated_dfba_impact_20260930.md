# Pf 交換反応キュレーションの dFBA への opt-in 接続と影響（2026-09-30）

## 変更点（既定 OFF）
- `main.py` のみ変更（+51/−6 行）。src/ 配下、config、results、SBML、atlas は触っていない。
  - L27–44 `PF_CURATED_ENV` と `pf_curation_requested(explicit)` を追加。`--pf-curated` が指定されていればそれを優先し、なければ環境変数 `PF_CURATED` を読む。未設定・0・off は OFF、1・true・on は ON、それ以外の値は ValueError。
  - L47–71 `load_requested_models(..., pf_curated=None)`: ON のときだけ、選ばれた Pf に `curate_pf_exchanges(close_toxic_sinks=True, inplace=True)` を適用する。Pf を含まない profile で ON にすると ValueError で止まる。
  - L90 と L101–105 `cultivation_options`: ON のときだけ `options['pf_curated']=True` を追加する。OFF の dict と `cultivation_configuration.json` は変わらない。
  - L129–130 と L244 `make_env` は `pf_curated` を受け渡し、env kwargs からは除外する。L314–315 `train_agent` も受け渡す。L620 で CLI 引数 `--pf-curated` を追加（train と evaluate）。
- 新規ファイル: `tests/test_pf_curated_dfba.py`（12 件）と本文書。
- 使い方
  - CLI: `python main.py train --consortium-profile pf-helper3 --pf-curated ...`
  - スクリプト（`load_requested_models` を呼ぶもの。physiology や equal-budget 系も含む）: `PF_CURATED=1 python ...`

## dFBA 側の挙動（コード変更なしで成立することを読んで確認し、実行でも確認した）
- 共有プールに入るか: 交換反応マップは `model.exchanges` から `canonical_metabolite_id` で作られる。キュレーション後の 195 本はすべてここに入る。
  - legacy（`_update_environment`）でも audited（`_integrate`）でも、排出量は `state.metabolites[pool]` に加算される。キーが未登録なら新しく作られる。
  - 翌 substep 以降は、他の種の `set_uptake_constraints` / `compute_uptake_limits` が同じプールを在庫として扱い、Monod 項と在庫按分で取り込み量を決める。
  - 未知のプールでもエラーにはならず、黙って無視されることもない。
- 取り込みの制限: 取り込みを許すのは、元の下限が負で、かつ培地に在庫がある場合に限られる。キュレーションは排出専用の上下限をそのまま引き継ぐので、新しいプールを Pf が取り込むことはない（テストで limit=0 を確認）。
- Pf 専用プール（74 個の `cpd*_e`）は、不活性な状態量として蓄積させる。
  - 理由: 取り込める種がおらず、無視するための特別扱いを入れるとコード変更が必要になる。蓄積させておけば、式のあるものは physiology の C/N 台帳に正しく入る。
  - 2 h 後の量はごく僅か（k10: 1.9e-5 mM、k2: 6.0e-5 mM）。
- 旧 sink について
  - OFF では、legacy も audited も `model.exchanges` の下限しか操作しない。そのため Pf の 4 sink（H2S/Hg/Cd/Pb）は下限 −1000 のまま、培地の制御の外で開いていた。
  - ON では Hg/Cd/Pb の取り込みを閉じ、H2S は培地の `h2s_e`（初期値 0）で制御される。

## 影響の計測
シナリオは `results/equal_budget_comparison_20260908` の screen 条件（three arm）と同じにした。
- PhysiologyDFBASimulator、design の `common_initial_medium`、初期菌体 OR16 0.5 / NS21 0.1 / Pf 0.03 g/L
- maintenance 0.1、death 0.01/0.1、pH 7、control dt 0.25 h、internal dt 0.025 h
- lactate "early"（0.5 mM/h）、NH4 "early"（0.1 mM/h）
- 2 h（80 substep）で、kLa 10 と kLa 2 の 2 条件
- 確率的な要素はない（seed なし）。flux は `_integrate` を包むスクラッチのプローブで記録した。

| 指標（2 h 時点） | k10 OFF | k10 ON | k2 OFF | k2 ON |
|---|---|---|---|---|
| Pf 生菌 g/L | 0.030406 | 0.030391 (−0.05%) | 0.032538 | 0.032482 (−0.17%) |
| NS21 生菌 g/L | 0.126446 | 0.126490 | 0.107730 | 0.107729 |
| OR16 生菌 g/L | 0.509410 | 0.509437 | 0.496567 | 0.496602 |
| PHA（生菌） g/L | 1.892e-4 | 1.988e-4 | 1.0013e-2 | 0.9937e-2 |
| lactate / propionate mM | 4.31e-4 / 1.980e-3 | 4.31e-4 / 1.950e-3 | 0.02716 / 0.10360 | 0.02701 / 0.10154 |
| acetate / CO2 mM | 0 / 5.8721 | 2.65e-5 / 5.8937 | 0 / 3.2206 | 1.66e-3 / 3.1801 |
| succinate / formate mM | 0 / 0 | 9.3e-6 / 3.08e-4 | 0 / 2.89e-3 | 7.70e-3 / 0.0658 |
| NH4 mM | 0.89752 | 0.89878 | 0.46437 | 0.41988 |
| Pf の C 取り込み（mmol C/L） | 0.1821 | 0.1820 | 0.5757 | 0.5737 |
| そのうち demand に消えた C | **0.0351 (19.3%)** | 0 | **0.1466 (25.5%)** | 0 |
| 旧 H2S sink からの取り込み mM | **2.02e-4** | 0 | **6.30e-4** | 0 |
| Pf の SO4 取り込み mM | 4.2e-6 | 2.03e-4 | 1.3e-5 | 6.32e-4 |

- 他の種による取り込み（ON、2 h 累計、mM）
  - k10: Pf が formate 0.0199 を出し、NS21 が 0.0196（98%）を取り込んだ。succinate は Pf 0.0028 に対し OR16 0.0023 と NS21 0.0005。acetate は NS21 が 5.1e-4。CO2 は OR16/NS21 とも排出側で、取り込みはなかった。
  - k2（低 O2）: NS21 は formate を取り込まない（OFF でも自分で排出している）ため、formate は 0.066 mM 蓄積した。succinate は OR16 が 0.0128 を取り込んだ。
- 旧 sink の使用
  - 使われていたのは H2S だけ。Pf は培地に無い H2S を sink から硫黄源として取り込み、SO4 の代わりにしていた。ON では同じ量を SO4 から取るようになった。
  - Hg/Cd/Pb の流束は 3 run すべてで 0 だった。
  - legacy dFBASimulator（separate, HiGHS, `get_initial_params`, 1 h）の OFF でも同じで、H2S sink の流束は −0.037 → −0.0035 mmol/g/h だった。
- 台帳: C/N の既知培地閉合残差は両条件とも 1e-14 以下だった。この台帳は exchange しか見ないため、OFF で demand に消えた C は残差に現れない。
- 計算コスト
  - k10: LP 要求数 3146 で同じ、実行時間 211 s → 212 s（2 run を並列実行）。
  - k2: LP 要求数 2233 → 2189、実行時間 124 s → 122 s。
  - ロードは +約 0.1 s（キュレーション）。

## OFF が変更前と同一であることの確認
- 変更前に k10 の 2 h を実行して保存し、変更後の OFF と比べた。`trajectory.csv` はバイト単位で一致し（`cmp`）、summary の数値もすべて一致した（時間計測を除く）。
- pytest の結果
  - `test_main_cultivation_profiles`, `test_cultivation_numerics`, `test_resolved_dfba`, `test_audited_dfba`, `test_pf_curation`, `test_pf_curated_dfba`: 153 passed（65 s）
  - `test_physiology_dfba`, `test_physiology_long_guards`: 29 passed

## リスク
- 公開済みの図や結果は OFF で作られたもの。ON にすると Pf の flux の選び方が変わる。parsimonious 選択で排出の絶対値に罰則がかかる対象に 172 本が加わるため、acetate 排出が現れ、propionate がわずかに減る。NH4 と PHA も変わる。k2 では Pf の生菌量 −0.17%、PHA −0.75%。OFF と ON の結果を混ぜないこと。
- `main.py` のハッシュが変わった。source_sha256 を検査する凍結スクリプト（`compare_equal_budget_20260908`, `validate_physiology_long_20260908` など、main.py を対象に含む 5 本以上）は、そのままでは再実行・続行できない。これは OFF でも同じ。
- ON では Pf の交換反応 id（`Ex_S_*`→`EX_*`）と反応の境界分類が変わる。影響を受けるもの:
  - RL の `initial_gem_identity` と policy contract
  - 協調サロゲートの凍結契約と fingerprint
  - 学習済みの `*pfreud*` 辞書と cooperative の frozen inputs
  - atlas の `model_fingerprint`
  これらは再構築が必要。ON のモデルで既存のチェックポイントを使ってはならない。
- `glcn__D_e`/`glcn_e` の不整合は未解決。formula_gaps は 14 → 20 に増えた（Pf 専用プールで式が欠けているもの）。

## 検証できなかったこと
- 24 h の全体（1 run 約 30–40 分）、refine の dt、kLa 50、two arm、cooperative/surrogate/GPU の各経路は ON で実行していない。
- RL の学習と評価（`--pf-curated`）は、フラグの受け渡しをテストしただけで実行していない。
- 生物学的な妥当性（NS21 の formate 利用や Pf の SO4 還元コスト）は検証していない。
