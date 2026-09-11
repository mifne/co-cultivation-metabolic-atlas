# 天然ゴム分解・PHA変換コンソーシアム dFBA-RL プロジェクト

## 概要
天然ゴム（poly(cis-1,4-isoprene)）を分解する3種の微生物コンソーシアムを構築し、動的フラックスバランス解析（dFBA）と強化学習（RL）を統合した制御システムを実装したプロジェクトです。

## プロジェクトの状況
OR16+NS21の2種コンソーシアムでの基盤開発は完了し、現在はP. freudenreichiiを
加えた3種共存の価値評価、GPU加速dFBA基盤の構築、湿式実験での校正に向けた
準備を並行して進めています。
プロジェクトの詳細なアーキテクチャや代謝モデル、管理状況に関する**最新の公式な情報は、すべて `docs/MASTER_INDEX.md` を起点として管理**されています。開発や調査を行う際は、必ずそちらを参照してください。

## コンソーシアム構成（現行 3 種、2026-09時点）
1. **Actinoplanes sp. OR16**: ゴム分解 (Lcp型) 担当
2. **Rhizobacter gummiphilus NS21**: ゴム分解 (Rox型) & PHA変換 担当
3. **Propionibacterium freudenreichii (Pf)**: 乳酸からのプロピオン酸供給によるPHBV組成(3HV)支援 担当

第3菌種は当初 *Lactobacillus plantarum* (WCFS1) を予定していましたが、計算研究の結果
明確なメリットが確認できなかったため、*P. freudenreichii* に変更しました。詳細な経緯と
現状の知見は [docs/OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md](docs/OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md)
を参照してください。3種の優位性は酸素供給条件に対して非単調(条件依存)であり、
実験的検証はまだ行われていません。

なお `models/sbml/final_consortium/Lactobacillus_plantarum.xml` は過去の構成の名残として
ファイルは残っていますが、現行コード(`main.py`)はこれを読み込まず、
`models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml` を
第3菌種として使用します。

## セットアップ

### 1. 依存関係のインストール
```bash
mamba env create -f environment.yml # もしくは pip install -r requirements.txt
mamba activate co-cultivation
```

### 2. シミュレーションの実行・検証
コードまたはモデルに変更を加えた後は、必ず以下のテストを実行し、プロジェクトの整合性が保たれているか確認してください。
```bash
mamba run -n co-cultivation pytest tests/test_project_integrity.py
```

### 3. GPUによるFBA rollout高速化

物質収支・全反応境界を検査するGPU辞書サロゲートを複数環境で共有できます。
分布外状態は厳密HiGHSへ戻し、最終科学評価もHiGHSで行います。構成、RTX 4060の
測定値、実行コマンドは
[GPU_SURROGATE_ARCHITECTURE.md](docs/GPU_SURROGATE_ARCHITECTURE.md)を参照してください。

### 4. 3種コンソーシアムの共生成立性監査

共有培地のmax–min LPで単独・2種・3種の同時増殖可能性を調べ、厳密HiGHS版dFBAで
各菌の残存、終点増殖率、pH、培地枯渇を確認します。成立しない場合は、正の共通増殖に
最低限必要な追加供給量と、条件付きで予測される代謝物授受を出力します。

```bash
python scripts/analysis/audit_consortium_coexistence.py \
  --dynamic triplet --hours 48 \
  --scenarios no_feed maintenance_feed candidate_rescue candidate_rescue_ph_control
```

結果は既定で `results/coexistence_audit/` のJSON、CSV、Markdown、PNGに保存されます。
これはin silicoの成立可能性の事前監査であり、長期継代や侵入可能性を含む実験的な
安定共生の証明ではありません。

RLを開始する前に、現行の行動・観測で共存とPHA生成へ到達できるかも検査できます。

```bash
python scripts/analysis/audit_rl_problem_feasibility.py --rollout-hours 48
```

判定基準と現時点の検証結果は
[RL_PRETRAINING_FEASIBILITY_AUDIT.md](docs/RL_PRETRAINING_FEASIBILITY_AUDIT.md)を参照してください。

## ドキュメント
- [MASTER_INDEX.md](docs/MASTER_INDEX.md) - プロジェクトの全体構造と各種ドキュメントへのリンク

## ライセンス
MIT License
