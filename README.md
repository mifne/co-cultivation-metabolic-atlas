# 天然ゴム分解・PHA変換コンソーシアム dFBA-RL プロジェクト

## 概要
天然ゴム（poly(cis-1,4-isoprene)）を分解する3種の微生物コンソーシアムを構築し、動的フラックスバランス解析（dFBA）と強化学習（RL）を統合した制御システムを実装したプロジェクトです。

## プロジェクトの状況
全主要開発フェーズは完了し、現在はメンテナンスおよびコードベースの最適化フェーズにあります。
プロジェクトの詳細なアーキテクチャや代謝モデル、管理状況に関する**最新の公式な情報は、すべて `docs/MASTER_INDEX.md` を起点として管理**されています。開発や調査を行う際は、必ずそちらを参照してください。

## コンソーシアム構成（精鋭 3 種）
1. **Actinoplanes sp. OR16**: ゴム分解 (Lcp型) 担当
2. **Rhizobacter gummiphilus NS21**: ゴム分解 (Rox型) & PHA変換 担当
3. **Lactobacillus plantarum**: 代謝バッファー (ビタミン・アミノ酸供給等) 担当

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
- [GEMINI.md](GEMINI.md) - プロジェクトの絶対的な開発・実行指針（Mandates）

## ライセンス
MIT License
