# 全3段階GPUバッチ化・辞書再構築の実測結果

## 結論

全3段階のGPU-first経路を実装し、後段辞書と段階別ルータを再構築した。元LP認証と短期閉ループの終点基準は維持したが、CPU単体より高速にはなっていない。32環境×8step×2組はCPU 60.6204秒、GPU併用73.0904秒（CPU/GPU=0.8294、20.57%減速）。固定辞書の網羅性不足が残り、辞書拡張を主方針として継続する根拠は弱い。

本計測はPPO学習時間ではなく、同じactionを与えたdFBAの閉ループ実行時間。1step=0.2 h、8step=1.6 hの仮想培養である。120step/24 hの評価・完全GPU環境・実培養妥当性・購入GPUの速度予測を示す結果ではない。

## 実装した変更

- maxmin→aggregate→exchangeの各段階を環境×候補のGPUバッチへ通すopt-in経路。GPU上で候補順位、復元、完全な元LP認証を行い、段階ごとの結果転送を1回に集約する。
- 認証が終わる前にCPU LPを開始しない。不合格行だけを同じ辞書初期化の厳密CPUへ送る。CPUの重複実行・参照解のGPU候補への流用はしない。
- CPU禁止の診断モードは未認証で失敗し、未完了を速度成功として集計しない。
- aggregate/exchangeも学習用480状態からbasisを採取し、候補の自己状態認証・全学習状態coverage・容量予算で選定する。
- 各段階用の候補順位モデルを学習。元LPの受理条件はモデルから独立したまま保持する。
- bank SHA変更時、未変更段階のルータだけを全artifact・特徴・候補順・GEM一致検証の上で明示的に再bindingする。
- 集計器は各段ルータの実体SHA、metadata、bank、特徴、GEM、履歴、setup費用を照合する。旧maxmin単独形式も検査できる。

PPO既定・GEM・反応・境界条件・目的関数・物理dtは変更していない。LP組立・状態更新・互換出力はまだhost側にある。

## 学習用辞書と順位モデル

教師は `pf_neural_basis_actual4x120_20260904`、seed 20287101–20287104の4軌跡×120step。以下は学習データ上の数であり、独立評価ではない。

| 段階 | 旧候補数→新候補数 | 全候補で認証できる状態 | nearest K4 | 学習済み順位モデルK4 |
|---|---:|---:|---:|---:|
| aggregate | 52→96 | 61→127 / 480 | 59→117 / 480 | 127 / 480 |
| exchange | 44→96 | 39→91 / 480 | 37→89 / 480 | 90 / 480 |

新規basis署名はaggregate476、exchange478 / 480状態。exchangeの新規候補64件は各々、全学習coverageの和集合へ主に1状態ずつしか追加しなかった。辞書作成元の状態に偏った改善である。内部4軌跡foldの選択後K4認証率はaggregate8.54%、exchange2.71%。これも辞書の作成から独立したholdoutではない。

既存教師は正規化LPキャッシュであり、過去raw入力のbit-exactな復元保証はない。collectorは固定行の2e-12以内の変動を扱う既存仕様を維持し、現在のGEMと各artifactのSHA、元LPの再認証を実施した。現在取得したSHAが過去時点の完全性を遡及保証するわけではない。

## 閉ループ比較

CPUはpersistent HiGHS、4worker×1thread、非同期後段pipeline。同じbank/順位モデルを使用。GPUは全段のbarrier batch、K=4、GPU修正0回、GPU不合格のみ厳密CPU。実行順は交互。schedulerが異なる実装全体の比較であり、ハードウェアだけを変えた単一要因比較ではない。

| 評価 | seed | CPU秒 | GPU併用秒 | GPU認証LP / 全LP | 終点・元LP基準 |
|---|---|---:|---:|---:|---|
| 旧後段辞書・4×3×2 | 20297101–20297108 | 3.4738 | 4.9444 | 15 / 72 | 合格 |
| 再構築後・4×3×2 | 20298101–20298108 | 3.4943 | 4.2879 | 23 / 72 | 合格 |
| 再構築後・32×8・CPU先行 | 20298201–20298232 | 29.6552 | 36.3488 | 123 / 768 | 合格 |
| 再構築後・32×8・GPU先行 | 20298233–20298264 | 30.9652 | 36.7416 | 118 / 768 | 合格 |

旧/新4×3は異なるseedなので、差を辞書変更の純粋な因果効果として扱わない。2組の計測から信頼区間・統計的有意性を主張しない。初期bankロード、教師作成、学習はonline時間とは別。各組の初回使用・転送・認証・CPU差戻し・状態更新は実測側に含む。

今回のoffline構築wallはaggregate138.24秒、exchange137.79秒、順位学習は各7.35秒・6.14秒。既存教師採取と旧辞書構築の費用を含む累積総費用ではない。

32×8×2の段階別内訳：

| 段階 | GPU認証 | CPU差戻し | GPU認証率 |
|---|---:|---:|---:|
| maxmin | 195 / 512 | 317 | 38.09% |
| aggregate | 28 / 512 | 484 | 5.47% |
| exchange | 18 / 512 | 494 | 3.52% |
| 全段 | 241 / 1536 | 1295 | 15.69% |

CPU差戻し1295件の実optimizer実行も1295回、未使用先行CPU解は0。元LP閾値はprimal∞≤1e-5、dual≤1e-7、relative KKT gap≤1e-7。終点最大差はPHA相対6.8204e-8、菌体量2.0159e-8 g/L、PHV分率1.1521e-7。いずれも既定0.01以下。CPU数値一致は実験的な生物学的妥当性とは別である。

## 候補不足と数値誤差の切り分け

既存開発診断traceの未学習seed 20296001、step1/2/4/8、aggregate/exchange計8 LPを使用。診断traceを学習へ転用せず、保存済みCPU参照x/yを読み込まない。

1. 旧候補群の凸包/affine空間におけるCPU小LP診断は、正の行スケーリング後も元LP認証0/16試行。単に重みのQPを追加する根拠は得られなかった。
2. 新96候補について、同じbasic/active/bound指定を現在の正規化行列でCPU sparse LU再因子化し、最大2回反復改良した。basis切替・LP optimizer・参照解は使わない。
3. 8 LP×96候補=768解釈で、GPU辞書もCPU再因子化も認証0件。3候補解釈は特異/不正なbasis。残りも主制約を満たさず、各LPの最良主残差は約0.003–3.73で閾値1e-5を超えた。

これは無限精度での不可能性証明ではない。しかし、今回の診断では単なるGPU丸め誤差や順位だけよりも、固定した有効制約の組合せそのものが適用できないことを強く示す。未知のbasisへ移動できる汎用GPU解法が必要である。

## 検証と成果物

対象回帰テストは別process群で289件＋22件、計311件が合格。同一processへ全てをまとめた実行では、診断LPテスト後に旧SciPy LP参照がNoneを返す5件の順序依存が残る。全suiteの単一process合格とは主張しない。実運用比較とCPU診断は別processで実施した。

- 計測: `results/pf_gpu_first_all3_4x3_20260905.json`、`results/pf_gpu_first_rebuilt_all3_4x3_20260905.json`、`results/pf_gpu_first_rebuilt_all3_32x8_20260905.json`（各source snapshot併設）
- 再構築: `results/pf_aggregate_rebuilt96_20260905`、`results/pf_all3_rebuilt96_20260905`
- 最終bank manifest SHA: `bcbc6bff85b797d176b5682cbc91bcf0be8135c286e8104a62036062753cba54`
- 順位学習: `results/pf_aggregate_router96_v2_20260905`、`results/pf_exchange_router96_20260905`
- 診断: `results/pf_downstream_candidate_hull8_scaled_20260905.json`、`results/pf_all3_refactor_diagnostic8_20260905.json`
- 集計: `scripts/summarize_gpu_first_batch.py`。全3実測ファイルの厳格検証を実施済み。

次の方式選定は [辞書非依存GPU方式の検討](DICTIONARY_FREE_GPU_APPROACH_REVIEW_20260905.md)を参照。無効な旧データの削除、新しい依存環境へのupgrade、PPO既定の変更は今回行っていない。
