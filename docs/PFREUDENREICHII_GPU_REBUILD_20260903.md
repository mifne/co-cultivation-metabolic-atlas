# P. freudenreichiiを含むGPU辞書の再構築

更新日: 2026-09-03

## 結論

OR16 + NS21 + *Propionibacterium freudenreichii* subsp. *shermanii* の3菌系について、32,768候補の厳密協調LP教師データ、GPU辞書、ニューラル順位付け層を再構築した。P. freudenreichiiは除外していない。乳酸供給はaction[2]から操作する。

ただし、24 h・3 seedの未学習作用系列による検証ではPHA終点誤差最大11.88%、GPU採択率平均72.22%であり、目標のPHA誤差最大1%・採択率90%以上を満たさない。成果物は `experimental_not_qualified` として保存し、検証を要求する通常の実行経路では自動採用しない。完全GPU移行・CPU相当精度を達成したという結論ではない。

## 菌の役割と科学的限定

- OR16: ゴム分解・可溶化。
- NS21: ゴム由来中間体の利用と3HB/3HV反復単位の蓄積。
- P. freudenreichii: 乳酸→プロピオン酸/酢酸変換による条件付き補助菌。B12供給は受容側の必須性が現GEMに十分表現されていないため、利益として加算しない。

P. freudenreichiiモデルは[Machado et al. (2020)の公開pan-GEM](https://doi.org/10.3390/genes11101115)由来で、1,351反応・1,197代謝物・719遺伝子を維持した。共有交換IDと既存反応の培地境界のみを正規化し、反応や遺伝子は追加していない。嫌気的乳酸→プロピオン酸表現型を想定して酸素交換を閉じている。したがって単一好気ジャーでの実際の酸素曝露・株特異的速度・B12要求・PHBV組成は実験較正が必要である。

修正済みNS21を使った20条件の逐次dFBA再評価では、最良3菌条件のPHAは0.11655 g/L、同じ乳酸流加速度の2菌条件に対して1.001倍だった。乳酸無添加の3菌系は2菌ゴム対照の0.984倍で、自然な共生利益は検出されなかった。これは実培養値ではなくモデル予測である。第3菌は「必須菌」ではなく、PHBV組成制御の候補として比較対象に含める。

## 今回修正した問題

1. 旧協調LP設定は共存増殖→交換最小化へ進み、窒素制限時のNS21製品目的を最適化していなかった。第1段階で共存増殖率を最大化し、その99%を保持した上で第2段階で現在の菌別目的、第3段階で不要交換を最小化する設定に変更した。
2. 旧初期NH4サンプリングは主に0.5–3 mMで、PHA開始条件の0.1 mM未満を覆わなかった。初期NH4を製品相0–0.06 mM（55%）、遷移相0.06–0.30 mM（25%）、増殖相0.30–3 mM（20%）へ層別化した。短軌道4–24 steps（60%）と最大120 stepsの長軌道を混合した。
3. PHB/PHVの変動教師が0件の場合、NS21ニューラル学習を明示的に停止するガードを追加した。
4. NPZデータとPT辞書が同じ名前幹でも、辞書要約がデータのメタデータを上書きしないよう `.summary.json` へ分離した。
5. GEM指紋・菌順序・反応順序・live-objectiveモードの不一致を実行時に拒否する。
6. PHBとPHVを86.09/100.12 mg/mmolで別々に質量換算する。GPU QPの目標合わせも旧PHB単独からPHB+PHV質量加重和へ拡張した。
7. 乱数seed固定、訓練80%/検証20%分割、CPU差し戻し回数、3HV組成誤差の記録を追加した。

最初に生成した製品目的なしの32,768点とその派生辞書は誤利用防止のため削除し、修正版へ置き換えた。

## 再構築済み成果物

| 項目 | 値 |
|---|---:|
| 菌構成 | OR16 / NS21 / P. freudenreichii |
| 反応次元 | 6,733 = 2,584 + 2,798 + 1,351 |
| 状態入力次元 | 20,599 |
| 候補数 | 32,768 |
| 検索・学習特徴数 | 74 |
| PHB非ゼロ教師 | 1,642 (5.01%) |
| PHV非ゼロ教師 | 11,214 (34.22%) |
| ニューラル決定対象 | 155反応、重要対象6、PHA目的2 |
| 訓練 / 内部検証 | 26,214 / 6,554 |
| CPU教師生成 | 約76分、16ワーカー |
| GPU学習 | RTX 4060 Laptop、100 epochs、約9.1秒 |
| 辞書容量 | 1,029.19 MiB |
| 全配列 | 32,768行、NaN/Infなし |

成果物（プロジェクトルートからの相対パス）:

- `models/cooperative_surrogate/production_32768_ns21_phbv_pfreud_20260903/`
- `models/cooperative_surrogate/cooperative_dictionary_32768_ns21_phbv_pfreud_20260903.pt`
- `models/cooperative_surrogate/cooperative_dictionary_32768_ns21_phbv_pfreud_20260903_neural.pt`
- `models/cooperative_surrogate/cooperative_dictionary_32768_ns21_phbv_pfreud_20260903_neural.validation.json`

SHA-256:

- 辞書: `616eed81ae3d2c797a4f3c516b03db9b796070158925c5ec6dffd565bf15d4d2`
- ニューラル: `755a7dbc5edca227b3bee53d137222c4b8575ddf4a725e80cfcb21853211800b`
- 教師メタデータ: `96defb715bca2aca6dc58b45f2397360697f758a81e6b372318756f82608c44f`

## 24 h外部検証

初期NH4 0.05 mM、dt=0.2 h、120 steps、各seedで同一作用系列。GPU候補512、再順位付け母集団2,048、PHB+PHV質量合わせ。以下は生物実験ではなくCPU厳密モデルへの一致性検証である。

| seed | CPU時間(s) | GPU経路時間(s) | PHA相対誤差 | 3HV組成差(ポイント) | GPU採択率 | CPU差し戻し |
|---|---:|---:|---:|---:|---:|---:|
| 20260911 | 128.46 | 75.54 | 2.75% | 2.73 | 59.17% | 49/120 |
| 20260912 | 129.65 | 68.59 | 2.68% | 4.26 | 65.00% | 42/120 |
| 20260913 | 130.28 | 31.77 | 11.88% | 11.65 | 92.50% | 9/120 |

GPU最大割当VRAMは951.07 MiB。GPU経路時間にはCPU差し戻しを含む。CPU参照は3並列、GPUは逐次実行したため、時間比は診断値であり、厳密な等負荷ベンチマークや購入性能主張には使わない。

関連回帰テスト26件はすべて成功した。ただしテスト成功は代理モデルの科学的適格性を意味しない。

## 残る問題と次の修正順

1. **長軌道の実行可能候補不足**: 乳酸、グルコース、プトレシン等の共有供給制約に対し局所候補の凸包が足りない。失敗軌道のCPU厳密スナップショットを選択的に追加する。候補数だけを一律拡大するより、失敗状態と低消費の実行可能アンカーを補う。
2. **実行可能性と最適性の区別**: QPが質量収支・境界を満たしてもCPU最適目的へ近いとは限らない。PHB/PHV別々の予測、総PHA目的差、菌別増殖目的差を使う最適性ガードが必要。
3. **本番統合は保留**: 最大PHA誤差1%、3HV組成誤差1ポイント、採択率90%以上を満たすまで、検証マニフェストの未適格状態を解除しない。
4. **生物学的採用判定**: 二菌系の独立対照、直接乳酸/プロピオン酸流加、第三菌添加を同じコスト制約で比較する。GEMで未表現のB12依存性や単一槽での酸素適合性は実験で較正する。

## 再現コマンド

```bash
python scripts/collect_cooperative_surrogate_dataset.py \
  --tier production --workers 16 --episode-steps 120 --seed 20260903 \
  --consortium pf-helper3 \
  --output models/cooperative_surrogate/production_32768_ns21_phbv_pfreud_20260903 \
  --dataset-format npy-directory

python scripts/validate_cooperative_neural_surrogate.py \
  models/cooperative_surrogate/cooperative_dictionary_32768_ns21_phbv_pfreud_20260903_neural.pt \
  --consortium pf-helper3 --steps 120 --seeds 20260911 20260912 20260913 \
  --initial-nh4 0.05 --gpu-qp-projection --gpu-qp-candidates 512 \
  --gpu-qp-rerank-pool 2048 --gpu-qp-match-pha --exact-workers 1
```

上記の検証再実行は等負荷比較のためCPU参照を1ワーカーにしている。今回保存済み結果は3ワーカーである。
