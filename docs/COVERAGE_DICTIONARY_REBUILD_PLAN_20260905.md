# 認証coverageを基準にしたGPU辞書再構築計画

## 判断の根拠

辞書を増やせば必ず速くなるとは仮定しない。前回のK4→K1ではGPU判定時間は減ったが認証率も落ち、総時間は安定してCPUを上回らなかった。今回は以下を分離する。

- **候補不足**：全辞書を診断用に試しても元LP証明に通る候補がない。
- **選択失敗**：有効候補は存在するが、少数のK候補へ絞ると取り逃がす。
- **費用問題**：認証率が上がっても転送・候補判定・CPU復帰を含む総時間が増える。

[SurfinFBA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007786)は有効な基底を再利用し、制約を満たさなくなる境界で再選択する根拠となる。[Akbari・BartonのmpFBA研究](https://arxiv.org/abs/1802.02567)はGEMの退化・多重最適解とactive setを扱う。本系はRHSだけでなく行列の一部も変化するため、固定行列の区分線形領域をそのまま流用せず、既存low-rank更新と**現在の元LP全制約・双対・gap証明**を維持する。論文の速度倍率を本系へ転用しない。

## データと精度の契約

対象はmaxminのみ。OR16＋NS21＋P. freudenreichii、後段aggregate/exchangeの辞書・目的・数値条件は変更しない。全LPで主残差≤1e−5、双対違反≤1e−7、相対gap≤1e−7。closed-loopではPHA相対≤1%、菌体量絶対≤0.01 g/L、PHV分率絶対≤0.01を維持する。

再構築には宣言済みtraining4軌跡×120step＝480状態だけを使う。旧44候補は保持し、既存結果は上書きしない。各chunkの現在SHA、旧bank/source manifest、全GEM fingerprint、学習seed、選定/投影/CPU証明設定を新規artifactへ保存する。旧cacheには個別chunk hashがないため、過去改変がなかったことを遡及保証したとは書かない。

routerのtrajectory分割はbank全体が独立なholdoutではない。旧bankは既に4軌跡由来の候補を含むため、内部router診断として扱い、独立評価seedを候補作成・選択・特徴尺度のfitへ混ぜない。

## 実行順序

1. 既存44候補×全480training状態を低メモリで認証し、全候補coverageとK1/K4の取り逃がしを分ける。これを先に測り、候補追加の必要性を確認する。
2. 元座標のpersistent HiGHSでtraining LPを再解し、元LP証明と完全basis statusを取得する。重複基底を除き、未被覆状態を優先してcompact mapを作成する。full inverse repair operatorは生成しない。
3. 新候補も全training状態へ適用して元LP証明coverageを測定する。旧44をmandatoryとし、候補数96を初期上限、byte予算付きgreedy set-coverで選ぶ。128への拡大は残存未被覆率・追加費用を見て決める。学習点を解けるだけで汎化成功としない。
4. 同じ最終辞書でnearestとmulti-labelの線形/小MLP routerを比較する。未認証best候補を正解ラベルにしない。all-false行も明示する。onlineはK=1/4固定、Torchを持ち込まずGPU FP32推論＋元LP FP64証明とする。
5. 独立traceで候補不足/選択失敗の変化を確認し、同じ辞書を与えたCPU対照を含む短期closed-loopで総wallを測る。初回setup・再構築・学習費用は別記する。精度・速度が改善した案だけを中盤/120stepへ拡張する。

各段階で計画を更新する。全候補coverageが十分なのにK4が悪い場合は候補追加を止めrouterを優先し、候補追加でも被覆が改善しない場合はfamily guard・数値安定性・モデルの変動範囲を調べる。未認証fluxはPPOへ渡さず、現行の安全なCPU経路を維持する。

## Revision 1

一次資料と旧builderの時点間引きを再確認。training loaderの入力hash/連続chunk/schema/scale復元ガードを実装開始し、streaming元LPcoverage計測とmulti-label routerを並行して実装する。新辞書・速度改善はまだ未確認。

## Revision 2 — 全480学習状態の既存候補coverage実測

`results/pf_coverage_baseline480_20260905/report.json`：既存44候補の全探索でも182/480 (37.92%)のみ認証。stateless最近傍ではK1=80/480 (16.67%)、K2=103/480 (21.46%)、K4=119/480 (24.79%)。候補不足298行、K4の選択失敗63行を区別できた。現時点ではrouterだけより候補追加を優先する根拠がある。

旧44を再投影せずbyte-identicalに保持する。追加poolは全480の元座標CPU証明済みbasisを重複除去し、未被覆のsource行を多く持つ基底から順にcompact化する。新候補ごとに全480行で証明し、自己学習行でも証明できない候補を無条件採用しない。全状態の被覆に達するか、明示したpool予算で停止し、最終96候補以内のgreedy selectionを行う。480独立標本とは扱わない。

loaderは現在GEM hash・source/old artifact hash・float64 schema・positive scales・chunk順・480行契約を検査し、推定seed/step表を保存。過去collectorの固定行差≤2e−12の省略とLP生成器全sourceの欠如は履歴上の制限として明記した。新しい未使用seedでの最終closed-loop評価はこのcache再構成の検証とは別に行う。

## Revision 3 — 再構築・router・新規120step入力評価まで完了

480 cached LPを元座標CPUで再解し全件証明、424種類の完全statusを得た。未被覆時点を持つ候補から144件をcompact化して全480時点に適用し、pool全体のcoverageは476/480。旧44候補を保持した総96候補のbyte-budget greedy選定では393/480 (81.875%)、投影配列208,378,720 bytes。これはGPUピークVRAMではない。全逆行列は作成していない。再構築は155.87秒（元LP再解・投影・GPUcoverage・I/Oを含む）。

multi-label線形/hidden32 MLPをtrajectory単位4-foldで内部比較し、K4平均57.29%/61.04%によりMLPを採用。固定設定のまま全480training行へ再fitした。全trainingでK1=375、K2=390、K4=393。bank構築に全軌跡を使うため、このfold値は独立なbank検証ではない。onlineはTorchなしのFP32推論、FP64元LP証明を維持。CPU対照にも同じNPZをNumPyで与え、既存warmbasisがある場合は従来どおり候補推論を省略する。

新規seed 20296001–04の120step入力を一切fit/選定へ混ぜず評価した。旧44の全候補coverage132/480、K4=72/480 (15.0%)。新96の全候補coverage292/480、nearest K4=130/480、learned K4=259/480 (53.96%)。新learned K1=193/480 (40.21%)。未知入力での少数候補認証率は改善したが、入力trace評価だけでclosed-loop速度・完全GPU化の成功とはしない。

4×3 closed-loop smokeは全元LP/終点合格（終点誤差0）だがCPU2.096秒、hybrid6.131秒で遅かった。そこで候補数や学習をさらに増やす前に、32環境のmatched CPU比較を実行する。初回capture費用と後続反復を区別し、総wallが不利なら既定PPOへの昇格はしない。

## Revision 4 — 少数候補の精度改善を確認、速度上の昇格は保留

同一の新辞書・学習済みrouterをCPU/GPU双方へ与え、32環境×8step×2組をK4/K1で比較した。全LP/終点ゲート合格、終点誤差0。

- K4：CPU58.9708秒、hybrid61.6208秒（4.49%増）。GPU認証208/512 maxmin、CPU取消76件、認証済みGPU解と重複して実行されたCPU解132件。
- K1：CPU57.9005秒、hybrid59.1804秒（2.21%増）。GPU認証93/512、CPU取消46件、重複CPU解47件。

K1の一方の組では差0.16%まで縮まったが、2組全体でCPU超えとは判断しない。32×8の早期閉ループと4×120のstateless入力評価は条件・順位付けが異なるため、54%認証率をそのまま短期runtimeへ転用しない。

今回の優先課題であるtraining-only辞書再構築・未知入力での少数候補認証率改善は確認できた。次は無条件の学習拡大より、(a) 上位候補と前回候補の優先順位による取り逃しの切り分け、(b) GPU提案・証明前のhost正規化/転送/順位取得の実測削減、(c) 証明時には既に完了してしまうCPU仕事の重複削減を優先する。後段CPU処理が全体時間の大部分を占めるのでmaxminだけの成功を完全GPU化と扱わない。

候補poolの未測定statusと旧44 mandatory制約は今後の選定改善余地として残る。現在のgreedyは評価済みpool内のheuristicであり、全424基底に対する最適96集合を見つけたわけではない。新bank/routerはopt-inのまま保存し、既定PPOを変更しない。
