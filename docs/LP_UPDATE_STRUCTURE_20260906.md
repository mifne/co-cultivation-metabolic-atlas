# Changing-state LP: input-only構造監査

## 範囲と結論

`results/pf_lp_trace_dev32x41_20260905`の先頭4環境、step 1/2/3、
maxmin/aggregate/exchangeの36入力を調べた。全32環境への一般化は未検証。
入力checksum・元LP hash・stage/step/environmentラベルを検査し、保存されたreference x/yは読んでいない。
LP solve、GPU、QRはいずれも0回。これは構造の診断であり、最適解認証・性能測定・閉ループ実行ではない。

主な結果は、**step 1→2は座標構造の変更が必要だが、step 2→3は同じ縮約構造を再利用できる候補**というもの。
各stageの各環境について独立に比較した結果であり、2→3以後も永続的に変化しないとは仮定しない。
記録済みaggregate/exchange入力には記録元の先行段階の計算結果が反映されている。
それをGPUへ直接渡すinput replayは、GPUが自分の解で次段階を作るcausal replayとは異なる。

## 観測された変更

同じstage・同じ環境の時間方向で比較。以下の変更数は1 LPあたり、RHSは4環境の範囲。
等式行列と等式RHSは全24遷移で不変、全stageでlower境界は不変だった。

| Stage | 遷移 | Aの変更係数数 | Aの変更行数 | CSR support追加/削除 | RHS変更数 | upper変更数 | cost変更数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maxmin | 1→2 | 639 | 397 | 0 / 0 | 41–42 | 2 | 0 |
| maxmin | 2→3 | 639 | 397 | 0 / 0 | 47–48 | 0 | 0 |
| aggregate | 1→2 | 639 | 397 | 0 / 0 | 41–42 | 2 | 3 |
| aggregate | 2→3 | 639 | 397 | 0 / 0 | 47–48 | 0 | 0 |
| exchange | 1→2 | 1,920 | 1,676 | 1 / 2 | 42–43 | 2 | 0 |
| exchange | 2→3 | 1,917 | 1,675 | 0 / 0 | 48–49 | 0 | 0 |

upperの変更箇所は全stageで元座標の列5366、5381（0-based）。明示的固定変数は11→13へ増える。
aggregateのcost変更箇所は5362、5366、5381。これらの列番号から生物学的reaction IDは推定しない。
元LPの`neq=4651`は不変。変更行は不等式ブロックに限られる。

縮約を`forest → zero-face(singleton有効) → second forest`とした形状は以下。
ExactEqualityReductionのQRや行削除は実行していないので、これは最終QR前の形状である。

| Stage | Step | 元LP rows × columns | first forest | zero-face | second forest |
| --- | --- | --- | --- | --- | --- |
| maxmin / aggregate | 1 | 5051 × 6734 | 2640 × 4323 | 2144 × 3914 | 2025 × 3795 |
| maxmin / aggregate | 2, 3 | 5051 × 6734 | 2640 × 4323 | 2142 × 3910 | 2025 × 3793 |
| exchange | 1 | 6330 × 7373 | 3919 × 4962 | 3423 × 4553 | 3304 × 4434 |
| exchange | 2, 3 | 6330 × 7373 | 3919 × 4962 | 3421 × 4549 | 3304 × 4432 |

second forest後の等式数は全条件1633。

- first forestのT/compression/dual-lift/行番号mapは24/24遷移で一致。
- 1→2の12/12遷移でzero-face、second forest、working bound topology、working CSR support、
  QR入力の等式+RHS fingerprintが変化。旧縮約座標のままupdateしてはいけない。
- 2→3の12/12遷移では上記のmap・topology・support・等式+RHS fingerprintが全て一致。
  Aの数値とRHSは変化するので、古い数値factorや古い認証結果は使えない。
- 同stage・同stepで環境0と環境1/2/3を比べた27比較では、forest/zero-face/second forestのmap、
  working topology/support、QR入力の等式+RHS fingerprintが全て一致。
  環境ごとの数値A/RHSは異なる。構造の共有は同一解・同一状態の複製ではない。

## 安全なcache契約

以下のhashは、同じcanonical CSRとFP64入力の厳密一致を調べる構造キーであり、
数値解の認証ではない。最終的な元LP精度ゲートは変更しない。
実装/設定versionと座標の由来もcache namespaceへ含める。

| 再利用対象 | 必要なキー/検査 | 状態更新時に必ず更新するもの |
| --- | --- | --- |
| 元CSRのindex/transpose scatter | stageの行列shape、neq、canonical indptr/indices、row/column ordering | A.data、A.Tの対応data、現在rhs/lo/hi/cと元認証buffer |
| first forestのT/dual lift | equality Aのshape/support/data、元LP shape/neq、等式RHSが厳密に0のmask、max_scale_ratio、実装version | A_kept T、T^T c、現在rhs、induced bounds、endpoint witness |
| zero-faceの全plan（保守的lookup） | 現在A全体/RHS/lo/hi、neq、duplicate/singleton設定。cは不要 | cが変わればreduced cとobjective offset |
| zero-faceのproof/座標のみ（細粒度） | 等式A/RHS/boxの証明とfixed集合/値、witness順序・係数・orientation、duplicate代表、保持行/列。削除不等式のzero-row条件と現在RHSも再検査 | fixed primal template、現在A/RHS/c/box、objective offset。map一致だけで古いplan全体を使わない |
| second forest | zero-face後の座標namespaceと、first forestと同様の構造キー | first forestと同様。上流mapの変化は別cache世代にする |
| ExactEqualityReductionの証明 | その入力座標のequality A+equality RHSのexact fingerprint、証明algorithm/config、保存proof integrity | 不等式A/RHSとbounds/cは現在値をコピー。primal/dual liftを新しい全chainへ接続 |
| bound→constraint変換 | working固定mask、finite lower mask、finite upper mask、変数ordering、neq | 固定値/finite endpoint、b/h、slack初期値 |
| cuDSS symbolic analysis | 最終KKT patternのshape/indptr/indices、batch lane構成、matrix type、condensation方式、index dtype/device/solver version | KKT係数とdiagonal、numeric factor、scaling。数値factorは新しいNewton matrixで再計算 |
| warm x/y/s/z | 独立環境ID、stage、episode/reset世代、座標mapの世代 | 新boundsに合わせたslack/dual初期化と現在元LP認証。CPU現在解/未来解は使用しない |

forestの`.equality_fingerprint`単体は等式RHSを含まない。
`reduce`は既に消した行のRHS=0を検査するが、以前非ゼロだったRHSが0になれば
新しいfresh planで追加のforest辺が選ばれうる。
完全に同じplanをlookupするキーにはhomogeneous maskも含めた。

zero-faceの保守的全入力キーは、今回2→3でもA/RHSが変わるので一致しない。
これを「mapは再利用不能」と読み替えない。細粒度の証明検査を実装すれば、
同じ等式/boxと保持行supportを用いてmapだけ再利用し、変更された不等式値をdeviceで更新できる。
削除されたゼロ行は、固定変数代入後のRHSが等式なら0、不等式なら非負であることを毎回保証する。
固定値が非ゼロの場合のRHS補正も忘れない。

exact equality fingerprintの一致は、**既存の正しい証明があれば再利用できる入力条件**を示すだけ。
本診断ではproofを作成しておらず、どの行を何本落とせるかは報告していない。
また、近似的なbounded-near-equalityを有効化する場合は別扱いで、現在boxでのdefectを再検証する。

同じCSR supportでもvalueが0を跨げばcanonical CSRは変わりうる。
あらかじめ宣言したsuperset patternへ0を明示格納してsymbolicを再利用する設計は可能だが、
各新しいentryがそのsuperset内かを検査し、data scatter位置を厳密に更新することが条件。
未知supportを黙って落とす方法は採用しない。

## 次の実装順序

1. `owner=(stage, environment_id, episode_generation)`を固定し、step 1→2でtopology変更を検出・再構築。
   full original座標の旧GPU解だけを新mapへ写したwarm proposalとして使い、元LPで再認証する。
2. step 2→3のようなmap一致経路へnumeric update APIを追加。
   元A/transpose、forest bound witness、working A/b/h/c、元認証bufferを一括してtransactional更新する。
   partial updateのままsolveやcertificateへ進めない。
3. cuDSS symbolicを保持しつつnumeric factorは更新。現在のIPM class群はconstructorで配列をsnapshotし、
   changing-state入力を受け取るupdate APIは未実装。旧`self.problems`の認証は絶対に残さない。
4. GPU上の状態・先行stageの自前解から次のLP入力を作成し、maxmin→aggregate→exchange→状態更新を検証。
   traceを順番に読むだけの実験とは結果ラベルを分ける。
5. 構造不変判定をdevice mask/比較へ移し、通常stepでhostへ全LPを戻す処理とhash再計算を避ける。
   初期構造証明/compile費用はsetup欄に残す。未知topologyはfail-closedまたは明示した再構築経路へ送る。
6. first4環境の診断だけで32環境を断定せず、32環境・長い時系列でcache hit/invalidateと認証率を確認する。
   CPUはHiGHS 1.15.1、1–16 workers、同じstable stage/environmentのbasis再利用を許す。

## 成果物と検証

- `scripts/probe_lp_update_structure.py`: bounded input-only監査、同一環境の時間差分と同時刻の環境差分。
- `tests/test_lp_update_structure.py`: 13 passed（1.00秒）。CSR support/valueの区別、forestのhomogeneous mask、
  bound witness、同じfinite topologyでもzero-faceが変わる例、QR/LP未呼出しを検査。
- `results/pf_lp_update_structure_dev4x3_20260906.json`: 36入力、24時間遷移、27環境比較、sparse map fingerprint。
- `results/pf_lp_update_inputs_dev4x3_20260906.json`: 同じ入力のmap省略版。各vectorの変更indexも記録。

既存CPU/GPU backend、モデル、精度閾値は変更していない。

## 実装追記: ZeroFaceReduction.rebind

`old_plan.rebind(current_problem)`は、現在入力に対して既存proofを再検査し、独立した新しい
`ZeroFaceReduction`を返すhost sparse-algebra APIとして実装した。
元planとcaller入力は変更しない。新planのoriginal/reduced、A/RHS/c/bounds、objective offset、
元LP/reduced hashは全て現在入力から構成する。行/列mapと証明の配列は独立したreadonly copy。

再利用条件は以下のとおり。

- 元shape/neqと等式A/RHSが厳密に一致。
- 明示固定集合と固定値が一致し、保持列のfinite lower/upper maskが一致。
- saved min/max/singleton witnessを現在bounds上で前進順に再証明。
- replay後の固定集合/値が一致し、新たな暗黙固定が1つも追加可能でないことを最終1passで確認。
- 固定値代入後のzero-row集合が一致し、削除行の現在RHSも可解な符号。
- 既存duplicate equalityの代表・符号・RHS identityが現在入力でも成立。

不成立は`ValueError`（不可能zero rowは`InfeasibleZeroRow`）として再構築を要求する。
不等式Aの数値/support変更そのものは許容するが、保持行がzeroになる/削除行がnonzeroになる変更は拒否する。
証明探索、LP solve、QRや数値的rank判定は行わない。
`validate_integrity`には既存snapshot検査に加え、map/witness/configのfingerprint検査を追加した。

新規`tests/test_lp_zero_face_rebind.py`と既存zero-face/構造診断/exact-equalityの4ファイルで
95 passed（2.41秒）。fresh constructorとの数値・座標比較、所有権、各拒否条件、
constructor/optimizer/QRを呼ばないことを小入力で確認した。
実traceのstep 2→3、先頭4環境×3stageの12入力でもrebindが全て成功し、fresh constructorとの
元LP/reduced hash、objective offset、行/列・固定map、witness/duplicate tuplesが全て一致した。
旧入力・現在入力の非変異と両planのintegrityも検査した。LP/GPU/QRは0回、終了コード0。
記録は`results/pf_zero_face_rebind_dev4_step2to3_20260906.json`。
参考host中央値はrebind 0.04188秒/LP、fresh構築0.17657秒/LPだった。
各stage/environment 1観測の開発診断で、ランダム化反復測定やGPU/E2E速度優位の証明には使わない。
最初のinline実行は構文解析時の括弧SyntaxErrorで計算前に停止し、修正後の実行が上記を完了した。
通常運用のGPU update接続、全device内完結、速度優位はこのhost APIだけでは達成していない。
