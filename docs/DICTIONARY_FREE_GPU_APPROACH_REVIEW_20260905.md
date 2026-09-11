# 辞書非依存GPU方式の検討

## 判断

固定辞書を正解の供給源とし、不一致をCPU LPへ戻す構成では、辞書の範囲外が残る限りCPU律速が残る。現在の後段認証率はaggregate5.47%、exchange3.52%。辞書のさらなる拡張を主方針にするのは停止する。

ただし、辞書方式一般が必ずCPUを必要とするわけではない。辞書の外も探索できるGPU最適化器を後段に置けば、CPU LPは設計上不要にできる。未収束の場合は同じGPU上で追加反復・再因子化・別GPU解法へ移るか、明示失敗として止める。未認証解で状態を進めることや「CPUを呼ばなかった」だけを成功と扱うことはしない。

主方針は「解を事前に網羅する」から「行列構造を共有し、毎回のLPをGPUで解く」へ変更する。辞書・MLP・GRUは補助的な初期値生成器としてのみ残し、無くても動作する解法を目指す。

## なぜ固定辞書の拡張では足りないか

培地・菌体量・流加actionによって上下限、目的、係数が変わる。固定候補が保持するbasic/active指定が、その時刻の制約に適合する保証はない。順位モデルは既存候補の順番を変えるだけで、新しい有効制約の組合せを作れない。

最新の診断は、各96候補を現在の行列でCPU sparse LU再因子化しても未学習8 LPで0件認証だった。元LP主残差の各LP最小値は約0.003–3.73。精度閾値1e-5の近傍だけの問題ではなく、単にGPUの小行列解法を高精度にすれば足りる、という仮説は支持されなかった。有限精度の診断なので数学的な不可能性証明とは区別する。

## 代替案と優先順位

| 方式 | 固定辞書外への対応 | 本系への適合性と条件 |
|---|---|---|
| 構造共有型GPUバッチ内点法 | 全変数・全制約を直接扱う | 第一候補。ただし疎分解のGPU費用・fill-in・数値安定性を先に測る。単純なcuOpt再呼出しではない |
| 動的GPU revised/dual simplex | basisを実行中に変更する | 時系列warm startに適する可能性。GPU pivot間の依存、退化、再因子化が難所。双対実行可能な初期basisを確認できる場合に限定して比較 |
| 高度化GPU PDLP/rHPDHG | 全空間の主双対変数を反復更新 | 大バッチ・低メモリに有利な候補。過去の未収束を踏まえ、restart・前処理・primal weight制御が実質的に変わる場合だけ再評価 |
| NN/GRU warm start＋上記のGPU解法 | 最終解法が辞書外を探索する | 上記の少なくとも1方式が収束してから追加。教師MSEより「元LP認証までの時間」を学習・採否指標にする |

### 内点法を第一候補として調べる理由

有効な反応集合を事前に選ぶ必要がなく、反復の主処理を疎行列分解・行列演算へまとめられる。GPU向けcondensed KKT方式や、ブロック構造を利用した内点法の研究がある。ただし、既存論文の対象は主として非線形最適化・電力系統等で、本dFBAの速度保証ではない。条件の悪い問題で頑健性や速度が低下する場合も報告されている。[Pacaud et al., GPU condensed IPM](https://arxiv.org/abs/2405.14236)、[block-structured IPM](https://arxiv.org/abs/2301.04869)。

cuDSSは共通の非零配置を持つuniform batchと構造解析の再利用を提供する。数値が変われば数値因子化は再実行する必要があり、古い因子の無条件流用はできない。本系ではまず段階別のCSR superpatternを検証し、状態に応じてゼロになる係数も同じ配置に保持する。培養環境間を独立にbatch化し、必要なら菌種内ブロックと共有制約のSchur補を検討する。共有結合が小さいと先に仮定せず、fill-inと費用で採否を決める。[cuDSS data types](https://docs.nvidia.com/cuda/cudss/types.html)、[cuDSS functions](https://docs.nvidia.com/cuda/cudss/functions.html)。

構造の再利用は、有限の「解候補辞書」と違って特定の栄養条件の正解を記憶することではない。完全な数式から得た非零配置を保持し、境界条件・目的の変化は毎回反映する。ノックアウト等でモデルが変更された場合は識別子を変えて再解析する。

cuDSSでもreorderingは通常CPUで行う。したがって、まず「初期構造解析はhost、定常的な数値更新・反復・認証はGPU」を明確な目標にする。host API制御まで無いという意味の完全GPU化とは呼ばない。[cuDSS execution phases](https://docs.nvidia.com/cuda/cudss/general.html)。ローカルはcuOpt26.8.0、cuDSS0.7.1.6。0.7.1ヘッダでUBATCH_SIZE/UBATCH_INDEX/REFACTORIZATIONの存在を確認済み。最新0.8固有の機能が導入済みとは仮定せず、依存環境をこの検討だけでupgradeしない。

### 過去の失敗を繰り返さない条件

- cuOpt Barrier、PDLP、正規化、複数presolve、縮約を過去に試しており、単なるソルバ交換は新しい提案ではない。`pf_gpu_block_barrier_dev4_20260905.json`の4環境・step1/41・3段階、3秒solve上限でも全block合格に至っていない。これは冷間固定入力診断で、warm dFBA全体の性能試験ではない。
- GPU simplexには、CPU LP0で精度が通るが非常に遅い実装が既にある。前回basisの保持だけではなく、GPU疎分解・方向計算・pivot更新の費用を具体的に削れる場合だけ再利用する。過去のdual simplex試作も初期双対実行可能性不足などで不採用だった。
- 新しいPDLP候補は単純反復の増量ではない。rHPDHG/cuPDLPxは再始動やprimal weight制御などを含む。旧2,048反復PDHGやcuOpt3秒試験の失敗がこれら全方式の不可能性を意味するわけではないが、同じ入力で認証までの費用が改善するかを確認する。[rHPDHG](https://arxiv.org/abs/2407.16144)、[cuPDLPx](https://arxiv.org/abs/2507.14051)。
- cuOptのConcurrentはCPU dual simplexも含む。GPU-only評価はPDLP/Barrierを明示し、crossoverやpresolveのhost処理、CPU LP呼出しを区別する。ライブラリの成功statusだけで元LP合格とはしない。[NVIDIA cuOpt methods](https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html)。

## 次の限定的な検証順

1. **保存LPの構造・残差監査。** 未学習のstep1/2/41を含む各段のLPで、CSR pattern、係数範囲、冗長等式、変数bounds、原LP残差を分類する。モデルの反応を学習出現頻度だけで削らない。
2. **GPU疎分解の小さな試験。** 同一CSR superpattern・異なる係数のbatchで、解析を1回だけ行い、数値更新・factor・solve・元残差・memoryを測定。単に分解済み行列のsolveだけをCPU LP全体と比較しない。
3. **方式選定。** 分解が十分安定かつ軽ければ構造共有内点法へ進む。分解が重すぎれば高度化PDLPを優先する。少数pivotで修正可能かつ双対実行可能な初期化が確保できる場合だけdynamic dual simplexを対抗にする。最初から3方式の本格実装を並走させない。
4. **coldで収束する解法を先に確立。** NN/GRUなしでも元LP基準に通るか確認し、次に前回GPU解、最後にNN/GRUの初期化を比較する。固定反復数後の残差・解距離を学ぶ設計は参考になるが、本系の保証ではない。[Sambharya et al., JMLR 2024](https://www.jmlr.org/beta/papers/v25/23-1174.html)。
5. **閉ループの総費用。** 4×3で機能・精度を確認した後、32×8の強いCPU対照で、CPU LP0のGPUモードと比較する。認証0、OOM、毎stepの構造再解析、CPUを超える修正費用なら長い120stepへ拡張しない。速度が確認できた構成だけ120step・独立条件・PPOへ進める。

LPの受理閾値もPHA終点1%基準もこの検討では緩和しない。単なるQP投影は実行可能性を改善しても、元LP目的の最適性を保証しない。小さな二次正則化を永続的に目的へ加える変更は同一LPではなく、別モデルとして扱う必要がある。

## 現在の達成状態

本書は方式選定・調査結果であり、新しいバッチ内点法の実装完了報告ではない。ユーザーの最新依頼は代替案の検討のため、新規ソルバ導入や生物モデル変更は行っていない。従来の全段GPU-first実装・辞書再構築・実測の完了内容は [実測レポート](GPU_ALL_STAGE_BATCH_RESULTS_20260905.md)に分離した。CPU速度超えは未達で、現時点で新GPU購入による解消を保証できない。
