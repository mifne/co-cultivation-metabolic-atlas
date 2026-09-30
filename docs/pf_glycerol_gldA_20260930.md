# Pf グリセロール経路と gldA（NAD 依存グリセロール脱水素酵素）の妥当性検証（2026-09-30）

対象: `models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml`（元: Machado et al. 2020 `P_sherm_model.xml`）。モデルファイルは変更せず、試験はすべてメモリ上のコピーで実施。
道具: NCBI BLAST+ 2.16.0（blastp/tblastn、ローカル）、Biopython 1.87、COBRApy 0.31.1。diamond/hmmer/mmseqs は無し。ネット不使用。

## 1. ローカルにある Pf 配列
- `models/genome/*.faa` は Actinoplanes (AP019371.1) と NS21 のみ。Pf の .faa は無い。
- Pf の全ゲノムは補足資料 `.../Supplementary file 4 S7/Genbank_files.zip` 内の `Propionibacterium_freudenreichii_subsp._shermanii_CIRM-BIA1.gbk`（2,616,386 bp、CDS 2406、翻訳・機能注釈つき）にある。LOCUS 行が壊れていて Biopython では読めないため、正規表現で FASTA に変換した。
- モデルの遺伝子 ID `...CIRM-BIA1.CDS.n` は gbk の `/protein_id` と 1:1 で一致する（CDS.1463 = Glycerol kinase、508 aa）。同じ zip に他の Propionibacterium 5 株の gbk もある。

## 2. 配列の証拠（gldA は見つからない）
- Pf の注釈に「glycerol dehydrogenase / EC 1.1.1.6 / gldA / dhaD」は無い。
- 手元で使える GldA 参照は Actinoplanes の BBH65341.1（repo 内の注釈のみが根拠、353 aa、GIGGGKTVDTAK モチーフあり）。これを blastp で Pf プロテオームに当てると、最良ヒットは **CDS.2215**（同一性 32.4%、275 aa 整列、クエリ被覆 78%、E=1.5e-20）。次点は CDS.1181（3-dehydroquinate synthase、27.5%、E=4e-6）。
- tblastn で Pf ゲノム全体を検索しても同じ 2 座位だけで、注釈されていない ORF は無い。
- CDS.2215 の注釈は「Predicted glycerol-1-phosphate dehydrogenase, arabinose operon」（458 aa）。リボキナーゼ/AraL/トランスケトラーゼのクラスター内にあり、P. acnes の G1PDH (EC 1.1.1.261) と 74% 同一。Machado S4 の種間クラスター 7999 でも G1PDH として扱われている。同じファミリーの別酵素なので gldA の証拠にはならない。CDS.2215 はモデルでは未使用。
- Machado S4 に「Glycerol_dehydrogenase (EC 1.1.1.6)」の行は 1 つだけあるが、これは P. acnes 11 株のクラスタリング欄（列 C–M）の行。Pf 列の値でも、最終機能行列の行でもない。
- Fe 型 ADH（CDS.932）や Zn 型 ADH（CDS.743、CDS.1890）は GldA 参照にヒットせず、すでに別の反応に割り当て済み（プロパノール/エタノール、rxn00763）。

## 3. モデル内容（関連反応）
- 取り込みと活性化: glpF CDS.1464（rxn05581）、GlpK CDS.1463（rxn00615）、MQ 依存 G3PDH（glpABC CDS.1320-1322、rxn08557）。NAD(P) 依存 GpsA（CDS.87）は DHAP→G3P の向きのみ。
- DHA キナーゼ: ATP 型 CDS.2223（rxn00744）と PEP 型 DhaKLM CDS.835/837/838（rxn00745）の 2 系統があり、DHA が入れば解糖系へ進める。DHA は H+ 共輸送（遺伝子なし）で、交換反応は分泌のみ。
- 既存の rxn00763（glycerol + NAD ⇌ D-glyceraldehyde + NADH、ADH 遺伝子）は、D-グリセルアルデヒド→F1P 以降が行き止まりのため流束は常に 0。
- dhaB 相当: rxn00768（PduCDE CDS.937-939、glycerol → 3-HPA）。3-HPA は酸化方向（PduP）にしか進まず、1,3-PDO 脱水素酵素（dhaT）と 1,3-PDO 自体はモデルに無い。
- 電子の逃げ先: MQH2 を消費するのはフマル酸還元酵素 rxnnew11 など。MQH2→NAD+ の経路は無い（rxn10123 は不可逆、rxnHYPOTHETICAL2 は (0,0) で閉鎖）。
- 式のバグ: プロパノール (cpd03559) と 3-HPA (cpd00714) の化学式が `0.0`、cpdnew29 が `-1.0`。ほかに CDS.1465（tRNA pseudouridine synthase）が 1,2-PDO 輸送の GPR に入っている。

## 4. FBA（メモリ上、glc/lac/ppa の取り込み 0、glycerol 取り込み ≤10 mmol/gDW/h、pFBA）
- 元モデルに glycerol を与えた場合: μ=0。ただし glycerol→propionate 自体は 10→10 で回り、ATP 1.5/glycerol が得られる。
- 増殖できない原因は MQH2 の余剰: MQH2→MQ の自由排出を仮に置くと μ=0.535。NADH や NADPH の自由排出では μ=0 のまま。したがって NADH を消費する出口（プロパノール、H2S）をいくら増やしても解決しない。
- NDH (rxn10123) を可逆にする（逆電子伝達）と μ=0.251。これも遺伝子上の根拠は無い。
- プロパノールの式を C3H8O に直すと rxn01710 の質量収支は合うが、μ は GldA 無しで 0、有りで 0.3348 と変化しない（FBA は化学式を使わないため当然）。電子トラップには無関係。
- 仮想 GldA `glycerol + NAD → DHA + NADH + H+`（bounds 0..1000、GPR 無し）を足すと μ=0.3348（glycerol 20 では 0.6695 で、監査時の約 0.66 はこの条件）。
  - 収率 Yx 0.364 gDW/g glycerol。生成物は mol/mol glycerol で propionate 0.28、propanol 0.24、acetate 0.09、H2S 0.024。
  - propanol と H2S の出口を両方閉じても μ=0.3085（propionate 0.41）。

## 5. 判断
- (a) 支持する材料: DHA キナーゼが 2 系統あること、実菌がグリセロールで良く育つこと。否定する材料: Pf ゲノムに gldA/dhaD の遺伝子が無いこと（注釈・blastp・tblastn）、唯一のヒットが G1PDH であること、S4 の GldA 行が P. acnes のものであること。
- (b) gldA は正当化できない。仮に入れるなら `GLYCDx_hyp: glycerol + NAD → DHA + NADH + H+`、(0,1000)、GPR 空、`no gene evidence` の注記つきで opt-in にする。ただしこの場合の生成物組成（プロパノールが主産物の一つになる）は、プロピオン酸が主産物という既知の表現型と合わない可能性が高い（文献値との定量比較は未実施）。
- (c) 確信度: 「CIRM-BIA1 の注釈済みゲノムに gldA は無い」は中〜高。ただし参照は Actinoplanes の注釈 1 本だけで、E. coli GldA P0A9S5 などの検証済み配列はローカルに無い。モデル欠陥の本体は MQH2→NAD(P) の不在（または G3P 酸化の電子受容体の割り当て）と考えられるが、どう直すのが正しいかは生化学的な確認が要る。
- (d) 推奨: モデルは変更しない。最も安い確認は、UniProt P0A9S5（E. coli GldA）と Klebsiella DhaD を CIRM-BIA1 のプロテオームとゲノムに当てること:
  `blastp -query gldA_refs.faa -subject pf.faa -evalue 1e-5 -outfmt "6 std qlen slen stitle"`
  `tblastn -query gldA_refs.faa -subject pf_genome.fna -evalue 1e-5 -outfmt 6`
  どちらも 30% 未満で CDS.2215（G1PDH）しかヒットしなければ「gldA 無し」で確定。その場合は、無細胞抽出液で glycerol+NAD+ の還元活性を測る実験が次の手になる。
