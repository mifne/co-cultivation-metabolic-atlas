"""Build a figure-heavy Word compilation of all biological dFBA experiments
run in this project to date (GPU/software benchmarks excluded per user
scope decision).

Content was synthesized by Claude from three DeepSeek V4 Flash extraction
passes (tmp/deepseek_phaseA.md, phaseB.md, phaseC.md), each covering one
chronological phase of the project. Key numbers were spot-checked by
Claude against the original results/*/REPORT_JA.md sources before use.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_PATH = ROOT / "docs" / "BIOLOGICAL_EXPERIMENT_DATA_COMPILATION_20260911.docx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
DARK_BLUE = RGBColor(0x1F, 0x4D, 0x78)
MUTED = RGBColor(0x5B, 0x65, 0x73)
GREEN = RGBColor(0x1B, 0x6E, 0x4F)
RED = RGBColor(0x9B, 0x1C, 0x1C)


def add_heading(doc, text, level=1, color=None):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.color.rgb = color or (NAVY if level <= 1 else DARK_BLUE)
    return h


def add_note(doc, text, color=MUTED, size=9.5):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(size)
    run.font.color.rgb = color
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_conclusion(doc, text, color=DARK_BLUE):
    p = doc.add_paragraph()
    run = p.add_run("結論: ")
    run.bold = True
    run.font.color.rgb = color
    run2 = p.add_run(text)
    run2.bold = True
    run2.font.color.rgb = color


def add_figure(doc, rel_path, caption, width_cm=15):
    full = RESULTS / rel_path
    if not full.exists():
        add_note(doc, f"[図が見つかりません: {rel_path}]", color=RED)
        return
    doc.add_picture(str(full), width=Cm(width_cm))
    last_p = doc.paragraphs[-1]
    last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = MUTED


def experiment(doc, title, *, source, date, purpose, method, results_list,
               figures=None, limitations=None, conclusion, judgment=None):
    add_heading(doc, title, level=3)
    meta = doc.add_paragraph()
    r = meta.add_run(f"出典: {source}　|　日付: {date}")
    r.font.size = Pt(8.5)
    r.font.color.rgb = MUTED
    if judgment:
        r2 = meta.add_run(f"　|　判定: {judgment}")
        r2.font.size = Pt(8.5)
        r2.bold = True
        r2.font.color.rgb = RED if "no_go" in judgment or "不成立" in judgment or "False" in judgment else GREEN

    p = doc.add_paragraph()
    p.add_run("目的: ").bold = True
    p.add_run(purpose)

    p2 = doc.add_paragraph()
    p2.add_run("条件・方法: ").bold = True
    p2.add_run(method)

    doc.add_paragraph("主要な数値結果:").runs[0].bold = True
    add_bullets(doc, results_list)

    if figures:
        for fig_path, cap in figures:
            add_figure(doc, fig_path, cap)

    if limitations:
        add_note(doc, f"限界・注意点: {limitations}")

    add_conclusion(doc, conclusion)
    doc.add_paragraph()


def main() -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Yu Gothic"
    normal.font.size = Pt(10.5)

    title = doc.add_heading(level=0)
    r = title.add_run("生物学的dFBA実験データ集")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("天然ゴム分解・PHA変換コンソーシアム — 全実験の図表付きまとめ　2026年9月11日時点")
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_heading(doc, "0. 本書について", level=1)
    doc.add_paragraph(
        "本書は、本プロジェクトでこれまでに実施した生物学的dFBA(動的フラックスバランス解析)"
        "計算実験を、実施順に3フェーズへ整理し、対応する図とともにまとめたものである。"
        "GPU高速化などのソフトウェア工学的ベンチマークは対象外とし、生物学・代謝モデルに"
        "関する知見に絞っている。"
    )
    add_note(
        doc,
        "重要な前提: 本書に含まれる結果はすべて計算機内(in silico)のGEM/dFBAシミュレーションで"
        "あり、湿式培養実験ではない。多くの実験で時間刻み・数値収束・解選択への依存性が"
        "確認されており、生物学的結論としては未検証の部分が多い。各実験の「限界・注意点」を"
        "必ず参照すること。",
        color=RED,
    )
    doc.add_paragraph(
        "フェーズ構成:"
    )
    add_bullets(
        doc,
        [
            "フェーズA — 初期検討: L. plantarum(WCFS1)を第3菌種としていた時期、および"
            "OR16+NS21の2種基礎検討(〜2026年9月2日)。",
            "フェーズB — 培養モデル監査・生理学的実装(2026年9月8日前後)。",
            "フェーズC — P. freudenreichii移行後の3種価値評価(2026年9月8日〜10日)。",
        ],
    )
    doc.add_paragraph(
        "本書はDeepSeek V4 Flash(OpenRouter経由)に22件の実験レポート原文を読ませて一次抽出し、"
        "主要な数値をClaudeが原文と突き合わせて検証した上で作成した。"
    )

    # ==================== PHASE A ====================
    add_heading(doc, "フェーズA: 初期検討(WCFS1時代・2種基礎検討)", level=1)
    add_note(
        doc,
        "この期間の第3菌種はL. plantarum(WCFS1)であった。後に計算研究の結果、"
        "明確なメリットが確認できず、P. freudenreichiiへ変更された"
        "(docs/PROJECT_MANAGEMENT.md Phase D参照)。",
    )

    experiment(
        doc,
        "A-1. 3種コンソーシアム(WCFS1含む)の共生成立性・事前監査",
        source="results/coexistence_audit_wcfs1_2022/coexistence_report.md",
        date="不明",
        purpose="L. plantarum WCFS1を含む3種コンソーシアムのin silico共生成立可能性を、"
                "共有培地max-min LPとdFBAで事前評価する。",
        method="静的LPで全菌構成の組合せの共通増殖率を計算。dFBA持続性試験、"
               "培地レスキュー試験(ile__L_e 0.01 mmol/L追加)も実施。",
        results_list=[
            "OR16+NS21+LPの3種共通増殖率: 0 1/h、静的成立判定: False",
            "OR16単独: 0.0792416 1/h(True)、NS21単独: 1.97415 1/h(True)、LP単独: 0 1/h(False)",
            "OR16+NS21: 0.107457 1/h(True)、OR16+LP・NS21+LP: いずれも0 1/h(False)",
            "培地レスキュー試験(ile__L_e追加)でも共通増殖率0 1/h、成立=False",
        ],
        figures=[("coexistence_audit_wcfs1_2022/coexistence_audit.png",
                  "図A-1: 共有培地max-min LPによる各菌構成の共通増殖率と成立判定")],
        limitations="静的LPは固定初期バイオマス比・共有培地供給上限を用いる。正の共通増殖は"
                    "成立可能性を示すのみで、侵入可能性・撹乱後回復・実験的安定性は別途検証が必要。",
        conclusion="試験条件ではWCFS1を含む3種共生は成立せず。",
        judgment="False(不成立)",
    )

    experiment(
        doc,
        "A-2. WCFS1修復後モデルによる流加液の詳細選定",
        source="results/feed_selection_wcfs1_fine/README.md",
        date="不明",
        purpose="WCFS1 2022修復モデルを含む3種コンソーシアムに対し、最少成分の流加液を"
                "決定論的FBA/dFBAで選定する。",
        method="合計438条件(一様濃度応答29、単一/二成分欠損55、全組合せ256、動的確認)を"
               "静的LPでスクリーニング。",
        results_list=[
            "最少候補: L-glutamate + L-isoleucine + pyridoxamine(3成分)",
            "24時間dFBAで3種すべての最終/初期バイオマス比: 約1.121",
            "供給量: Glu 5.00983e-4、Ile 2.15423e-4、pyridoxamine 1.00197e-5 mmol/L/h",
            "24h添加質量: 0.002488 g/L(Defined-10基準液3.154899 g/Lの約1,268分の1)",
        ],
        figures=[
            ("feed_selection_wcfs1_fine/Figure_feed_selection_static.png", "図A-2a: 一様濃度応答・成分欠損試験"),
            ("feed_selection_wcfs1_fine/Figure_feed_selection_decision.png", "図A-2b: 全組合せ探索の通過数と選定供給量"),
            ("feed_selection_wcfs1_fine/Figure_feed_selection_dynamic.png", "図A-2c: 動的スクリーニングと24時間確認"),
        ],
        limitations="モデルの理想的取り込みと完全混合を仮定した値。購入・培養手順に直接用いず、"
                    "濃度系列で実測する必要がある。",
        conclusion="モデル上の最少流加候補はGlu+Ile+pyridoxamineの3成分。",
    )

    experiment(
        doc,
        "A-3. 3種共存系の代謝フラックス・栄養補正監査",
        source="results/flux_nutrition_wcfs1_2022/flux_nutrition_audit_report.md",
        date="不明",
        purpose="ポリマー炭素の二重計上、Lcp/RoxのGPR、RoxA経路トポロジーを修正し、"
                "培地候補比較に使えるモデル状態にする。",
        method="生SBMLの交換反応ゼロフラックスチェック、ポリマー分解モデルの元素収支/GPR監査、"
               "可溶性培地の組合せスクリーニング。",
        results_list=[
            "zero-feasible基準条件の共通成長率: 0.00000 h⁻¹",
            "Defined-10流加: 0.14483 h⁻¹",
            "コハク酸+グルタミン酸+マンニトールは現モデルのmax-min成長率を改善せず",
        ],
        figures=[
            ("flux_nutrition_wcfs1_2022/Figure_flux_nutrition_audit.png", "図A-3a: 培地候補の組合せ比較"),
            ("flux_nutrition_wcfs1_2022/Figure_polymer_model_audit.png", "図A-3b: ポリマー分解モデルの監査"),
        ],
        limitations="ゴムからPHAへの絶対収率には未校正の細胞外酵素速度・仮説的輸送が残る。"
                    "酵母エキス近似はg/Lに校正されていない。",
        conclusion="モデル修正後、培地候補の比較は可能だが絶対収率には未校正パラメータが残る。",
    )

    experiment(
        doc,
        "A-4. WCFS1 3種コンソーシアムの共生戦略dFBA",
        source="results/wcfs1_coexistence_strategy/wcfs1_coexistence_strategy.json",
        date="不明",
        purpose="無給餌協調・Defined-10分離/協調・Complete-16分離の4シナリオで、WCFS1を"
                "含む3種コンソーシアムの24時間dFBA持続性を評価する。",
        method="24時間、dt=0.5h、pH setpoint 6.5。初期バイオマスOR16 0.5/NS21 0.1/LP 0.1 g/L。",
        results_list=[
            "No feed+cooperative: LP正増殖率割合0.0、active_coexistence=false",
            "Defined-10+separate: LP正増殖率割合0.0、active_coexistence=false",
            "Defined-10+cooperative: LP正増殖率割合1.0、active_coexistence=true、最小終点/初期比1.121",
            "Complete-16(4x)+separate: active_coexistence=true、最小終点/初期比3.273",
        ],
        figures=[("wcfs1_coexistence_strategy/wcfs1_coexistence_strategy.png",
                  "図A-4: 4シナリオでの3種バイオマス推移と共存判定")],
        limitations="予測された交叉摂食は協調LP設計仮説であり、実測された分泌速度ではない。",
        conclusion="Defined-10協調モードでのみWCFS1の正の増殖が維持され、活性共存が成立した。",
    )

    experiment(
        doc,
        "A-5. 第3菌(WCFS1)の代謝ニッチ検証(総PHB目的)",
        source="results/third_species_niche_20260902/third_species_niche_report.md",
        date="不明",
        purpose="L. plantarum WCFS1を第3菌としてOR16+NS21二種系に追加した際の代謝ニッチと"
                "総PHB生産性への影響を評価する。",
        method="公開GEMを用いた嫌気条件dFBA。乳酸取り込み・プロピオン酸分泌・PHA出力を評価。",
        results_list=[
            "最良条件 three_lactate_b0.01_r0.50: PHAモデル出力0.08047 g/L-as-PHB",
            "同一流加速度の直接流加(two_lactate_0.50)比: 0.958",
            "無乳酸の第3菌添加は二種ゴム対照の0.984倍",
        ],
        figures=[("third_species_niche_20260902/Figure_third_species_metabolic_niche.png",
                  "図A-5: 第3菌の代謝ニッチ(乳酸取込・プロピオン酸分泌経路)")],
        limitations="主解析は逐次dFBA。P. freudenreichiiモデルは公開pan-model由来。",
        conclusion="総PHB目的では第3菌追加の利益は検出されず、不採用。",
        judgment="no_go_for_total_phb",
    )

    experiment(
        doc,
        "A-6. 第3菌(P. freudenreichii)の代謝ニッチ再評価(PHBV組成目的)",
        source="results/third_species_phbv_reassessment_20260903/third_species_niche_report.md",
        date="不明",
        purpose="第3菌としてP. freudenreichiiを評価し、総PHA質量と3HV組成を併用した判定を行う。",
        method="公開GEMを用いた嫌気条件dFBA。3HV mol%を含めて評価。",
        results_list=[
            "最良条件 three_lactate_b0.01_r0.50: PHAモデル出力0.11655 g/L、3HV 72.07 mol%",
            "同一流加速度の直接流加比: 1.001",
            "無乳酸の第3菌添加は二種ゴム対照の0.984倍",
        ],
        figures=[("third_species_phbv_reassessment_20260903/Figure_third_species_metabolic_niche.png",
                  "図A-6: PHBV組成(3HV比)を含めた代謝ニッチ再評価")],
        limitations="主解析は逐次dFBA。OR16/NS21の既計算では乳酸分泌がほぼ0。",
        conclusion="総PHA量では直接流加以下だが、PHBV組成操作の候補として実験的裏付けを待つ"
                   "(この知見がPf移行の初期材料となった)。",
        judgment="no_go_without_experimental_support",
    )

    experiment(
        doc,
        "A-7. OR16+NS21二種系の培養・律速トレース",
        source="results/or16_ns21_limitation_trace_20260902/OR16_NS21_limitation_trace_report.md",
        date="2026-09-02",
        purpose="OR16とNS21のみの二種系で、追加流加なし・pH-stat有効のin silico培養を行い、"
                "律速段階を診断する。",
        method="24時間dFBA。ゴム分解段階とPHA段階の律速を判定。",
        results_list=[
            "終点(24h): ゴム 10.0000→8.8362 g/L、PHAモデルプール 0→0.011026 mmol/L(極めて小さい)",
            "OR16菌体 0.5→0.55713 g/L、NS21菌体 0.1→0.11142 g/L",
            "ゴム分解段階: allocated_oxygen_limited、PHA段階: nitrogen_replete_growth_phase",
            "最大の単独介入: O2補充(共通増殖容量30.1%増)",
        ],
        figures=[("or16_ns21_limitation_trace_20260902/Figure_OR16_NS21_limitation_trace.png",
                  "図A-7: OR16+NS21二種系の律速段階トレース")],
        limitations="ゴム分解酵素速度・酸素配分・NH4によるPHA切替閾値は実測校正前。",
        conclusion="二種系では酸素配分が律速であり、PHA蓄積は極めて少ない。",
    )

    experiment(
        doc,
        "A-8. 非生産側ヘルパー候補(B. subtilis 168)の選定と流加dFBA",
        source="results/nonproducer_helper_screen_20260902/nonproducer_helper_report.md",
        date="2026-09-02",
        purpose="PHA非生産菌のヘルパー候補を選定し、OR16+NS21二種系への添加効果をdFBAで評価する。",
        method="複数候補をスコアリングし最良候補B. subtilis 168(GEM: iYO844)を選定、"
               "スターチ流加条件で三種dFBAを実施。",
        results_list=[
            "候補比較首位: Bacillus subtilis 168(スコア19)",
            "最良三種条件 three_starch_b0.01_r0.10: PHAモデル出力0.12306 g/L-as-PHB",
            "ゴムのみ二種対照比: +653.1%だが、同一速度グルコース二種対照比: -25.7%",
            "ゴム除去量のゴムのみ対照比: -1.0%",
        ],
        figures=[("nonproducer_helper_screen_20260902/Figure_nonproducer_helper_screen.png",
                  "図A-8: ヘルパー候補スコアリングとPHA生産比較")],
        limitations="スターチ由来炭素がPHAへ入るため、「ゴム由来PHA」と主張するには13C追跡が必要。",
        conclusion="生菌ヘルパーとしての採用は見送り(炭素源を直接与える方が優れる)。",
        judgment="no_go_as_live_third_strain",
    )

    experiment(
        doc,
        "A-9. 第三菌候補(B. subtilis)の窒素・酸素移動感度",
        source="results/nonproducer_helper_sensitivity_20260902/sensitivity_report.md",
        date="2026-09-02",
        purpose="B. subtilisを第三菌とした際の、初期NH4量とkLaに対するPHA生産性の感度を評価する。",
        method="12時間、流加0.10 mmol/L/h、第三菌初期量0.01 g/L。初期NH4量・kLaの軸上評価。",
        results_list=[
            "第三菌/最良直接流加のPHA比: 0.000〜0.747",
            "直接流加を5%以上上回った条件: 0/5",
        ],
        figures=[("nonproducer_helper_sensitivity_20260902/Figure_nonproducer_helper_sensitivity.png",
                  "図A-9: NH4・kLa軸上でのPHA生産感度")],
        limitations="実測速度論による校正はない。",
        conclusion="全条件で直接流加を上回らず、生菌ヘルパー採用は見送り。",
        judgment="no_go_as_live_third_strain",
    )

    experiment(
        doc,
        "A-10. L. plantarum WCFS1モデル比較(iBT721 vs 2022 deposit vs 2022 repaired)",
        source="results/lplantarum_model_replacement/model_comparison.json",
        date="不明",
        purpose="3種コンソーシアムで使用するWCFS1の3つのGEMバージョンを比較し、"
                "修復モデルの増殖要件を確認する。",
        method="各モデルの反応数・代謝物数・遺伝子数、現培地/全開放/目標増殖速度、"
               "必要追加供給を比較。",
        results_list=[
            "iBT721: 反応778、現培地成長率0.0、全開放0.2970。必要追加10成分(val, malt, ile等)",
            "2022 deposit: 反応1208、全開放でも成長率0.0(要修復)",
            "2022 repaired: 反応1209、全開放10.0。必要追加はval/ile/metの3成分のみ",
        ],
        figures=[("lplantarum_model_replacement/lplantarum_model_replacement_audit.png",
                  "図A-10: WCFS1モデル3版の比較監査")],
        conclusion="修復モデルは全開放で増殖可能だが、現培地ではval・ile・metの微量追加が必要。",
    )

    # ==================== PHASE B ====================
    add_heading(doc, "フェーズB: 培養モデル監査・生理学的実装(2026年9月8日前後)", level=1)
    add_note(
        doc,
        "この期間に第3菌種はP. freudenreichii(Pf)へ切り替わっている。培養モデル自体の"
        "実装不整合(酸素配分、PHA上限処理等)を修正するフェーズであり、3種優位性の"
        "有無そのものはまだ確定していない。",
    )

    experiment(
        doc,
        "B-1. 旧酸素予約方式による3種共培養監査再評価",
        source="results/audited_symbiosis_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="数値不整合を修正した監査用モデルaudited_cultivation_v2で、基本21ケースの"
                "3種共培養(OR16/NS21/Pf)を評価する。",
        method="酸素計算に旧方式(quota_mean_v2)を使用。12時間、初期ゴム10g/L、kLa 50h⁻¹。"
               "3種(0.5/0.1/0.03 g/L)と総接種量一致の2種対照(0.525/0.105 g/L)を比較。",
        results_list=[
            "基準刻みで探索基準(PHA5%超増・ゴム除去悪化5%以内)を満たす条件: 0/5",
            "基準条件: 2種PHA 0.515436 g/L、3種PHA 0.501777 g/L(-2.65%)",
            "低kLa(細刻み0.0125h): 2種PHA 0.091248 g/L、3種PHA 0.094759 g/L(+3.85%)",
            "全162,574ソルバー呼出し中、不適合拒否7,349回",
        ],
        figures=[("audited_symbiosis_20260908/comparison.png", "図B-1: 旧酸素予約方式での2種/3種PHA比較")],
        limitations="定常DOに大きな時間刻み依存が後に判明。この結果を新方式の結論として流用しない。",
        conclusion="旧酸素予約方式では、調べた5条件全てで3種の優位性は確認されなかった。",
    )

    experiment(
        doc,
        "B-2. 共有酸素方式による3種共培養監査再評価",
        source="results/audited_symbiosis_shared_oxygen_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="酸素計算をshared_endpoint_v3に変更した監査用モデルで、23ケースを再評価する。",
        method="旧方式の酸素予約機構を廃止。全菌が同じDOを参照し、酸素消費の総和から"
               "DOを連立して解く。他条件はB-1と同一。",
        results_list=[
            "基準条件: 2種PHA 0.515515 g/L、3種PHA 0.501749 g/L(-2.67%)",
            "低kLa: 2種PHA 0.096416 g/L、3種PHA 0.100172 g/L(+3.89%)",
            "細刻み(0.00625h)低kLa: 3種PHA差 +3.71%",
            "全217,784ソルバー呼出し中、不適合拒否7,232回",
        ],
        figures=[
            ("audited_symbiosis_shared_oxygen_20260908/comparison.png", "図B-2a: 共有酸素方式での2種/3種PHA比較"),
            ("audited_symbiosis_shared_oxygen_20260908/oxygen_convergence.png", "図B-2b: 酸素収束の時間刻み依存性"),
        ],
        limitations="1段階の半減で一致しても数学的収束の証明ではない。",
        conclusion="共有酸素方式でも、調べた5条件全てで3種の優位性は確認されなかった。",
    )

    experiment(
        doc,
        "B-3. 3種共生の根拠再評価(旧計算フレームワーク)",
        source="results/symbiosis_reassessment_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="旧計算フレームワークで5条件の3種共培養を2種対照と比較し、事前基準を満たす"
                "条件を探索する。",
        method="旧dFBA実装。12時間、ゴム10g/L、pH7固定。3種(0.5/0.1/0.03g/L)、"
               "2種対照(0.525/0.105g/L)。",
        results_list=[
            "PHAが5%超改善しゴム除去悪化5%以内の条件: 1/5(低酸素移動係数oxygen_low)",
            "低kLa: 2種PHA 0.464098 g/L、3種PHA 0.504520 g/L(+8.71%)",
            "低kLa細刻み(0.0125h): 差+3.16%に縮小",
            "低kLaで3種はNS21単独(1.319957 g/L)に対しPHA -61.78%",
        ],
        figures=[("symbiosis_reassessment_20260908/comparison.png", "図B-3: 低kLa条件での3種優位の細刻み再検証")],
        limitations="決定論的モデル探索であり統計的有意差ではない。低kLaの陽性は探索的解析で"
                    "独立確認試験ではない。",
        conclusion="低kLa条件下でのみ3種がPHA増産で2種対照を上回る可能性を示したが、"
                   "NS21単独には大きく劣る。",
    )

    experiment(
        doc,
        "B-4. 生理モデルの長時間検証・PHA蓄積期",
        source="results/physiology_long_validation_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="修正された生理モデルで、リッチ条件・窒素枯渇条件での長時間(24h/12h)PHA蓄積を"
                "検証し、数値安定性と時間刻み収束性を確認する。",
        method="11ケース(リッチ3/2種、細刻み、高維持代謝、高死滅、供給停止等)。判定基準: "
               "終点・ピークPHA差2%以内、菌種別生菌体差2%以内、3HV絶対差0.1ポイント以内。",
        results_list=[
            "長時間ケース: 11/11完了(全4主比較で「合格」)",
            "リッチ3種(24h): 生菌中PHAピーク0.704072 g/L、終点0.693589 g/L",
            "リッチ2種(24h): 終点PHA 0.707884 g/L",
            "リッチ条件2種vs3種: 3種が-2.019%",
            "N枯渇3種(12h): 生菌中PHAピーク0.356546 g/L",
        ],
        figures=[
            ("physiology_long_validation_20260908/trajectories.png", "図B-4a: PHA蓄積の時系列軌道"),
            ("physiology_long_validation_20260908/sensitivity.png", "図B-4b: 時間刻み・パラメータ感度"),
        ],
        limitations="判定は数学的収束の証明や任意条件での精度保証ではない。PHA貯蔵上限は"
                    "実測校正値ではない。",
        conclusion="今回の設計範囲内で生理モデルの長時間数値検証は合格基準を満たした。",
    )

    experiment(
        doc,
        "B-5. 酸素とNH4による時間刻み依存の改善",
        source="results/resolved_dynamics_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="旧dFBA実装の酸素・NH4モデルの時間刻み依存を改善する新CPU参照経路"
                "(ResolvedDFBASimulator)を実装・検証する。",
        method="制御間隔と内部刻みを分離。酸素は解析解で更新。NS21の増殖/PHA配分に"
               "Michaelis-Menten式(K_N=0.1 mmol/L)を導入。",
        results_list=[
            "3種(内部刻み0.025h): PHA 0.44407716 g/L → (0.0125h): PHA 0.44974787 g/L",
            "内部刻み3→1.5分: PHA+2.604%、1.5→0.75分: PHA+1.277%(収束傾向)",
            "制御間隔1→0.2h(内部刻み固定): PHA差 -4.846e-14(実質ゼロ)",
            "検証テスト9件合格(解析解、酸素収支等)",
        ],
        figures=[("resolved_dynamics_20260908/validation.png", "図B-5: 新旧経路の時間刻み収束比較")],
        limitations="NS21の配分則は生物学的作業仮説でK_N=0.1 mmol/Lも未校正。新経路はGPU"
                    "サロゲート等に未接続。",
        conclusion="新経路は酸素・NH4の時間刻み依存を改善し、基本検証を通過した。",
    )

    experiment(
        doc,
        "B-6. 1Lジャー仮想スモークテスト",
        source="results/one_l_jar_smoke/smoke_results.json, docs/ONE_L_JAR_SMOKE_REPORT.md",
        date="不明",
        purpose="4系統送液・pH-stat・サンプリング・液量変化・kLa・3種GEM逐次FBAを結合した"
                "6時間シミュレーションで、ソフトウェア経路の動作と数値安定性を確認する。",
        method="4制御方策×4仮想パラメータ=16条件+再現性確認。HiGHS dual simplex、"
               "6h、刻み0.5h。",
        results_list=[
            "合格: 17/17実行、pytest全79件合格",
            "FBAソルバー成功率: 全条件1.000。負濃度0件、制約違反0件",
            "液量収支最大絶対誤差: 5.55×10⁻¹³ mL",
            "最終PHA範囲: 0.0000〜0.01712 mmol/L",
        ],
        figures=[("one_l_jar_smoke/Figure_one_l_jar_smoke.png", "図B-6: 1Lジャー仮想スモークテストの推移")],
        limitations="初期値・kLa較正式・ポンプ原液濃度は仮想値を含む。現時点の数値を実験"
                    "予測値や購入根拠として扱ってはならない。",
        conclusion="ソフトウェア経路は最低限の動作確認を通過し、次段階へ進行可能となった。",
    )

    add_note(
        doc,
        "フェーズBには上記6実験に加え、外部研究(Rhodococcus sp. RDE2組換株、"
        "C/N比とPHA含有率の関係)を定性的参照として用いた検討が1件あるが"
        "(results/timestep_reference_20260908)、第三者由来の画像のため図は本書に"
        "転載していない。",
    )

    # ==================== PHASE C ====================
    add_heading(doc, "フェーズC: P. freudenreichii移行後の3種価値評価(2026年9月8日〜10日)", level=1)
    add_note(
        doc,
        "この一連の研究が、docs/THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docxで詳述した"
        "「3種優位性は酸素条件に対して非単調」という結論に至る過程である。本章は同文書と"
        "一部重複するが、各実験の図をすべて掲載する点で異なる。",
    )

    experiment(
        doc,
        "C-1. P. freudenreichii追加価値の再評価監査",
        source="results/pfreud_value_audit_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="Pfの乳酸→プロピオン酸変換が3種コンソーシアムの総PHA量に与える価値を、"
                "過去の陽性候補を再現しつつ検証する。",
        method="12時間、ゴム10g/L、kLa=50h⁻¹。中心条件でdt=1/0.2/0.1hを比較。",
        results_list=[
            "dt=1h: 3種PHA 0.030650 g/L、生産菌量固定対照比+20.19%",
            "dt=0.2h: 3種PHA 0.140931 g/L、対照比-27.12%",
            "dt=0.1h: 3種PHA 0.631859 g/L、対照比-4.53%(符号が刻みで反転)",
            "周辺条件4件は全て暫定基準(5%超増)を満たさず",
        ],
        figures=[("pfreud_value_audit_20260908/pfreud_value_audit.png", "図C-1: 時間刻みによる3種優位性の反転")],
        limitations="1時間刻みの約20%増は再現したが、細刻みで符号が変わり3種の優位性として"
                    "採用できない。",
        conclusion="1時間刻み以外では3種の優位性は確認できず、刻み幅依存性が主課題。",
    )

    experiment(
        doc,
        "C-2. 1時間刻み陽性原因の分解検証",
        source="results/pfreud_causal_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="1時間刻みで観測された3種PHA増加(約20%)の原因を、窒素スイッチのタイミングと"
                "プロピオン酸交換に焦点を当てて検証する。",
        method="乳酸供給continuous/early/lateの3種。刻みを1〜0.00625hまで変化。",
        results_list=[
            "1時間刻み陽性の主因: NH4<0.1mmol/Lによる生産モード切替時間の差"
            "(反実仮想分解でスケジュール成分が90.40%)",
            "0.05h連続供給の陽性(+42.60%)はPPA除去介入で消失 → PPA経由の動態が必要",
            "最も細かい22.5秒刻み(0.00625h): 3種PHA 0.458632 g/L、生産菌一致2種比-2.62%",
        ],
        figures=[("pfreud_causal_20260908/causal_audit.png", "図C-2: 陽性原因のスケジュール/構成成分分解")],
        limitations="局所的な機序診断であり、全条件探索・RL最適化・実測検証は行っていない。",
        conclusion="NH4閾値とPPA経由の動態が1時間刻み陽性の主因だが、細刻みでは優位性は"
                   "維持されない。",
    )

    experiment(
        doc,
        "C-3. 同一資源予算における2種・3種供給探索",
        source="results/equal_budget_comparison_20260908/REPORT_JA.md",
        date="2026-09-08",
        purpose="同一資源(乳酸6mmol/L、NH4 0.4mmol/L、24時間)の下で2種・3種それぞれに"
                "最適な供給条件を探索し、公平に比較する。",
        method="kLa=2,10,50の各条件で供給パターン全6候補を探索。上位候補を細刻み"
               "(0.0125h)で再評価。全54ケース完了。",
        results_list=[
            "kLa=2: 2種0.048818 g/L、3種0.023622 g/L(-51.612%)",
            "kLa=10: 2種0.815198 g/L、3種0.800928 g/L(-1.750%)",
            "kLa=50: 2種0.770125 g/L、3種0.759744 g/L(-1.348%)",
            "5%超増産・ゴム除去悪化5%以内を満たす細刻み候補: なし",
        ],
        figures=[("equal_budget_comparison_20260908/best.png", "図C-3: 同一資源予算下での2種・3種最良供給比較")],
        limitations="有限候補の探索であり全球最適解ではない。維持代謝・死滅率・Pf酸素表現型は"
                    "未校正。",
        conclusion="この有限な供給探索範囲では、3種が2種を明確に上回る条件は確認されなかった。",
    )

    experiment(
        doc,
        "C-4. プロピオン酸直接添加との比較評価",
        source="results/direct_propionate_comparison_20260909/REPORT_JA.md, "
               "direct_propionate_refinement_20260909/FINAL_REPORT_JA.md",
        date="2026-09-09",
        purpose="Pfによるプロピオン酸供給の価値を評価するため、同じ炭素量・NH4条件で"
                "乳酸+プロピオン酸を直接与えた2種系と3種系のPHA生産を比較する。",
        method="全72ケース+最終監査36ケース(0.00625h)。構成×供給パターン×kLa×刻みの"
               "組合せ。指標はpha_live_g_l、phv_live_g_l。",
        results_list=[
            "kLa=10: 3種0.800928 g/L(3HV 0.080mol%)、2種プロピオン酸のみ0.909441 g/L"
            "(3HV 4.629%)、2種混合0.874454 g/L",
            "kLa=2: 3種0.023622 g/L(3HV 61.978%)、2種プロピオン酸のみ0.003205 g/L"
            "(3HV 100.000%)",
            "最終監査(収束合格16/36件): kLa=10で3種0.801197 g/L、2種プロピオン酸のみ"
            "0.909513 g/L",
        ],
        figures=[("direct_propionate_comparison_20260909/comparison.png",
                  "図C-4: 3種 vs プロピオン酸直接添加2種のPHA・3HV比較")],
        limitations="投入炭素は同じだが還元当量・費用は同じではない。Pf酸素表現型・PHV生成"
                    "制御は未校正。",
        conclusion="3種のPHA総量は直接プロピオン酸添加の2種を上回らない。3HV組成が高い"
                   "だけでは優位性とはならない。",
    )

    experiment(
        doc,
        "C-5. 第3菌価値評価の統合サマリ(酸素スケジュール軸)",
        source="results/third_species_value_20260910/summary.json",
        date="2026-09-10",
        purpose="酸素条件と供給条件を変えた一連の比較実験を統合し、Pfの有無による"
                "PHA生産への影響を整理する。",
        method="equal_budget_comparison、oxygen_pulse_validation、oxygen_schedule_extension"
               "から収束した10条件を統合(pha_relative=0.02, biomass_relative=0.02, "
               "hv_fraction_absolute=0.001で判定)。",
        results_list=[
            "PHA増加条件: const_k4, const_k6, const_k8, pulse_2_10, pulse_10_2, "
            "pulse4_2_10, pulse4_10_2",
            "PHA減少条件: const_k2, const_k50",
            "最大PHA増加率: pulse4_2_10条件で+25.35%(3種0.434311 g/L vs 2種0.346469 g/L)",
            "最大3HV増加: pulse_2_10条件で+7.6852ポイント",
        ],
        figures=[
            ("third_species_value_20260910/fig1_third_species_axis.png", "図C-5a: 酸素条件別のPHA差・3HV変化"),
            ("third_species_value_20260910/fig2_equal_budget_family.png", "図C-5b: 等曝気予算下での5候補比較"),
            ("third_species_value_20260910/fig3_representative_trajectories.png", "図C-5c: 代表軌道(PHA/3HV/DO/Pf生菌)"),
        ],
        limitations="全条件が決定論的1軌道で反復・統計検定なし。kLa・維持代謝・死滅率・Pf"
                    "表現型は未校正。",
        conclusion="特定の酸素パルス条件で3種が2種より最大25%のPHA増加を示すが、モデル"
                   "未校正点が多く生物学的検証は未実施(現時点の最新知見)。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
