"""Report only the completed final physiology cohort and preserve its sources."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/physiology_implementation_20260908'
summary=json.loads((OUT/'verification_summary.json').read_text())
assert len(summary['cases'])==8
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in summary['source_sha256'].items())
log=(OUT/'tests.log').read_text();passed=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',log)
assert passed and ' failed' not in log
cases={r['id']:json.loads((OUT/(r['id']+'.json')).read_text()) for r in summary['cases']}
assert all(r['nitrogen_allocation']['Rhizobacter_gummiphilus_NS21']['relief_pools']==['nh4_e'] for r in cases.values())
rows=[]
for cid,r in cases.items():
    allocation=r['nitrogen_allocation']['Rhizobacter_gummiphilus_NS21']
    pha_text='<1e-9' if abs(r['pha_g_l'])<1e-9 else format(r['pha_g_l'],'.9f')
    rows.append(f"|{cid}|{r['internal_dt']}|{pha_text}|{allocation['growth_fraction']:.6f}|{r['seconds']:.2f}|")
refinement=[];previous=None
for cid in ['three_low_maintenance','three_fine','three_finer','three_finest']:
    r=cases[cid];pha=r['pha_g_l'];x=sum(r['final_biomass'].values())
    delta=None if previous is None else pha-previous['pha']
    relative=None if previous is None or previous['pha']<1e-9 else 100*delta/previous['pha']
    refinement.append(dict(id=cid,internal_dt=r['internal_dt'],pha_g_l=pha,pha_absolute_change_g_l=delta,
        pha_relative_change_percent=relative,biomass_g_l=x,
        biomass_relative_change_percent=None if previous is None else 100*(x/previous['x']-1)))
    previous=dict(pha=pha,x=x)
max_residual=max(abs(v) for r in cases.values() for v in r['diagnostics']['physiology']['elements']['known_medium_closure_residual'].values())
ftable='\n'.join(f"|{r['internal_dt']}|{r['pha_g_l']:.3e}|{format(r['pha_relative_change_percent'],'.2f')+'%' if r['pha_relative_change_percent'] is not None else '算定せず'}|{r['biomass_g_l']:.9f}|{format(r['biomass_relative_change_percent'],'.4f')+'%' if r['biomass_relative_change_percent'] is not None else '—'}|" for r in refinement)
(OUT/'REPORT_JA.md').write_text(f'''# 生理モデル実装の検証結果（2026-09-08）

**実装テスト{passed[1]}合格、0失敗、{passed[2]}警告（{passed[3]}秒）。実GEMの8ケースが完了。**
警告は既存Pfテストのmultiprocessing forkに関するDeprecationWarning。
実装と前提は [実装説明](../../docs/PHYSIOLOGY_IMPLEMENTATION_20260908.md) を参照。

同定済みアンモニウムだけを緩和する最終方式で全8ケースを計算し、実行前後のソース一致を確認した。
最初の分子式自動選択方式の6ケースは `initial_probe/` に保管し、混在させない。

## 実GEMでの短時間検査

全ケース0.1時間、初期ゴム10 g/L、kLa 10 h⁻¹、乳酸0.25 mmol/L/h、pH 7。
3種はOR16/NS21/Pfを0.5/0.1/0.03 g/L、2種は0.525/0.105 g/L。
基礎死滅率は全て0.01 h⁻¹。維持ATPなし/0.1/1 mmol/g/hの設定感度を確認。
維持ATPを指定した場合の最大飢餓死滅率は0.1 h⁻¹。全て未校正の検証用仮定。
NH4初期濃度は0.05 mmol/L、three_no_nh4のみ0。その他の背景培地は一致。

|ケース|内部刻み h|生菌中PHA g/L|終点の窒素配分係数|実行秒|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

既知分子式部分の培地C/N台帳残差は最大{max_residual:.3g} mmol/L。
この閉鎖は台帳の整合性を示すもので、分子式の正しさや全GEMの元素保存の証明ではない。
未分類成分、菌種別取込・分泌、死菌区画、維持ATPの要求・実現・不足は各JSONに保存した。
窒素配分係数は仮想NH4補充条件とのGEM増殖能力比であり、測定された窒素充足率ではない。

## 時間刻みの感度

維持ATP0.1の3種について、同じ0.1時間で4刻みを比較した。

|内部刻み h|PHA g/L（微小値も表示）|PHA相対差|総生菌体 g/L|生菌体の前段階からの相対差|
|---|---:|---:|---:|---:|
{ftable}

1e-9 g/L未満のPHAは報告上のゼロ近傍として相対変化率を算出しない。この閾値は測定装置の検出限界ではない。
アンモニウムの仮想補充で増殖が改善しない条件では、現配分則は増殖を優先する。PHAがほぼ0の比較から、有意な蓄積期のPHA予測まで収束したとは主張できない。
今回の短時間確認だけで長期の誤差を上限保証できない。初期過渡、NH4変化、貯蔵・PHVゲートを含む長時間の刻み比較が教師の本収集前に必要。
旧数値参照の12時間比較で得た刻み感度を、この生理方式の収束根拠へ流用しない。

## B12関連の変更

NS21のMETSから、FASTAでpseudo=trueのA4W93_27845を単純OR規則の根拠として除外した。
A4W93_24875は残した。反応式とフラックス境界は変更していない。
OR16のmetH候補とcobalamin-binding候補、NS21の残る酵素候補の完全な機能は未検証。
B12外部依存やPf必須性を仮定として加えていない。

## 使用上の範囲

`--dynamics physiology --physiology-config ...` で新実装を使用できる。
`example_uncalibrated_physiology.json` は実際にこの確認で使用した設定であり、校正済みの培養条件ではない。
新生理設定は方策契約へ保存される。旧512軌道の収集やRL学習は実行していない。
3種優位・安定共生・実測一致・PHAのゴム由来炭素割合は本検査では評価していない。
''',encoding='utf-8')
(OUT/'refinement.json').write_text(json.dumps(refinement,indent=2))
files=list(summary['source_sha256'])+['tests/test_physiology_dfba.py','docs/PHYSIOLOGY_IMPLEMENTATION_20260908.md',
    'scripts/analysis/audit_physiology_evidence_20260908.py','scripts/analysis/report_physiology_implementation_20260908.py',
    'models/genome/AP019371.1.faa','models/genome/NS21.faa',
    'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',
    'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml',
    'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml']
snapshots=OUT/'sources';snapshots.mkdir(exist_ok=True)
manifest={}
for p in files:
    source=ROOT/p;manifest[p]=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(source,snapshots/p.replace('/','__'))
(OUT/'completion.json').write_text(json.dumps(dict(tests_passed=int(passed[1]),warnings=int(passed[2]),
    real_gem_cases=8,source_and_input_sha256=manifest,max_known_element_ledger_residual=max_residual,
    long_horizon_convergence_validated=False,biology_calibrated=False),indent=2))
print(json.dumps(dict(tests=int(passed[1]),cases=8,refinement=refinement),indent=2))
