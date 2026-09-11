"""Aggregate all preregistered cases, without selecting only favorable outputs."""
from pathlib import Path
import sys,json,csv,math,hashlib
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis.reassess_symbiosis_20260908 import cases
OUT=ROOT/'results/symbiosis_reassessment_20260908'
PF='Propionibacterium_freudenreichii_shermanii';NS='Rhizobacter_gummiphilus_NS21';OR='Actinoplanes_sp_OR16_lcp'
conditions=['base','no_lactate','lactate_rich','nitrogen_staged','oxygen_low']
labels=['Baseline','No lactate','4x lactate','Early NH4 feed','Low kLa']
jlabels=['基準条件','乳酸なし','乳酸4倍','初期4時間NH4供給','低酸素移動係数']
def read(path):return json.loads(path.read_text())
def hv_percent(d):
    f=d['final'];hb=max(0.,(f['pha']-f['phv']*.10012)/.08609)
    return 100*f['phv']/(hb+f['phv']) if hb+f['phv']>1e-12 else 0.
data={c['id']:read(OUT/c['id']/'result.json') for c in cases()}
checks={}
for key,d in data.items():
    rows=list(csv.DictReader((OUT/key/'trajectory.csv').open()))
    finite=all(math.isfinite(float(v)) and float(v)>=-1e-7 for r in rows for v in r.values())
    diag=d['diagnostics'];rd=diag['resolved_dynamics'];feed=rd['continuous_feed'];o=rd['oxygen']
    checks[key]=dict(finite_nonnegative=finite,final_time=abs(d['final']['time']-12)<1e-9,
        lactate_budget=abs(feed.get('lac__L_e',0)-12*d['settings']['lactate_rate'])<1e-9,
        extra_nh4_budget=abs(feed.get('nh4_e',0)-(.4 if d['settings']['case']['condition']=='nitrogen_staged' else 0))<1e-9,
        oxygen_balance=abs(d['final']['o2']-.25-o['transferred']+o['polymer_consumed']+o['cellular_consumed'])<1e-8,
        lp_success=diag['solve_success_rate']==1.)
design=read(OUT/'design.json')
checks['source_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in design['sources'].items()}
assert all(all(v.values()) for v in checks.values()),checks
(OUT/'integrity.json').write_text(json.dumps(checks,indent=2))
paired=[]
for c in conditions:
    two=data[c+'_two_equal_total'];three=data[c+'_three']
    a=two['final'];b=three['final']
    gain=100*(b['pha']/a['pha']-1);rubber=100*((10-b['rubber'])/(10-a['rubber'])-1)
    paired.append(dict(condition=c,two_pha_g_l=a['pha'],three_pha_g_l=b['pha'],pha_gain_percent=gain,
        two_rubber_removed_g_l=10-a['rubber'],three_rubber_removed_g_l=10-b['rubber'],rubber_gain_percent=rubber,
        two_3hv_mol_percent=hv_percent(two),three_3hv_mol_percent=hv_percent(three),
        three_pf_fold=three['fold_growth'][PF],screen_positive=gain>5 and rubber>=-5))
with (OUT/'paired_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)
exchange_rows=[]
for key,d in data.items():
    for species,flux in d['exchanges'].items():
        for met,value in flux.items():exchange_rows.append(dict(case=key,species=species,exchange=met,integrated_mmol_l=value))
with (OUT/'integrated_exchanges.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=exchange_rows[0]);w.writeheader();w.writerows(exchange_rows)
base=data['base_three'];pf=base['exchanges'][PF];ns=base['exchanges'][NS]
carbon=3*pf.get('ppa_e_out',0)+2*pf.get('ac_e_out',0)
upper_pha=carbon/4*.08609
fine=[]
for folder in sorted(OUT.glob('refinement_*')):
    if not all((folder/a/'result.json').exists() for a in ['two_equal_total','three']):raise RuntimeError('Unfinished refinement '+str(folder))
    a=read(folder/'two_equal_total/result.json');b=read(folder/'three/result.json')
    for arm,d in [('two_equal_total',a),('three',b)]:
        rows=list(csv.DictReader((folder/arm/'trajectory.csv').open()))
        rd=d['diagnostics']['resolved_dynamics'];o=rd['oxygen'];feed=rd['continuous_feed']
        checks[folder.name+'/'+arm]=dict(
            finite_nonnegative=all(math.isfinite(float(v)) and float(v)>=-1e-7 for r in rows for v in r.values()),
            final_time=abs(d['final']['time']-12)<1e-8,
            lp_success=d['diagnostics']['solve_success_rate']==1,
            lactate_budget=abs(feed.get('lac__L_e',0)-12*d['settings']['lactate_rate'])<1e-9,
            extra_nh4_budget=abs(feed.get('nh4_e',0)-(.4 if d['settings']['case']['condition']=='nitrogen_staged' else 0))<1e-9,
            oxygen_balance=abs(d['final']['o2']-.25-o['transferred']+o['polymer_consumed']+o['cellular_consumed'])<1e-8)
    checks[folder.name+'/source_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in read(folder/'design.json')['sources'].items()}
    fine.append(dict(condition=folder.name.replace('refinement_',''),two=a['final']['pha'],three=b['final']['pha'],gain=100*(b['final']['pha']/a['final']['pha']-1),
        rubber_gain=100*((10-b['final']['rubber'])/(10-a['final']['rubber'])-1),two_3hv=hv_percent(a),three_3hv=hv_percent(b)))
controls={arm:read(OUT/'low_oxygen_controls'/arm/'result.json') for arm in ['two_fixed','ns21_alone','inert_pf']}
for arm,d in controls.items():
    rows=list(csv.DictReader((OUT/'low_oxygen_controls'/arm/'trajectory.csv').open()))
    rd=d['diagnostics']['resolved_dynamics'];o=rd['oxygen']
    checks['low_oxygen_controls/'+arm]=dict(
        finite_nonnegative=all(math.isfinite(float(v)) and float(v)>=-1e-7 for r in rows for v in r.values()),
        final_time=abs(d['final']['time']-12)<1e-8,lp_success=d['diagnostics']['solve_success_rate']==1,
        lactate_budget=abs(rd['continuous_feed'].get('lac__L_e',0)-3)<1e-9,
        oxygen_balance=abs(d['final']['o2']-.25-o['transferred']+o['polymer_consumed']+o['cellular_consumed'])<1e-8)
checks['low_oxygen_controls/source_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in read(OUT/'low_oxygen_controls/design.json')['sources'].items()}
assert all(all(v.values()) for v in checks.values()),checks
(OUT/'integrity.json').write_text(json.dumps(checks,indent=2))
(OUT/'summary.json').write_text(json.dumps(dict(paired=paired,refinements=fine,low_oxygen_controls={a:dict(pha=d['final']['pha'],hv_percent=hv_percent(d),rubber_removed=10-d['final']['rubber']) for a,d in controls.items()},base_secreted_carbon_mmol_l=carbon,base_direct_pha_upper_bound_g_l=upper_pha),indent=2))
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
x=np.arange(5);ax=axs[0,0]
ax.bar(x-.18,[r['two_pha_g_l'] for r in paired],.36,label='OR16 + NS21',color='#718096')
ax.bar(x+.18,[r['three_pha_g_l'] for r in paired],.36,label='OR16 + NS21 + Pf',color='#138a72')
ax.set(xticks=x,xticklabels=labels,ylabel='PHA at 12 h (g/L)',title='Equal total initial biomass, same feed per pair');ax.tick_params(axis='x',rotation=15);ax.legend()
ax=axs[0,1];ax.axhline(0,color='black',lw=.8);ax.axhline(5,color='#999999',ls='--',lw=.8)
ax.bar(x,[r['pha_gain_percent'] for r in paired],color=['#138a72' if r['pha_gain_percent']>0 else '#c26542' for r in paired]);ax.set(xticks=x,xticklabels=labels,ylabel='3-species PHA difference (%)',title='Production benefit is a separate question');ax.tick_params(axis='x',rotation=15)
arms=['ns21_alone','ns21_pf','two_equal_total','three'];ax=axs[1,0]
ax.bar(range(4),[data['base_'+a]['final']['pha'] for a in arms],color=['#4976a8','#7b68a6','#718096','#138a72'])
ax.set(xticks=range(4),xticklabels=['NS21','NS21 + Pf','OR16 + NS21','All 3'],ylabel='PHA at 12 h (g/L)',title='Alternative inoculum allocations (total 0.63 g/L)')
ax=axs[1,1];growtharms=['pf_alone','ns21_pf','three'];ax.bar(range(3),[data['base_'+a]['fold_growth'][PF] for a in growtharms],color='#138a72');ax.axhline(1,color='black',lw=.8)
ax.set(xticks=range(3),xticklabels=['Pf alone','NS21 + Pf','All 3'],ylabel='Pf final / initial biomass',title='Growth is not evidence of stable coexistence')
fig.suptitle('Exploratory model audit: 12 h, internal step 0.025 h, uncalibrated kinetics',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
table='\n'.join(f"|{label}|{r['two_pha_g_l']:.6f}|{r['three_pha_g_l']:.6f}|{r['pha_gain_percent']:+.2f}%|{r['rubber_gain_percent']:+.2f}%|" for label,r in zip(jlabels,paired))
alternate='\n'.join(f"|{a}|{data['base_'+a]['final']['pha']:.6f}|{10-data['base_'+a]['final']['rubber']:.6f}|" for a in arms+['two_fixed','pf_alone'])
finetext='\n'.join(f"- {r['condition']}、内部刻み0.0125 h：2種 {r['two']:.6f}、3種 {r['three']:.6f} g/L、差 {r['gain']:+.2f}%。ゴム除去差 {r['rubber_gain']:+.2f}%。3HV mol%は2種 {r['two_3hv']:.2f}、3種 {r['three_3hv']:.2f}。" for r in fine)
low=data['oxygen_low_three'];lowtwo=data['oxygen_low_two_equal_total']
lowtable='\n'.join(f"|{label}|{d['final']['pha']:.6f}|{hv_percent(d):.2f}|{10-d['final']['rubber']:.6f}|" for label,d in [('2種・総量一致',lowtwo),('3種',low),('2種・OR16/NS21量固定',controls['two_fixed']),('NS21単独',controls['ns21_alone']),('3種・Pf全反応停止（反実仮想）',controls['inert_pf'])])
growth='\n'.join(f"- {a}: Pf {data['base_'+a]['fold_growth'][PF]:.6f}倍、平均log増殖率 {data['base_'+a]['mean_net_growth_h'][PF]:.6f} h⁻¹。" for a in growtharms)
positives=[r['condition'] for r in paired if r['screen_positive']]
report=f'''# 3種共生の根拠の再評価（2026-09-08）

探索した5条件のうち、事前に定めた「PHAが5%超改善し、ゴム除去の悪化が5%以内」を満たす条件は **{len(positives)} / 5**：{positives}。
これは決定論的なモデル探索であり、統計的有意差や実培養の効果量ではない。条件網羅・最適化はしていない。

## 比較条件と生産結果

12時間、ゴム10 g/L、初期NH4 0.05 mmol/L、pH 7固定。基準の乳酸供給0.25 mmol/L/h、kLa 50 h⁻¹。
3種の初期菌体はOR16 0.5、NS21 0.1、Pf 0.03 g/L。主対照はOR16 0.525、NS21 0.105 g/Lで、総量0.63 g/Lをそろえた。
乳酸なしは供給0、4倍は1 mmol/L/h、NH4供給は最初の4時間のみ0.1 mmol/L/h、低kLaは10 h⁻¹。他条件は基準と共通。
各ペア内の投入量をそろえた。条件間では乳酸・窒素の投入量が異なり、単純な費用効率比較ではない。

|条件|2種PHA g/L|3種PHA g/L|PHA差|ゴム除去量差|
|---|---:|---:|---:|---:|
{table}

基準条件でOR16とNS21の接種量を固定してPfだけを足す対照（総量0.60対0.63 g/L）の2種PHAは {data['base_two_fixed']['final']['pha']:.6f} g/L。
追加菌体の効果とPfの種としての価値を混同しないため、総量をそろえる対照と併記した。

{finetext}

全14ケースで有限・非負の記録値、12時間到達、供給量、酸素収支、LP成功率100%、ソースハッシュ一致を検証した（integrity.json）。
数値収束確認は生物学的妥当性や完全な炭素・窒素収支の証明ではない。NH4依存の連続的な成長・貯蔵配分は未校正である。

## 低kLaで見つかった候補と追加対照

この追加対照は低kLaでの陽性結果を見た後に選んだ探索的解析であり、独立の確認試験ではない。

|構成（kLa 10 h⁻¹）|PHA g/L|3HV mol%|ゴム除去 g/L|
|---|---:|---:|---:|
{lowtable}

3種はOR16/NS21の接種量を固定した2種に対してPHA {100*(low['final']['pha']/controls['two_fixed']['final']['pha']-1):+.2f}%。
Pfの全反応を0にした仮想対照に対しては {100*(low['final']['pha']/controls['inert_pf']['final']['pha']-1):+.2f}%。
後者ではPf菌体のプール・分配上の存在を残して代謝だけを除いた。これは生物学的な変異株や遺伝子ノックアウトの予測ではない。
全反応停止は炭素分泌だけでなく窒素取り込み等も止めるため、差をプロピオン酸単独の効果とは呼べない。
低kLaの3種はNS21単独に対してPHA {100*(low['final']['pha']/controls['ns21_alone']['final']['pha']-1):+.2f}%。総PHA、組成、ゴム処理量を分けて判断する必要がある。
基準条件では3HVが0だが、低kLaでは2種でも3HVが出る。Pfによる新規能力の付与と、既存能力の配分変化を区別する。
酸素・NH4応答、接種比、乳酸供給、pH制御の仮定を固定した候補であり、「3種の普遍的優位」や「安定共生」の証拠ではない。

## Pfの機能は存在するが、共生の証明とは異なる

基準条件の全内部ステップ積分（mmol/L）：

- Pf乳酸取り込み {pf.get('lac__L_e_in',0):.8f}、外部乳酸投入3.0の {100*pf.get('lac__L_e_in',0)/3:.2f}%に相当。
- Pfプロピオン酸分泌 {pf.get('ppa_e_out',0):.8f}、NS21取り込み {ns.get('ppa_e_in',0):.8f}。
- Pf酢酸分泌 {pf.get('ac_e_out',0):.8f}、NS21取り込み {ns.get('ac_e_in',0):.8f}。
- Pf B12分泌 {pf.get('b12_e_out',0):.8f}。
- NS21は乳酸を直接 {ns.get('lac__L_e_in',0):.8f} 取り込んでおり、Pfを経由しなければ乳酸を利用できない構成ではない。

共有プールでの分泌・取り込みはモデル内の物質移動を示す。同位体追跡でも、分泌物遮断による因果効果測定でもない。
Pf分泌プロピオン酸・酢酸は合計 {carbon:.8f} mmol-C/L。その炭素を100%PHBにできたとしても直接寄与は最大 {upper_pha:.6f} g/L（基準PHAの {100*upper_pha/base['final']['pha']:.3f}%）。
これは窒素消費などによる間接効果の上限ではない。基準条件の小さな炭素受け渡しを、大きなPHA増産の直接原因とは呼べない。

Pfの増殖：

{growth}

Pf単独と3種はPf初期量を0.03 g/Lでそろえたが、他種追加に伴う総菌体量・競争は変わる。NS21+Pfは総菌体量を合わせたためPf初期量0.145385 g/Lであり、単純な相互利益の対照にはならない。
乳酸無供給の3種でPfは {data['no_lactate_three']['fold_growth'][PF]:.6f} 倍。培地にはアミノ酸・核酸塩基等が残っているため「外部有機物なし」試験ではない。
12時間に菌が残ることや増えることは安定共存の証明ではない。本試験は希釈・継代・侵入可能性・長期摂動後の回復を試していない。
特に現在の菌体更新ではpH 7における死亡項は0であり、増殖フラックス0の菌体はそのまま残る。乳酸なしでPfが残存するだけでは、生存維持機構の裏付けにならない。

## 1種・別の2種構成も含めた反証用対照

|構成|PHA g/L|ゴム除去 g/L|
|---|---:|---:|
{alternate}

NS21単独・NS21+Pf・主対照2種・3種は総菌体量0.63 g/Lだが、NS21への割当量が異なる。
したがって、これらは運用上の候補比較であり、種間相互作用だけを分離する実験ではない。OR16+Pfおよび全接種比は未探索。
実装上のPHV量も各result.jsonに残した。現在のPHA目的関数から得た結果を、実株で知られるPHBV組成の再現とはみなせない。

## 文献とモデル構造から分かること

1. OR16のゴム分解とLcpの酸素消費は一次研究で支持される。ただしこの3種の優位の実証ではない。
   [OR16のLcp解析](https://pmc.ncbi.nlm.nih.gov/articles/PMC7413915/)
2. NS21は単独で天然ゴムからPHBV、グルコースからPHBを作り、窒素制限でPHBV蓄積が増える。NS21単独という強い対照を省けない。
   文献ではPiscinibacter gummiphilus NS21T、コードでは従来名Rhizobacterを使用。
   [Tamamura et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39216800/)
3. Pfの乳酸代謝は実証されるが、微好気条件では乳酸枯渇後にプロピオン酸・酢酸を再消費する。常に他菌へ有機酸を供給し続けるとは限らない。
   [Dank et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33955639/)
4. 微好気代謝とB12生産の関連は研究されている。しかしB12生産可能性、菌外への供給、OR16/NS21の需要は別の証拠を要する。
   [Pfの微好気代謝とB12](https://pubmed.ncbi.nlm.nih.gov/36307780/)

今回確認した一次文献では、OR16+NS21+Pfという同じ組合せが単独・2種より優れる直接の実験根拠は見つからなかった。
structure_audit.jsonではOR16/NS21に明示的なB12受容インターフェースがない。前駆体まで含めるとOR16にはadenosyl cobinamide等の2代謝物があるが、これはPfからのB12供給が成長・PHAを改善する回路の証明ではない。NS21では今回の関連語検索に該当する代謝物は見つからなかった。
これは実菌のB12非依存性を証明しないが、現在のモデルでPfのB12供給価値を評価できないことを示す。
Pf側でもB12関連反応rxnnew81_c0のGPRはOpen_problem、輸送反応rxnnew43_c0はTransporters_added_without_gene_assignmentという元モデル由来の注記であり、遺伝子に裏付けられた供給予測としては不十分である。
またPfの酸素取り込みはモデル準備時に閉じており、同じ曝気槽の溶存酸素に対するPfの応答は表現されていない。
酸素移動の数値計算改善と、この生理学的仮定の検証は別問題である。

## 次に優先すべき根拠づくり

- **まずPHBV品質仮説**：プロピオン酸がNS21の3HV比率へ寄与する経路・発現・速度を一次データで確認する。総PHAだけでなく3HV比率を目的にする場合にPfの役割を再評価する。未確認の反応や必須性を追加して3種を有利にしない。
- **B12は受け手の需要が先**：OR16/NS21の依存反応、代替酵素、取り込みをゲノム・文献で点検し、必要な場合だけ実測の制約を導入する。Pfの生産能だけで相利共生と呼ばない。
- **外部乳酸を含む混合廃棄物流として評価**：Pfが使う炭素源を他の2種が供給できるか、外部共基質なのかを明記する。外部基質ならゴム由来炭素収率と総投入炭素収率を分ける。
- **実用性は対照を保って探索**：NS21単独、OR16+NS21、NS21+Pf、3種で同じ供給・接種量制約を置き、直接プロピオン酸供給も比較する。乳酸除去、PHA量、3HV比率、酸素と培地コストのどの改善を求めるか固定する。
- **安定性は別段階で検証**：短期生産の有望条件が得られた後、希釈や繰返し供給、初期比率・供給誤差への頑健性を調べる。現段階の計算を「安定3種共生の実証」として教師データにラベル付けしない。

添付RDE2報告は別菌種・別基質・日単位の実験であり、この3種共培養の直接の裏付けやK_Nの校正には使用していない。
既存512軌道の収集器・旧データ・元GEMは本再評価では変更していない。
'''
(OUT/'REPORT_JA.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(paired=paired,refinements=fine,checks_passed=True),indent=2))
