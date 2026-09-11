"""Independent budget audit and comparison of independently selected feed profiles."""
from pathlib import Path
import json,csv,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/equal_budget_comparison_20260908'
def read(p):return json.loads(p.read_text())
def csvread(p):
    with p.open() as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
def profile(c):return c['profile']['lactate']+'/'+c['profile']['nitrogen']

def main():
    d=read(OUT/'design.json');execution=read(OUT/'execution_verification.json');selection=read(OUT/'selection.json')
    assert execution['all_completed']
    assert d['source_sha256']==execution['source_sha256']=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in d['source_sha256']}
    cases=d['screen_cases']+selection['refinement_cases'];results={};series={};checks={};oxygen_saturations=set()
    assert len(cases)==execution['screen_count']+execution['refinement_count']
    for c in cases:
        cid=c['id'];p=OUT/cid;r=read(p/'result.json');s=csvread(p/'trajectory.csv');settings=read(p/'settings.json')
        assert read(p/'progress.json')['status']=='complete' and c==r['case']==settings['case']
        assert settings['initial_medium']==d['common_initial_medium']
        oxygen_saturations.add(settings['oxygen_saturation_mmol_l'])
        assert abs(sum(settings['initial_biomass'].values())-.63)<1e-12
        assert len(s)==97 and all(abs(row['time_h']-.25*i)<1e-9 for i,row in enumerate(s))
        assert all(math.isfinite(v) for row in s for v in row.values())
        assert all(v>=-1e-8 for row in s for k,v in row.items() if k.endswith('_g_l') or k.endswith('_mmol_l'))
        for row in s:
            t=row['time_h'];l=c['profile']['lactate'];n=c['profile']['nitrogen']
            expected_l=.25*t if l=='uniform' else .5*(min(t,12.) if l=='early' else max(0.,t-12.))
            expected_n=.4*min(1.,t/(4. if n=='early' else 12.))
            assert abs(row['lactate_delivered']-expected_l)<1e-8 and abs(row['nh4_delivered']-expected_n)<1e-8
            assert row['oxygen_transferred']<=c['kla']*t*settings['oxygen_saturation_mmol_l']+1e-8
            assert abs(row['pha_total_g_l']-row['pha_live_g_l']-row['pha_dead_g_l'])<1e-9
        a=r['diagnostics']['audited_cultivation']['accounting']
        checks[cid]=dict(known_element_residual=max(abs(row[k]) for row in s for k in ['C_residual','N_residual']),
            oxygen_balance=max(row['oxygen_balance_error'] for row in s),oxygen_kinetic=max(row['oxygen_kinetic_error'] for row in s),
            storage_excess=max(row['storage_excess'] for row in s),lp=max(a[k] for k in ['max_lp_residual','max_dual_residual','max_relative_duality_gap']))
        assert checks[cid]['known_element_residual']<1e-6 and checks[cid]['oxygen_balance']<=1e-8
        assert checks[cid]['oxygen_kinetic']<=1e-6 and checks[cid]['storage_excess']<=1e-7 and checks[cid]['lp']<=1e-7
        assert a['failed_steps']==0 and a['lp_requests']==a['lp_certified_requests']+a.get('superseded_uncertified_requests',0)
        assert r['final']==s[-1]
        results[cid]=r;series[cid]=s
    assert len(oxygen_saturations)==1
    refinement=[]
    for c in selection['refinement_cases']:
        fine=c['id'];coarse=fine.replace('refine_','screen_',1);a=results[coarse];b=results[fine]
        end=abs(a['final']['pha_live_g_l']-b['final']['pha_live_g_l'])/max(.001,abs(b['final']['pha_live_g_l']))
        peak=abs(a['peak_pha_g_l']-b['peak_pha_g_l'])/max(.001,abs(b['peak_pha_g_l']))
        biomass=max(abs(a['final'][k]-v)/max(1e-6,abs(v)) for k,v in b['final'].items() if k.endswith('_live_g_l') and k!='pha_live_g_l')
        refinement.append(dict(id=fine,endpoint_pha_error=end,peak_pha_error=peak,biomass_error=biomass,
            max_hv_fraction_difference=max((abs(x['hv_mol_fraction']-y['hv_mol_fraction']) for x,y in zip(series[coarse],series[fine])
                if min(x['pha_live_g_l'],y['pha_live_g_l'])>.001),default=0.),
            passed=end<=.02 and biomass<=.02))
    paired=[];best=[]
    for kla in sorted({c['kla'] for c in cases}):
        for phase in ['screen','refine']:
            candidates={arm:[r for r in results.values() if r['case']['kla']==kla and r['case']['phase']==phase and r['case']['arm']==arm] for arm in ['two','three']}
            assert {profile(r['case']) for r in candidates['two']}=={profile(r['case']) for r in candidates['three']}
            winner={arm:sorted(rr,key=lambda r:(-r['final']['pha_live_g_l'],r['case']['id']))[0] for arm,rr in candidates.items()}
            a=winner['two'];b=winner['three'];gain=(b['final']['pha_live_g_l']-a['final']['pha_live_g_l'])/max(.001,a['final']['pha_live_g_l'])
            rubber_loss=(a['final']['rubber_removed_g_l']-b['final']['rubber_removed_g_l'])/max(1e-6,a['final']['rubber_removed_g_l'])
            best.append(dict(kla=kla,phase=phase,evaluations_per_arm=len(candidates['two']),
                two_id=a['case']['id'],three_id=b['case']['id'],two_profile=profile(a['case']),three_profile=profile(b['case']),
                two_pha=a['final']['pha_live_g_l'],three_pha=b['final']['pha_live_g_l'],gain=gain,rubber_loss=rubber_loss,
                two_total_pha=a['final']['pha_total_g_l'],three_total_pha=b['final']['pha_total_g_l'],
                two_oxygen_transfer=a['final']['oxygen_transferred'],three_oxygen_transfer=b['final']['oxygen_transferred'],
                engineering_candidate=min(a['final']['pha_live_g_l'],b['final']['pha_live_g_l'])>.001 and gain>.05 and rubber_loss<=.05))
            for a in candidates['two']:
                b=next(r for r in candidates['three'] if profile(r['case'])==profile(a['case']))
                paired.append(dict(kla=kla,phase=phase,profile=profile(a['case']),two_pha=a['final']['pha_live_g_l'],
                    three_pha=b['final']['pha_live_g_l'],gain=(b['final']['pha_live_g_l']-a['final']['pha_live_g_l'])/max(.001,a['final']['pha_live_g_l'])))
    needs=[r['id'] for r in refinement if not r['passed']]
    near_best=[]
    for r in best:
        for arm in ['two','three']:
            near_best.append(dict(kla=r['kla'],phase=r['phase'],arm=arm,
                profiles=[profile(v['case']) for v in results.values() if v['case']['kla']==r['kla'] and v['case']['phase']==r['phase']
                    and v['case']['arm']==arm and v['final']['pha_live_g_l']>=.99*r[arm+'_pha']]))
    exchanges={}
    for r in best:
        for cid in [r['two_id'],r['three_id']]:
            molecular=results[cid]['diagnostics']['physiology']['elements']['molecular_exchanges']
            exchanges[cid]={name:{direction:{p:pools.get(p,0.) for p in ['lac__L_e','ppa_e','nh4_e','o2_e']}
                for direction,pools in directions.items()} for name,directions in molecular.items()}
    summary=dict(completed=len(results),screen_count=execution['screen_count'],refinement_count=execution['refinement_count'],
        checks=checks,refinement=refinement,best=best,same_feed_pairs=paired,near_best_within_one_percent=near_best,best_integrated_exchanges_mmol_l=exchanges,needs_additional_refinement=needs,
        biological_validation=False,global_optimality_claim=False,teacher_collection_started=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    for filename,rows in [('best.csv',best),('same_feed_pairs.csv',paired),('refinement.csv',refinement)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig,(ax,gain_ax)=plt.subplots(1,2,figsize=(12,5),layout='constrained');fine=[r for r in best if r['phase']=='refine'];x=np.arange(len(fine))
    ax.bar(x-.18,[r['two_pha'] for r in fine],.36,label='Two species');ax.bar(x+.18,[r['three_pha'] for r in fine],.36,label='Three species')
    ax.set_xticks(x,[str(r['kla']) for r in fine]);ax.set(xlabel='Shared kLa budget (1/h)',ylabel='Best final live-cell PHA (g/L)',title='Equal feed budgets; independently selected schedules')
    ax.legend()
    gain_ax.bar(x,[100*r['gain'] for r in fine],color=['#318b65' if r['gain']>0 else '#a74e4e' for r in fine])
    gain_ax.axhline(0,color='black',lw=.8);gain_ax.axhline(5,color='gray',ls='--',label='5% engineering threshold')
    gain_ax.set_xticks(x,[str(r['kla']) for r in fine]);gain_ax.set(xlabel='Shared kLa budget (1/h)',ylabel='Three minus two (%)',title='PHA gain after separate feed selection');gain_ax.legend(fontsize=8)
    fig.savefig(OUT/'best.png',dpi=170);fig.savefig(OUT/'best.pdf');plt.close(fig)
    table='\n'.join(f"|{r['kla']:g}|{r['evaluations_per_arm']}|{r['two_profile']}|{r['two_pha']:.6f}|{r['three_profile']}|{r['three_pha']:.6f}|{100*r['gain']:+.3f}%|" for r in fine)
    candidates=[r['kla'] for r in fine if r['engineering_candidate']]
    (OUT/'REPORT_JA.md').write_text(f'''# 同一資源予算での2種・3種の供給探索

全{len(results)}ケース完了。探索{execution['screen_count']}、対称な細刻み再評価{execution['refinement_count']}。
生菌中PHAの5%超増産・ゴム除去の低下5%以内を満たす細刻み候補のkLa：{candidates if candidates else 'なし'}。
数値基準未達の追加細分対象：{needs if needs else 'なし'}。未達がある場合、優位性の判定は確定しない。

## 公平性と探索範囲

各酸素条件で、乳酸6 mmol/L、追加NH4 0.4 mmol/L、初期ゴム10 g/L、背景培地、総接種量0.63 g/L、24時間を共通化した。
kLa=2/10/50 h^-1を別々に比較。各行で同じ曝気能力と飽和濃度を与え、実酸素移動量・消費量は別記した。酸素実消費を同量に強制した比較ではない。
接種比は固定（3種0.5/0.1/0.03、2種0.525/0.105 g/L）。pH7の維持に必要な酸・塩基の費用は未評価。
乳酸uniform=24 h均等、early=前半12 h、late=後半12 h。窒素early=最初4 h、spread=最初12 hに全量供給。
全6候補を各構成・各kLaで評価した。内部刻み0.025 hで上位2候補と最良から4%以内を抽出し、その和集合を両構成で0.0125 hにして同数評価した。
これは有限候補の探索であり、全球最適解・接種比の最適解・RL学習の結果ではない。

## それぞれで選んだ供給条件の比較

|kLa h^-1|細刻み評価数/構成|2種の選択供給（乳酸/窒素）|2種PHA g/L|3種の選択供給|3種PHA g/L|3種−2種|
|---|---:|---|---:|---|---:|---:|
{table}

各構成が自分に最も有利な供給条件を選んだ結果。同じ供給条件での差は `same_feed_pairs.csv` に全件保存した。
表示した供給が唯一の最適解とは限らない。最良値から1%以内の供給候補も `summary.json` に保存し、微小差による順位を生物学的な差と解釈しない。
総PHA（死菌分を含む）、実酸素移動、ゴム除去の低下率は `best.csv`。主指標は結果を見る前に生菌中PHAと定め、他の指標へ置き換えていない。
選択された条件の菌種別乳酸・プロピオン酸・NH4・酸素交換の積分量は `summary.json` の `best_integrated_exchanges_mmol_l`。共有プールへの分泌と取込の観測であり、因果効果や炭素起源の証明ではない。

## 数値検証と限界

全保存時点の有限・非負、既知C/N台帳、酸素収支・連立残差、貯蔵上限、LP認証、投入の閉形式との一致、24 h終点を確認した。
背景培地が完全に同じこと、候補集合・探索数・細刻み候補集合が両構成で同じことを独立に照合した。ソース・GEMの実行前後ハッシュも一致。
刻み差の詳細は `refinement.csv`。終点PHA・菌種別終点菌体の差2%以内を数値基準とし、ピークPHA差と3HV差も併記する。
PHA貯蔵上限、維持代謝・死滅率、B12依存、Pf酸素表現型、PHVゲートは未校正であり、この比較を生物学的必然性や統計的有意差と呼ばない。総PHAのゴム由来割合も未特定。
この有限供給探索で利点がなければ、その範囲では3種固定の根拠は得られなかったと扱う。全条件でPfが無価値とは断定しない。
大量教師収集は行っていない。事前設計は [設計文書](../../docs/EQUAL_BUDGET_COMPARISON_20260908.md)。
''',encoding='utf-8')
    print(json.dumps(dict(completed=len(results),best=fine,needs_additional_refinement=needs),indent=2))

if __name__=='__main__':main()
