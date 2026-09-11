"""Report all three low-oxygen resolutions without promoting a favorable sign."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_20260908'
EXTRA=OUT/'third_refinement'
def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open() as stream:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
def main():
    design=read(EXTRA/'design.json');attestation=read(EXTRA/'execution_source_verification.json')
    assert attestation==design['sources']
    results=read(EXTRA/'progress.json');assert len(results)==2 and all(r['returncode']==0 for r in results)
    metadata={r['case']['id']:r for r in results}
    checks={};all_rows=[];data={};series={}
    for dt,suffix in [(.025,''),(.0125,'_fine'),(.00625,'_finer')]:
        for arm in ['two_equal_total','three']:
            key='oxygen_low_'+arm+suffix
            location=OUT/key if dt!=.00625 else OUT/metadata[key]['relative_directory']
            d=read(location/'result.json');diag=d['diagnostics'];acc=diag['audited_cultivation']['accounting']
            time_series=rows(location/'trajectory.csv');internal=rows(location/'internal_trajectory.csv')
            if dt==.00625:
                patch=read(location/'numerical_patch.json')
                assert all(design['sources'].get(file)==sha for file,sha in patch['sources'].items())
                assert patch['threshold_mmol_l']==1e-12
            o=diag['resolved_dynamics']['oxygen'];initial=d['settings']['initial_medium']
            oxygen_error=d['final']['o2']-initial['o2_e']-o['transferred']+o['cellular_consumed']+o['polymer_consumed']
            checks[key]=dict(finite=all(math.isfinite(v) and v>=-1e-7 for row in time_series for v in row.values()),
                complete=abs(d['final']['time']-12)<1e-8,
                valid=diag['audited_cultivation']['valid'] and acc['failed_steps']==0,
                lp=acc['lp_requests']==acc['lp_certified_requests']+acc.get('superseded_uncertified_requests',0),
                residual=acc['max_lp_residual']<=1e-7 and acc['max_dual_residual']<=1e-7 and acc['max_relative_duality_gap']<=1e-7,
                oxygen=abs(oxygen_error)<1e-8 and acc['max_oxygen_mean_residual']<=1e-6,
                carbon=max(abs(row['polymer_carbon_error']) for row in internal)<1e-8,
                storage=acc['max_storage_excess_g_l']<=1e-7,
                feed=abs(diag['resolved_dynamics']['continuous_feed'].get('lac__L_e',0)-3)<1e-9)
            data[dt,arm]=d;series[dt,arm]=(time_series,internal)
            prior=data.get((dt*2,arm))
            relative_change=None if prior is None else 100*(d['final']['pha']/prior['final']['pha']-1)
            peak=max(internal,key=lambda row:row['nh4'])
            depleted=next((row['time'] for row in internal if row['time']>peak['time'] and row['nh4']<.001),None)
            all_rows.append(dict(internal_dt_h=dt,arm=arm,pha_g_l=d['final']['pha'],
                change_from_double_step_percent=relative_change,rubber_removed_g_l=10-d['final']['rubber'],
                hv_mol_percent=100*d['final']['phv_mol_fraction'],peak_nh4_mmol_l=peak['nh4'],
                post_peak_nh4_below_001_time_h=depleted,lp_attempts=diag['solve_attempts'],seconds=d['seconds']))
        a=data[dt,'two_equal_total']['settings'];b=data[dt,'three']['settings']
        assert a['initial_medium']==b['initial_medium'] and a['internal_dt']==b['internal_dt']==dt
        assert a['kla']==b['kla']==10 and a['lactate_rate']==b['lactate_rate']==.25
        assert abs(sum(a['initial_biomass'].values())-sum(b['initial_biomass'].values()))<1e-12
    assert all(all(c.values()) for c in checks.values()),checks
    gains=[dict(internal_dt_h=dt,pha_gain_percent=100*(data[dt,'three']['final']['pha']/data[dt,'two_equal_total']['final']['pha']-1))
           for dt in [.025,.0125,.00625]]
    differences={}
    for arm in ['two_equal_total','three']:
        values=[data[dt,arm]['final']['pha'] for dt in [.025,.0125,.00625]]
        first=values[1]-values[0];second=values[2]-values[1]
        differences[arm]=dict(first_difference_g_l=first,second_difference_g_l=second,
            difference_shrank=abs(second)<abs(first),same_direction=first*second>0,
            observed_order=(math.log2(abs(first/second)) if first*second>0 and second else None))
    summary=dict(rows=all_rows,paired_gains=gains,successive_differences=differences,
        mathematical_convergence_proven=False,checks=checks,
        source_attestation='execution_source_verification.json')
    (EXTRA/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (EXTRA/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=all_rows[0]);writer.writeheader();writer.writerows(all_rows)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for dt,color in [(.025,'#718096'),(.0125,'#138a72'),(.00625,'#d38322')]:
        ts,internal=series[dt,'three']
        axs[0,0].plot([r['time'] for r in ts],[r['pha'] for r in ts],label=f'{dt} h',color=color)
        axs[0,1].plot([r['time'] for r in internal],[r['nh4'] for r in internal],label=f'{dt} h',color=color)
        axs[1,0].plot([r['time'] for r in internal],[r['mean_o2'] for r in internal],label=f'{dt} h',color=color)
    axs[0,0].set(title='Three species: PHA',ylabel='g/L',xlabel='h');axs[0,0].legend()
    axs[0,1].set(title='Three species: NH4',ylabel='mmol/L',xlabel='h')
    axs[1,0].set(title='Three species: interval mean DO',ylabel='mmol/L',xlabel='h')
    axs[1,1].bar(range(3),[r['pha_gain_percent'] for r in gains],color=['#718096','#138a72','#d38322'])
    axs[1,1].axhline(0,color='black',lw=.8);axs[1,1].axhline(5,color='#999999',ls='--')
    axs[1,1].set(title='Three versus two, equal total inoculum',xticks=range(3),xticklabels=['.025 h','.0125 h','.00625 h'],ylabel='PHA difference (%)')
    fig.suptitle('Low kLa: additional numerical refinement, uncalibrated biology')
    fig.savefig(EXTRA/'comparison.png',dpi=180);fig.savefig(EXTRA/'comparison.pdf');plt.close(fig)
    table='\n'.join(f"|{r['internal_dt_h']}|{r['arm']}|{r['pha_g_l']:.6f}|{('—' if r['change_from_double_step_percent'] is None else format(r['change_from_double_step_percent'],'+.2f')+'%')}|{r['rubber_removed_g_l']:.6f}|{r['hv_mol_percent']:.4f}|" for r in all_rows)
    gtable='\n'.join(f"|{r['internal_dt_h']}|{r['pha_gain_percent']:+.3f}%|" for r in gains)
    report=f'''# 低kLa条件の追加刻み検証

0.025→0.0125 hでPHAが約7%変わったため、同じ2種・3種を0.00625 hでも12時間計算した。陽性を増やす目的で条件を選び直していない。基本21ケースと合わせて23ケースであり、独立した培養反復ではない。

|内部刻み h|構成|PHA g/L|一段粗い刻みからの変化|ゴム除去 g/L|3HV mol%|
|---|---|---:|---:|---:|---:|
{table}

|内部刻み h|同一総接種量の2種に対する3種のPHA差|
|---|---:|
{gtable}

連続する刻み変更の差が縮小したか：2種 {'はい' if differences['two_equal_total']['difference_shrank'] else 'いいえ'}、3種 {'はい' if differences['three']['difference_shrank'] else 'いいえ'}。数値列からの観測次数はsummary.jsonに残したが、真の解が既知ではなく、有限個の刻み比較で数学的収束を証明したとはしない。順位の変動と生産量自体の変動を区別する。

全6結果で終点・投入量・有限非負・酸素収支・ゴム分解部分の炭素収支・LP受理解・貯蔵上限を検査した。追加2ケースの実行ソースの一致はexecution_source_verification.jsonへ記録した。C30丸め補正と一次目的の再計算の適用有無は各result.jsonに残る。基本21ケースの基準条件の細刻み比較は親のREPORT_JA.mdを参照。

NH4やDOの系列の違いは、まだ残る離散化誤差と生理モデルの仮定を調べるための診断であり、特定の機構だけが原因であることの証明ではない。初期NH4が0.05 mmol/Lでも他の窒素成分から再分泌が起きるため、初期NH4だけで全窒素状態を決めない。

旧1時間刻みでの陽性には、刻みごとに酸素在庫の固定比率を酵素へ予約する処理や菌体・交換量の不一致が含まれていた。監査のA02/A06等でそれぞれの人工的な効果を確認し修正した。ただし複数の実装と解選択を同時に変更した新旧比較から、昔の増産幅の何%が各不具合によるものかを分解したとは言わない。

本結果は校正前のモデル内比較である。Pfの酸素表現型、NH4依存の貯蔵配分、酵素速度、維持代謝などの未確認事項が残り、安定共生や実験での有意差を示さない。
'''
    (EXTRA/'REPORT_JA.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(gains=gains,differences=differences),indent=2))
if __name__=='__main__':main()
