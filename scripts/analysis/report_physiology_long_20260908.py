"""Independent trajectory checks and predeclared long-study comparisons."""
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/physiology_long_validation_20260908'

def read(p):return json.loads(p.read_text())
def trajectories(p):
    with p.open() as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]

def main():
    design=read(OUT/'design.json');execution=read(OUT/'execution_verification.json')
    assert execution['source_sha256']==design['source_sha256']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in design['source_sha256'].items())
    assert len(execution['results'])==len(design['cases'])
    results={};series={};checks={};failures=[];residual_maxima={}
    for case in design['cases']:
        cid=case['id'];p=OUT/cid;status=read(p/'progress.json')
        if status['status']!='complete':failures.append(status);continue
        r=read(p/'result.json');s=trajectories(p/'trajectory.csv');settings=read(p/'settings.json')
        assert r['case']==case==settings['case']
        a=r['diagnostics']['audited_cultivation']['accounting']
        checks[cid]=dict(final_time=abs(s[-1]['time_h']-case['hours'])<1e-9,
            times=all(abs(b['time_h']-a['time_h']-.25)<1e-9 for a,b in zip(s,s[1:])),
            finite=all(math.isfinite(v) for row in s for v in row.values()),
            nonnegative=all(row[k]>=-1e-8 for row in s for k in row if
                k.endswith('_g_l') or k.endswith('_mmol_l') or k.endswith('_maintenance_deficit')),
            element_ledger=all(max(abs(row[k]) for k in ['C_ledger_residual','N_ledger_residual'])<1e-6 for row in s),
            oxygen=all(row['oxygen_balance_error']<=1e-8 and row['oxygen_kinetic_error']<=1e-6 for row in s),
            storage=all(row['storage_excess']<=1e-7 for row in s),
            lp=a['max_lp_residual']<=1e-7 and a['max_dual_residual']<=1e-7 and a['max_relative_duality_gap']<=1e-7,
            lp_accounting=a['lp_requests']==a['lp_certified_requests']+a.get('superseded_uncertified_requests',0),
            lactate_feed=abs(s[-1]['lactate_delivered']-.25*min(case['hours'],case.get('feed_stop',1e6)))<1e-8,
            nh4_feed=abs(s[-1]['nh4_delivered']-.1*case.get('nh4_until',0.))<1e-8,
            pha_components=all(abs(row['pha_live_g_l']+row['pha_dead_g_l']-row['pha_total_g_l'])<1e-9 for row in s),
            no_failed_steps=a['failed_steps']==0)
        assert all(checks[cid].values()),(cid,checks[cid])
        residual_maxima[cid]={k:max(abs(row[k]) for row in s) for k in
            ['C_ledger_residual','N_ledger_residual','oxygen_balance_error','oxygen_kinetic_error','lp_residual','storage_excess']}
        residual_maxima[cid].update(max_dual_residual=a['max_dual_residual'],max_relative_duality_gap=a['max_relative_duality_gap'])
        results[cid]=r;series[cid]=s
    if 'rich_three' in series and 'rich_three_feed_stop' in series:
        # A future feed stop cannot change the preceding trajectory.
        keys=['pha_live_g_l','pha_dead_g_l','rubber_g_l','nh4_mmol_l','o2_mmol_l',
              'OR16_live_g_l','NS21_live_g_l','Pf_live_g_l','lactate_delivered']
        checks['feed_stop_prefix']={'same_before_event':all(abs(a[k]-b[k])<=1e-9
            for a,b in zip(series['rich_three'],series['rich_three_feed_stop']) if a['time_h']<=12. for k in keys)}
        assert checks['feed_stop_prefix']['same_before_event']
    pairs=[('rich_three','rich_three_fine'),('rich_two','rich_two_fine'),('n_depleted_three','n_depleted_three_fine'),
           ('rich_three_coarse','rich_three')]
    comparisons=[]
    for a,b in pairs:
        if a not in results or b not in results:continue
        ra,rb=results[a],results[b];sa,sb=series[a],series[b]
        assert len(sa)==len(sb) and all(x['time_h']==y['time_h'] for x,y in zip(sa,sb))
        assert read(OUT/a/'settings.json')['initial_medium']==read(OUT/b/'settings.json')['initial_medium']
        peak_scale=max(.001,ra['peak_pha_g_l'],rb['peak_pha_g_l'])
        endpoint_error=abs(ra['final']['pha_live_g_l']-rb['final']['pha_live_g_l'])/max(.001,abs(rb['final']['pha_live_g_l']))
        peak_error=abs(ra['peak_pha_g_l']-rb['peak_pha_g_l'])/max(.001,rb['peak_pha_g_l'])
        biomass={k:abs(ra['final'][k]-rb['final'][k])/max(1e-6,abs(rb['final'][k]))
            for k in rb['final'] if k.endswith('_live_g_l') and k!='pha_live_g_l'}
        hv=max((abs(x['hv_mol_fraction']-y['hv_mol_fraction']) for x,y in zip(sa,sb)
            if min(x['pha_live_g_l'],y['pha_live_g_l'])>.001),default=0.)
        accumulation=min(ra['peak_pha_g_l'],rb['peak_pha_g_l'])>.001
        passes=(endpoint_error<=.02 and peak_error<=.02 and max(biomass.values())<=.02 and hv<=.001)
        comparisons.append(dict(coarse=a,fine=b,endpoint_pha_relative_error=endpoint_error,
            peak_pha_relative_error=peak_error,endpoint_biomass_relative_errors=biomass,
            accumulation_onset_difference_h=(abs(ra['accumulation_onset_h']-rb['accumulation_onset_h'])
                if ra['accumulation_onset_h'] is not None and rb['accumulation_onset_h'] is not None else None),
            max_trajectory_pha_error_normalized_by_peak=max(abs(x['pha_live_g_l']-y['pha_live_g_l']) for x,y in zip(sa,sb))/peak_scale,
            max_hv_fraction_difference_during_accumulation=hv,accumulation_observed=accumulation,
            predeclared_numerical_screen_passed=passes,pha_accumulation_validation_passed=passes and accumulation))
    physiology_coverage={cid:dict(
        maximum_maintenance_deficit={k:max(row[k] for row in series[cid])
            for k in series[cid][0] if k.endswith('_maintenance_deficit')},
        cumulative=r['diagnostics']['physiology']['cumulative']) for cid,r in results.items()}
    consortium_comparisons=[]
    for two,three in [('rich_two','rich_three'),('rich_two_fine','rich_three_fine')]:
        if two in results and three in results:
            p2=results[two]['final']['pha_live_g_l'];p3=results[three]['final']['pha_live_g_l']
            consortium_comparisons.append(dict(two=two,three=three,dt=results[three]['case']['dt'],
                pha_two_g_l=p2,pha_three_g_l=p3,relative_three_minus_two=(p3-p2)/max(.001,abs(p2))))
    qualified=(len(results)==len(design['cases']) and not failures and len(comparisons)==len(pairs)
        and all(r['pha_accumulation_validation_passed'] for r in comparisons))
    summary=dict(completed=len(results),failed=failures,checks=checks,comparisons=comparisons,
        predeclared_long_validation_passed=qualified,
        physiology_coverage=physiology_coverage,
        consortium_comparisons=consortium_comparisons,
        residual_maxima=residual_maxima,
        cases={cid:dict(peak_pha_g_l=r['peak_pha_g_l'],accumulation_onset_h=r['accumulation_onset_h'],final=r['final'],seconds=r['seconds']) for cid,r in results.items()},
        biological_calibration_completed=False,mass_teacher_collection_started=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for column,ids in enumerate([['rich_three_coarse','rich_three','rich_three_fine'],['n_depleted_three','n_depleted_three_fine']]):
        for cid in ids:
            if cid not in series:continue
            s=series[cid];t=[r['time_h'] for r in s];dt=results[cid]['case']['dt']
            for ax,key in zip(axes[:,column],['pha_live_g_l','nh4_mmol_l',None]):
                y=[r[key] if key else sum(v for k,v in r.items() if k in ['OR16_live_g_l','NS21_live_g_l','Pf_live_g_l']) for r in s]
                ax.plot(t,y,label=f'dt={dt} h')
        for ax in axes[:,column]:ax.legend();ax.set_xlabel('Culture time (h)')
    axes[0,0].set_title('Rich background, three species');axes[0,1].set_title('N-depleted computational control')
    axes[1,1].set_title('NH4 numerical residue (~1e-11 mmol/L)',fontsize=10)
    for row,label in enumerate(['Live-cell PHA (g/L)','NH4 (mmol/L)','Total live biomass (g/L)']):
        for ax in axes[row]:ax.set_ylabel(label)
    fig.suptitle('Long physiology validation: uncalibrated parameters')
    fig.savefig(OUT/'trajectories.png',dpi=160);fig.savefig(OUT/'trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for cid in ['rich_three','rich_three_high_maintenance','rich_three_high_death','rich_three_feed_stop','rich_three_nh4_staged']:
        if cid not in series:continue
        s=series[cid];t=[r['time_h'] for r in s]
        for ax,key in zip(axes.flat,['pha_live_g_l','pha_dead_g_l','Pf_live_g_l','nh4_mmol_l']):
            ax.plot(t,[r[key] for r in s],label=cid.replace('rich_three','baseline'))
    for ax,key in zip(axes.flat,['Live-cell PHA (g/L)','Dead-cell PHA (g/L)','Live Pf (g/L)','NH4 (mmol/L)']):
        ax.set(xlabel='Culture time (h)',ylabel=key);ax.legend(fontsize=7)
    fig.suptitle('Assumption and feed sensitivity; not evidence of biological stability')
    fig.savefig(OUT/'sensitivity.png',dpi=160);fig.savefig(OUT/'sensitivity.pdf');plt.close(fig)
    ctable='\n'.join(f"|{cid}|{r['case']['hours']:g}|{r['peak_pha_g_l']:.6f}|{r['final']['pha_live_g_l']:.6f}|{r['final']['pha_dead_g_l']:.6f}|{r['accumulation_onset_h'] if r['accumulation_onset_h'] is not None else '未到達'}|" for cid,r in results.items())
    ptable='\n'.join(f"|{r['coarse']} → {r['fine']}|{100*r['endpoint_pha_relative_error']:.3f}%|{100*r['peak_pha_relative_error']:.3f}%|{100*max(r['endpoint_biomass_relative_errors'].values()):.3f}%|{100*r['max_hv_fraction_difference_during_accumulation']:.4f}ポイント|{'合格' if r['pha_accumulation_validation_passed'] else '未達'}|" for r in comparisons)
    failed_text='なし' if not failures else '、'.join(f['id'] for f in failures)
    consortium_table='\n'.join(f"|{r['dt']:g}|{r['pha_two_g_l']:.6f}|{r['pha_three_g_l']:.6f}|{100*r['relative_three_minus_two']:+.3f}%|" for r in consortium_comparisons)
    needed=[r['coarse'] for r in comparisons if r['fine'].endswith('_fine') and not r['pha_accumulation_validation_passed']]
    (OUT/'REPORT_JA.md').write_text(f'''# 生理モデルの追加監査・PHA蓄積期の長時間検証

**今回の設計に対する長時間数値検証：{'合格' if qualified else '未達'}。** 任意の培地・操作範囲での精度や生物学的校正を保証する判定ではない。

追加監査でRLの版判定、逆向き貯蔵境界、ATPMの入力情報不足を修正した。
修正・計算前の基準は [検証設計](../../docs/PHYSIOLOGY_LONG_VALIDATION_20260908.md) に記録した。
この実行ソースでの事前回帰試験は347合格、0失敗（既存fork警告6件）。ログは `preflight_tests.log`。

長時間ケースは{len(results)}/{len(design['cases'])}完了。停止したケース：{failed_text}。
完了ケースでは全記録時点の有限・非負、LP認証、酸素、既知C/N台帳、PHA貯蔵上限、投入量、終点時刻を確認した。
実行前後のソース・GEM・ゲノムのSHA256は一致する。失敗や微小PHAを0に置き換えて成功扱いにしていない。
残差の実測最大値と生理機能の作動状況は `summary.json` の `residual_maxima` と `physiology_coverage` に記録した。

## PHA蓄積と死菌区画

|条件|計算時間 h|生菌中PHAピーク g/L|終点生菌中PHA g/L|終点死菌中PHA g/L|0.001 g/L超への到達 h|
|---|---:|---:|---:|---:|---:|
{ctable}

初期PHAは0。0.001 g/Lは数値検証で蓄積期を区別するための閾値で、実験的有意差や検出限界ではない。
ピーク値・到達時刻・時系列差は0.25時間間隔の保存点で評価した。
設定したPHA貯蔵上限は全乾燥重量の80%で、実測校正値ではない。上限付近の終点値だけで生産過程の一致を判断しない。
死菌体のPHAは別区画に保存し、現RLのPHA報酬へは含めない。総抽出可能PHAの最適化が完成したことを意味しない。
窒素を除いた対照は計算上の洗浄菌体条件であり、菌を培養・維持できる実機培地の提案ではない。

## 時間刻み比較

判定は、終点・ピークPHA差2%以内、菌種別終点生菌体差2%以内、蓄積中3HVの絶対差0.1 mol%ポイント以内。
PHAの相対差分母には0.001 g/L、菌体には1e-6 g/Lの下限を設ける。

|対応比較|終点PHA差|ピークPHA差|菌種別菌体差の最大|3HV差の最大|蓄積期を含む判定|
|---|---:|---:|---:|---:|---|
{ptable}

追加検証が必要な主比較：{'、'.join(needed) if needed else 'なし（この設計の数値基準に限る）'}。
全時系列のPHA差も `summary.json` に保存した。判定は数学的収束の証明や任意の供給・パラメータでの精度保証ではない。
維持代謝・死滅・供給停止等の条件は単一刻みでのストレス確認であり、その全域が細刻みで認証されたという意味ではない。
3HV差は2種の22.75 hで最大0.09485 mol%ポイントとなり、基準0.1に近い。3種でも22.5 hに0.06918ポイントの差がある。
PHA貯蔵上限に到達する終盤の組成は、総PHA量より刻みに敏感である。この結果を高精度な組成制御の認証に使わず、組成を報酬にする段階で追加の細刻み・配分検証を行う。
N除去対照のNH4図は約1e-11 mmol/Lの数値残差を拡大表示している。生物学的な窒素供給・窒素生成を意味しない。

## この条件での2種・3種比較

初期総菌体量0.63 g/Lをそろえた標準条件の結果。Pfを加えた分、2種側のOR16/NS21接種量も調整している。

|内部刻み h|2種の終点生菌中PHA g/L|3種の終点生菌中PHA g/L|3種−2種の相対差|
|---|---:|---:|---:|
{consortium_table}

この比較は固定条件での計算予測であり、培地供給を双方で最適化した比較、統計的有意差、3種共生の必然性の評価ではない。

## 大量教師収集への扱い

この結果だけで任意のRL探索領域を一括で承認しない。対象の培地・生理パラメータ・操作範囲を固定し、必要な細刻み確認と少量の教師収集・再生検査を経る。
旧512軌道の教師収集と新しい大量収集は開始していない。
GEMの不完全な分子式、B12依存、Pfの酸素表現型、PHVゲート、維持・死滅の実測校正、PHA炭素起源は依然として別の課題。
今回の2種・3種比較を実験的な安定共生やPf必須性と同一視しない。

図：`trajectories.png`、`sensitivity.png`。全軌道CSV、設定、診断、失敗記録は各ケースのディレクトリに保存した。
''',encoding='utf-8')
    print(json.dumps(dict(completed=len(results),failed=len(failures),needs_additional=needed,comparisons=comparisons),indent=2))

if __name__=='__main__':main()
