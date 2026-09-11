"""Fill the repair log only from completed shared-oxygen results and tests."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/audited_symbiosis_shared_oxygen_20260908'
AUDIT=ROOT/'results/cultivation_model_audit_20260908'
def read(path):return json.loads(path.read_text())
summary=read(OUT/'summary.json');integrity=read(OUT/'integrity.json');design=read(OUT/'design.json')
assert summary['all_integrity_checks_passed'] and len(design['cases'])==23
assert all(hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha for file,sha in design['sources'].items())
log=(AUDIT/'oxygen_quota_followup/verification_final.log').read_text()
assert 'failed' not in log.split('short test summary info')[-1]
passed=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',log)
assert passed,log
base=next(r for r in summary['refinement'] if r['condition']=='base')
low=next(r for r in summary['refinement'] if r['condition']=='oxygen_low')
third={r['arm']:r for r in summary['third_refinement']}
ns=read(OUT/'base_ns21_alone/result.json')['final']
active=read(OUT/'oxygen_low_three/result.json')['final'];inert=read(OUT/'oxygen_low_inert_pf/result.json')['final']
ph=100*(active['pha']/inert['pha']-1)
maxima={name:max(abs(m[name]) for m in integrity['metrics'].values()) for name in
    ['oxygen_kinetic_residual','oxygen_transfer_law_error_mmol_l','polymer_c5_error_mmol_l','lp_residual']}
table={
'最終テスト実行コマンド・ログ':'`results/cultivation_model_audit_20260908/verification_command.txt` と `oxygen_quota_followup/verification_final.log`。数値・培養器・教師来歴・RL・GEM curationを含む21テストファイル。',
'最終合格/失敗/警告件数':f'{passed[1]}合格、0失敗、{passed[2]}警告、{passed[3]}秒。警告は既存Pfテストのmultiprocessing forkに関するDeprecationWarning。',
'最終ソース・モデルのハッシュ記録':'`results/audited_symbiosis_shared_oxygen_20260908/design.json`、`sources/`、`execution_source_verification.json`。実GEMの短時間確認は `actual_model_verification.json`。',
'修正後の完了/失敗ケース数':'新しい共有酸素方式で23/23ケース完了。採用結果に未解決の停止なし。前段の失敗・補正試行・酸素予約方式の23ケースは別フォルダに保存。',
'時間刻み収束・同一操作での制御間隔比較':f"基準のPHA刻み半減差は2種 {base['two_change_percent']:+.3f}%、3種 {base['three_change_percent']:+.3f}%。低kLaは0.025→0.0125 hで {low['two_change_percent']:+.3f}% / {low['three_change_percent']:+.3f}%、さらに0.00625 hで {third['two_equal_total']['pha_change_percent']:+.3f}% / {third['three']['pha_change_percent']:+.3f}%。本GEM比較は制御間隔1h固定。入力・報酬の分割不変性は別の単体試験。数学的収束を証明したとはしない。",
'酸素・供給・貯蔵・ゴム分解部分の収支検査':f"23ケースで合格。酸素連立残差最大 {maxima['oxygen_kinetic_residual']:.3g} mmol/L、LP残差最大 {maxima['lp_residual']:.3g}、ゴム分解C5当量誤差最大 {maxima['polymer_c5_error_mmol_l']:.3g} mmol/L。記録した平均DOからkLa×(Cs−DO)を積分して酸素移動量も照合。",
'同一総接種量の2種対照との生産量比較':f"基準細刻みで3種PHA差 {base['gain_percent']:+.3f}%。低kLaの最細刻みで {summary['third_step_pha_gain_percent']:+.3f}%。全5条件の比較は新結果のREPORT_JA.mdとpaired_comparison.csv。",
'Pfの機能・代謝停止対照・単独/部分構成の比較':f"基準NS21単独PHA {ns['pha']:.6f} g/L。低kLaで代謝停止Pf対照に対する活性PfのPHA差 {ph:+.3f}%。菌量配分やPHA/ゴム除去/3HV品質を区別する。交換量はintegrated_exchanges.csv。",
'確認された3種優位条件と未解決の解釈':f"0.025/0.0125 hの両刻みで5%超の探索基準を満たす条件：{summary['conditions_passing_screen_at_both_steps'] or 'なし'}。独立実験や安定共生の証明ではない。生理・培地・維持代謝の未確認事項は残る。"
}
path=ROOT/'docs/CULTIVATION_MODEL_REPAIR_20260908.md';text=path.read_text()
text=text.replace('最終テスト件数と修正後の再計算結果は末尾に追記する。途中の追加変更前に得られた暫定252件という件数を、最終コードの検証結果として扱わない。',
    '最終ソースでのテストと23ケースの再計算結果を末尾に記録した。途中の旧版に対する検証件数と区別する。')
text=text.replace('この変更は同一LPの解法変更とは異なるため、全比較を新しい同一ソースで再計算する。',
    'この変更は同一LPの解法変更とは異なるため、全比較を新しい同一ソースで再計算した。')
text=text.replace('最終的には通常の `AuditedDFBASimulator` へ統合し、通常版と補正対照の12h全時系列の一致を検証する。方策契約にも修正リビジョンを含める。',
    '通常の `AuditedDFBASimulator` へ統合し、その時点の酸素予約方式で通常版と補正対照の12h全時系列の一致を検証した。これは次節の共有酸素方式への変更前の同等性検査である。方策契約にも修正リビジョンを含めた。')
text=text.replace('以下は最終ソースを固定した検証・再計算の完了後に記入する。未記入を合格、0件、優位なしという意味には扱わない。',
    '以下は共有酸素方式の実行済み検証記録。実装の検査と生理モデルの校正・実験での共生検証は区別する。')
for label,value in table.items():
    pattern=r'^\|'+re.escape(label)+r'\|.*\|$'
    text,count=re.subn(pattern,lambda _: '|'+label+'|'+value+'|',text,flags=re.M)
    assert count==1,label
text=text.replace('全23ケースを `results/audited_symbiosis_shared_oxygen_20260908` へ同じ新ソースで再計算する。',
                  '全23ケースを `results/audited_symbiosis_shared_oxygen_20260908` へ同じ新ソースで再計算した。')
path.write_text(text)
(AUDIT/'CURRENT_STATUS_JA.md').write_text(f'''# 培養モデル監査の完了記録

2026-09-08。問題一覧化、実装の修正、修正後の3種共培養の再評価を完了した。

- 修正前の41項目は `docs/CULTIVATION_MODEL_AUDIT_20260908.md` に保存。
- 各項目の修正・未解決の生理仮定は `docs/CULTIVATION_MODEL_REPAIR_20260908.md` に対応付けた。
- `shared_endpoint_v3` の同一ソースで23/23ケースが完了。最終テストは{passed[1]}合格、0失敗、{passed[2]}警告。
- 最終結果は `results/audited_symbiosis_shared_oxygen_20260908/REPORT_JA.md`。ソース一致と全ケースの収支・LP・供給・貯蔵の検査は同フォルダの `integrity.json`。
- 低kLaで最細刻みの3種PHA差は主対照2種に対して{summary['third_step_pha_gain_percent']:+.3f}%。工学的候補基準を両刻みで満たした条件は{summary['conditions_passing_screen_at_both_steps'] or 'なし'}。統計的有意差や安定共生の実証ではない。
- 旧酸素予約方式の23ケースと失敗試行は保存。新結果へ混ぜていない。
- 旧512軌道の教師収集は再開していない。新参照でのGNN+GRU教師収集は未接続であり、今回の共培養再評価の完了とは区別する。

未解決事項は、NH4以外の窒素源、Pfの酸素応答、PHV生成条件、B12受容、維持代謝・死滅、背景炭素からのPHA生成、実測との校正など。根拠のない値に変更して3種を有利にしていない。
''',encoding='utf-8')
old_status=ROOT/'results/audited_symbiosis_20260908/POST_QUOTA_AUDIT_STATUS.md'
old_status.write_text(old_status.read_text().replace('に再計算中。','に再計算済み。')+
    '\n新方式の最終結果は [再評価報告書](../audited_symbiosis_shared_oxygen_20260908/REPORT_JA.md) を参照。\n',encoding='utf-8')
(AUDIT/'FINAL_COMPLETION.json').write_text(json.dumps(dict(test_passed=int(passed[1]),warnings=int(passed[2]),
    cases=23,oxygen_scheme='shared_endpoint_v3',report=str((OUT/'REPORT_JA.md').relative_to(ROOT)),
    sources_after_final_checks={file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in design['sources']}),indent=2))
print('Repair document completed from final verified results.')
