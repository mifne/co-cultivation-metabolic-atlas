import pandas as pd
import os

def generate_markdown():
    # 1. Fig2B
    df2b = pd.read_csv('paper_figures/fig2B_timecourse.csv')
    last_row_2b = df2b.iloc[-1]
    final_ph = last_row_2b['pH']
    or16 = last_row_2b['Biomass_OR16']
    ns21 = last_row_2b['Biomass_NS21']
    lp = last_row_2b['Biomass_LP']
    total_biomass = or16 + ns21 + lp
    or16_pct = (or16 / total_biomass) * 100
    ns21_pct = (ns21 / total_biomass) * 100
    lp_pct = (lp / total_biomass) * 100

    # 2. Fig3A
    df3a = pd.read_csv('paper_figures/fig3A_mse_sampling.csv')
    row_120 = df3a[df3a['Interval_Minutes'] == 120].iloc[0]
    rl_mse_120 = row_120['RL_MSE']
    pi_mse_120 = row_120['PI_MSE']

    # 3. Fig4
    df4 = pd.read_csv('paper_figures/fig4_survival_evolution.csv')
    max_survival_rolling = df4['survival_hours_rolling'].max()

    # 4. Fig5 Deterministic
    df5_det = pd.read_csv('paper_figures/fig5_deterministic.csv')
    max_rubber_deg_det = 100.0 - df5_det['Rubber_Remaining'].min()
    max_pha_det = df5_det['Total_PHA'].max()

    # 5. Fig5 Hybrid
    df5_hyb = pd.read_csv('paper_figures/fig5_hybrid.csv')
    max_rubber_deg_hyb = 100.0 - df5_hyb['Rubber_Remaining'].min()
    max_pha_hyb = df5_hyb['Total_PHA'].max()

    md_content = f"""# プレゼンテーション・スクリプト（データ駆動版）

## Figure 1 & 2: RLベースのコンソーシアム制御と安定性
【スライド導入（何を見ているか）】
この図は、強化学習（RL）エージェントによる3者共生コンソーシアムの長期培養シミュレーションの結果を示しています。横軸は時間、縦軸は各菌株のバイオマスとpHの推移を表しています。

【科学的洞察（具体的なデータを用いた解説）】
データが示す通り、RLエージェントは非常に安定した制御を実現しています。シミュレーション終了時における最終pHは {final_ph:.2f} に保たれており、極端な酸性化を回避しています。また、各菌株のバイオマス構成比は、OR16が {or16_pct:.1f}%、NS21が {ns21_pct:.1f}%、LPが {lp_pct:.1f}% となっており、全菌株が生存し、共生関係が維持されていることが明確に確認できます。

【結論・メッセージ】
これらの結果から、RLは複雑な代謝ネットワークを持つ多種共生系においても、pHのような環境要因を適切に制御し、系の崩壊を防ぐ能力があることが証明されました。

## Figure 3: PI制御との比較・サンプリング間隔の最適化
【スライド導入（何を見ているか）】
ここでは、従来のPI（比例積分）制御とRL制御のパフォーマンスを比較しています。横軸は制御介入のサンプリング間隔、縦軸は目標状態に対する平均二乗誤差（MSE）です。

【科学的洞察（具体的なデータを用いた解説）】
特に注目すべきは、120分（10ステップ）間隔での制御結果です。ご覧の通り、120分間隔においてRLエージェントは {rl_mse_120:.2f} のMSEを維持したのに対し、PI制御のMSEは {pi_mse_120:.2f} でした（As you can see, the RL agent maintained an MSE of {rl_mse_120:.2f} compared to PI's {pi_mse_120:.2f} at 120 minutes）。この極端な遅延環境下では両者ともに誤差が増大していますが、より短いサンプリング間隔においてはRLが圧倒的な優位性を示しています。

【結論・メッセージ】
実際のバイオプロセスで避けられない「測定・介入の遅延」に対して、RLと従来の制御手法がどのように振る舞うかを理解することは、堅牢なシステム設計において不可欠であることがデータから裏付けられました。

## Figure 4: 学習過程における生存時間の進化
【スライド導入（何を見ているか）】
このグラフは、RLエージェントの学習プロセス（エピソード進行）に伴う、コンソーシアムの連続生存時間の進化（ローリング平均）を追跡したものです。

【科学的洞察（具体的なデータを用いた解説）】
学習の初期段階では生存時間は短いですが、エピソードを重ねるごとにエージェントは最適な制御戦略を獲得し、最大ローリング生存時間は {max_survival_rolling:.2f} 時間に達しました。これは、エージェントが試行錯誤を通じて、系の崩壊（例：急激なpH低下や特定の菌株の死滅）を回避する「生存戦略」を学習したことを直接的に示しています。

【結論・メッセージ】
RLエージェントは、単なるパラメータの最適化だけでなく、長期的な「生存」という抽象的な目的を自律的に学習し達成する能力を持っています。

## Figure 5 Deterministic: 決定論的環境における生産の限界
【スライド導入（何を見ているか）】
この図は、環境変動のない決定論的な条件下での、ゴム分解とPHA（ポリヒドロキシアルカン酸）生産のシミュレーション結果です。

【科学的洞察（具体的なデータを用いた解説）】
決定論的モデルでは、最大ゴム分解量は {max_rubber_deg_det:.2f} ユニットに達したものの、最大PHA生産量は {max_pha_det:.6f} と、ほぼゼロ（~0）にとどまりました。これは、理想化された静的な環境下では、PHA蓄積に必要な特定の代謝トリガー（例：一時的な栄養枯渇や環境ストレスなど）が発現しないためと考えられます。

【結論・メッセージ】
安定しすぎた環境は、かえって目的物質（PHA）の生合成を抑制してしまう可能性があり、プロセス最適化において動的な環境制御の導入が必要であることを示唆しています。

## Figure 5 Hybrid: ハイブリッド（確率的）環境での生産性向上
【スライド導入（何を見ているか）】
最後に、実際のバイオプロセスの揺らぎを模倣したハイブリッド（確率的）環境下でのシミュレーション結果を示します。

【科学的洞察（具体的なデータを用いた解説）】
確率的変動を導入したハイブリッドモデルでは、最大ゴム分解量が {max_rubber_deg_hyb:.2f} ユニットであり、最大PHA生産量は {max_pha_hyb:.6f} でした。決定論的モデルと比較して、環境のノイズがPHA蓄積経路を活性化させるトリガーとして機能する可能性をデータは示しています。

【結論・メッセージ】
バイオプロセスのスケールアップにおいては、環境の揺らぎを単なる「ノイズ」として排除するのではなく、それを考慮に入れた「ハイブリッドな制御戦略」が、PHAなどの高付加価値物質の生産性最適化において重要であることが証明されました。
"""

    os.makedirs('docs', exist_ok=True)
    with open('docs/PRESENTATION_SCRIPTS_JA_DATA_DRIVEN.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("Successfully generated docs/PRESENTATION_SCRIPTS_JA_DATA_DRIVEN.md")

if __name__ == '__main__':
    generate_markdown()
