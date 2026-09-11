# Gene ID registry audit

GPR内では株別locus tagを使用し、`CCG_<organism>_<locus_tag>`をプロジェクト全体の一意キーとする。元のSBML/FASTA IDとNCBI protein accessionは別名表に保持する。

| Organism | Before | After | Renamed aliases | Merged duplicates | Retired IDs | Simplified GPRs | Quarantined history | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---|---|
| OR16 | 1596 | 1596 | 0 | 0 | 0 | 0 | none | none |
| NS21 | 1392 | 1392 | 0 | 0 | 0 | 0 | none | none |
| WCFS1 | 913 | 913 | 0 | 0 | 0 | 0 | lp1406 | none |

未解決IDは自動推測せず、配列または原著の根拠を追加してから設定ファイルで解決する。
