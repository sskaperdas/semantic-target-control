# Public source consistency audit

Checks: 27
Failures: 0

- PASS — `source:scripts/preprocessing/preprocess_dbpedia.py`: required public source is present
- PASS — `source:scripts/preprocessing/preprocess_drugbank_xml.py`: required public source is present
- PASS — `source:scripts/preprocessing/preprocess_eurostatkg.py`: required public source is present
- PASS — `source:scripts/preprocessing/build_dbpedia_eval_subset.py`: required public source is present
- PASS — `source:scripts/training/train_base_kgc.py`: required public source is present
- PASS — `source:scripts/training/train_base_kgc_ddp.py`: required public source is present
- PASS — `source:scripts/training/train_schema_aware_portfolio_multigpu.py`: required public source is present
- PASS — `source:scripts/run_exp08_global_quota_return_semantics.py`: required public source is present
- PASS — `source:scripts/run_exp12_quota_matched_baselines.py`: required public source is present
- PASS — `source:scripts/run_exp14_factual_utility.py`: required public source is present
- PASS — `source:scripts/run_exp19_bootstrap_ci.py`: required public source is present
- PASS — `source:scripts/run_exp21_runtime_breakdown.py`: required public source is present
- PASS — `source:scripts/run_exp22_learned_reranker_baseline.py`: required public source is present
- PASS — `cli:scripts/preprocessing/preprocess_drugbank_xml.py:dataset-name`: defines --dataset-name
- PASS — `cli:scripts/preprocessing/preprocess_drugbank_xml.py:output-dir`: defines --output-dir
- PASS — `cli:scripts/preprocessing/preprocess_eurostatkg.py:dataset-name`: defines --dataset-name
- PASS — `cli:scripts/preprocessing/preprocess_eurostatkg.py:output-dir`: defines --output-dir
- PASS — `cli:scripts/training/train_base_kgc.py:dataset-name`: defines --dataset-name
- PASS — `cli:scripts/training/train_base_kgc.py:output-dir`: defines --output-dir
- PASS — `optq:zero-branch`: explicit zero-pressure branch
- PASS — `optq:b-plus-one`: selects the (b+1)-th largest crossing in zero-based indexing
- PASS — `optq:no-old-max-formula`: correct threshold-order-statistic construction is used
- PASS — `env:pykeen`: PyKEEN version recorded
- PASS — `env:torch-cu121`: historical PyTorch/CUDA lock recorded
- PASS — `env:no-nul`: environment lock is clean UTF-8
- PASS — `config:datasets`: all three canonical datasets are present
- PASS — `config:topm`: canonical Top-M values are recorded
