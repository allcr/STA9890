#!/usr/bin/env fish

uv run lightgbm_moe.py --mode tune --tier global
uv run lightgbm_moe.py --mode tune --tier region
uv run lightgbm_moe.py --mode tune --tier district_type
uv run lightgbm_moe.py --mode tune --tier macro
uv run lightgbm_moe.py --mode tune --tier assessment
uv run lightgbm_moe.py --mode tune --tier subgroup
uv run lightgbm_moe.py --mode tune --tier county

uv run catboost_moe.py --mode tune --enable-tune --tier global
#uv run catboost_moe.py --mode tune --enable-tune --tier region
#uv run catboost_moe.py --mode tune --enable-tune --tier district_type
#uv run catboost_moe.py --mode tune --enable-tune --tier assessment
#uv run catboost_moe.py --mode tune --enable-tune --tier macro
#uv run catboost_moe.py --mode tune --enable-tune --tier county
uv run catboost_moe.py --mode tune --enable-tune --tier subgroup

uv run lightgbm_moe.py --mode stack 
uv run catboost_moe.py --mode stack 
uv run resnet_moe.py --mode tune --tier global
uv run resnet_moe.py --mode stack 
uv run ft_transformer_moe.py --mode stack 
uv run moe_submission.py
cp submission_meta_top3_ensemble.csv submission_(date '+%Y%m%d_%H%M%S').csv

