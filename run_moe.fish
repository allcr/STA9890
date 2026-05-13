#!/usr/bin/env fish


set total_iters 2

uv run data_processing.py

for i in (seq 1 $total_iters)
    echo "== [iter $i] starting at "(date '+%H:%M:%S')

    uv run resnet_moe.py --mode tune --tier global
    uv run lightgbm_moe.py --mode tune --tier global

    uv run lightgbm_moe.py --mode tune --tier region
    uv run lightgbm_moe.py --mode tune --tier district_type
    uv run lightgbm_moe.py --mode tune --tier macro
    uv run lightgbm_moe.py --mode tune --tier assessment
    uv run lightgbm_moe.py --mode tune --tier subgroup
    uv run lightgbm_moe.py --mode tune --tier county

    uv run catboost_moe.py --mode tune --enable-tune --tier region
    uv run catboost_moe.py --mode tune --enable-tune --tier district_type
    uv run catboost_moe.py --mode tune --enable-tune --tier assessment
    uv run catboost_moe.py --mode tune --enable-tune --tier macro
    uv run catboost_moe.py --mode tune --enable-tune --tier county
    uv run catboost_moe.py --mode tune --enable-tune --tier subgroup    
    
    if test $i -eq $total_iters
        uv run lightgbm_moe.py --mode stack --force
        uv run catboost_moe.py --mode stack --force
        uv run resnet_moe.py --mode stack --force 
        uv run ft_transformer_moe.py --mode stack --force
        uv run moe_submission.py
        cp submission_meta_top3_ensemble.csv submission_(date '+%Y%m%d_%H%M%S').csv
        # uv run ft_transformer_simple.py
    end

    echo "iter $i completed at "(date '+%H:%M:%S')
end

echo ""
echo "==> all stages complete at "(date '+%H:%M:%S')
