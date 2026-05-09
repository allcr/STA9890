#!/usr/bin/env fish


set total_iters 5

for i in (seq 1 $total_iters)
    echo "== [iter $i] starting at "(date '+%H:%M:%S')

    if test $i -eq 1
       uv run resnet_moe.py --mode stack --force
       uv run lightgbm_moe.py --mode stack --force
       uv run catboost_moe.py --mode stack
       uv run ft_transformer_moe.py --mode stack
       uv run moe_submission.py
    end
    
    uv run resnet_moe.py --mode tune --tier global
    uv run resnet_moe.py --mode tune --tier region
    uv run resnet_moe.py --mode tune --tier district_type
    uv run resnet_moe.py --mode tune --tier macro
    uv run resnet_moe.py --mode tune --tier assessment
    uv run resnet_moe.py --mode tune --tier subgroup

    uv run lightgbm_moe.py --mode tune --tier global
    uv run lightgbm_moe.py --mode tune --tier region
    uv run lightgbm_moe.py --mode tune --tier district_type
    uv run lightgbm_moe.py --mode tune --tier macro
    uv run lightgbm_moe.py --mode tune --tier assessment
    uv run lightgbm_moe.py --mode tune --tier subgroup
    uv run lightgbm_moe.py --mode tune --tier county
    
    
   
    uv run lightgbm_moe.py --mode stack

    if test $i -eq $total_iters
        uv run ft_transformer_moe.py --mode tune --tier global
        uv run ft_transformer_moe.py --mode stack
        uv run resnet_moe.py --mode submit-global --n-seeds 10
        uv run resnet_moe.py --mode stack
        uv run moe_submission.py
    end

    echo "iter $i completed at "(date '+%H:%M:%S')
end

echo ""
echo "==> all stages complete at "(date '+%H:%M:%S')
