#!/usr/bin/env fish
set script_dir (dirname (status filename))
nohup fish $script_dir/run_moe.fish 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' >> moe.log & disown
