#!/usr/bin/env bash

echo "start ================="
oc exec -it anjali-torch-spyre-may14 -n cicd-project -- bash -c 'source /home/senuser/torch-spyre/.venv/bin/activate && export TORCH_ROOT=/home/senuser/pytorch && export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1  &&  export TORCH_SPYRE_DEBUG=1 && export TORCH_COMPILE_DEBUG=1 && cd /home/senuser/torch-spyre/tests && bash run_test.sh ./configs/upstream_tests/test_profiler_config.yaml'
echo $?
echo "END ==========:"
