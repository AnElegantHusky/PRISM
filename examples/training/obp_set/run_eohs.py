import os
import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eohs import EoHS,EoHSProfiler


def main():

    llm = HttpsApi(host=os.environ['LLM_API_HOST'],
                   key=os.environ['LLM_API_KEY'],
                   model=os.environ['LLM_MODEL'],
                   timeout=60)

    task = OBPSEvaluation(
        timeout_seconds=120,
        dataset='./dataset_100_2k_128_5_80_training.pkl',
        return_list=True)

    method = EoHS(llm=llm,
                 profiler=EoHSProfiler(log_dir='logs/eohs', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=2000,
                 max_generations=1000,
                 pop_size=10,
                 num_samplers=4,
                 num_evaluators=4,
                 debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
