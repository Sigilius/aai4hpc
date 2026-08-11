"""
single_agent_baseline/run_suite.py

Single-agent baseline: fresh run per query.
"""

from tqdm.autonotebook import tqdm
import sys
sys.path.insert(0, "single_agent_baseline")
from agent import SingleAgent


def run_suite(label: str, queries: list[str], verbose=False):
    sep = "=" * 62
    print(f"\n{sep}\n  {label}\n{sep}")
    
    agent = SingleAgent(verbose=verbose)
    
    for q in tqdm(queries):
        print(f"\nQuery: {q}")
        print("-" * 62)
        result = agent.run(q)
        print(f"Answer: {result}")
        print()

# ── Eval queries ──────────────────────────────────────────────────
# SQL_QUERIES = [
#     "How many jobs ran in 2022-01?",
#     "How many large jobs (more than 384 nodes) ran in 2023-02?",
#     "How many jobs did user usr_1898 submit in total?",
#     "How many compute-bound jobs are in the dataset?",
#     "How many distinct users submitted jobs to Fugaku?",
#     "How many jobs failed on Fugaku?",
#     "What is the average number of nodes requested per job?",
#     "What is the average power consumption of large jobs (more than 384 nodes)?",
#     "What is the average number of nodes used by memory-bound jobs?",
#     "What is the average job duration in seconds?",
#     "Who are the top 3 users by total number of jobs submitted?",
#     "Which 3 users consumed the most total energy?",
#     "What are the top 5 most common job names submitted by usr_2111?",
#     "How many jobs were submitted for each job class (pclass)?",
#     "What is the monthly job submission count for 2023?",
#     "What is the distribution of jobs by node size bucket?",
#     "How many jobs used exactly 6 nodes?",
#     "What percentage of jobs requested more than 384 nodes?",
#     "What is the maximum number of nodes ever used by a single job?",
#     "How many jobs were submitted using jobenv_req = 'jobenv_req_0'?",
#     "What is the average wait time in 2023?",
#     "Which 5 users submitted the most memory-bound jobs?",
#     "What is the average energy consumption for failed jobs?",
#     "How many jobs ran with exactly 1 node?",
#     "What is the monthly average job duration for 2023?"
# ]

# PA_QUERIES = [
#     "Will a 4-node compute-bound job running for 1 hour fail?",
#     "What is the failure risk for a 16-node memory-bound job with a 2-hour walltime?",
#     "Is it safe to submit a 128-node compute-bound job overnight?",
#     "Estimate the energy consumption of a 256-node memory-bound job running for 4 hours.",
#     "What is the expected runtime for a 512-node compute-bound job with a 12-hour walltime?",
#     "What is the failure risk for usr_1229 submitting a 512-node compute-bound job for 24 hours?",
#     "Will usr_1229's 8-node memory-bound 1-hour job fail?",
#     "Should usr_1229 submit a 64-node compute-bound job for 6 hours?",
#     "What is the failure probability for usr_1898 submitting a typical 4-node job for 2 hours?",
#     "Predict the runtime of a 32-node memory-bound job submitted by usr_1898 with a 3-hour walltime.",
#     "Is it safe for usr_1898 to submit a 1024-node compute-bound job for 8 hours?",
#     "What is the risk for a single-node compute-bound job running for 30 minutes?",
#     "Estimate failure risk for a 4096-node memory-bound job running for 24 hours.",
#     "Will a 384-node compute-bound job running for exactly 6 hours fail?",
#     "Estimate the energy consumption of a compute-bound job on 1000 nodes running for 2 hours.",
#     "How long will a 64-node memory-bound job with a 4-hour walltime actually run?",
#     "What is the expected energy usage for usr_2111 running a 128-node memory-bound job for 3 hours?",
#     "Is it safe to submit a large 256-node overnight job?",
#     "I want to run an MD simulation on 512 nodes for 8 hours, will it succeed?",
#     "How risky is a 2048-node memory-bound job for usr_1916 with a 12-hour walltime?",
#     "What is the failure probability for a 1024-node memory-bound job with a 6-hour walltime?",
#     "Predict the risk level for usr_2111 running a 32-node compute-bound job for 90 minutes.",
#     "How much energy will a 1-node memory-bound test job running for 10 minutes consume?",
#     "Is there high risk for a 2048-node compute-bound job submitted by a new user for 12 hours?",
#     "What is the expected runtime and failure risk for usr_1916's 128-node memory-bound job with 2-hour walltime?"
# ]

# DOC_QUERIES = [
#     "What is the purpose of setting up a VNC server in the Pre/Post Environment?",
#     "How do I compile a C program with OpenMP using the Arm Compiler on a login node?",
#     "What happens when both SINGULARITYENV and %environment are not set?",
#     "How do you execute a benchmark scene in POV-Ray using Spack?",
#     "Can a program compiled by the Fujitsu compiler be executed on the pre/post environment?",
#     "What command is used to compile the sample SSL2 Fortran program?",
#     "How can I install the uberftp command on Windows or Mac?",
#     "What happens if the OMP_NUM_THREADS and PARALLEL environment variables do not match when using the Fujitsu OpenMP library?",
#     "How can I view the resource consumption for a specific theme in Fugaku?",
#     "What does the '-g' option specify in the resourceinfo command?",
#     "What does the '#PJM -L \"node=8\"' line specify in the job script?",
#     "What happens if the value specified in `--mpi proc` is larger than the value specified in `--mpi shape` x 48?",
#     "What happens if the power consumption exceeds the set value during job execution on Fugaku?",
#     "What does the directive '#PJM -L \"node=2x2\"' specify in a job script?",
#     "What is the purpose of the configuration file in VeloC?",
#     "What command can be used to check the total number of available and free nodes?",
#     "How do I specify a two-dimensional node shape in a Fugaku job script?",
#     "What is Open OnDemand in the context of the Fugaku website?",
#     "What does the -r option do in the globus-url-copy command?",
#     "What should I do if the Select Client Certificate dialog box appears on the Fugaku website?",
#     "What version of the CUDA Toolkit is used in the system configuration?",
#     "How can a system administrator display job information for all users?",
#     "How can I build proot and talloc for use with Singularity on FUGAKU?",
#     "How does Lens3 handle file access requests for buckets?",
#     "How can I make packages from the public instance available in my private instance using Spack?",
#     "How can I specify the transfer port range when transferring data from the Fugaku global file system to my server using SSH public-key authentication?",
#     "How can I upload a file to Fugaku using the Passenger Apps interface?",
#     "What should I do if a warning dialog appears during the FortiClient VPN installation on MacOS?",
#     "What was modified in the system environment on September 20, 2020?",
#     "How can variables and arrays be registered with VeloC in Fortran?"
# ]

MIXED_QUERIES = [
    "How many large jobs (more than 384 nodes) ran in February 2023, and what is the failure probability and expected wasted node-hours for a 4096-node memory-bound job running for 24 hours?",
    "How many jobs did user usr_1898 submit in total, and what is the predicted runtime for a 32-node memory-bound job submitted by the same user with a 3-hour walltime?",
    "What happens if the power consumption exceeds the set value during job execution on Fugaku, and what is the estimated energy consumption of a 256-node memory-bound job running for 4 hours?",
    "What is the average power consumption of large jobs (more than 384 nodes) on Fugaku, and how can I view the resource consumption for a specific theme?",
    "What is the expected runtime for a 512-node compute-bound job with a 12-hour walltime, how do I specify a two-dimensional node shape in a Fugaku job script, and what was the monthly job submission count for 2023?",
    "Is it safe to submit a 128-node compute-bound job overnight, what happens if --mpi proc exceeds --mpi shape × 48, and what percentage of jobs requested more than 384 nodes on Fugaku?",
    "What is the failure risk and associated metrics for usr_1229 submitting a 512-node compute-bound job for 24 hours, how many jobs used exactly 6 nodes on Fugaku, and what is the purpose of setting up a VNC server in the Pre/Post Environment?",
    "Will a 4-node compute-bound job running for 1 hour fail, and can a program compiled by the Fujitsu compiler be executed on the pre/post environment?",
    "What is the maximum number of nodes ever used by a single job on Fugaku, and what command can be used to check the total number of available and free nodes?",
    "How risky is a 2048-node memory-bound job for usr_1916 with a 12-hour walltime, what is the average number of nodes used by memory-bound jobs, and what is Open OnDemand in the context of the Fugaku website?",
    "What is the failure risk and failure probability for a 16-node memory-bound job with a 2-hour walltime, and what is the average energy consumption for failed jobs on the system?",
    "What is the failure probability for a 1024-node memory-bound job with a 6-hour walltime, how many jobs have failed on Fugaku overall, and what happens if power consumption exceeds the set limit during job execution?",
    "Should usr_1229 submit a 64-node compute-bound job for 6 hours given the predicted failure risk and expected runtime/energy, and what is the average wait time in 2023 on the system?",
    "What is the average number of nodes requested per job on Fugaku, and for a single-node compute-bound job running for 30 minutes, what are the expected runtime, energy consumption, failure probability, and wasted node-hours?",
    "What is the average number of nodes requested per job on Fugaku, and what does the directive '#PJM -L \"node=2x2\"' specify in a job script?",
    "How many jobs ran with exactly 1 node on Fugaku, and how does Lens3 handle file access requests for buckets?",
    "Is there high risk for a 2048-node compute-bound job submitted by a new user for 12 hours, and how can a system administrator display job information for all users?",
    "What is the expected runtime and failure risk for usr_1916's 128-node memory-bound job with 2-hour walltime, and how can I make packages from the public Spack instance available in my private instance?",
    "For usr_2111 running a 32-node compute-bound job for 90 minutes, what is the predicted risk level, failure probability, runtime, energy consumption, and wasted node-hours, and what system environment modification was made on September 20, 2020?",
    "What is the monthly average job duration for 2023 on Fugaku, and what version of the CUDA Toolkit is used in the system configuration?",
    "How many jobs were submitted using jobenv_req = 'jobenv_req_0', how can I build proot and talloc for use with Singularity on FUGAKU, and how much energy will a 1-node memory-bound test job running for 10 minutes consume?",
    "For a 64-node memory-bound job with a 4-hour walltime, how long will it actually run, which users have submitted the most memory-bound jobs on Fugaku, and how can variables and arrays be registered with VeloC in Fortran?",
    "For user usr_1898, how many total jobs have they submitted, and based on that context, what is the failure probability for a typical 4-node compute-bound job running for 2 hours, and is it safe for them to submit a 1024-node compute-bound job for 8 hours?",
    "What does the '#PJM -L \"node=8\"' and '#PJM -L \"node=2x2\"' directives specify in Fugaku job scripts, and what is the distribution of jobs across different node size buckets on the system?",
    "For memory-bound jobs on Fugaku, what is the average number of nodes used and the average job duration, and based on that, how long will a 64-node memory-bound job with a 4-hour walltime actually run?"
]

if __name__ == "__main__":
    # Define your test queries here
    run_suite("Single Agent Baseline", MIXED_QUERIES, verbose=False)