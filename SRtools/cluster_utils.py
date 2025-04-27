import sys
import os
import re
import subprocess

def get_failed_jobs(job_id, job_name):
    """
    This function gets the failed jobs from the job_id and job_name.
    it returns a list of tuples with the job index and the exit code.
    """
    # Get the list of ended jobs
    bjobs = f"bjobs -w {job_id}"
    output = os.popen(bjobs).read()
    lines = output.split("\n")
    failed_jobs = []
    for line in lines:
        if "EXIT" in line:
            job_index = re.search(rf'{job_name}\[(\d+)\]', line).group(1)
            #To get the exit codes the program will run bjobs -l {job id}[extracted job index] for each job
            bjobs_l = f"bjobs -l {job_id}[{job_index}]"
            output_l = os.popen(bjobs_l).read()
            lines_l = output_l.split("\n")
            for line_l in lines_l:
                if "Exited with exit code" in line_l:
                    exit_code = re.search(r'Exited with exit code (\d+)', line_l).group(1)
            failed_jobs.append((job_index, exit_code))
    return failed_jobs

def rerun_if_exit_codes(exit_codes, failed_jobs, run_params,size = 30):
    """
    This function reruns an array job with the indexes that failed with the specifide exit code.
    """
    codes_str = ",".join(exit_codes)
    print(f'rerun_if_exit_code {codes_str}')
    failed_jobs_code =[]
    new_job_ids = []
    if len(failed_jobs) > 0:
        failed_jobs_code = [job[0] for job in failed_jobs if job[1] in exit_codes]
        failed_jobs_strings = []
        #split to batches of 30 jobs:
        indexes = [i for i in range(0, len(failed_jobs_code), size)]
        if len(failed_jobs_code)%size>0:
            indexes.append(len(failed_jobs_code))
        for i in range(1,len(indexes)):
            failed_jobs_strings.append(",".join(failed_jobs_code[indexes[i-1]:indexes[i]]))
            print (f"Failed jobs {exit_codes}: {failed_jobs_strings[i-1]}")
        if len(failed_jobs_code) > 0:
            for failed_jobs_str in [failed_jobs_strings[0]]: #I only send one job. otherwise I get issues...
                print(f"Rerunning failed jobs {exit_codes}: {failed_jobs_str}")
                rerun = f"bsub -J \"{run_params['job_name']}[{failed_jobs_str}]\" -R 'rusage[mem={run_params['memory']}GB]' -oo {run_params['out_folder']}/%J_%I.o -eo {run_params['e_folder']}/%J_%I.e -q short {run_params['run_file']} {run_params['log_folder']} {run_params['config_path']} {run_params['results_folder']} "
                output = subprocess.run(rerun, shell=True, capture_output=True, text=True)
                #echo the output of the rerun command
                print('rerun command',rerun)
                print('output',output.stdout)
                os.system(f'echo "rerun failed jobs "')
                os.system(f'echo "{output.stdout}"')
                new_job_id = re.search(r'Job <(\d+)>', output.stdout).group(1)
                new_job_ids.append(new_job_id)
        else:
            print(f"No failed jobs with exit code {codes_str}")
            new_job_ids = None
    else:
        print(f"No failed jobs at all")
        new_job_ids = None

    return new_job_ids

def send_email(subject, message, receiver_email = "Naveh.Raz@weizmann.ac.il", when_a_jobe_ends = None):
    """
    This function sends an email to the user with the given subject and message.
    """
     # Create the email command
    email_command = f'echo \\"{message}\\" | mail -s \\"{subject}\\" {receiver_email}'

    # Submit the job with bsub
    if when_a_jobe_ends is not None:
        bsub_command = f'bsub -q short -N -u {receiver_email} -w "ended({when_a_jobe_ends})" "{email_command}"'
    else:
        bsub_command = f'bsub -q short -N -u {receiver_email} "{email_command}"'
    os.system(bsub_command)

def get_n_jobs(job_id):
    """
    This function gets the number of jobs that are in the array job.
    """
    bjobs = f"bjobs -w {job_id}"
    output = os.popen(bjobs).read()
    lines = output.split("\n")
    n_jobs = 0
    for line in lines:
        n_jobs += 1
    if n_jobs > 0:
        n_jobs -= 1
    return n_jobs