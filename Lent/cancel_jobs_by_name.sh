#!/bin/bash
user="cxw22"
job_name="t-cifar"

for job_id in $(squeue -u $user -o "%i %j" --noheader | grep "$job_name" | awk '{print $1}'); do
	scancel $job_id
done
