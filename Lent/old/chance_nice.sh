#!/bin/bash

# Replace '<your_username>' with your actual username and '<new_nice_value>' with a positive integer (e.g., 1000)

your_username="cxw22"
new_nice_value="0"
job_name="t-cifar"

for job_id in $(squeue -u $your_username -t PENDING -o %i --noheader | grep "$job_name" | awk '{print $1}'); do
	scontrol update JobId=$job_id Nice=$new_nice_value
done
