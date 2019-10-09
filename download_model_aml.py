#!/usr/bin/env python
# coding: utf-8
###############################################################################
# Michal Marusan
#
# Usage:
#   python download_model_aml.py --exp_name gc-yolo3-test 
#   python download_model_aml.py --exp_name gc-yolo3-test --run_id gc-yolo3-test_1570647368_2e949e7c
###############################################################################

import colorsys
import math
import os
import random
import re
import sys
import time
import argparse

import azureml.core
import numpy as np
from azureml.core import (Datastore, Environment, Experiment, Run,
                          ScriptRunConfig, Workspace)
from azureml.core.authentication import InteractiveLoginAuthentication
print("SDK version:", azureml.core.VERSION)


parser = argparse.ArgumentParser(description='AML Service - get model from run.',argument_default=argparse.SUPPRESS)
parser.add_argument('--exp_name', type=str, dest='exp_name', help='exp_name...', default="drones-yolo3")
parser.add_argument('--run_id', type=str, dest='run_id', help='run_id...', default="#NA#")
# parser.add_argument('--prj_upd', help='update also aml project', action='store_true')
args = parser.parse_args()

run_id = args.run_id
exp_name = args.exp_name



# due to diferent tenant -> typically customer tenant
interactive_auth = InteractiveLoginAuthentication(tenant_id="0f277086-d4e0-4971-bc1a-bbc5df0eb246")

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')


exp = ws.experiments[exp_name]


def getAMLRun(exp, runId = None):
    runs = exp.get_runs()
    found = False
    if runId:
        for run in runs:
            xrun = run.get_details()
            if (xrun["runId"] == runId):
                found = True
                xrun = run
                break
        if (not found):
            xrun = None 
    # runId not set -> get last run
    else:
        xrun = next(runs)
    
    return xrun

def getAMLLastRun(exp):
    return getAMLRun(exp, None)


if (run_id == "#NA#"):
    print(f"getting last run from experiment: {exp_name}")
    xrun = getAMLLastRun(exp)
else:
    print(f"getting run by id ({run_id}) from experiment: {exp_name}")
    xrun = getAMLRun(exp, run_id)

# xrun = getAMLLastRun(exp)

print()
print(f"fetched run: {xrun}")

# download model -> download rather to blob
if not(os.path.exists(os.path.join("outputs","trained_weights_final.h5"))):
    print(f"downloading model from experiment {exp.name}...")
    xrun.download_file(name="outputs/trained_weights_final.h5", output_file_path="outputs")
    # xrun.download_file(name="outputs/ep024-loss246.116-val_loss246.895.h5", output_file_path="outputs")
    print("done.")
else:
    print(f'target file already exists {os.path.exists(os.path.join("outputs","trained_weights_final.h5"))} - download skipped')

