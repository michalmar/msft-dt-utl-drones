#!/usr/bin/env python
# coding: utf-8


import colorsys
import math
import os
import random
import re
import sys
import time

import azureml.core
import numpy as np
from azureml.core import (Datastore, Environment, Experiment, Run,
                          ScriptRunConfig, Workspace)
from azureml.core.authentication import InteractiveLoginAuthentication
print("SDK version:", azureml.core.VERSION)

# due to diferent tenant -> typically customer tenant
# interactive_auth = InteractiveLoginAuthentication(tenant_id="0f277086-d4e0-4971-bc1a-bbc5df0eb246")
interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47") # MSFT tenant
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')


exp = ws.experiments['drones-yolo3']


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


xrun = getAMLLastRun(exp)

# download model -> download rather to blob
if not(os.path.exists(os.path.join("outputs","trained_weights_final.h5"))):
    print(f"downloading model from experiment {exp.name}...")
    xrun.download_file(name="logs/000/trained_weights_final.h5", output_file_path="outputs")
    print("done.")
else:
    print(f'target file already exists {os.path.exists(os.path.join("outputs","trained_weights_final.h5"))}')

