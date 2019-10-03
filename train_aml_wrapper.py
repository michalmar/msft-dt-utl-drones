###############################################################################
# Michal Marusan
#
# Usage:
#   python train_aml_wrapper.py --aml_compute amlgpu-low
#   python train_aml_wrapper.py --aml_compute amlgpu-ded --annotation_path 'annotations.txt' --log_dir 'logs/000/' --classes_path 'classes.txt' --anchors_path 'yolo_anchors.txt' --epochs_frozen 3 --epochs_unfrozen 4

#   python train_aml_wrapper.py --aml_compute amlgpu-ded
#   python train_aml_wrapper.py --aml_compute amlgpu-ded-new2
###############################################################################

# Check core SDK version number
import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace,Datastore

print("SDK version:", azureml.core.VERSION)

import argparse
# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument('--aml_compute', type=str, dest='aml_compute', help='Set specific AML Compute, using default if not set.', default="###")
parser.add_argument('--annotation_path', type=str, dest='annotation_path', help='path to training files annotation', default='vott-json-export-20190808/annotations.txt')
parser.add_argument('--log_dir', type=str, dest='log_dir', help='where logs and intermediate model are placed', default='logs/000/' )
parser.add_argument('--classes_path', type=str, dest='classes_path', help='path to training classes', default='vott-json-export-20190808/classes.txt') 
parser.add_argument('--anchors_path', type=str, dest='anchors_path', help='path to training anchors', default='vott-json-export-20190808/yolo_anchors.txt') 
parser.add_argument('--epochs_frozen', type=str, dest='epochs_frozen', help='epochs on frozen heads', default=33) 
parser.add_argument('--epochs_unfrozen', type=str, dest='epochs_unfrozen', help='epochs on unfrozen heads - all net', default=66) 

args = parser.parse_args()

# due to diferent tenant -> typically customer tenant
interactive_auth = InteractiveLoginAuthentication(tenant_id="0f277086-d4e0-4971-bc1a-bbc5df0eb246")

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')


# ws.datastores

dsname = 'gatecontrol_storage'
if (dsname in ws.datastores):
    print(f"found existing datastore: {dsname}")
    ds = ws.datastores[dsname]
else:
    print(f"datastore: {dsname} not found.")
    quit()

# experiment_name = 'gc-yolo3-clean'
experiment_name = 'gc-yolo3'
exp = Experiment(workspace=ws, name=experiment_name)

import os
script_folder = os.path.join(os.getcwd(), "aml_prj")
os.makedirs(script_folder, exist_ok=True)

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

amlcompute_cluster_name = "amlgpu-low" #Name your cluster

if args.aml_compute == "###":
    print(f"Using default AML Compute: {amlcompute_cluster_name}")
else:
    print(f"setting AML compute to: {args.aml_compute}")
    amlcompute_cluster_name = args.aml_compute

try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print(f"Found existing compute target: {amlcompute_cluster_name}")
except:
    print(f"compute target {amlcompute_cluster_name} not found...quit")
    quit()

# Use the 'status' property to get a detailed status for the current cluster. 
cts = compute_target.status.serialize()
print(f'(running)currentNodeCount: {cts["currentNodeCount"]}, vmPriority:{cts["vmPriority"]}, vmSize: {cts["vmSize"]}')

from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['numpy','keras','Pillow','matplotlib'])
# myenv.add_tensorflow_conda_package(core_type="gpu", version=None) # doesn’t work -> uses CPU
myenv.add_tensorflow_pip_package(core_type="gpu", version=None) # works -> uses GPU


with open("aml_prj/myenv-gpu.yml","w") as f:
    f.write(myenv.serialize_to_string())

from azureml.train.estimator import Estimator

script_params = {
    # '--data-folder': ds.path('vott-json-export-20190618').as_mount(),
    '--data-folder': ds.path('vott-json-export-20190808-merged').as_mount(),
    '--model-folder': ds.path('model_data').as_mount(),
    '--annotation_path': args.annotation_path,
    '--log_dir': args.log_dir,
    '--classes_path': args.classes_path, 
    '--anchors_path': args.anchors_path, 
    '--epochs_frozen': args.epochs_frozen, #50
    '--epochs_unfrozen': args.epochs_unfrozen # 150
}

est = Estimator(source_directory=script_folder,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                use_gpu=True,
                conda_dependencies_file_path="myenv-gpu.yml",
                # conda_packages=['numpy','tensorflow-gpu','keras','Pillow','matplotlib']
                max_run_duration_seconds=60*60*3 # 4 hours
                )


run = exp.submit(config=est)
print("submited.")
print(run)
