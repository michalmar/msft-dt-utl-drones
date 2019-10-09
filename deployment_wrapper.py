print()
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
args = parser.parse_args()

# due to diferent tenant -> typically customer tenant
# interactive_auth = InteractiveLoginAuthentication(tenant_id="0f277086-d4e0-4971-bc1a-bbc5df0eb246") # VFN tenant
interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47") # MSFT tenant

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')




# ws.models
experiment_name = 'drones-yolo3'
exp = Experiment(workspace=ws, name=experiment_name)



from azureml.core import Run

def getAMLRun(exp, runId = None):
    runs = exp.get_runs()
    found = False
    if runId:
        for run in runs:
            xrun = run.get_details()
            if (xrun["runId"] == runId):
                # print("found")
                found = True
                # print(f'run: {xrun["target"]}')
                # found -> stop iteration
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


# runId = 'drones-yolo3_1570484309705'

# xrun = getAMLRun(exp, runId = 'gc-yolo3_1559594992_51dfe9be')
# xrun = getAMLRun(exp, runId)
xrun = getAMLLastRun(exp)

print(f"run:{xrun}")

xrun.download_file(name="outputs/trained_weights_final.h5", output_file_path="outputs")

########################################################################################



# Register model
from azureml.core.model import Model
model = Model.register(model_path = "outputs/ep024-loss246.116-val_loss246.895.h5",
                       model_name = "dronesv2.h5",
                       tags = {'type': "yolov3"},
                       description = "DT Utility - Drones Demo - V2",
                       workspace = ws)

print(model.name, model.description, model.version, sep = '\t')

try:
    print(model.name, model.description, model.version, sep = '\t')
except:
    ## get model from registered models (last version)
    # if model doesnt exists
    models = Model.list(workspace=ws)
    for m in models:
        print("Name:", m.name,"\tVersion:", m.version, "\tDescription:", m.description, m.tags)
    model = m
    print(model.name, model.description, model.version, sep = '\t')




from azureml.core.conda_dependencies import CondaDependencies 

import os
script_folder = os.path.join(os.getcwd(), "aml_deploy_prj")
os.makedirs(script_folder, exist_ok=True)

myenv = CondaDependencies.create(conda_packages=['numpy','keras','Pillow','matplotlib'])
myenv.add_tensorflow_pip_package(core_type="cpu", version=None)

with open(os.path.join(script_folder,"myenv-cpu.yml"),"w") as f:
    f.write(myenv.serialize_to_string())



from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="score.py",
                                 conda_file="aml_deploy_prj/myenv-cpu.yml",
                                 dependencies=["yolo.py", "yolo3","aml_deploy_prj", "font"],
                                 tags = {'type': "yolov3"},
                                 description = "Image with Drones detection model")

image = Image.create(name = "dronesv2-img",
                     # this is the model object. note you can pass in 0-n models via this list-type parameter
                     # in case you need to reference multiple models, or none at all, in your scoring script.
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)




# # List images by tag and find out the detailed build log for debugging.
# for i in Image.list(workspace = ws):
#     print('{}(v.{} [{}]) stored at {} with build log {}'.format(i.name, i.version, i.creation_state, i.image_location, i.image_build_log_uri))
# print()




################ AKS #########################################
from azureml.core.compute.aks import AksCompute
from azureml.core.webservice.aks import AksWebservice
from azureml.core.webservice import Webservice

aks_target = AksCompute(ws,"myaks-drone1")
aks_service_name = "drone-aks-svc"
# If deploying to a cluster configured for dev/test, ensure that it was created with enough
# cores and memory to handle this deployment configuration. Note that memory is also used by
# things such as dependencies and AML components.
aksconfig = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 2)

aks_service = Webservice.deploy_from_image(deployment_config = aksconfig,
                                           deployment_target=aks_target,
                                           image = image,
                                           name = aks_service_name,
                                           workspace = ws)
aks_service.wait_for_deployment(True)
print(aks_service.state)

print(aks_service.get_logs())



# ################ ACI - without IMAGE #########################################

# from azureml.core.model import InferenceConfig

# inference_config = InferenceConfig(runtime= "python",
#                                    entry_script="score.py",
#                                    conda_file="aml_deploy_prj/myenv-cpu.yml")

# from azureml.core.webservice import AciWebservice, Webservice
# from azureml.exceptions import WebserviceException

# deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

# aci_service_name = 'drone-aci-svc'

# service = Model.deploy(ws, aci_service_name, [model], inference_config, deployment_config)

# service.wait_for_deployment(True)
# print(service.state)

# print(service.get_logs())