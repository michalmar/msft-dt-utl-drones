print()
# Check core SDK version number
import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace,Datastore

print("SDK version:", azureml.core.VERSION)


REGISTER_MODEL = False

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
experiment_name = 'drones-yolo3-clean'
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



# xrun.download_file(name="outputs/trained_weights_final.h5", output_file_path="outputs")
import os
# download model -> download rather to blob
if not(os.path.exists(os.path.join("outputs","trained_weights_final.h5"))):
    print(f"downloading model from experiment {exp.name}...")
    xrun.download_file(name="outputs/trained_weights_final.h5", output_file_path="outputs")
    print("done.")
else:
    print(f'target file already exists {os.path.exists(os.path.join("outputs","trained_weights_final.h5"))} - download skipped')


########################################################################################



# Register model
from azureml.core.model import Model

if (REGISTER_MODEL):
    print(f"registering model...")
    model = Model.register(model_path = "outputs/trained_weights_final.h5",
                        model_name = "dronesv2.h5",
                        tags = {'type': "yolov3"},
                        description = "DT Utility - Drones Demo - V2",
                        workspace = ws)

# print(model.name, model.description, model.version, sep = '\t')

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

USE_GPU = True
if (USE_GPU):
        
    # myenv = CondaDependencies.create(conda_packages=['cudatoolkit==9.0','cudnn=7.1.2','numpy==1.14.5','keras==2.2.4','Pillow','matplotlib'])
    myenv = CondaDependencies.create(conda_packages=['numpy','keras','Pillow','matplotlib'])
    myenv.add_tensorflow_pip_package(core_type="gpu", version="1.13.1")

    with open(os.path.join(script_folder,"myenv-gpu.yml"),"w") as f:
        f.write(myenv.serialize_to_string())

else:
        
    myenv = CondaDependencies.create(conda_packages=['numpy','keras','Pillow','matplotlib'])
    myenv.add_tensorflow_pip_package(core_type="cpu", version=None)

    with open(os.path.join(script_folder,"myenv-cpu.yml"),"w") as f:
        f.write(myenv.serialize_to_string())


USE_GPU = True
from azureml.core.image import Image, ContainerImage
if (USE_GPU):
    print("GPU Image")
    # the my-gpu:latest image must exist in ACR -> it has been done manually
    image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="score_deploy.py",
                                #  conda_file="aml_deploy_prj/myenv-gpu.yml",
                                 dependencies=["aml_deploy_prj", "font"],
                                 tags = {'type': "yolov3"},
                                 base_image="dtutldronesv588b2bf1.azurecr.io/my-gpu:latest",
                                #  docker_file = "./aml_deploy_prj/docker_file_steps_gpu.txt",
                                 description = "Image with Drones detection model")
    # image_config.base_image = xrun.properties["AzureML.DerivedImageName"]
    image = Image.create(name = "dronesv2-img-gpu",
                     # this is the model object. note you can pass in 0-n models via this list-type parameter
                     # in case you need to reference multiple models, or none at all, in your scoring script.
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

else:
    print("CPU Image")
    image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="score_deploy.py",
                                 conda_file="aml_deploy_prj/myenv-cpu.yml",
                                 dependencies=["aml_deploy_prj", "font"],
                                 tags = {'type': "yolov3"},
                                 description = "Image with Drones detection model")


    image = Image.create(name = "dronesv2-img",
                     # this is the model object. note you can pass in 0-n models via this list-type parameter
                     # in case you need to reference multiple models, or none at all, in your scoring script.
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)



image.wait_for_creation(show_output = True)





# ################ AKS #########################################
# from azureml.core.compute.aks import AksCompute
# from azureml.core.webservice.aks import AksWebservice
# from azureml.core.webservice import Webservice

# aks_target = AksCompute(ws,"aks-gpu1")
# aks_service_name = "drone-aks-svc-gpu-2"
# # If deploying to a cluster configured for dev/test, ensure that it was created with enough
# # cores and memory to handle this deployment configuration. Note that memory is also used by
# # things such as dependencies and AML components.
# aksconfig = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 4)

# aks_service = Webservice.deploy_from_image(deployment_config = aksconfig,
#                                            deployment_target=aks_target,
#                                            image = image,
#                                            name = aks_service_name,
#                                            workspace = ws)
# aks_service.wait_for_deployment(True)
# print(aks_service.state)

# print(aks_service.get_logs())



# ##########ACI##################################################################

# from azureml.core.model import InferenceConfig

# inference_config = InferenceConfig(source_directory="aml_deploy_prj",
#                                    runtime= "python", 
#                                    entry_script="score.py",
#                                 #    conda_file="myenv-cpu.yml"#, 
#                                    #extra_docker_file_steps="helloworld.txt"
#                                    )



# from azureml.core.webservice import AciWebservice, Webservice
# from azureml.exceptions import WebserviceException

# deployment_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=4   )

# aci_service_name = 'drone-aci-svc-gpu'
# aci_service = Webservice.deploy_from_image(deployment_config = deployment_config,
#                                         #    deployment_target=aks_target,
#                                            image = image,
#                                            name = aci_service_name,
#                                            workspace = ws)
# aci_service.wait_for_deployment(True)
# print(aci_service.state)

# print(aci_service.get_logs())


# aci_service_name = 'drone-aci-svc6'

# try:
#     # if you want to get existing service below is the command
#     # since aci name needs to be unique in subscription deleting existing aci if any
#     # we use aci_service_name to create azure aci
#     service = Webservice(ws, name=aci_service_name)
#     if service:
#         print(f"OK: Found service {aci_service_name} in state: {service.state} - deleting")
#         service.delete()
# except WebserviceException as e:
#     print(f"OK: service {aci_service_name} doesn't exist - no need to delete")

# service = Model.deploy(ws, aci_service_name, [model], inference_config, deployment_config)

# service.wait_for_deployment(True)
# print(service.state)

# if service.state != 'Healthy':
#     # run this command for debugging.
#     print(service.get_logs())
#     service.delete()

# # print(service.get_logs())





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

# #######################################################################################
# # AKS GPU
# # https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-inferencing-gpus
# # WORKING BUT SLOW????

# from azureml.core.compute import ComputeTarget, AksCompute
# from azureml.exceptions import ComputeTargetException

# # Choose a name for your cluster
# aks_name = "aks-gpu1"

# # Check to see if the cluster already exists
# try:
#     aks_target = ComputeTarget(workspace=ws, name=aks_name)
#     print('Found existing compute target')
# except ComputeTargetException:
#     print('Creating a new compute target...')
#     # Provision AKS cluster with GPU machine
#     prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6s_v3")

#     # Create the cluster
#     aks_target = ComputeTarget.create(
#         workspace=ws, name=aks_name, provisioning_configuration=prov_config
#     )

#     aks_target.wait_for_completion(show_output=True)


# from azureml.core.webservice import AksWebservice

# gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,
#                                                     num_replicas=3,
#                                                     cpu_cores=2,
#                                                     memory_gb=4)




# from azureml.core.model import InferenceConfig


# inference_config = InferenceConfig(source_directory="aml_deploy_prj",
#                                    runtime= "python", 
#                                    base_image="dtutldronesv588b2bf1.azurecr.io/my-gpu:latest",
#                                    entry_script="score.py",
#                                 #    conda_file="myenv-gpu.yml",
#                                    enable_gpu=True
#                                    )



# # Name of the web service that is deployed
# aks_service_name = 'drones-aks-gpu3'

# # Deploy the model
# aks_service = Model.deploy(ws,
#                            models=[model],
#                            inference_config=inference_config,
#                            deployment_config=gpu_aks_config,
#                            deployment_target=aks_target,
#                            name=aks_service_name)

# aks_service.wait_for_deployment(show_output=True)
# print(aks_service.state)

