###############################################################################
# Michal Marusan
#
# Usage:
#   python aml_setup.py
###############################################################################


import os
import warnings
import argparse


# Check core SDK version number
import azureml.core
import yaml
from azureml.core import (Datastore, Environment, Experiment, ScriptRunConfig,
                          Workspace)
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget

warnings.filterwarnings("ignore")
print("SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser(description='AML Service Workspace setup.',argument_default=argparse.SUPPRESS)
parser.add_argument('--config_file', type=str, dest='cfg_file', help='Set specific config file.', default="aml_setup.yml")
args = parser.parse_args()

cfg_file = args.cfg_file

###############################################################################
# Fetch YAML config
###############################################################################
    # YAML file for setup AML environment
    # 
    # Environment
    #   subscription_id: ***
    #   tenant_id: ***
    #   resource_group: ***
    #   workspace_name: ***
    #   workspace_region: ***
    #
    # Datasources:
    #   datasource1:
    #     name: ***
    #     container_name: ***
    #     account_name: ***
    #     account_key: ***
    #   ...
    #
    # ProjectFolder:
    #   folder: aml_prj
    #
    # Computes:
    #   AMLCompute1:
    #     name: ***
    #     vm_size : *** 
    #     vm_priority : ***
    #     min_nodes : 0
    #     max_nodes : 1
    #     idle_seconds_before_scaledown: 1200
    #
    #   ...

with open(cfg_file) as f:
    amlsetup = yaml.safe_load(f)

###############################################################################
# Login to Azure
###############################################################################
interactive_auth = InteractiveLoginAuthentication(tenant_id=amlsetup["Environment"]["tenant_id"])
# sp_auth = ServicePrincipalAuthentication(tenant_id="***", service_principal_id="***", service_principal_password="***", _enable_caching=False)

###############################################################################
# Create workspace
###############################################################################
print(f"Setting up workspace:")


subscription_id = os.getenv("SUBSCRIPTION_ID", default=amlsetup["Environment"]["subscription_id"])
resource_group = os.getenv("RESOURCE_GROUP", default=amlsetup["Environment"]["resource_group"])
workspace_name = os.getenv("WORKSPACE_NAME", default=amlsetup["Environment"]["workspace_name"])
workspace_region = os.getenv("WORKSPACE_REGION", default=amlsetup["Environment"]["workspace_region"])

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("\tWorkspace configuration succeeded. Skip the workspace creation steps below")
except:
    print(f"\tWorkspace {workspace_name} doesn't exist - creating...")
    # Create the workspace using the specified parameters
    ws = Workspace.create(name = workspace_name,
                        subscription_id = subscription_id,
                        resource_group = resource_group, 
                        location = workspace_region,
                        create_resource_group = True,
                        exist_ok = False)
    ws.get_details()
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()

###############################################################################
# Create & Attach datastores
###############################################################################

print(f"Setting up datasources:")
# read from YAML and create
for k,v in amlsetup["Datasources"].items():
    # ds = ws.get_default_datastore()
    dsname = v["name"]
    if (dsname in ws.datastores):
        print(f"\tfound existing datastore: {dsname}")
        ds = ws.datastores[dsname]
    else:
        print(f"\tcreating new datastore: {dsname}...")
        ds = Datastore.register_azure_blob_container(workspace=ws,
                                                datastore_name=dsname,
                                                container_name=v["container_name"],
                                                account_name=v["account_name"],
                                                account_key=v["account_key"],
                                                create_if_not_exists=True)

    for attr, value in ds.__dict__.items():
        if (attr in ['name', 'datastore_type', 'container_name', 'account_name']):
            print(f"\t\t{attr}: {value}")
print("\tstorage initialized.")



###############################################################################
# Create project folder
###############################################################################
    
script_folder = os.path.join(os.getcwd(), amlsetup["ProjectFolder"]["folder"])
os.makedirs(script_folder, exist_ok=True)


###############################################################################
# Create & Attach Compute targets
###############################################################################
print(f"Setting up compute targets:")   

i=0
for k,aml_cmp in amlsetup["Computes"].items():
    # print(v)
# for aml_cmp in aml_computes_setup:
    try:
        compute_target = ComputeTarget(workspace=ws, name=aml_cmp["name"])
        print(f'\tFound existing compute target: {aml_cmp["name"]}')
    except:
        print(f'\tCompute target {aml_cmp["name"]} NOT FOUND - creating...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = aml_cmp["vm_size"], 
                                                                    vm_priority = aml_cmp["vm_priority"],
                                                                    min_nodes = aml_cmp["min_nodes"],
                                                                    max_nodes = aml_cmp["max_nodes"],
                                                                    idle_seconds_before_scaledown=aml_cmp["idle_seconds_before_scaledown"],
                                                                    #admin_username="vfn",
                                                                    #admin_user_password="******"
                                                                    )


        compute_target = ComputeTarget.create(ws, aml_cmp["name"], provisioning_config)

        # Can poll for a minimum number of nodes and for a specific timeout.
        # If no min_node_count is provided, it will use the scale settings for the cluster.
        compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)   

        # Use the 'status' property to get a detailed status for the current cluster. 
        cts = compute_target.status.serialize()
        print(f'\t(running)currentNodeCount: {cts["currentNodeCount"]}, vmPriority:{cts["vmPriority"]}, vmSize: {cts["vmSize"]}')
        i+=1

print(f'\tCompute initialized. Newly created {i} of {len(amlsetup["Computes"].items())} instace(s).')


print()
print(f"Azure Machine Learning service Workspace Environment for '{workspace_name}' setup completed.")
print()
