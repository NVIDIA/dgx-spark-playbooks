# Multi spark cluster setup script

## Usage

### Step 1. Clone the repo

Clone the dgx-spark-playbooks repo from GitHub

### Step 2. Switch to the multi spark cluster setup scripts directory

```bash
cd dgx-spark-playbooks/nvidia/multi-sparks-through-switch/assets/spark_cluster_setup
```

### Step 3. Create or edit a JSON config file with your cluster information

```bash
# Create or edit JSON config file under the `config` directory with the ssh credentials for your nodes.
# Adjust the number of nodes in "nodes_info" list based on the number of nodes in your cluster

# Example: (config/spark_config_b2b.json):
# {
#     "nodes_info": [
#         {
#             "ip_address": "10.0.0.1",
#             "port": 22,
#             "user": "nvidia",
#             "password": "nvidia123"
#         },
#         {
#             "ip_address": "10.0.0.2",
#             "port": 22,
#             "user": "nvidia",
#             "password": "nvidia123"
#         }
#
```

### Step 4. Run the cluster setup script with your json config file

The script can be run with different options as mentioned below

```bash
# To run validation, cluster setup and NCCL bandwidth test (all steps)

bash spark_cluster_setup.sh -c <JSON config file> --run-setup

# To only run pre-setup validation steps

bash spark_cluster_setup.sh -c <JSON config file> --pre-validate-only

# To run NCCL test and skip cluster setup (use this after cluster is already set up)

bash spark_cluster_setup.sh -c <JSON config file> --run-nccl-test

```

> [!NOTE]
> The full cluster setup (first command above) will do the following
> 1. Create a python virtual env and install required packages
> 2. Validate the environment and cluster config
> 3. Detect the topology and configure the IP addresses
> 4. Configure password-less ssh between the cluster nodes
> 5. Run NCCL BW test