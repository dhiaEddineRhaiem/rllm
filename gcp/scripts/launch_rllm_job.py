#!/usr/bin/env python
import os
import subprocess
import tempfile
import hashlib
import shutil
import yaml
from pathlib import Path
from fire import Fire

def cleanup_debug_jobs(namespace: str, job_prefix: str = "rllm-debug"):
    """Clean up any existing jobs that start with the given prefix"""
    print(f"\n[Pre-deploy] Checking for existing {job_prefix}* jobs to cleanup...")

    # List all helm releases in the namespace
    helm_list_output = run_command(
        f"helm list --namespace {namespace} --output json",
        check=False
    )

    if not helm_list_output:
        print(f"  No existing releases found in namespace {namespace}")
        return

    import json
    try:
        releases = json.loads(helm_list_output)
    except json.JSONDecodeError:
        print(f"  Could not parse helm list output")
        return

    # Filter releases that start with the prefix
    matching_releases = [r for r in releases if "rllm" in r['name']]

    if not matching_releases:
        print(f"  No {job_prefix}* jobs found")
        return

    print(f"  Found {len(matching_releases)} {job_prefix}* job(s) to cleanup:")
    for release in matching_releases:
        print(f"    - {release['name']} (status: {release['status']})")

    # Uninstall each matching release
    for release in matching_releases:
        release_name = release['name']
        print(f"  Uninstalling {release_name}...")
        try:
            run_command(
                f"helm uninstall {release_name} --namespace {namespace}",
                check=True
            )
            print(f"    ✓ Successfully uninstalled {release_name}")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed to uninstall {release_name}: {e}")
            # Continue with other releases even if one fails

    print(f"  Cleanup complete\n")

def run_command(command, check=True):
    """Run a shell command and return its output"""
    result = subprocess.run(
        command,
        shell=True,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 and check:
        print(f"Error running command: {command}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout.strip()


def generate_job_id():
    """Generate a random 8-character hash for JOB_ID"""
    random_bytes = os.urandom(32)
    job_id = hashlib.sha256(random_bytes).hexdigest()[:8]
    return job_id


def override_values_yaml(
    values_yaml_path: str,
    n_nodes: int,
    wandb_key_name: str,
    region: str,
):
    """Override values.yaml with runtime parameters"""
    with open(values_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Set number of nodes
    data["nodes"] = n_nodes

    # Set wandb secret
    data["secrets"]["WANDB_API_KEY"]["secret_name"] = wandb_key_name

    # Fix localhost for single node
    if n_nodes == 1:
        data["workload"]["script"] = data["workload"]["script"].replace(
            "$MASTER_ADDR", "localhost"
        )

    # Update checkpoint bucket region if needed
    for mounted_bucket in data["workload"]["volumes"]["gcsMounts"]:
        bucket_name = mounted_bucket["bucket"]
        if bucket_name.startswith("tii-aiccu-checkpoints") and region not in bucket_name:
            mounted_bucket["bucket"] = f"tii-aiccu-checkpoints-{region}"

    # Write to temporary file
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
    yaml.dump(data, temp, default_flow_style=False)
    temp.close()

    return temp.name


def main(
    rllm_code_path: str,
    config_yaml_path: str,
    values_yaml_path: str,
    charts_path: str,
    gcs_bucket_path: str,
    job_name: str,
    n_nodes: int = 1,
    wandb_key_name: str = "wandb-credentials",
    region: str = "us-central1",
    namespace: str = "falcon-mamba",
    cleanup_debug_jobs_first: bool = True,

):
    """
    Launch RLLM training job on Kubernetes

    Args:
        rllm_code_path: Local path to RLLM repository
        config_yaml_path: Path to training config (e.g., configs/qwen3-1b.yaml)
        values_yaml_path: Path to Helm values.yaml
        charts_path: Path to Helm charts directory
        gcs_bucket_path: GCS bucket path (e.g., gs://tii-aiccu-falcon-mamba-us-central1)
        job_name: Name for the job
        n_nodes: Number of nodes (default: 1)
        wandb_key_name: K8s secret name for wandb (default: wandb-credentials)
        region: GCP region (default: us-central1)
        namespace: K8s namespace (default: falcon-mamba)
    """

    # Generate job ID
    job_id = generate_job_id()
    full_job_name = f"{job_name}-{job_id}"

    # Remove trailing slash from GCS path
    gcs_bucket_path = gcs_bucket_path.rstrip("/")

    # Define GCS paths
    gcs_code_path = f"{gcs_bucket_path}/job-folders/job-{job_id}/rllm.tar.gz"
    gcs_config_path = f"{gcs_bucket_path}/job-folders/job-{job_id}/config.yaml"

    print("=" * 60)
    print(f"Launching RLLM Job")
    print("=" * 60)
    print(f"Job ID:       {job_id}")
    print(f"Job Name:     {full_job_name}")
    print(f"Nodes:        {n_nodes}")
    print(f"Region:       {region}")
    print(f"Namespace:    {namespace}")
    print(f"Config:       {config_yaml_path}")
    print("=" * 60)

    # Cleanup existing debug jobs if requested
    if "rllm" in job_name:
        cleanup_debug_jobs(namespace, job_prefix="rllm-debug")

    # 1. Compress RLLM codebase
    print("\n[1/6] Compressing RLLM codebase...")
    rllm_path = Path(rllm_code_path).resolve()
    parent_dir = rllm_path.parent
    dir_name = rllm_path.name

    tmp_tar_fd, tmp_tar_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(tmp_tar_fd)

    # FIXED: Properly exclude .git directories recursively but keep all source code
    print(f"  Creating tar from: {rllm_path}")
    print(f"  Checking for R2E-Gym directory...")
    r2e_gym_path = rllm_path / "R2E-Gym"
    if r2e_gym_path.exists():
        print(f"  ✓ R2E-Gym found at: {r2e_gym_path}")
    else:
        print(f"  ✗ WARNING: R2E-Gym not found at: {r2e_gym_path}")

    # Create tar excluding .git recursively and Python cache files
    # Note: Using --exclude pattern matches anywhere in the path
    tar_cmd = (
        f"tar "
        f"--exclude='*/.git' "
        f"--exclude='*/.git/*' "
        f"--exclude='*.pyc' "
        f"--exclude='*/__pycache__' "
        f"--exclude='*/__pycache__/*' "
        f"-czf {tmp_tar_path} "
        f"-C {parent_dir} {dir_name}"
    )
    print(f"  Running: {tar_cmd}")
    run_command(tar_cmd)

    # Verify tar contents
    print(f"  Verifying tar contents...")
    tar_list = run_command(f"tar -tzf {tmp_tar_path} | head -20")
    print(f"  First 20 entries in tar:\n{tar_list}")

    # Specifically check for R2E-Gym
    r2e_check = run_command(f"tar -tzf {tmp_tar_path} | grep -c 'R2E-Gym' || echo '0'", check=False)
    if int(r2e_check) > 0:
        print(f"  ✓ R2E-Gym IS included in tar ({r2e_check} entries)")
    else:
        print(f"  ✗ WARNING: R2E-Gym NOT found in tar!")
        print(f"  This may cause issues during deployment.")

    # 2. Upload code to GCS
    print(f"[2/6] Uploading code to GCS: {gcs_code_path}")
    run_command(f"gsutil cp {tmp_tar_path} {gcs_code_path}")
    os.remove(tmp_tar_path)

    # 3. Upload config to GCS
    print(f"[3/6] Uploading config to GCS: {gcs_config_path}")
    run_command(f"gsutil cp {config_yaml_path} {gcs_config_path}")

    # 4. Override values.yaml
    print("[4/6] Generating values.yaml...")
    tmp_values_path = override_values_yaml(
        values_yaml_path,
        n_nodes,
        wandb_key_name,
        region
    )

    # 5. Check for existing job and uninstall if found
    print(f"[5/6] Checking for existing job: {full_job_name}...")
    helm_list = run_command(f"helm list --namespace {namespace}", check=False)
    if full_job_name in helm_list:
        print(f"    Found existing job. Uninstalling...")
        run_command(f"helm uninstall {full_job_name} --namespace {namespace}")

    # 6. Deploy with Helm
    print(f"[6/6] Deploying job with Helm...")
    helm_cmd = (
        f"helm install {full_job_name} {charts_path} "
        f"-f {tmp_values_path} "
        f"--namespace {namespace} "
        f"--set workload.extra_env.TII_GCP_JOB_ID={job_id} "
        f"--set workload.extra_env.TII_RLLM_JOB_NAME={full_job_name}"
    )
    run_command(helm_cmd)

    # Cleanup
    os.remove(tmp_values_path)

    print("\n" + "=" * 60)
    print("✓ Deployment completed successfully!")
    print("=" * 60)
    print(f"Job Name:     {full_job_name}")
    print(f"Job ID:       {job_id}")
    print(f"\nMonitor logs with:")
    print(f"  kubectl logs -f -n {namespace} -l job-name={full_job_name}")
    print("\nVerify R2E-Gym in pod:")
    print(f"  kubectl exec -it -n {namespace} <pod-name> -- ls -la /workspace/rllm/R2E-Gym")
    print("=" * 60)


if __name__ == "__main__":
    Fire(main)

# python launch_rllm_job.py \
#   --rllm_code_path /home/aiccu/rllm \
#   --config_yaml_path /home/aiccu/rllm/gcp/configs/qwen3-1b.yaml \
#   --values_yaml_path /home/aiccu/rllm/gcp/values/rllm-values-main.yaml \
#   --charts_path /home/aiccu/charts/mambatron \
#   --gcs_bucket_path gs://tii-aiccu-falcon-mamba-us-central1 \
#   --job_name rllm-debug-qwen1b \
#   --n_nodes 1 \
#   --region us-central1