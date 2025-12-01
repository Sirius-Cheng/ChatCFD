import config, json, os, subprocess
from urllib.parse import parse_qs, urlparse, urlunparse


def _normalize_base_url(raw_url: str):
    """Strip query strings and /chat/completions suffix from custom endpoints."""
    if not raw_url:
        return raw_url, None

    parsed = urlparse(raw_url)
    query = parse_qs(parsed.query)
    api_version = query.get("api-version", [None])[0]

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]

    normalized = parsed._replace(path=path, query="", params="", fragment="")
    return urlunparse(normalized), api_version

def read_in_config():
    config_data = []
    with open(f'{config.Base_PATH}/inputs/chatcfd_config.json', 'r', encoding='utf-8') as file:
        config_data = json.load(file)

    os.environ["DEEPSEEK_V3_KEY"] = config_data["DEEPSEEK_V3_KEY"]
    v3_base_url, v3_api_version_from_url = _normalize_base_url(config_data["DEEPSEEK_V3_BASE_URL"])
    os.environ["DEEPSEEK_V3_BASE_URL"] = v3_base_url
    v3_api_version = config_data.get("DEEPSEEK_V3_API_VERSION") or v3_api_version_from_url
    if v3_api_version:
        os.environ["DEEPSEEK_V3_API_VERSION"] = v3_api_version
    else:
        os.environ.pop("DEEPSEEK_V3_API_VERSION", None)
    os.environ["DEEPSEEK_V3_MODEL_NAME"] = config_data["DEEPSEEK_V3_MODEL_NAME"]
    config.V3_temperature = config_data["V3_temperature"]

    os.environ["DEEPSEEK_R1_KEY"] = config_data["DEEPSEEK_R1_KEY"]
    r1_base_url, r1_api_version_from_url = _normalize_base_url(config_data["DEEPSEEK_R1_BASE_URL"])
    os.environ["DEEPSEEK_R1_BASE_URL"] = r1_base_url
    r1_api_version = config_data.get("DEEPSEEK_R1_API_VERSION") or r1_api_version_from_url
    if r1_api_version:
        os.environ["DEEPSEEK_R1_API_VERSION"] = r1_api_version
    else:
        os.environ.pop("DEEPSEEK_R1_API_VERSION", None)
    os.environ["DEEPSEEK_R1_MODEL_NAME"] = config_data["DEEPSEEK_R1_MODEL_NAME"]
    config.R1_temperature = config_data["R1_temperature"]

    config.run_time = config_data["run_time"]
    config.OpenFOAM_path = config_data["OpenFOAM_path"]
    config.OpenFOAM_tutorial_path = config_data["OpenFOAM_tutorial_path"]
    config.OpenFOAM_use_docker = config_data.get("OpenFOAM_use_docker", False)
    config.OpenFOAM_docker_exec = config_data.get("OpenFOAM_docker_exec", "")
    config.max_running_test_round = config_data["max_running_test_round"]
    config.pdf_chunk_d = config_data["pdf_chunk_d"]

def load_openfoam_environment():
    """Load OpenFOAM environment variables into the current Python process at once"""
    if getattr(config, "OpenFOAM_use_docker", False):
        print("Docker-based OpenFOAM is enabled, skip loading host OpenFOAM environment variables.")
        return
    try:
        # Get environment variables after sourcing through bash
        command =  f'source {config.OpenFOAM_path}/etc/bashrc && env'
        output = subprocess.run(
            command,
            shell=True,
            executable="/usr/bin/bash",  # Ensure using Bash
            check=True,  # Check if command was successful
            text=True,
            capture_output=True,
        )
        # Inject environment variables
        for line in output.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
    except subprocess.CalledProcessError as e:
        print(f"Failed to load OpenFOAM environment: {e.stderr}")
        raise