import logging
import os
import docker
import requests
import socket
import time
from contextlib import closing
from typing import Dict, Any, Tuple, List, Optional

# Mapping of status codes to strings
status_map = {
    0: "No prediction requested",
    1: "Prediction requested",
    2: "Prediction in progress",
    3: "Prediction completed",
    4: "Prediction failed"
}

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def is_running_in_container() -> bool:
    """
    Detect if the code is running inside a Docker container.
    
    Returns
    -------
    bool
        True if running inside a container, False otherwise.
    """
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Check cgroup
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        return False

def get_docker_host() -> str:
    """
    Get the appropriate hostname for connecting to Docker-exposed ports.
    
    Returns
    -------
    str
        The hostname to use for connections (localhost or host.docker.internal).
    """
    # Allow environment variable override
    docker_host = os.getenv('DOCKER_HOST_INTERNAL')
    if docker_host:
        return docker_host
    
    # Auto-detect based on environment
    if is_running_in_container():
        # When running inside a container, use host.docker.internal
        # This works on Docker Desktop for Mac/Windows
        return "host.docker.internal"
    else:
        # When running directly on host, use localhost
        return "localhost"

def find_free_port() -> int:
    """
    Find an available port on the host system.

    Returns
    -------
    int
        An available port number.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def start_docker_container(image_name: str, internal_port: int = 8000) -> Tuple[docker.models.containers.Container, int, Optional[str]]:
    """
    Pull the Docker image (if needed) and start the container on a random free port.

    Parameters
    ----------
    image_name : str
        Name of the Docker image.
    internal_port : int, optional
        The container's exposed port, by default 8000.

    Returns
    -------
    Tuple[docker.models.containers.Container, int]
        The container instance and the bound host port.
    
    Raises
    ------
    RuntimeError
        If Docker is not available or container fails to start.
    """
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        # Provide more specific error messages based on the error type
        error_msg = str(e).lower()
        if "no such file or directory" in error_msg and "docker.sock" in error_msg:
            raise RuntimeError(
                "Docker socket not found. When running in a container, ensure Docker socket is mounted:\n"
                "  -v /var/run/docker.sock:/var/run/docker.sock\n"
                "See DEPLOYMENT.md for detailed instructions."
            ) from e
        elif "permission denied" in error_msg:
            raise RuntimeError(
                "Permission denied accessing Docker. Possible solutions:\n"
                "1. Add user to docker group\n"
                "2. Run with appropriate permissions\n"
                "3. Check Docker socket permissions\n"
                f"Original error: {e}"
            ) from e
        else:
            raise RuntimeError(
                f"Docker is not available or not running. Please ensure Docker Desktop is installed and running. Error: {e}"
            ) from e

    try:
        client.images.pull(image_name)
        logging.debug("Successfully pulled image: %s", image_name)
    except docker.errors.ImageNotFound:
        raise RuntimeError(
            f"Docker image '{image_name}' not found. Please check the image name and ensure it exists."
        )
    except docker.errors.APIError as e:
        if "pull access denied" in str(e):
            raise RuntimeError(
                f"Access denied when pulling image '{image_name}'. Please check your Docker Hub credentials or image permissions."
            ) from e
        logging.warning("Could not pull image %s, attempting to use local copy: %s", image_name, e)
    except Exception as exc:
        logging.warning("Could not pull image %s, continuing locally: %s", image_name, exc)

    # Get the image SHA256 digest
    image_sha256 = None
    try:
        image = client.images.get(image_name)
        image_sha256 = image.id  # Returns format like "sha256:abc123..."
        logging.debug("Image SHA256: %s", image_sha256)
    except Exception as exc:
        logging.warning("Could not get image SHA256: %s", exc)

    host_port = find_free_port()
    logging.debug("Launching container on host port %d...", host_port)

    try:
        container = client.containers.run(
            image=image_name,
            detach=True,
            remove=True,
            ports={f"{internal_port}/tcp": host_port}
        )
    except docker.errors.ImageNotFound:
        raise RuntimeError(
            f"Docker image '{image_name}' not found locally and could not be pulled. "
            "Please ensure the image name is correct and you have internet connectivity."
        )
    except docker.errors.APIError as e:
        raise RuntimeError(
            f"Failed to start Docker container for image '{image_name}': {e}. "
            "Check Docker daemon logs for more details."
        ) from e
    
    time.sleep(1)

    container.reload()
    logging.debug("Container status: %s", container.status)

    if container.status != "running":
        logs = container.logs().decode(errors="ignore")
        logging.error("Container not in 'running' state. Logs:\n%s", logs)
        container.stop()
        raise RuntimeError(
            f"Container failed to start properly. Status: {container.status}. \n"
            f"Container logs:\n{logs[:500]}..." if len(logs) > 500 else f"Container logs:\n{logs}"
        )

    return container, host_port, image_sha256

def wait_for_container(host_port: int, timeout: Optional[int] = None, container: Optional[docker.models.containers.Container] = None) -> None:
    """
    Wait for the container to respond on the given host port.

    Parameters
    ----------
    host_port : int
        The bound host port.
    timeout : int, optional
        Maximum wait time in seconds. If None, uses CONTAINER_STARTUP_TIMEOUT env var or 60 seconds.
    container : docker.models.containers.Container, optional
        Container instance for getting logs if needed.

    Raises
    ------
    RuntimeError
        If the container doesn't respond within the timeout.
    """
    if timeout is None:
        timeout = int(os.getenv("CONTAINER_STARTUP_TIMEOUT", "60"))
    
    logging.info(f"Waiting up to {timeout} seconds for container to become ready...")
    start = time.time()
    last_log_time = start
    
    while (time.time() - start) < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex((get_docker_host(), host_port)) == 0:
                logging.debug("Container is responding on port %d", host_port)
                # Give the container a moment to fully initialize after port is open
                time.sleep(1)
                return
        
        # Log progress every 5 seconds
        if time.time() - last_log_time > 5:
            elapsed = int(time.time() - start)
            logging.info(f"Still waiting for container... ({elapsed}/{timeout} seconds)")
            
            # Check container status and logs if available
            if container:
                try:
                    container.reload()
                    if container.status != "running":
                        logs = container.logs(tail=20).decode(errors="ignore")
                        raise RuntimeError(
                            f"Container stopped unexpectedly. Status: {container.status}\n"
                            f"Recent logs:\n{logs}"
                        )
                except Exception as e:
                    logging.warning(f"Could not check container status: {e}")
            
            last_log_time = time.time()
        
        time.sleep(0.5)
    
    # Timeout reached - provide helpful error message
    error_msg = (
        f"Docker container did not become ready within {timeout} seconds. "
        f"The container may need more time to start.\n\n"
        f"Possible solutions:\n"
        f"1. Increase timeout by setting CONTAINER_STARTUP_TIMEOUT environment variable\n"
        f"2. Check if the model image needs to be downloaded (first run)\n"
        f"3. Verify the model container starts correctly: docker run -p 8000:8000 <image>\n"
    )
    
    if container:
        try:
            logs = container.logs(tail=30).decode(errors="ignore")
            error_msg += f"\nRecent container logs:\n{logs}"
        except:
            pass
    
    raise RuntimeError(error_msg)

def request_prediction(base_url: str, payload: list[dict[str, Any]], timeout: int = 360) -> None:
    """
    Send a POST request to /predict with the input payload.

    Parameters
    ----------
    base_url : str
        Base URL of the container, e.g., 'http://localhost:12345'
    payload : list[dict[str, Any]]
        JSON payload to send to /predict endpoint. It represents a list of inputs
        to the model, where each input is a dictionary with keys matching the model's
        input labels.
    timeout : int, optional
        Request timeout in seconds, by default 360.

    Raises
    ------
    RuntimeError
        If the response status is not successful, with detailed error information.
    """
    logging.debug("Sending payload to %s/predict: %s", base_url, payload)
    resp = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)

    if not resp.ok:
        # Try to get error details from response body
        error_body = ""
        try:
            error_body = resp.text
        except Exception:
            pass

        # Determine if this is a validation/data error (4xx) or server error (5xx)
        if 400 <= resp.status_code < 500:
            raise ValueError(
                f"Model rejected the input data (HTTP {resp.status_code}). "
                f"This may indicate invalid or out-of-range values in your data. "
                f"Model response: {error_body[:1000]}"
            )
        else:
            raise RuntimeError(
                f"Model container returned error (HTTP {resp.status_code}). "
                f"Model response: {error_body[:1000]}"
            )

def get_status_info(base_url: str) -> Tuple[int, str]:
    """
    Retrieve model status code and status message from /status.

    Supports both:
      {"status": <int>, "message": <str>}
    and:
      {"status": {"code": <int>, "message": <str>}}
    """
    resp = requests.get(f"{base_url}/status")
    if resp.ok:
        try:
            data = resp.json()
            status = data.get("status", -1)
            message = data.get("message", "")

            if isinstance(status, dict):
                code = status.get("code", -1)
                message = status.get("message", message)
            else:
                code = status

            try:
                code = int(code)
            except (TypeError, ValueError):
                code = -1

            logging.debug("Status code: %s, message: %s", code, message)
            return code, str(message) if message is not None else ""
        except Exception as ex:
            logging.warning("Could not parse JSON from /status: %s", ex)
    else:
        logging.warning("/status request failed with code %d.", resp.status_code)
    return -1, ""

def retrieve_result(base_url: str) -> list[float]:
    """
    Get numeric result(s) from /result as a list.

    The endpoint may return either a single numeric value or a list of numerics.

    Returns
    -------
    list[float]
        The numeric model result(s) as a list of floats.

    Raises
    ------
    RuntimeError
        If result cannot be parsed.
    """
    resp = requests.get(f"{base_url}/result")
    resp.raise_for_status()

    try:
        data = resp.json()
        return parse_ordered_response(data)
    except Exception as ex:
        raise RuntimeError(f"Failed to parse result from /result: {ex}")



def parse_ordered_response(data: dict) -> List[float]:
    """
    Parse API response with keys that may have prefixes like 'prediction_0'
    """
    # check if keys have a prefix like 'prediction_'
    if any(not key.isdigit() for key in data.keys()):
        # extract numbers from keys like 'prediction_0' (not sure whether we need a more robust way to deal wtth schema or this is fine. TODO: need to discuss with vedran)
        result = []
        for key in sorted(data.keys(), key=lambda k: int(k.split('_')[-1]) if '_' in k else int(k)):
            result.append(data[key])
        return result
    else:
        # Original behavior for simple numeric keys
        return [data[str(i)] for i in sorted(map(int, data.keys()))]

def stop_docker_container(container: docker.models.containers.Container) -> None:
    """
    Stop the Docker container.

    Parameters
    ----------
    container : docker.models.containers.Container
        The running container instance.
    """
    logging.debug("Stopping container...")
    container.stop()

def execute_model(metadata: Any, input_payload: list[dict[str, Any]], timeout = 360) -> dict:
    """
    Multi-step model execution:
      1) Start container.
      2) Wait for readiness.
      3) POST /predict.
      4) Poll /status for code == 3 (completed) or 4 (failed).
      5) If completed, GET /result.
      6) Stop container.

    Parameters
    ----------
    metadata : Any
        Metadata with 'docker_image' key or attribute.
    input_payload : list[dict[str, Any]]
        A list of inputs to send to the model, formatted as JSON.
    timeout : int, optional
        Maximum time (sec) to wait for model completion, by default 360.

    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': list[float] - Numeric prediction result(s).
        - 'docker_image_sha256': Optional[str] - SHA256 digest of the Docker image used.

    Raises
    ------
    RuntimeError
        If container fails or if it times out waiting for completion.
    """
    if not hasattr(metadata, 'docker_image') or not metadata.docker_image:
        raise RuntimeError(
            "Model metadata does not contain a 'docker_image' field. "
            "Please ensure the model metadata includes the Docker image name."
        )
    
    container = None
    image_sha256 = None
    try:
        container, port, image_sha256 = start_docker_container(metadata.docker_image)
        docker_host = get_docker_host()
        base_url = f"http://{docker_host}:{port}"
        logging.info(f"Connecting to model container at {base_url}")

        wait_for_container(port, container=container)
        
        # Retry the prediction request a few times in case the container needs a moment
        max_retries = 3
        for attempt in range(max_retries):
            try:
                request_prediction(base_url, input_payload, timeout)
                break
            except ValueError:
                # Data validation errors should not be retried - re-raise immediately
                raise
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Connection failed on attempt {attempt + 1}, retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Failed to connect to model container after {max_retries} attempts. "
                        f"The container may not be fully initialized yet."
                    ) from e

        start_time = time.time()
        failed_status_count = 0
        while time.time() - start_time < timeout:
            code, status_message = get_status_info(base_url)
            if code == 3:
                val = retrieve_result(base_url)
                logging.debug("Final numeric result: %s", val)
                return {"predictions": val, "docker_image_sha256": image_sha256}
            elif code == 4:
                # Some model servers briefly report failure while the result can still be fetched.
                # Try reading the result first, then require repeated failed states before raising.
                try:
                    val = retrieve_result(base_url)
                    logging.warning("Status endpoint returned 4 but /result succeeded; using returned result.")
                    return {"predictions": val, "docker_image_sha256": image_sha256}
                except Exception:
                    failed_status_count += 1
                    if failed_status_count >= 3:
                        logs = container.logs(tail=100).decode(errors="ignore")
                        detail = f"Model execution failed (status code 4)."
                        if status_message:
                            detail += f" Status message: {status_message}."
                        raise RuntimeError(f"{detail} Container logs:\n{logs}")
            elif code == 0:
                raise RuntimeError(
                    "Model did not process the prediction request. "
                    "The model may not have received the input data correctly."
                )
            time.sleep(1)

        elapsed = time.time() - start_time
        raise RuntimeError(
            f"Model execution timed out after {elapsed:.0f} seconds (timeout: {timeout}s). "
            f"The model may be taking longer than expected to process the input data."
        )
    except Exception as e:
        # Log container logs for debugging if available
        if container:
            try:
                logs = container.logs(tail=100).decode(errors="ignore")
                logging.error("Container logs at time of error:\n%s", logs)
            except:
                pass
        raise
    finally:
        if container:
            stop_docker_container(container)
