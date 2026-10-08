import argparse
import sys

import requests


def get_workers(router_url):
    try:
        response = requests.get(f"{router_url}/workers")
        response.raise_for_status()
        return response.json().get("workers", [])
    except Exception as e:
        print(f"Error fetching workers from router: {e}")
        return []


def start_profile(worker_url):
    try:
        print(f"Starting profile on {worker_url}...")
        response = requests.post(f"{worker_url}/start_profile", json={})
        response.raise_for_status()
        print(f"Successfully started profile on {worker_url}")
    except Exception as e:
        print(f"Failed to start profile on {worker_url}: {e}")


def stop_profile(worker_url):
    try:
        print(f"Stopping profile on {worker_url}...")
        response = requests.post(f"{worker_url}/stop_profile", json={})
        response.raise_for_status()
        print(f"Successfully stopped profile on {worker_url}")
    except Exception as e:
        print(f"Failed to stop profile on {worker_url}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Automate vLLM profiling across all workers via router.")
    parser.add_argument("--router-url", type=str, required=True, help="Router URL (e.g., http://127.0.0.1:3000)")
    parser.add_argument("--action", type=str, choices=["start", "stop"], default="start", help="Action to perform")

    args = parser.parse_args()

    workers = get_workers(args.router_url)
    if not workers:
        print("No workers found. Ensure the router is running and workers are registered.")
        sys.exit(1)

    print(f"Found {len(workers)} workers.")

    for worker in workers:
        worker_url = worker.get("url")
        if not worker_url:
            continue

        if args.action == "start":
            start_profile(worker_url)
        else:
            stop_profile(worker_url)


if __name__ == "__main__":
    main()
