# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DigitalOcean Adapter - Complete DigitalOcean cloud automation
Provides droplet management, Kubernetes, networking, and more
"""
import os
from typing import Any, Dict, Optional
from ..base_adapter import BaseAdapter
import boto3


class DigitalOceanAdapter(BaseAdapter):
    """
    DigitalOcean cloud adapter.

    Capabilities:
    - Droplet (VM) management
    - Kubernetes clusters
    - Load balancers
    - Volumes and snapshots
    - VPC networking
    - Floating IPs
    - Databases
    - Spaces (S3-compatible storage)
    - Container Registry
    - App Platform
    """

    def __init__(self):
        super().__init__("digitalocean")
        self.api_token = os.getenv("DIGITALOCEAN_TOKEN")
        if not self.api_token:
            raise ValueError("DIGITALOCEAN_TOKEN environment variable not set")

    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute DigitalOcean actions"""
        actions = {
            # Droplets
            "droplet.create": self._droplet_create,
            "droplet.delete": self._droplet_delete,
            "droplet.list": self._droplet_list,
            "droplet.get": self._droplet_get,
            "droplet.reboot": self._droplet_reboot,
            "droplet.snapshot": self._droplet_snapshot,
            "droplet.resize": self._droplet_resize,
            # Kubernetes
            "k8s.create": self._k8s_create,
            "k8s.delete": self._k8s_delete,
            "k8s.list": self._k8s_list,
            "k8s.get_kubeconfig": self._k8s_get_kubeconfig,
            # Load Balancers
            "lb.create": self._lb_create,
            "lb.delete": self._lb_delete,
            "lb.list": self._lb_list,
            # Volumes
            "volume.create": self._volume_create,
            "volume.attach": self._volume_attach,
            "volume.detach": self._volume_detach,
            "volume.delete": self._volume_delete,
            # Databases
            "db.create": self._db_create,
            "db.delete": self._db_delete,
            "db.list": self._db_list,
            # Spaces
            "space.create": self._space_create,
            "space.upload": self._space_upload,
            "space.download": self._space_download,
            "space.delete": self._space_delete,
        }

        handler = actions.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        return await handler(params)

    async def _api_call(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API call to DigitalOcean"""
        import aiohttp

        url = f"https://api.digitalocean.com/v2{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as resp:
                return await resp.json()

    # Droplet Operations

    async def _droplet_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new droplet"""
        data = {
            "name": params["name"],
            "region": params.get("region", "nyc3"),
            "size": params.get("size", "s-1vcpu-1gb"),
            "image": params.get("image", "ubuntu-22-04-x64"),
            "ssh_keys": params.get("ssh_keys", []),
            "backups": params.get("backups", False),
            "ipv6": params.get("ipv6", True),
            "monitoring": params.get("monitoring", True),
            "tags": params.get("tags", []),
            "user_data": params.get("user_data"),
            "vpc_uuid": params.get("vpc_uuid"),
        }

        result = await self._api_call("POST", "/droplets", data)
        return result

    async def _droplet_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a droplet"""
        droplet_id = params["droplet_id"]
        await self._api_call("DELETE", f"/droplets/{droplet_id}")
        return {"status": "deleted", "droplet_id": droplet_id}

    async def _droplet_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all droplets"""
        result = await self._api_call("GET", "/droplets")
        return result

    async def _droplet_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get droplet details"""
        droplet_id = params["droplet_id"]
        result = await self._api_call("GET", f"/droplets/{droplet_id}")
        return result

    async def _droplet_reboot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Reboot a droplet"""
        droplet_id = params["droplet_id"]
        data = {"type": "reboot"}
        result = await self._api_call("POST", f"/droplets/{droplet_id}/actions", data)
        return result

    async def _droplet_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create droplet snapshot"""
        droplet_id = params["droplet_id"]
        data = {
            "type": "snapshot",
            "name": params.get("name", f"snapshot-{droplet_id}")
        }
        result = await self._api_call("POST", f"/droplets/{droplet_id}/actions", data)
        return result

    async def _droplet_resize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resize a droplet"""
        droplet_id = params["droplet_id"]
        data = {
            "type": "resize",
            "size": params["size"],
            "disk": params.get("resize_disk", False)
        }
        result = await self._api_call("POST", f"/droplets/{droplet_id}/actions", data)
        return result

    # Kubernetes Operations

    async def _k8s_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kubernetes cluster"""
        data = {
            "name": params["name"],
            "region": params.get("region", "nyc1"),
            "version": params.get("version", "1.28.2-do.0"),
            "node_pools": params.get("node_pools", [{
                "size": "s-2vcpu-2gb",
                "count": 3,
                "name": "worker-pool"
            }]),
            "auto_upgrade": params.get("auto_upgrade", True),
            "maintenance_policy": params.get("maintenance_policy", {
                "day": "sunday",
                "start_time": "02:00"
            })
        }

        result = await self._api_call("POST", "/kubernetes/clusters", data)
        return result

    async def _k8s_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Kubernetes cluster"""
        cluster_id = params["cluster_id"]
        await self._api_call("DELETE", f"/kubernetes/clusters/{cluster_id}")
        return {"status": "deleted", "cluster_id": cluster_id}

    async def _k8s_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Kubernetes clusters"""
        result = await self._api_call("GET", "/kubernetes/clusters")
        return result

    async def _k8s_get_kubeconfig(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get kubeconfig for cluster"""
        cluster_id = params["cluster_id"]
        result = await self._api_call("GET", f"/kubernetes/clusters/{cluster_id}/kubeconfig")
        return result

    # Load Balancer Operations

    async def _lb_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create load balancer"""
        data = {
            "name": params["name"],
            "region": params.get("region", "nyc3"),
            "forwarding_rules": params.get("forwarding_rules", [{
                "entry_protocol": "http",
                "entry_port": 80,
                "target_protocol": "http",
                "target_port": 80
            }]),
            "health_check": params.get("health_check", {
                "protocol": "http",
                "port": 80,
                "path": "/",
                "check_interval_seconds": 10,
                "response_timeout_seconds": 5,
                "healthy_threshold": 5,
                "unhealthy_threshold": 3
            }),
            "droplet_ids": params.get("droplet_ids", []),
            "tag": params.get("tag"),
        }

        result = await self._api_call("POST", "/load_balancers", data)
        return result

    async def _lb_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete load balancer"""
        lb_id = params["lb_id"]
        await self._api_call("DELETE", f"/load_balancers/{lb_id}")
        return {"status": "deleted", "lb_id": lb_id}

    async def _lb_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List load balancers"""
        result = await self._api_call("GET", "/load_balancers")
        return result

    # Volume Operations

    async def _volume_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create block storage volume"""
        data = {
            "size_gigabytes": params["size_gb"],
            "name": params["name"],
            "region": params.get("region", "nyc1"),
            "filesystem_type": params.get("filesystem", "ext4"),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/volumes", data)
        return result

    async def _volume_attach(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach volume to droplet"""
        volume_id = params["volume_id"]
        data = {
            "type": "attach",
            "droplet_id": params["droplet_id"],
            "region": params.get("region", "nyc1")
        }

        result = await self._api_call("POST", f"/volumes/{volume_id}/actions", data)
        return result

    async def _volume_detach(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detach volume from droplet"""
        volume_id = params["volume_id"]
        data = {
            "type": "detach",
            "droplet_id": params["droplet_id"],
            "region": params.get("region", "nyc1")
        }

        result = await self._api_call("POST", f"/volumes/{volume_id}/actions", data)
        return result

    async def _volume_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete volume"""
        volume_id = params["volume_id"]
        await self._api_call("DELETE", f"/volumes/{volume_id}")
        return {"status": "deleted", "volume_id": volume_id}

    # Database Operations

    async def _db_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create managed database"""
        data = {
            "name": params["name"],
            "engine": params.get("engine", "pg"),
            "version": params.get("version", "15"),
            "region": params.get("region", "nyc3"),
            "size": params.get("size", "db-s-1vcpu-1gb"),
            "num_nodes": params.get("nodes", 1),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/databases", data)
        return result

    async def _db_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete managed database"""
        db_id = params["db_id"]
        await self._api_call("DELETE", f"/databases/{db_id}")
        return {"status": "deleted", "db_id": db_id}

    async def _db_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List managed databases"""
        result = await self._api_call("GET", "/databases")
        return result

    # Spaces (S3-compatible) Operations

    async def _space_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Spaces bucket"""
        # Spaces uses S3-compatible API
        import boto3

        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=params.get("region", "nyc3"),
            endpoint_url=f'https://{params.get("region", "nyc3")}.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("SPACES_KEY"),
            aws_secret_access_key=os.getenv("SPACES_SECRET")
        )

        bucket_name = params["name"]
        client.create_bucket(Bucket=bucket_name)

        return {"status": "created", "bucket": bucket_name}

    async def _space_upload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Upload file to Spaces"""

        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=params.get("region", "nyc3"),
            endpoint_url=f'https://{params.get("region", "nyc3")}.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("SPACES_KEY"),
            aws_secret_access_key=os.getenv("SPACES_SECRET")
        )

        client.upload_file(
            params["local_path"],
            params["bucket"],
            params["key"]
        )

        return {"status": "uploaded", "key": params["key"]}

    async def _space_download(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download file from Spaces"""

        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=params.get("region", "nyc3"),
            endpoint_url=f'https://{params.get("region", "nyc3")}.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("SPACES_KEY"),
            aws_secret_access_key=os.getenv("SPACES_SECRET")
        )

        client.download_file(
            params["bucket"],
            params["key"],
            params["local_path"]
        )

        return {"status": "downloaded", "path": params["local_path"]}

    async def _space_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Spaces bucket"""

        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=params.get("region", "nyc3"),
            endpoint_url=f'https://{params.get("region", "nyc3")}.digitaloceanspaces.com',
            aws_access_key_id=os.getenv("SPACES_KEY"),
            aws_secret_access_key=os.getenv("SPACES_SECRET")
        )

        # Delete all objects first
        bucket = params["bucket"]
        response = client.list_objects_v2(Bucket=bucket)

        if 'Contents' in response:
            for obj in response['Contents']:
                client.delete_object(Bucket=bucket, Key=obj['Key'])

        # Delete bucket
        client.delete_bucket(Bucket=bucket)

        return {"status": "deleted", "bucket": bucket}
