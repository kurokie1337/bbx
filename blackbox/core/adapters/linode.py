"""
Linode Adapter - Complete Linode cloud automation
Provides Linode instance management, Kubernetes, networking, and more
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from ..base_adapter import BaseAdapter


class LinodeAdapter(BaseAdapter):
    """
    Linode cloud adapter.

    Capabilities:
    - Linode (VM) management
    - Kubernetes clusters (LKE)
    - NodeBalancers (load balancers)
    - Block storage volumes
    - Object storage
    - Firewalls
    - VLANs
    - Longview monitoring
    """

    def __init__(self):
        super().__init__("linode")
        self.api_token = os.getenv("LINODE_TOKEN")
        if not self.api_token:
            raise ValueError("LINODE_TOKEN environment variable not set")

    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute Linode actions"""
        actions = {
            # Linodes
            "linode.create": self._linode_create,
            "linode.delete": self._linode_delete,
            "linode.list": self._linode_list,
            "linode.get": self._linode_get,
            "linode.reboot": self._linode_reboot,
            "linode.resize": self._linode_resize,
            "linode.snapshot": self._linode_snapshot,
            # Kubernetes
            "k8s.create": self._k8s_create,
            "k8s.delete": self._k8s_delete,
            "k8s.list": self._k8s_list,
            "k8s.get_kubeconfig": self._k8s_get_kubeconfig,
            # NodeBalancers
            "lb.create": self._lb_create,
            "lb.delete": self._lb_delete,
            "lb.list": self._lb_list,
            # Volumes
            "volume.create": self._volume_create,
            "volume.attach": self._volume_attach,
            "volume.detach": self._volume_detach,
            "volume.delete": self._volume_delete,
            # Object Storage
            "object.create_bucket": self._object_create_bucket,
            "object.upload": self._object_upload,
            "object.download": self._object_download,
            "object.delete_bucket": self._object_delete_bucket,
            # Firewalls
            "firewall.create": self._firewall_create,
            "firewall.delete": self._firewall_delete,
            "firewall.list": self._firewall_list,
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
        """Make API call to Linode"""
        import aiohttp

        url = f"https://api.linode.com/v4{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as resp:
                if resp.status == 204:
                    return {"status": "success"}
                return await resp.json()

    # Linode Operations

    async def _linode_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Linode instance"""
        data = {
            "label": params["label"],
            "region": params.get("region", "us-east"),
            "type": params.get("type", "g6-nanode-1"),
            "image": params.get("image", "linode/ubuntu22.04"),
            "root_pass": params.get("root_pass"),
            "authorized_keys": params.get("ssh_keys", []),
            "backups_enabled": params.get("backups", False),
            "private_ip": params.get("private_ip", True),
            "tags": params.get("tags", []),
            "metadata": {
                "user_data": params.get("user_data")
            } if params.get("user_data") else None
        }

        result = await self._api_call("POST", "/linode/instances", data)
        return result

    async def _linode_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a Linode instance"""
        linode_id = params["linode_id"]
        result = await self._api_call("DELETE", f"/linode/instances/{linode_id}")
        return {"status": "deleted", "linode_id": linode_id}

    async def _linode_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all Linode instances"""
        result = await self._api_call("GET", "/linode/instances")
        return result

    async def _linode_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Linode instance details"""
        linode_id = params["linode_id"]
        result = await self._api_call("GET", f"/linode/instances/{linode_id}")
        return result

    async def _linode_reboot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Reboot a Linode instance"""
        linode_id = params["linode_id"]
        result = await self._api_call("POST", f"/linode/instances/{linode_id}/reboot")
        return result

    async def _linode_resize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resize a Linode instance"""
        linode_id = params["linode_id"]
        data = {
            "type": params["type"],
            "allow_auto_disk_resize": params.get("auto_resize_disk", True)
        }
        result = await self._api_call("POST", f"/linode/instances/{linode_id}/resize", data)
        return result

    async def _linode_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Linode snapshot (backup)"""
        linode_id = params["linode_id"]
        data = {
            "label": params.get("label", f"snapshot-{linode_id}")
        }
        result = await self._api_call("POST", f"/linode/instances/{linode_id}/backups", data)
        return result

    # Kubernetes Operations

    async def _k8s_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create LKE (Linode Kubernetes Engine) cluster"""
        data = {
            "label": params["label"],
            "region": params.get("region", "us-east"),
            "k8s_version": params.get("version", "1.28"),
            "node_pools": params.get("node_pools", [{
                "type": "g6-standard-2",
                "count": 3,
            }]),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/lke/clusters", data)
        return result

    async def _k8s_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete LKE cluster"""
        cluster_id = params["cluster_id"]
        result = await self._api_call("DELETE", f"/lke/clusters/{cluster_id}")
        return {"status": "deleted", "cluster_id": cluster_id}

    async def _k8s_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List LKE clusters"""
        result = await self._api_call("GET", "/lke/clusters")
        return result

    async def _k8s_get_kubeconfig(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get kubeconfig for LKE cluster"""
        cluster_id = params["cluster_id"]
        result = await self._api_call("GET", f"/lke/clusters/{cluster_id}/kubeconfig")
        return result

    # NodeBalancer (Load Balancer) Operations

    async def _lb_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create NodeBalancer"""
        data = {
            "label": params["label"],
            "region": params.get("region", "us-east"),
            "client_conn_throttle": params.get("throttle", 0),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/nodebalancers", data)
        nb_id = result["id"]

        # Create config (port forwarding rules)
        if params.get("configs"):
            for config in params["configs"]:
                config_data = {
                    "port": config.get("port", 80),
                    "protocol": config.get("protocol", "http"),
                    "algorithm": config.get("algorithm", "roundrobin"),
                    "stickiness": config.get("stickiness", "none"),
                    "check": config.get("health_check", "connection"),
                    "check_interval": config.get("check_interval", 5),
                    "check_timeout": config.get("check_timeout", 3),
                    "check_attempts": config.get("check_attempts", 2)
                }

                await self._api_call("POST", f"/nodebalancers/{nb_id}/configs", config_data)

        return result

    async def _lb_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete NodeBalancer"""
        nb_id = params["nb_id"]
        result = await self._api_call("DELETE", f"/nodebalancers/{nb_id}")
        return {"status": "deleted", "nb_id": nb_id}

    async def _lb_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List NodeBalancers"""
        result = await self._api_call("GET", "/nodebalancers")
        return result

    # Volume Operations

    async def _volume_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create block storage volume"""
        data = {
            "label": params["label"],
            "size": params["size_gb"],
            "region": params.get("region", "us-east"),
            "linode_id": params.get("linode_id"),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/volumes", data)
        return result

    async def _volume_attach(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach volume to Linode"""
        volume_id = params["volume_id"]
        data = {
            "linode_id": params["linode_id"]
        }

        result = await self._api_call("POST", f"/volumes/{volume_id}/attach", data)
        return result

    async def _volume_detach(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detach volume from Linode"""
        volume_id = params["volume_id"]
        result = await self._api_call("POST", f"/volumes/{volume_id}/detach")
        return result

    async def _volume_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete volume"""
        volume_id = params["volume_id"]
        result = await self._api_call("DELETE", f"/volumes/{volume_id}")
        return {"status": "deleted", "volume_id": volume_id}

    # Object Storage Operations

    async def _object_create_bucket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Object Storage bucket"""
        data = {
            "label": params["label"],
            "cluster": params.get("cluster", "us-east-1")
        }

        result = await self._api_call("POST", "/object-storage/buckets", data)
        return result

    async def _object_upload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Upload file to Object Storage"""
        # Object storage uses S3-compatible API
        import boto3

        cluster = params.get("cluster", "us-east-1")
        access_key = os.getenv("LINODE_OBJECT_ACCESS_KEY")
        secret_key = os.getenv("LINODE_OBJECT_SECRET_KEY")

        session = boto3.session.Session()
        client = session.client(
            's3',
            endpoint_url=f'https://{cluster}.linodeobjects.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        client.upload_file(
            params["local_path"],
            params["bucket"],
            params["key"]
        )

        return {"status": "uploaded", "key": params["key"]}

    async def _object_download(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download file from Object Storage"""

        cluster = params.get("cluster", "us-east-1")
        access_key = os.getenv("LINODE_OBJECT_ACCESS_KEY")
        secret_key = os.getenv("LINODE_OBJECT_SECRET_KEY")

        session = boto3.session.Session()
        client = session.client(
            's3',
            endpoint_url=f'https://{cluster}.linodeobjects.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        client.download_file(
            params["bucket"],
            params["key"],
            params["local_path"]
        )

        return {"status": "downloaded", "path": params["local_path"]}

    async def _object_delete_bucket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Object Storage bucket"""
        cluster = params.get("cluster", "us-east-1")
        label = params["label"]

        result = await self._api_call("DELETE", f"/object-storage/buckets/{cluster}/{label}")
        return {"status": "deleted", "bucket": label}

    # Firewall Operations

    async def _firewall_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create firewall"""
        data = {
            "label": params["label"],
            "rules": params.get("rules", {
                "inbound": [
                    {
                        "action": "ACCEPT",
                        "protocol": "TCP",
                        "ports": "22",
                        "addresses": {"ipv4": ["0.0.0.0/0"]}
                    },
                    {
                        "action": "ACCEPT",
                        "protocol": "TCP",
                        "ports": "80, 443",
                        "addresses": {"ipv4": ["0.0.0.0/0"]}
                    }
                ],
                "outbound": [
                    {
                        "action": "ACCEPT",
                        "protocol": "TCP",
                        "ports": "1-65535",
                        "addresses": {"ipv4": ["0.0.0.0/0"]}
                    }
                ]
            }),
            "tags": params.get("tags", [])
        }

        result = await self._api_call("POST", "/networking/firewalls", data)

        # Attach to Linodes if specified
        if params.get("linode_ids"):
            firewall_id = result["id"]
            devices_data = {
                "linodes": params["linode_ids"]
            }
            await self._api_call("POST", f"/networking/firewalls/{firewall_id}/devices", devices_data)

        return result

    async def _firewall_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete firewall"""
        firewall_id = params["firewall_id"]
        result = await self._api_call("DELETE", f"/networking/firewalls/{firewall_id}")
        return {"status": "deleted", "firewall_id": firewall_id}

    async def _firewall_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List firewalls"""
        result = await self._api_call("GET", "/networking/firewalls")
        return result
