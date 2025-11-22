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
BBX AWS Adapter

Provides complete AWS automation:
- EC2 instance management
- S3 bucket operations
- Lambda functions
- CloudFormation stacks
- RDS databases
- ECS/EKS containers

Examples:
    # Launch EC2 instance
    - id: launch_instance
      mcp: bbx.aws
      method: ec2_launch
      inputs:
        image_id: "ami-12345678"
        instance_type: "t3.micro"
        key_name: "my-key"

    # Deploy Lambda
    - id: deploy_lambda
      mcp: bbx.aws
      method: lambda_deploy
      inputs:
        function_name: "my-function"
        runtime: "python3.11"
        code: "./lambda.zip"
"""

import json
import tempfile
from typing import Dict, Any
from pathlib import Path
import os
from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse


class AWSAdapter(DockerizedAdapter):
    """BBX Adapter for AWS operations using AWS CLI (Dockerized)"""

    def __init__(self):
        super().__init__(
            adapter_name="AWS",
            docker_image="amazon/aws-cli:latest",
            cli_tool="aws",
            version_args=["--version"],
            required=True
        )

    def run_command(self, *args, **kwargs):
        """Override run_command to inject AWS credentials"""
        env = kwargs.get("env", {}) or {}
        
        # Pass AWS credentials from host environment
        aws_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION"
        ]
        
        for var in aws_vars:
            if os.environ.get(var):
                env[var] = os.environ[var]
                
        kwargs["env"] = env
        return super().run_command(*args, **kwargs)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute AWS method"""
        self.log_execution(method, inputs)

        # Map method to handler
        handlers = {
            # EC2
            "ec2_launch": self._ec2_launch,
            "ec2_terminate": self._ec2_terminate,
            "ec2_describe": self._ec2_describe,
            # S3
            "s3_create_bucket": self._s3_create_bucket,
            "s3_upload": self._s3_upload,
            "s3_list": self._s3_list,
            # Lambda
            "lambda_deploy": self._lambda_deploy,
            "lambda_invoke": self._lambda_invoke,
            # CloudFormation
            "cfn_deploy": self._cfn_deploy,
            "cfn_delete": self._cfn_delete,
            # RDS
            "rds_create": self._rds_create,
        }

        handler = handlers.get(method)
        if not handler:
            return AdapterResponse.error_response(
                error=f"Unknown method: {method}",
            ).to_dict()

        try:
            result = await handler(inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            return AdapterResponse.error_response(error=str(e)).to_dict()

    # EC2 Operations

    async def _ec2_launch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Launch EC2 instance"""
        image_id = inputs.get("image_id")
        if not image_id:
            return AdapterResponse.error_response(
                error="image_id is required"
            ).to_dict()

        instance_type = inputs.get("instance_type", "t3.micro")

        args = [
            "ec2", "run-instances",
            "--image-id", image_id,
            "--instance-type", instance_type,
            "--count", "1"
        ]

        if "key_name" in inputs:
            args.extend(["--key-name", inputs["key_name"]])

        if "security_groups" in inputs:
            for sg in inputs["security_groups"]:
                args.extend(["--security-groups", sg])

        response = self.run_command(*args)

        if response.success:
            instances = response.data.get("Instances", [])
            if instances:
                instance = instances[0]
                return AdapterResponse.success_response(
                    data={
                        "instance_id": instance.get("InstanceId"),
                        "state": instance.get("State", {}).get("Name"),
                        "instance": instance
                    },
                    status="launched"
                ).to_dict()

        return response.to_dict()

    async def _ec2_terminate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate EC2 instance"""
        instance_ids = inputs.get("instance_ids")
        if not instance_ids:
            return AdapterResponse.error_response(
                error="instance_ids is required"
            ).to_dict()

        if isinstance(instance_ids, str):
            instance_ids = [instance_ids]

        args = ["ec2", "terminate-instances", "--instance-ids"] + instance_ids
        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="terminated"
            ).to_dict()

        return response.to_dict()

    async def _ec2_describe(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Describe EC2 instances"""
        args = ["ec2", "describe-instances"]

        if "instance_ids" in inputs:
            ids = inputs["instance_ids"]
            if isinstance(ids, str):
                ids = [ids]
            args.extend(["--instance-ids"] + ids)

        response = self.run_command(*args)

        if response.success:
            reservations = response.data.get("Reservations", [])
            instances = []
            for reservation in reservations:
                instances.extend(reservation.get("Instances", []))

            return AdapterResponse.success_response(
                data={
                    "instances": instances,
                    "count": len(instances)
                },
                status="ok"
            ).to_dict()

        return response.to_dict()

    # S3 Operations

    async def _s3_create_bucket(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create S3 bucket"""
        bucket_name = inputs.get("bucket_name")
        if not bucket_name:
            return AdapterResponse.error_response(
                error="bucket_name is required"
            ).to_dict()

        region = inputs.get("region", "us-east-1")

        args = ["s3api", "create-bucket", "--bucket", bucket_name]

        if region != "us-east-1":
            args.extend([
                "--create-bucket-configuration",
                f"LocationConstraint={region}"
            ])

        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data={"bucket": bucket_name, "region": region},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _s3_upload(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Upload file to S3"""
        bucket = inputs.get("bucket")
        key = inputs.get("key")
        file_path = inputs.get("file")

        if not all([bucket, key, file_path]):
            return AdapterResponse.error_response(
                error="bucket, key, and file are required"
            ).to_dict()

        args = ["s3", "cp", file_path, f"s3://{bucket}/{key}"]
        response = self.run_command(*args, output_format="text")

        if response.success:
            return AdapterResponse.success_response(
                data={"bucket": bucket, "key": key},
                status="uploaded"
            ).to_dict()

        return response.to_dict()

    async def _s3_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List S3 objects"""
        bucket = inputs.get("bucket")
        if not bucket:
            return AdapterResponse.error_response(
                error="bucket is required"
            ).to_dict()

        prefix = inputs.get("prefix", "")

        args = ["s3api", "list-objects-v2", "--bucket", bucket]
        if prefix:
            args.extend(["--prefix", prefix])

        response = self.run_command(*args)

        if response.success:
            contents = response.data.get("Contents", [])
            return AdapterResponse.success_response(
                data={
                    "objects": contents,
                    "count": len(contents)
                },
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Lambda Operations

    async def _lambda_deploy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy Lambda function"""
        function_name = inputs.get("function_name")
        role = inputs.get("role")
        code_zip = inputs.get("code")

        if not all([function_name, role, code_zip]):
            return AdapterResponse.error_response(
                error="function_name, role, and code are required"
            ).to_dict()

        runtime = inputs.get("runtime", "python3.11")
        handler = inputs.get("handler", "lambda_function.lambda_handler")

        # Check if function exists
        check_args = ["lambda", "get-function", "--function-name", function_name]
        check_response = self.run_command(*check_args)

        if check_response.success:
            # Update existing function
            args = [
                "lambda", "update-function-code",
                "--function-name", function_name,
                "--zip-file", f"fileb://{code_zip}"
            ]
        else:
            # Create new function
            args = [
                "lambda", "create-function",
                "--function-name", function_name,
                "--runtime", runtime,
                "--role", role,
                "--handler", handler,
                "--zip-file", f"fileb://{code_zip}"
            ]

        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data={"function_name": function_name},
                status="deployed"
            ).to_dict()

        return response.to_dict()

    async def _lambda_invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke Lambda function"""
        function_name = inputs.get("function_name")
        if not function_name:
            return AdapterResponse.error_response(
                error="function_name is required"
            ).to_dict()

        payload = inputs.get("payload", {})

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(payload, f)
            payload_file = f.name

        # Mount the payload file to /tmp in the container
        container_payload_path = f"/tmp/{Path(payload_file).name}"
        volumes = {payload_file: container_payload_path}

        args = [
            "lambda", "invoke",
            "--function-name", function_name,
            "--payload", f"file://{container_payload_path}",
            "/dev/stdout"
        ]

        response = self.run_command(*args, volumes=volumes)

        # Cleanup
        Path(payload_file).unlink(missing_ok=True)

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="invoked"
            ).to_dict()

        return response.to_dict()

    # CloudFormation Operations

    async def _cfn_deploy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy CloudFormation stack"""
        stack_name = inputs.get("stack_name")
        template = inputs.get("template")

        if not all([stack_name, template]):
            return AdapterResponse.error_response(
                error="stack_name and template are required"
            ).to_dict()

        args = [
            "cloudformation", "deploy",
            "--stack-name", stack_name,
            "--template-file", template,
            "--capabilities", "CAPABILITY_IAM"
        ]

        if "parameters" in inputs:
            param_list = []
            for key, value in inputs["parameters"].items():
                param_list.append(f"ParameterKey={key},ParameterValue={value}")
            args.extend(["--parameter-overrides"] + param_list)

        response = self.run_command(*args, timeout=900)

        if response.success:
            return AdapterResponse.success_response(
                data={"stack_name": stack_name},
                status="deployed"
            ).to_dict()

        return response.to_dict()

    async def _cfn_delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delete CloudFormation stack"""
        stack_name = inputs.get("stack_name")
        if not stack_name:
            return AdapterResponse.error_response(
                error="stack_name is required"
            ).to_dict()

        args = ["cloudformation", "delete-stack", "--stack-name", stack_name]
        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data={"stack_name": stack_name},
                status="deleting"
            ).to_dict()

        return response.to_dict()

    # RDS Operations

    async def _rds_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create RDS database"""
        db_instance_id = inputs.get("db_instance_id")
        master_username = inputs.get("master_username")
        master_password = inputs.get("master_password")

        if not all([db_instance_id, master_username, master_password]):
            return AdapterResponse.error_response(
                error="db_instance_id, master_username, and master_password are required"
            ).to_dict()

        db_class = inputs.get("db_instance_class", "db.t3.micro")
        engine = inputs.get("engine", "postgres")
        storage = inputs.get("storage", 20)

        args = [
            "rds", "create-db-instance",
            "--db-instance-identifier", db_instance_id,
            "--db-instance-class", db_class,
            "--engine", engine,
            "--allocated-storage", str(storage),
            "--master-username", master_username,
            "--master-user-password", master_password
        ]

        response = self.run_command(*args, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"db_instance_id": db_instance_id},
                status="creating"
            ).to_dict()

        return response.to_dict()
