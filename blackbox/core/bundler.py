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
BBX Bundler - Create single-file executable .bbx app bundles
Packages workflows with dependencies into standalone executables
"""
import asyncio
import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("bbx.bundler")


class BundleBuilder:
    """
    Build single-file .bbx bundles containing:
    - Workflow definition
    - Dependencies
    - Assets and resources
    - Metadata
    - Embedded Python runtime (optional)
    """

    def __init__(self, workflow_path: Path):
        from blackbox.core.config import get_config

        self.workflow_path = Path(workflow_path)
        self.bundle_dir = get_config().paths.bundle_dir
        self.bundle_dir.mkdir(exist_ok=True)

    def parse_workflow(self) -> Dict[str, Any]:
        """Parse workflow file"""
        content = self.workflow_path.read_text()

        if self.workflow_path.suffix == ".json":
            return json.loads(content)
        else:
            return yaml.safe_load(content)

    def collect_dependencies(self, workflow: Dict[str, Any]) -> List[str]:
        """Collect all dependencies from workflow"""
        dependencies = set()

        # Check adapters used
        for node in workflow.get("nodes", {}).values():
            adapter = node.get("adapter")
            if adapter:
                dependencies.add(f"bbx-adapter-{adapter}")

        # Check explicit dependencies
        if "dependencies" in workflow:
            for dep in workflow["dependencies"]:
                dependencies.add(dep)

        return list(dependencies)

    def collect_assets(self, workflow: Dict[str, Any]) -> List[Path]:
        """Collect asset files referenced in workflow"""
        assets = []

        def find_file_refs(obj):
            """Recursively find file references"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ["file", "path", "template", "script"] and isinstance(
                        value, str
                    ):
                        file_path = Path(value)
                        if file_path.exists() and file_path.is_file():
                            assets.append(file_path)
                    else:
                        find_file_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_file_refs(item)

        find_file_refs(workflow)
        return assets

    def create_manifest(
        self, workflow: Dict[str, Any], dependencies: List[str], assets: List[Path]
    ) -> Dict[str, Any]:
        """Create bundle manifest"""
        return {
            "version": "1.0",
            "name": workflow.get("name", "Untitled"),
            "description": workflow.get("description", ""),
            "bbx_version": workflow.get("version", "6.0"),
            "author": workflow.get("author", ""),
            "license": workflow.get("license", "Apache-2.0"),
            "dependencies": dependencies,
            "assets": [str(asset) for asset in assets],
            "created_at": __import__("datetime").datetime.now().isoformat(),
            "checksum": "",  # Will be filled later
        }

    def create_bundle(
        self,
        output_path: Optional[Path] = None,
        include_runtime: bool = False,
        compress: bool = True,
    ) -> Path:
        """Create the bundle file"""
        workflow = self.parse_workflow()
        dependencies = self.collect_dependencies(workflow)
        assets = self.collect_assets(workflow)

        manifest = self.create_manifest(workflow, dependencies, assets)

        if not output_path:
            output_path = Path.cwd() / f"{workflow.get('name', 'bundle')}.bbx"

        # Create bundle archive
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED,
        ) as bundle:
            # Add workflow
            bundle.writestr("workflow.yaml", yaml.dump(workflow))

            # Add manifest
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2))

            # Add assets
            for asset in assets:
                bundle.write(asset, f"assets/{asset.name}")

            # Add Python runtime (optional)
            if include_runtime:
                self._add_runtime(bundle)

            # Add launcher script
            launcher = self._create_launcher(workflow)
            bundle.writestr("__main__.py", launcher)

        # Calculate and update checksum
        checksum = self._calculate_checksum(output_path)
        self._update_checksum(output_path, checksum)

        return output_path

    def _add_runtime(self, bundle: zipfile.ZipFile):
        """Add embedded Python runtime to bundle"""
        # This would include a minimal Python runtime
        # For now, we'll add a note
        bundle.writestr("runtime/README.txt", "Embedded Python runtime would go here")

    def _create_launcher(self, workflow: Dict[str, Any]) -> str:
        """Create launcher script for the bundle"""
        return f'''#!/usr/bin/env python3
"""
BBX Bundle Launcher
Auto-generated launcher for {workflow.get("name", "bundle")}
"""
import sys
import json
import zipfile
from pathlib import Path

def main():
    # Get bundle path
    if getattr(sys, 'frozen', False):
        bundle_path = Path(sys.executable)
    else:
        bundle_path = Path(__file__).parent

    # Extract and run workflow
    with zipfile.ZipFile(bundle_path, 'r') as bundle:
        # Read workflow
        workflow_data = bundle.read('workflow.yaml').decode()

        # Read manifest
        manifest_data = json.loads(bundle.read('manifest.json').decode())

        print(f"Running: {{manifest_data['name']}}")
        print(f"Version: {{manifest_data['bbx_version']}}")

        # Import and run BBX runtime
        try:
            from blackbox.core.runtime import BBXRuntime

            runtime = BBXRuntime()
            runtime.load_workflow_from_string(workflow_data)
            result = runtime.execute()

            print("\\nExecution completed successfully")
            print(json.dumps(result, indent=2))

        except ImportError:
            print("Error: BBX runtime not found")
            print("Install with: pip install blackbox-workflow")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {{e}}")
            sys.exit(1)

if __name__ == "__main__":
    main()
'''

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of bundle"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _update_checksum(self, bundle_path: Path, checksum: str):
        """Update checksum in manifest"""
        temp_path = bundle_path.with_suffix(".tmp")

        with zipfile.ZipFile(bundle_path, "r") as source:
            with zipfile.ZipFile(temp_path, "w") as target:
                for item in source.infolist():
                    data = source.read(item.filename)

                    if item.filename == "manifest.json":
                        manifest = json.loads(data.decode())
                        manifest["checksum"] = checksum
                        data = json.dumps(manifest, indent=2).encode()

                    target.writestr(item, data)

        temp_path.replace(bundle_path)


class BundleExtractor:
    """Extract and run .bbx bundles"""

    def __init__(self, bundle_path: Path):
        self.bundle_path = Path(bundle_path)

    def validate(self) -> bool:
        """Validate bundle integrity"""
        try:
            with zipfile.ZipFile(self.bundle_path, "r") as bundle:
                # Check required files
                required = ["workflow.yaml", "manifest.json", "__main__.py"]
                files = bundle.namelist()

                for req in required:
                    if req not in files:
                        return False

                # Read and validate manifest
                manifest = json.loads(bundle.read("manifest.json").decode())

                # Verify checksum (if present)
                if manifest.get("checksum"):
                    expected_checksum = manifest["checksum"]
                    # Calculate actual checksum (excluding manifest itself to avoid recursion)
                    import hashlib

                    sha256 = hashlib.sha256()

                    for item in sorted(bundle.namelist()):
                        if item != "manifest.json":
                            data = bundle.read(item)
                            sha256.update(data)

                    actual_checksum = sha256.hexdigest()

                    if actual_checksum != expected_checksum:
                        logger.warning(
                            f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                        )
                        return False

                return True

        except Exception:
            return False

    def extract(self, output_dir: Optional[Path] = None) -> Path:
        """Extract bundle contents"""
        if not output_dir:
            output_dir = Path.cwd() / f".bbx_extracted_{self.bundle_path.stem}"

        output_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(self.bundle_path, "r") as bundle:
            bundle.extractall(output_dir)

        return output_dir

    def get_manifest(self) -> Dict[str, Any]:
        """Get bundle manifest"""
        with zipfile.ZipFile(self.bundle_path, "r") as bundle:
            return json.loads(bundle.read("manifest.json").decode())

    def get_workflow(self) -> Dict[str, Any]:
        """Get workflow definition"""
        with zipfile.ZipFile(self.bundle_path, "r") as bundle:
            workflow_data = bundle.read("workflow.yaml").decode()
            return yaml.safe_load(workflow_data)

    async def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the bundled workflow"""
        from .runtime import BBXRuntime  # type: ignore

        workflow = self.get_workflow()

        runtime = BBXRuntime()  # type: ignore
        runtime.load_workflow(workflow)

        if inputs:
            for key, value in inputs.items():
                runtime.set_input(key, value)

        return await runtime.execute()


def create_executable_bundle(
    workflow_path: Path,
    output_path: Optional[Path] = None,
    include_runtime: bool = False,
    platform: str = "current",
) -> Path:
    """
    Create a standalone executable bundle using PyInstaller or similar.

    This creates a true single-file executable that doesn't require
    Python to be installed on the target system.
    """
    builder = BundleBuilder(workflow_path)
    bundle_path = builder.create_bundle(include_runtime=include_runtime)

    if platform == "current":
        # Use PyInstaller to create executable
        try:
            import PyInstaller.__main__  # type: ignore

            PyInstaller.__main__.run(
                [  # type: ignore
                    str(bundle_path),
                    "--onefile",
                    "--name",
                    bundle_path.stem,
                    "--add-data",
                    f"{bundle_path}:.",
                ]
            )

            # Executable will be in dist/ directory
            exe_path = Path("dist") / bundle_path.stem
            if platform == "windows":
                exe_path = exe_path.with_suffix(".exe")

            if not output_path:
                output_path = Path.cwd() / exe_path.name

            if exe_path.exists():
                import shutil

                shutil.move(str(exe_path), str(output_path))

            return output_path

        except ImportError:
            print("PyInstaller not found. Returning zip bundle.")
            return bundle_path

    return bundle_path


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="BBX Bundle Builder")
    subparsers = parser.add_subparsers(dest="command")

    # Build command
    build = subparsers.add_parser("build", help="Build a bundle")
    build.add_argument("workflow", type=Path, help="Workflow file to bundle")
    build.add_argument("-o", "--output", type=Path, help="Output bundle path")
    build.add_argument("--runtime", action="store_true", help="Include Python runtime")
    build.add_argument("--executable", action="store_true", help="Create executable")
    build.add_argument("--no-compress", action="store_true", help="Don't compress")

    # Extract command
    extract = subparsers.add_parser("extract", help="Extract a bundle")
    extract.add_argument("bundle", type=Path, help="Bundle file to extract")
    extract.add_argument("-o", "--output", type=Path, help="Output directory")

    # Info command
    info = subparsers.add_parser("info", help="Show bundle info")
    info.add_argument("bundle", type=Path, help="Bundle file")

    # Run command
    run = subparsers.add_parser("run", help="Run a bundle")
    run.add_argument("bundle", type=Path, help="Bundle file to run")
    run.add_argument("--input", action="append", help="Input values (key=value)")

    args = parser.parse_args()

    if args.command == "build":
        builder = BundleBuilder(args.workflow)

        if args.executable:
            output = create_executable_bundle(
                args.workflow, args.output, include_runtime=args.runtime
            )
        else:
            output = builder.create_bundle(
                args.output, include_runtime=args.runtime, compress=not args.no_compress
            )

        print(f"Bundle created: {output}")
        print(f"Size: {output.stat().st_size} bytes")

    elif args.command == "extract":
        extractor = BundleExtractor(args.bundle)
        output = extractor.extract(args.output)
        print(f"Extracted to: {output}")

    elif args.command == "info":
        extractor = BundleExtractor(args.bundle)
        manifest = extractor.get_manifest()
        print(json.dumps(manifest, indent=2))

    elif args.command == "run":
        extractor = BundleExtractor(args.bundle)

        # Parse inputs
        inputs = {}
        if args.input:
            for inp in args.input:
                key, value = inp.split("=", 1)
                inputs[key] = value

        result = asyncio.run(extractor.run(inputs))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
