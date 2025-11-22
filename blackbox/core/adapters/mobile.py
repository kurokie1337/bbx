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
Mobile Emulator Adapter - Android & iOS support for BBX workflows
Provides integration with Android Emulator and iOS Simulator
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from ..base_adapter import BaseAdapter


class MobileAdapter(BaseAdapter):
    """
    Mobile emulator adapter supporting Android and iOS platforms.

    Capabilities:
    - Launch/stop emulators
    - Install/uninstall apps
    - Execute commands on device
    - Capture screenshots
    - Record screen
    - Manage device state
    - Network traffic monitoring
    - Performance profiling
    """

    def __init__(self):
        super().__init__("mobile")
        self.android_sdk = self._detect_android_sdk()
        self.xcode_path = self._detect_xcode()

    def _detect_android_sdk(self) -> Optional[Path]:
        """Detect Android SDK installation"""
        # Check common locations
        common_paths = [
            Path.home() / "Android" / "Sdk",
            Path.home() / "Library" / "Android" / "sdk",
            Path("/usr/local/android-sdk"),
        ]

        for path in common_paths:
            if path.exists():
                return path
        return None

    def _detect_xcode(self) -> Optional[Path]:
        """Detect Xcode installation (macOS only)"""
        xcode_path = Path("/Applications/Xcode.app")
        if xcode_path.exists():
            return xcode_path
        return None

    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute mobile emulator actions"""
        actions = {
            "android.launch": self._android_launch,
            "android.stop": self._android_stop,
            "android.install": self._android_install,
            "android.uninstall": self._android_uninstall,
            "android.exec": self._android_exec,
            "android.screenshot": self._android_screenshot,
            "android.record": self._android_record,
            "android.list": self._android_list,
            "ios.launch": self._ios_launch,
            "ios.stop": self._ios_stop,
            "ios.install": self._ios_install,
            "ios.uninstall": self._ios_uninstall,
            "ios.exec": self._ios_exec,
            "ios.screenshot": self._ios_screenshot,
            "ios.record": self._ios_record,
            "ios.list": self._ios_list,
        }

        handler = actions.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        return await handler(params)

    # Android Emulator Methods

    async def _android_launch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Launch Android emulator"""
        if not self.android_sdk:
            raise RuntimeError("Android SDK not found")

        emulator = params.get("emulator", "Pixel_5_API_31")
        wait_boot = params.get("wait_boot", True)

        emulator_bin = self.android_sdk / "emulator" / "emulator"

        cmd = [str(emulator_bin), "-avd", emulator, "-no-snapshot-save"]

        # Additional options
        if params.get("headless"):
            cmd.extend(["-no-window", "-no-audio"])
        if params.get("writable_system"):
            cmd.append("-writable-system")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        if wait_boot:
            await self._wait_for_boot()

        return {
            "status": "launched",
            "emulator": emulator,
            "pid": proc.pid
        }

    async def _wait_for_boot(self, timeout: int = 120):
        """Wait for Android device to finish booting"""
        adb = self.android_sdk / "platform-tools" / "adb"

        start = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError("Device boot timeout")

            proc = await asyncio.create_subprocess_exec(
                str(adb), "shell", "getprop", "sys.boot_completed",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()

            if stdout.decode().strip() == "1":
                break

            await asyncio.sleep(2)

    async def _android_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop Android emulator"""
        adb = self.android_sdk / "platform-tools" / "adb"

        device_id = params.get("device_id")
        cmd = [str(adb)]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["emu", "kill"])

        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.communicate()

        return {"status": "stopped"}

    async def _android_install(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Install APK on Android device"""
        adb = self.android_sdk / "platform-tools" / "adb"
        apk_path = params["apk_path"]

        device_id = params.get("device_id")
        cmd = [str(adb)]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["install", "-r", apk_path])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "status": "installed",
            "output": stdout.decode(),
            "error": stderr.decode()
        }

    async def _android_uninstall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Uninstall app from Android device"""
        adb = self.android_sdk / "platform-tools" / "adb"
        package = params["package"]

        device_id = params.get("device_id")
        cmd = [str(adb)]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["uninstall", package])

        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.communicate()

        return {"status": "uninstalled"}

    async def _android_exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command on Android device"""
        adb = self.android_sdk / "platform-tools" / "adb"
        command = params["command"]

        device_id = params.get("device_id")
        cmd = [str(adb)]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["shell"] + command.split())

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode
        }

    async def _android_screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture screenshot from Android device"""
        adb = self.android_sdk / "platform-tools" / "adb"
        output_path = params.get("output", "screenshot.png")

        device_id = params.get("device_id")

        # Capture on device
        cmd1 = [str(adb)]
        if device_id:
            cmd1.extend(["-s", device_id])
        cmd1.extend(["shell", "screencap", "/sdcard/screenshot.png"])

        proc = await asyncio.create_subprocess_exec(*cmd1)
        await proc.communicate()

        # Pull to local
        cmd2 = [str(adb)]
        if device_id:
            cmd2.extend(["-s", device_id])
        cmd2.extend(["pull", "/sdcard/screenshot.png", output_path])

        proc = await asyncio.create_subprocess_exec(*cmd2)
        await proc.communicate()

        return {"status": "captured", "path": output_path}

    async def _android_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record screen on Android device"""
        adb = self.android_sdk / "platform-tools" / "adb"
        duration = params.get("duration", 180)
        output_path = params.get("output", "recording.mp4")

        device_id = params.get("device_id")

        # Start recording
        cmd1 = [str(adb)]
        if device_id:
            cmd1.extend(["-s", device_id])
        cmd1.extend(["shell", "screenrecord", f"--time-limit={duration}", "/sdcard/recording.mp4"])

        proc = await asyncio.create_subprocess_exec(*cmd1)
        await proc.communicate()

        # Pull to local
        cmd2 = [str(adb)]
        if device_id:
            cmd2.extend(["-s", device_id])
        cmd2.extend(["pull", "/sdcard/recording.mp4", output_path])

        proc = await asyncio.create_subprocess_exec(*cmd2)
        await proc.communicate()

        return {"status": "recorded", "path": output_path}

    async def _android_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List Android devices and emulators"""
        adb = self.android_sdk / "platform-tools" / "adb"

        proc = await asyncio.create_subprocess_exec(
            str(adb), "devices", "-l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        devices = []
        lines = stdout.decode().strip().split("\n")[1:]  # Skip header
        for line in lines:
            if line.strip():
                parts = line.split()
                devices.append({
                    "id": parts[0],
                    "status": parts[1],
                    "info": " ".join(parts[2:]) if len(parts) > 2 else ""
                })

        return {"devices": devices}

    # iOS Simulator Methods

    async def _ios_launch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Launch iOS simulator"""
        if not self.xcode_path:
            raise RuntimeError("Xcode not found (macOS only)")

        device = params.get("device", "iPhone 14")

        # Get device UDID
        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "list", "devices", "-j",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        devices_data = json.loads(stdout.decode())

        device_udid = None
        for runtime, devices in devices_data.get("devices", {}).items():
            for dev in devices:
                if dev["name"] == device and dev["isAvailable"]:
                    device_udid = dev["udid"]
                    break
            if device_udid:
                break

        if not device_udid:
            raise ValueError(f"Device not found: {device}")

        # Boot simulator
        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "boot", device_udid
        )
        await proc.communicate()

        # Open Simulator app
        proc = await asyncio.create_subprocess_exec(
            "open", "-a", "Simulator"
        )
        await proc.communicate()

        return {
            "status": "launched",
            "device": device,
            "udid": device_udid
        }

    async def _ios_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Shutdown iOS simulator"""
        device_udid = params.get("device_udid", "all")

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "shutdown", device_udid
        )
        await proc.communicate()

        return {"status": "stopped"}

    async def _ios_install(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Install app on iOS simulator"""
        device_udid = params["device_udid"]
        app_path = params["app_path"]

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "install", device_udid, app_path
        )
        await proc.communicate()

        return {"status": "installed"}

    async def _ios_uninstall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Uninstall app from iOS simulator"""
        device_udid = params["device_udid"]
        bundle_id = params["bundle_id"]

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "uninstall", device_udid, bundle_id
        )
        await proc.communicate()

        return {"status": "uninstalled"}

    async def _ios_exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command on iOS simulator"""
        device_udid = params["device_udid"]
        command = params["command"]

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "spawn", device_udid, *command.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode
        }

    async def _ios_screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture screenshot from iOS simulator"""
        device_udid = params["device_udid"]
        output_path = params.get("output", "screenshot.png")

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "io", device_udid, "screenshot", output_path
        )
        await proc.communicate()

        return {"status": "captured", "path": output_path}

    async def _ios_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record screen on iOS simulator"""
        device_udid = params["device_udid"]
        duration = params.get("duration", 30)
        output_path = params.get("output", "recording.mp4")

        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "io", device_udid, "recordVideo", output_path
        )

        # Wait for duration
        await asyncio.sleep(duration)

        # Stop recording (Ctrl+C)
        proc.terminate()
        await proc.communicate()

        return {"status": "recorded", "path": output_path}

    async def _ios_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List iOS simulators"""
        proc = await asyncio.create_subprocess_exec(
            "xcrun", "simctl", "list", "devices", "-j",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        devices_data = json.loads(stdout.decode())

        devices = []
        for runtime, devs in devices_data.get("devices", {}).items():
            for dev in devs:
                devices.append({
                    "name": dev["name"],
                    "udid": dev["udid"],
                    "state": dev["state"],
                    "available": dev["isAvailable"],
                    "runtime": runtime
                })

        return {"devices": devices}
