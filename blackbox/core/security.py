# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
Docker Image Security Scanner Integration
Supports: Trivy, Grype, Snyk
"""

import logging
import subprocess
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class SecurityVulnerability:
    """Security vulnerability found in image."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    vuln_id: str
    package: str
    fixed_version: Optional[str]
    description: str

class ImageSecurityScanner:
    """
    Scans Docker images for security vulnerabilities.
    
    Supports multiple scanners:
    - Trivy (recommended, open source)
    - Grype (Anchore)
    - Snyk (requires API key)
    """
    
    def __init__(self, scanner: str = "trivy"):
        """
        Initialize scanner.
        
        Args:
            scanner: Scanner to use ('trivy', 'grype', 'snyk')
        """
        self.scanner = scanner
        self.logger = logging.getLogger("bbx.security")
    
    def scan_image(self, image: str, severity_threshold: str = "HIGH") -> Dict[str, Any]:
        """
        Scan Docker image for vulnerabilities.
        
        Args:
            image: Docker image to scan
            severity_threshold: Minimum severity to report
            
        Returns:
            Scan results with vulnerabilities
        """
        if self.scanner == "trivy":
            return self._scan_with_trivy(image, severity_threshold)
        elif self.scanner == "grype":
            return self._scan_with_grype(image, severity_threshold)
        elif self.scanner == "snyk":
            return self._scan_with_snyk(image, severity_threshold)
        else:
            raise ValueError(f"Unknown scanner: {self.scanner}")
    
    def _scan_with_trivy(self, image: str, severity: str) -> Dict[str, Any]:
        """Scan with Trivy."""
        try:
            self.logger.info(f"🔍 Scanning {image} with Trivy...")
            
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "aquasec/trivy:latest",
                    "image",
                    "--severity", severity,
                    "--format", "json",
                    image
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "vulnerabilities": []
                }
            
            scan_data = json.loads(result.stdout)
            vulnerabilities = self._parse_trivy_output(scan_data)
            
            return {
                "success": True,
                "scanner": "trivy",
                "image": image,
                "vulnerability_count": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "critical_count": sum(1 for v in vulnerabilities if v.severity == "CRITICAL"),
                "high_count": sum(1 for v in vulnerabilities if v.severity == "HIGH")
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Scan timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_trivy_output(self, data: dict) -> List[SecurityVulnerability]:
        """Parse Trivy JSON output."""
        vulnerabilities = []
        
        for result in data.get("Results", []):
            for vuln in result.get("Vulnerabilities", []):
                vulnerabilities.append(SecurityVulnerability(
                    severity=vuln.get("Severity", "UNKNOWN"),
                    vuln_id=vuln.get("VulnerabilityID", ""),
                    package=vuln.get("PkgName", ""),
                    fixed_version=vuln.get("FixedVersion"),
                    description=vuln.get("Title", "")
                ))
        
        return vulnerabilities
    
    def _scan_with_grype(self, image: str, severity: str) -> Dict[str, Any]:
        """Scan with Grype (Anchore)."""
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "anchore/grype:latest",
                    image,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            
            # Parse Grype output (simplified)
            scan_data = json.loads(result.stdout)
            return {
                "success": True,
                "scanner": "grype",
                "image": image,
                "vulnerability_count": len(scan_data.get("matches", [])),
                "raw_data": scan_data
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scan_with_snyk(self, image: str, severity: str) -> Dict[str, Any]:
        """Scan with Snyk (requires SNYK_TOKEN env var)."""
        import os
        
        if not os.getenv("SNYK_TOKEN"):
            return {"success": False, "error": "SNYK_TOKEN not set"}
        
        # Placeholder - would require Snyk CLI integration
        return {
            "success": False,
            "error": "Snyk integration not yet implemented"
        }
    
    def is_image_safe(self, image: str, max_critical: int = 0, max_high: int = 5) -> bool:
        """
        Check if image meets security thresholds.
        
        Args:
            image: Image to check
            max_critical: Maximum allowed critical vulnerabilities
            max_high: Maximum allowed high vulnerabilities
            
        Returns:
            True if image is safe
        """
        scan_result = self.scan_image(image, "HIGH")
        
        if not scan_result.get("success"):
            self.logger.warning(f"Security scan failed for {image}")
            return False
        
        critical = scan_result.get("critical_count", 0)
        high = scan_result.get("high_count", 0)
        
        is_safe = critical <= max_critical and high <= max_high
        
        if not is_safe:
            self.logger.warning(
                f"⚠️  {image} has {critical} CRITICAL and {high} HIGH vulnerabilities"
            )
        
        return is_safe
