# BBX Troubleshooting Guide

## Common Issues and Solutions

---

## Docker Issues

### ❌ Problem: "Docker daemon not running"

**Symptoms:**
```
Error: Cannot connect to the Docker daemon
```

**Solutions:**

1. **Start Docker Desktop** (Windows/Mac)
   ```bash
   # Check if Docker is running
   docker info
   ```

2. **Start Docker Service** (Linux)
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **Check Docker socket permissions** (Linux)
   ```bash
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

**Verification:**
```bash
python -m blackbox.cli.main system check
# Should show: "Docker daemon running"
```

---

### ❌ Problem: "Docker pull timeout"

**Symptoms:**
```
Error: Failed to pull image alpine:latest
Timeout after 60 seconds
```

**Solutions:**

1. **Increase timeout** in test configuration
   ```python
   # In tests/e2e/test_01_system_requirements.py
   timeout=120  # Increase from 60 to 120
   ```

2. **Pre-pull common images**
   ```bash
   docker pull alpine:latest
   docker pull python:3.11
   docker pull node:18
   ```

3. **Check network connectivity**
   ```bash
   ping docker.io
   ```

4. **Use Docker mirror** (China/slow regions)
   ```json
   // In Docker Desktop settings -> Docker Engine
   {
     "registry-mirrors": ["https://mirror.example.com"]
   }
   ```

---

### ❌ Problem: "Permission denied" on volume mounting

**Symptoms:**
```
Error: Permission denied accessing /host/path
```

**Solutions:**

1. **Windows - Enable file sharing**
   - Docker Desktop → Settings → Resources → File Sharing
   - Add your project directory

2. **Linux - Fix permissions**
   ```bash
   chmod 755 /path/to/directory
   # Or run Docker with user namespace
   ```

3. **Use absolute paths**
   ```yaml
   volumes:
     "C:/Users/User/data": "/data"  # Windows
     "/home/user/data": "/data"     # Linux
   ```

---

## Timeout Issues

### ❌ Problem: "Command timed out after X seconds"

**Symptoms:**
```
Error: Command timed out after 2 seconds
```

**Solutions:**

1. **Increase timeout in workflow**
   ```yaml
   steps:
     - id: long_task
       inputs:
         timeout: 300  # 5 minutes
   ```

2. **Check command is not hanging**
   ```bash
   # Test command manually
   docker run alpine:latest your-command
   ```

3. **Reduce operation scope**
   ```yaml
   # Instead of processing all files
   cmd: ["process", "--limit", "100"]
   ```

---

## Memory Issues

### ❌ Problem: "Out of memory" error

**Symptoms:**
```
Error: Container killed due to OOM
```

**Solutions:**

1. **Increase Docker memory limit**
   - Docker Desktop → Settings → Resources → Memory
   - Increase to 4GB+

2. **Add memory limit to workflow**
   ```yaml
   resources:
     memory: "2g"
     cpu: "1.5"
   ```

3. **Process data in chunks**
   ```yaml
   cmd: ["process", "--batch-size", "1000"]
   ```

---

## Workflow Issues

### ❌ Problem: "Workflow validation failed"

**Symptoms:**
```
Error: Invalid workflow format
```

**Solutions:**

1. **Check YAML syntax**
   ```bash
   # Use online YAML validator
   # or Python yaml library
   python -c "import yaml; yaml.safe_load(open('workflow.bbx'))"
   ```

2. **Verify required fields**
   ```yaml
   name: Required          # Must have name
   version: 1.0           # Must have version
   steps:                 # Must have steps
     - id: step1          # Each step needs id
       mcp: universal     # Must specify adapter
   ```

3. **Check step dependencies**
   ```yaml
   # Ensure depends_on references exist
   - id: step2
     depends_on: [step1]  # step1 must exist
   ```

---

### ❌ Problem: "Step not executing"

**Symptoms:**
- Step appears skipped
- No output from step

**Solutions:**

1. **Check conditionals**
   ```yaml
   # Verify condition evaluates to true
   condition: "{{ inputs.deploy == true }}"
   ```

2. **Check dependencies**
   ```yaml
   # Ensure parent steps completed
   depends_on: [step1, step2]
   ```

3. **Enable debug logging**
   ```bash
   python -m blackbox.cli.main run workflow.bbx --verbose
   ```

---

## Template Issues

### ❌ Problem: "Template rendering error"

**Symptoms:**
```
Error: Undefined variable 'inputs'
jinja2.exceptions.UndefinedError
```

**Solutions:**

1. **Check variable exists**
   ```yaml
   # In workflow inputs
   inputs:
     my_var:
       type: string
       default: "value"
   
   # In step
   cmd: ["echo", "{{ inputs.my_var }}"]
   ```

2. **Use safe defaults**
   ```yaml
   cmd: ["echo", "{{ inputs.my_var | default('fallback') }}"]
   ```

3. **Check syntax**
   ```yaml
   # Correct
   "{{ inputs.value }}"
   
   # Wrong
   "{{ input.value }}"  # Missing 's'
   "${{ inputs.value }}" # Wrong syntax
   ```

---

## Package Issues

### ❌ Problem: "Package not found"

**Symptoms:**
```
Error: Package 'terraform' not found
```

**Solutions:**

1. **List available packages**
   ```bash
   python -m blackbox.cli.main package list
   ```

2. **Install package**
   ```bash
   python -m blackbox.cli.main package install terraform
   ```

3. **Check package location**
   ```bash
   ls blackbox/library/*.yaml
   ```

---

## Performance Issues

### ❌ Problem: "Workflow runs slowly"

**Solutions:**

1. **Enable parallel execution**
   ```yaml
   # Steps without depends_on run in parallel
   steps:
     - id: task1
       # runs in parallel
     - id: task2
       # runs in parallel
     - id: final
       depends_on: [task1, task2]
   ```

2. **Reduce Docker image pulls**
   ```bash
   # Pre-pull images
   docker pull alpine:latest
   docker pull python:3.11
   ```

3. **Use workflow cache**
   ```python
   results = await run_file(workflow_path, use_cache=True)
   ```

---

## Windows-Specific Issues

### ❌ Problem: "Path not found" on Windows

**Solutions:**

1. **Use forward slashes**
   ```yaml
   volumes:
     "C:/Users/User/data": "/data"  # Correct
     # Not: "C:\Users\User\data"
   ```

2. **Use WSL2 backend**
   - Docker Desktop → Settings → General
   - Enable "Use WSL 2 based engine"

3. **Check file sharing**
   - Settings → Resources → File Sharing
   - Add C:\Users directory

---

### ❌ Problem: "Line ending issues" (CRLF vs LF)

**Solutions:**

1. **Configure git**
   ```bash
   git config --global core.autocrlf input
   ```

2. **Use .gitattributes**
   ```
   *.bbx text eol=lf
   *.sh text eol=lf
   ```

---

## Testing Issues

### ❌ Problem: "Tests timing out"

**Solutions:**

1. **Increase pytest timeout**
   ```bash
   pytest tests/e2e/ --timeout=300
   ```

2. **Run specific test**
   ```bash
   pytest tests/e2e/test_02_universal_adapter.py -v
   ```

3. **Skip slow tests**
   ```bash
   pytest -m "not slow"
   ```

---

## Network Issues

### ❌ Problem: "Cannot reach external API"

**Solutions:**

1. **Check Docker network**
   ```bash
   docker run alpine:latest ping -c 3 google.com
   ```

2. **Configure proxy** (corporate network)
   ```yaml
   env:
     HTTP_PROXY: "http://proxy:8080"
     HTTPS_PROXY: "http://proxy:8080"
   ```

3. **Use host network** (Linux only)
   ```bash
   docker run --network host ...
   ```

---

## Debugging Tips

### Enable Verbose Logging

```bash
# CLI
python -m blackbox.cli.main run workflow.bbx --verbose

# Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Docker Logs

```bash
# View container logs
docker logs <container_id>

# Follow logs in real-time
docker logs -f <container_id>
```

### Inspect Workflow State

```python
from blackbox.core.runtime import run_file

results = await run_file('workflow.bbx')
print(results)  # See all step results
```

### Test Adapter Manually

```python
from blackbox.core.universal_v2 import UniversalAdapterV2

definition = {
    "id": "test",
    "uses": "docker://alpine:latest",
    "cmd": ["echo", "test"]
}

adapter = UniversalAdapterV2(definition)
result = await adapter.execute(method='run', inputs={})
print(result)
```

---

## Getting Help

### Health Check

```bash
python blackbox/core/health.py
```

### Run Test Suite

```bash
python run_e2e_tests.py
```

### Check Logs

```bash
# Enable logging
export BBX_LOG_LEVEL=DEBUG
python -m blackbox.cli.main run workflow.bbx
```

### Community Support

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** `docs/` directory
- **Examples:** `examples/` directory

---

## FAQ

**Q: Why is Docker required?**  
A: BBX uses Docker for consistent, isolated execution environments.

**Q: Can I run BBX without Docker?**  
A: No, Docker is a core requirement for the Universal Adapter.

**Q: What Python version is required?**  
A: Python 3.9 or higher.

**Q: How do I update BBX?**  
A: `git pull` and reinstall dependencies.

**Q: Where are workflows cached?**  
A: In memory during execution. No persistent cache by default.

**Q: Can I use private Docker images?**  
A: Yes, ensure `docker login` is configured.

---

## 📖 See Also

- **[Getting Started](GETTING_STARTED.md)** - Installation and first steps
- **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Workflow format reference
- **[Universal Adapter Guide](UNIVERSAL_ADAPTER.md)** - Docker-based execution
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[API Reference](API_REFERENCE.md)** - Adapter documentation
- **[Documentation Index](INDEX.md)** - Complete documentation navigation

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0

*Last Updated: November 2025*
