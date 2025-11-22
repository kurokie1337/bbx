# BBX Test Suite - Updated README

## Test Structure (Bottom-Up)

### Unit Tests (`tests/unit/`)
**No external dependencies** - Can run without Docker, network, or credentials.

- `test_auth_providers.py` - Auth provider logic
- `test_schema_validation.py` - YAML schema validation  
- `test_package_manager.py` - Package manager (file system only)

**Run:** `pytest tests/unit/ -v`

### Integration Tests (`tests/integration/`)
**Requires minimal setup** - May need Docker installed for some tests.

- `test_cli_commands.py` - CLI command execution

**Run:** `pytest tests/integration/ -v`

### E2E Tests (`tests/e2e/`)
**Full system tests** - Requires Docker, will pull images and run containers.

Tests organized in progressive layers:

1. **Layer 1: System Requirements** (`test_01_system_requirements.py`)
   - Python version check
   - Docker availability
   - Can pull images
   - Can run containers
   - Project structure validation

2. **Layer 2: Universal Adapter** (`test_02_universal_adapter.py`)
   - Real Docker execution
   - Command templating
   - Error handling
   - JSON output parsing
   - Timeout enforcement

3. **Layer 3: Package Manager** (`test_03_package_manager.py`)
   - Real library package loading
   - Package validation
   - Install/reload operations

4. **Layer 4: CLI Execution** (`test_04_cli_execution.py`)
   - Real CLI subprocess calls
   - Package commands
   - System health check
   - Workflow execution

**Run:** `python run_e2e_tests.py` (runs in stages, stops on required failures)

---

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Layer
```bash
# Unit only (fast, no Docker)
pytest tests/unit/ -v

# Integration (needs Docker)
pytest tests/integration/ -v

# E2E (full system, takes time)
python run_e2e_tests.py
```

### E2E by Stage
```bash
# Just system checks
pytest tests/e2e/test_01_system_requirements.py -v

# Just Universal Adapter
pytest tests/e2e/test_02_universal_adapter.py -v
```

### With Timeout Protection
```bash
pytest tests/ -v --timeout=60
```

### With Coverage
```bash
pytest tests/ --cov=blackbox --cov-report=html
```

---

## E2E Test Philosophy

1. **Progressive Difficulty:** Each layer tests more complex functionality
2. **Fail Fast:** Required layers stop execution if they fail
3. **Real Execution:** E2E tests use actual Docker containers
4. **Timeout Protected:** All E2E tests have timeouts to prevent hangs

---

## Estimated Runtimes

| Test Suite | Time | Docker Required |
|------------|------|-----------------|
| Unit | ~2-5s | No |
| Integration | ~5-10s | Optional |
| E2E Layer 1 | ~30-60s | Yes (pulls alpine) |
| E2E Layer 2 | ~30-90s | Yes |
| E2E Layer 3 | ~10-20s | No |
| E2E Layer 4 | ~60-120s | Yes |

---

## What Each Test Layer Validates

### Unit Tests
✅ Python code syntax and logic  
✅ Import structure  
✅ Function/class behavior  
❌ No Docker  
❌ No external services  

### Integration Tests
✅ Click CLI framework  
✅ Command routing  
✅ Argument parsing  
⚠️ May use Docker if available  

### E2E Tests
✅ Full system integration  
✅ Real Docker execution  
✅ Actual image pulls  
✅ Container lifecycle  
✅ End-user workflows  

---

## Troubleshooting

### E2E Tests Fail at Layer 1
**Problem:** Docker not available  
**Solution:** Install Docker Desktop and ensure daemon is running

### E2E Tests Timeout
**Problem:** Slow network or Docker  
**Solution:** Pre-pull images: `docker pull alpine:latest`

### E2E Tests Skip
**Problem:** Missing prerequisites  
**Solution:** Check pytest output for skip reasons

---

## Next Steps

After all E2E tests pass:
1. Run real workflows from `examples/`
2. Test with production-like data
3. Performance benchmarking
4. Security scanning integration
