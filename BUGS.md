# BUGS.md - Potential Issues and Silent Failures

**Generated:** 2025-11-14
**Repository:** quantum-pipeline
**Analysis Scope:** Complete codebase review including all Python modules, configurations, tests, and infrastructure

---

## Table of Contents
1. [Critical Bugs](#critical-bugs)
2. [Silent Failures](#silent-failures)
3. [Logic Errors](#logic-errors)
4. [Race Conditions & Concurrency Issues](#race-conditions--concurrency-issues)
5. [Data Integrity Issues](#data-integrity-issues)
6. [Configuration & Deployment Bugs](#configuration--deployment-bugs)
7. [Testing Gaps](#testing-gaps)

---

## Critical Bugs

### 1. Optimizer Configuration Mutually Exclusive Parameters Not Enforced

**Location:** `quantum_pipeline/solvers/optimizer_config.py:20-24`

```python
if max_iterations is not None and convergence_threshold is not None:
    raise ValueError(
        "max_iterations and convergence_threshold are mutually exclusive. "
        "Please specify only one."
    )
```

**Issue:** While the OptimizerConfig class raises an error, the VQESolver itself (`quantum_pipeline/solvers/vqe_solver.py`) accepts both parameters without validation and passes them through. This means the validation is bypassed when parameters are set directly on VQESolver.

**Impact:** Users can set both parameters, leading to undefined optimizer behavior where priority rules are silently applied instead of explicit errors.

**Reproduction:**
```python
solver = VQESolver(
    qubit_op=op,
    backend_config=config,
    max_iterations=100,
    convergence_threshold=1e-6,  # Both set - should error but doesn't
    optimizer='COBYLA'
)
```

**Fix:** Add validation in VQESolver.__init__() before passing to optimizer configuration.

---

### 2. Schema Registry Cache Inconsistency

**Location:** `quantum_pipeline/utils/schema_registry.py:154-157`

```python
if schema_name not in self.registry_schema_existence:
    schema_in_registry = self.is_schema_in_registry(schema_name)
else:
    schema_in_registry = self.registry_schema_existence[schema_name]
```

**Issue:** The cache `registry_schema_existence` is never invalidated. If a schema is added to the registry externally or by another process, the cache will continue to report it as missing.

**Impact:**
- Schemas uploaded to registry won't be detected
- Multiple containers may attempt to register the same schema
- Schema synchronization failures in distributed environments

**Silent Failure Mode:** The system falls back to local files without informing the user that the registry has newer versions.

**Fix:** Implement cache TTL or add a force-refresh parameter.

---

### 3. Airflow Variable Overwrite Without Verification

**Location:** `docker/airflow/quantum_processing_dag.py:67-68`

```python
Variable.set('MINIO_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY'))
Variable.set('MINIO_SECRET_KEY', os.getenv('MINIO_SECRET_KEY'))
```

**Issue:** Variables are set on every DAG load without checking if `os.getenv()` returns None. If environment variables are missing, this overwrites existing valid Airflow variables with None.

**Impact:**
- Silent credential loss on DAG reload
- Spark jobs fail with authentication errors
- No error until job execution

**Reproduction:**
1. Set valid credentials in Airflow UI
2. Restart Airflow without env vars set
3. DAG loads and overwrites with None
4. Spark job fails with cryptic S3 auth error

**Fix:** Only set if environment variable exists:
```python
if os.getenv('MINIO_ACCESS_KEY'):
    Variable.set('MINIO_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY'))
```

---

### 4. Spark DataFrame Empty Check After Cache

**Location:** `docker/airflow/scripts/quantum_incremental_processing.py:161-162`

```python
if num_partitions:
    df = df.repartition(num_partitions)
return df.cache()
```

**Issue:** DataFrame is cached before checking if it's empty. Later code calls `df.isEmpty()` on cached DataFrame, but if the check is never performed and the DataFrame is used, it can cause expensive operations on empty data.

**Impact:**
- Wasted memory caching empty DataFrames
- Unnecessary Spark transformations
- Delayed error detection

---

### 5. Kafka Producer Close in __exit__ Without Suppressing Exceptions

**Location:** `quantum_pipeline/stream/kafka_interface.py:217-218`

```python
def __exit__(self, exc_type, exc_value, traceback):
    self.close()
```

**Issue:** If `close()` raises an exception, it will mask any exception that occurred in the `with` block. This is a common Python anti-pattern.

**Impact:**
- Original exceptions are lost
- Debugging becomes difficult
- Silent failures when both operations fail

**Fix:**
```python
def __exit__(self, exc_type, exc_value, traceback):
    try:
        self.close()
    except Exception:
        if exc_type is None:  # Only raise if no previous exception
            raise
        # Otherwise, suppress close exception and let original propagate
```

---

## Silent Failures

### 1. Performance Monitoring Disabled by Default

**Location:** `quantum_pipeline/configs/settings.py:68`

```python
PERFORMANCE_MONITORING_ENABLED = False  # Global toggle
```

**Issue:** Performance monitoring is disabled by default, meaning users won't collect metrics unless they explicitly enable it. This is a silent failure because:
- No warning is logged when metrics aren't being collected
- VQE runs complete "successfully" without performance data
- Grafana dashboards remain empty
- Users may not realize monitoring needs explicit enabling

**Impact:**
- Loss of valuable performance data for optimization
- Thesis comparisons incomplete without GPU/CPU metrics
- No way to retroactively gather metrics

---

### 2. Molecule Loader Missing 'masses' Field Silently Uses Default

**Location:** `quantum_pipeline/drivers/molecule_loader.py:50`

```python
masses=mol.get('masses', None),
```

**Issue:** When 'masses' is missing from molecule JSON, it silently defaults to None. PySCF then calculates masses automatically, but users aren't warned that their input was incomplete.

**Impact:**
- Users may expect specific isotope masses
- Results differ from expectations without explanation
- Debugging why energies don't match references is difficult

**Recommendation:** Add logging when defaults are used:
```python
if 'masses' not in mol:
    logger.warning(f"Masses not specified for molecule {mol['symbols']}, using atomic defaults")
```

---

### 3. Schema Registry Unavailable Falls Back Silently

**Location:** `quantum_pipeline/utils/schema_registry.py:64-67`

```python
if not self.is_schema_registry_available():
    self.logger.warning('Schema registry is not available.')
    self.registry_schema_existence[schema_name] = False
    return False
```

**Issue:** When schema registry is unavailable, the system logs a WARNING and continues using local schemas. In a production environment, this means:
- Schema versioning is lost
- Multiple producers may use different schema versions
- No error is raised, leading to potential deserialization issues downstream

**Impact:**
- Kafka consumers may fail to deserialize messages
- Schema evolution breaks without notification
- Data corruption risk in Kafka topics

**Recommendation:** Make schema registry availability configurable (strict vs. permissive mode).

---

### 4. Docker Stats Collection Failure Is Silent

**Location:** `quantum_pipeline/monitoring/performance_monitor.py:279-280`

```python
if result.returncode != 0:
    return {'container_name': container_name, 'docker_stats_available': False}
```

**Issue:** When Docker stats fail to collect (common when running without Docker socket access), the system silently returns `docker_stats_available: False` without logging why it failed.

**Impact:**
- Users don't know why container metrics are missing
- `result.stderr` is ignored, hiding useful error messages
- Monitoring appears to work but is incomplete

**Fix:** Log the stderr:
```python
if result.returncode != 0:
    self.logger.warning(f'Docker stats failed: {result.stderr}')
    return {'container_name': container_name, 'docker_stats_available': False}
```

---

### 5. Spark Incremental Processing Skips Empty New Records Without Notification

**Location:** `docker/airflow/scripts/quantum_incremental_processing.py:316-319`

```python
new_record_count = truly_new_data.count()
if new_record_count == 0:
    print(f'No new records found for table {table_name}')
    return None, 0
```

**Issue:** When no new records are found, function returns silently with 0 count. The calling code doesn't distinguish between:
- No new data available (expected)
- All data was already processed (expected)
- Data processing failed (unexpected)

**Impact:**
- Monitoring dashboards can't distinguish between normal and abnormal zero counts
- Silent data loss if filtering logic is broken
- Debugging requires log analysis

---

## Logic Errors

### 1. Iteration ID Collision Risk

**Location:** `docker/airflow/scripts/quantum_incremental_processing.py:569-572`

```python
'iteration_id',
expr(
    "concat(experiment_id, '_iter_', cast(hash(concat(experiment_id, cast(iteration_step as string))) % 1000000 as string))"
),
```

**Issue:** Hash collision risk - hash modulo 1000000 can produce duplicates. If two iterations in the same experiment happen to hash to the same value, their iteration_ids will collide.

**Probability:** ~0.0001% for 100 iterations, but increases with more experiments.

**Impact:**
- Primary key violations in iteration_parameters table
- Data loss or corruption
- Spark job failures

**Fix:** Use a proper UUID or monotonic sequence:
```python
'iteration_id', expr("concat(experiment_id, '_iter_', cast(iteration_step as string))")
```

---

### 2. Incorrect Priority Logic Comment

**Location:** `quantum_pipeline/solvers/vqe_solver.py:128-132`

```python
if self.convergence_threshold and self.max_iterations:
    self.logger.info(
        f'Starting VQE optimization with max iterations {self.max_iterations} taking priority over '
        f'convergence threshold {self.convergence_threshold}'
    )
```

**Issue:** The comment claims max_iterations takes priority, but looking at the optimizer_config.py, this is only true for specific optimizers (L-BFGS-B). For COBYLA and SLSQP, both parameters are passed and scipy decides priority.

**Impact:**
- Misleading documentation
- Users expect consistent behavior across optimizers
- Actual behavior varies by optimizer

---

### 3. Incorrect String Interpolation in Log Statement

**Location:** `quantum_pipeline/solvers/vqe_solver.py:180`

```python
f'Calculations on quantum hardware via IBMQ completed in {t.elapsed:}.6f seconds.'
```

**Issue:** The format string is malformed - should be `{t.elapsed:.6f}` not `{t.elapsed:}.6f`. The dot is outside the format spec.

**Impact:**
- Incorrect formatting in logs
- May cause formatting errors in some Python versions

---

### 4. Optimizer Options Mutation in Factory Pattern

**Location:** `quantum_pipeline/solvers/optimizer_config.py`

**Issue:** The `get_options()` method builds a dictionary that is returned directly. If the calling code modifies this dictionary, it doesn't affect the config object, but users might expect it to. This violates the principle of least surprise.

**Recommendation:** Document that returned dict should be treated as immutable, or return a copy:
```python
return options.copy()
```

---

## Race Conditions & Concurrency Issues

### 1. Global Performance Monitor Singleton Without Thread Lock

**Location:** `quantum_pipeline/monitoring/performance_monitor.py:626-631`

```python
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(**kwargs)
    return _global_monitor
```

**Issue:** Classic check-then-act race condition. If two threads call `get_performance_monitor()` simultaneously, both might see `_global_monitor is None` and create two instances.

**Impact:**
- Multiple monitoring threads started
- Duplicate metrics exports
- Prometheus PushGateway pollution
- Memory leaks

**Fix:** Use double-checked locking:
```python
import threading
_monitor_lock = threading.Lock()

def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    global _global_monitor
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor(**kwargs)
    return _global_monitor
```

---

### 2. Schema Registry Cache Access Without Locks

**Location:** `quantum_pipeline/utils/schema_registry.py`

**Issue:** Multiple attributes are modified without thread synchronization:
- `self.schema_cache` (line 16)
- `self.id_cache` (line 17)
- `self.registry_schema_existence` (line 23)

If multiple Kafka producers run in threads (which is common), concurrent access to these dicts can cause:
- Race conditions
- Cache corruption
- Missing schema registrations

**Fix:** Use `threading.Lock` or `collections.defaultdict` with thread-safe operations.

---

### 3. Monitoring Thread Stop Signal Check Frequency

**Location:** `quantum_pipeline/monitoring/performance_monitor.py:335-336`

```python
if self.stop_monitoring.wait(self.collection_interval):
    break
```

**Issue:** If `collection_interval` is set to a large value (e.g., 60 seconds), the monitoring thread won't check the stop signal for up to 60 seconds, causing slow shutdown.

**Impact:**
- Context manager exit hangs for up to collection_interval seconds
- Application shutdown delays
- Docker container stop delays (may cause SIGKILL)

**Fix:** Check stop signal more frequently:
```python
# Wait in small increments, checking stop signal
for _ in range(self.collection_interval):
    if self.stop_monitoring.wait(1):
        break
```

---

## Data Integrity Issues

### 1. No Validation of Energy Values

**Issue:** VQE can produce invalid energy values (NaN, Inf) when optimization fails, but there's no validation before:
- Storing in Kafka
- Writing to Iceberg tables
- Exporting to Prometheus

**Impact:**
- Invalid data pollutes data lake
- Grafana charts break with NaN values
- ML models trained on bad data

**Recommendation:** Add validation in `VQEResult`:
```python
if not np.isfinite(res.fun):
    raise ValueError(f"Invalid energy value: {res.fun}")
```

---

### 2. Molecule Coordinates Not Validated

**Location:** `quantum_pipeline/drivers/molecule_loader.py:42-44`

```python
molecules.append(
    MoleculeInfo(
        symbols=mol['symbols'],
        coords=mol['coords'],
```

**Issue:** No validation that:
- `coords` is a properly shaped array
- `len(coords)` matches `len(symbols)`
- Coordinates are finite numbers
- Units are valid

**Impact:**
- PySCF fails deep in the stack with cryptic errors
- Impossible to determine which molecule is malformed
- Wastes compute time on invalid inputs

---

### 3. Iceberg Table Version Tags Can Overwrite

**Location:** `docker/airflow/scripts/quantum_incremental_processing.py:295-298`

```python
spark.sql(f"""
ALTER TABLE quantum_catalog.quantum_features.{table_name}
CREATE TAG {version_tag} AS OF VERSION {snapshot_id}
""")
```

**Issue:** If the same `processing_batch_id` is somehow reused (UUID collision, clock rollback, etc.), the tag creation will fail or overwrite.

**Impact:**
- Lost time-travel capability
- Confusion about which version is which
- Silent data loss if overwrite succeeds

**Fix:** Use `CREATE TAG IF NOT EXISTS` or check for existence first.

---

## Configuration & Deployment Bugs

### 1. Hardcoded Python Version in Conda Environment

**Location:** `docker/Dockerfile.gpu:43`

```dockerfile
conda create -y -n qenv python=3.12.9 && conda clean -afy
```

**Issue:** Hardcoded to 3.12.9 while CPU dockerfile uses 3.12-slim. Version drift between GPU and CPU containers.

**Impact:**
- Subtle behavior differences between GPU and CPU
- Dependency resolution differences
- Hard to maintain version parity

---

### 2. Missing Healthchecks in docker-compose

**Location:** `examples/docker-compose-kafka.yml`

**Issue:** No healthcheck configured for Kafka service. The quantum-pipeline container starts immediately without waiting for Kafka to be ready.

**Impact:**
- Race condition on startup
- Producer initialization fails
- Requires manual restarts or retry logic

**Fix:** Add healthcheck and depends_on conditions:
```yaml
kafka:
  healthcheck:
    test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server=localhost:9092"]
    interval: 10s
    timeout: 5s
    retries: 5
```

---

### 3. Flake8 Max Complexity Warning Only (Not Enforced)

**Location:** `.github/workflows/docker-publish.yml:42`

```yaml
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

**Issue:** `--exit-zero` means complexity violations don't fail the build. The VQERunner.__init__() has complexity > 10 (line 25 marked with `# noqa: C901`).

**Impact:**
- Code complexity grows unchecked
- Maintenance burden increases
- Bugs hide in complex functions

---

### 4. Docker Build Context Copies Entire Repo

**Location:** `docker/Dockerfile.cpu:9`

```dockerfile
COPY . .
```

**Issue:** Copies everything including `.git/`, `tests/`, `examples/`, etc. This increases:
- Build time
- Image size
- Security risk (secrets in .git history)

**Fix:** Use `.dockerignore`:
```
.git
tests
examples
*.md
.github
gen/
```

---

## Testing Gaps

### 1. No Integration Tests for Spark Jobs

**Issue:** The Spark processing script (`quantum_incremental_processing.py`) has no tests. This is 844 lines of critical data transformation logic with zero test coverage.

**Impact:**
- Regressions go undetected
- Refactoring is risky
- Data corruption can go unnoticed until production

---

### 2. Kafka Producer Tests Skip When Kafka Unavailable

**Location:** `tests/stream/test_kafka_interface.py:136`

```python
except NoBrokersAvailable:
    pytest.skip('Skipping test: No Kafka brokers available')
```

**Issue:** Tests are skipped in CI/CD where Kafka isn't running, providing false confidence.

**Recommendation:** Run tests against embedded Kafka or Docker Compose in CI.

---

### 3. No Tests for Optimizer Mutually Exclusive Parameters

**Issue:** While `test_optimizer_config.py` tests the factory, there are no tests verifying that VQESolver enforces the mutual exclusivity.

**Evidence:** Looking at `test_vqe_solver.py:249-262`, there's a test that creates a solver with both parameters but doesn't verify an error is raised.

---

### 4. Performance Monitor Tests Use Mocks Instead of Real Subprocess

**Issue:** Tests in `test_performance_monitor.py` don't test the actual Docker stats collection, meaning bugs in parsing Docker output won't be caught.

---

### 5. No End-to-End Test

**Issue:** No test that:
1. Loads a molecule
2. Runs VQE
3. Publishes to Kafka
4. Verifies Avro serialization
5. Checks schema registry
6. Validates Prometheus metrics

**Impact:** Integration bugs only discovered in production.

---

## Recommendations

### Priority 1 (Critical)
1. Fix Airflow variable overwrite bug (Section 3 of Critical Bugs)
2. Add thread safety to global monitor singleton (Section 1 of Race Conditions)
3. Implement energy value validation (Section 1 of Data Integrity)

### Priority 2 (Important)
1. Fix schema registry cache invalidation
2. Add proper exception handling in Kafka producer __exit__
3. Fix iteration_id collision risk
4. Add Docker healthchecks

### Priority 3 (Quality of Life)
1. Add logging for silent fallbacks
2. Improve test coverage for Spark jobs
3. Fix string formatting in log statements
4. Add .dockerignore files

---

## Verification Steps

For each bug fix:
1. Write a failing test that demonstrates the bug
2. Implement the fix
3. Verify the test passes
4. Add regression test to CI/CD
5. Document the fix in CHANGELOG.md

---

**End of Bug Report**
