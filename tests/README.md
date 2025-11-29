# Tests Directory

This directory contains comprehensive tests for the enhanced autonomous driving RL system, organized by test type and functionality.

## 📁 Test Organization

```
tests/
├── unit/                     # Basic functionality tests
│   ├── test_smoke.py        # Quick smoke tests
│   └── test_env_initialization.py  # Environment setup tests
├── integration/             # Feature integration tests
│   ├── test_enhanced_rewards.py     # Enhanced reward structure
│   ├── test_collision_avoidance.py  # Collision avoidance behavior
│   └── test_generalization.py       # Cross-scenario generalization
├── analysis/                # Analysis and exploration tools
│   ├── analyze_exploration.py       # PPO exploration analysis
│   └── check_exploration.py         # Exploration quality checks
├── demonstration/           # Demo and showcase scripts
│   └── demonstrate_safety_constraints.py  # Safety features demo
├── __init__.py             # Package initialization
└── README.md               # This file
```

## 🧪 Test Categories

### Unit Tests (`unit/`)
**Purpose**: Verify basic functionality and setup
**Scope**: Individual components, environment initialization
**Run Frequency**: On every code change

#### `test_smoke.py`
- Basic training pipeline functionality
- Configuration factory validation
- Core component initialization

#### `test_env_initialization.py`
- Environment creation and setup
- Observation/action space validation
- Basic environment interactions

### Integration Tests (`integration/`)
**Purpose**: Validate feature combinations and end-to-end workflows
**Scope**: Multi-component interactions, full training loops
**Run Frequency**: Before releases, after major changes

#### `test_enhanced_rewards.py`
- Scenario-aware speed optimization
- Lane position reward validation
- Progress and completion bonuses
- Overall reward structure coherence

#### `test_collision_avoidance.py`
- Safety constraint effectiveness
- Collision avoidance behavior analysis
- Hard constraint vs soft reward comparison
- Risk assessment validation

#### `test_generalization.py`
- Cross-scenario performance (highway → merge → intersection)
- Multi-modality generalization (lidar → grayscale → both)
- Curriculum learning effectiveness
- Transfer learning validation

### Analysis Tools (`analysis/`)
**Purpose**: Deep analysis of agent behavior and learning dynamics
**Scope**: Exploration quality, learning efficiency, behavioral patterns
**Run Frequency**: During development, performance debugging

#### `analyze_exploration.py`
- PPO entropy analysis across configurations
- Exploration vs exploitation balance
- Scenario-specific exploration patterns
- Action distribution analysis

#### `check_exploration.py`
- Quick exploration quality assessment
- Entropy validation across scenarios
- Behavioral diversity checks

### Demonstration Scripts (`demonstration/`)
**Purpose**: Showcase system capabilities and features
**Scope**: Visual demonstrations, feature highlights
**Run Frequency**: Presentations, documentation

#### `demonstrate_safety_constraints.py`
- Safety constraint visualization
- Emergency braking demonstrations
- Proximity detection showcases
- Hard vs soft constraint comparisons

## 🚀 Running Tests

### Run All Tests
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run all integration tests
python -m pytest tests/integration/ -v

# Run specific test
python tests/unit/test_smoke.py
```

### Run Analysis Tools
```bash
# Analyze exploration
python tests/analysis/analyze_exploration.py

# Quick exploration check
python tests/analysis/check_exploration.py
```

### Run Demonstrations
```bash
# Safety constraints demo
python tests/demonstration/demonstrate_safety_constraints.py
```

## 📊 Key Test Metrics

### Performance Metrics
- **Reward**: Average episode reward (higher is better)
- **Success Rate**: Percentage of successful episodes (target: >80%)
- **Crash Rate**: Safety violations (target: <20%)
- **Completion Rate**: Task completion percentage

### Safety Metrics
- **Constraint Activations**: How often safety overrides trigger
- **Proximity Warnings**: Early collision detection events
- **Emergency Actions**: Hard constraint interventions

### Learning Metrics
- **Entropy**: Action distribution randomness (exploration measure)
- **Convergence**: Training stability and final performance
- **Generalization**: Performance across unseen scenarios

## 🏗️ Test Architecture

### Test Fixtures
```python
@pytest.fixture
def enhanced_env():
    """Enhanced environment with safety constraints."""
    return EnhancedUrbanJunctionEnv(scenario="highway", modality="lidar")

@pytest.fixture
def trained_model(enhanced_env):
    """Pre-trained model for testing."""
    model = PPO("MlpPolicy", enhanced_env, verbose=0)
    model.learn(total_timesteps=1000)
    return model
```

### Test Structure
```python
def test_feature_name():
    """Test description and expected behavior."""
    # Arrange
    env = create_test_env()

    # Act
    result = perform_test_action(env)

    # Assert
    assert condition_met(result), "Expected behavior not observed"
```

## 🔧 Test Configuration

### Environment Settings
```python
TEST_CONFIG = {
    "scenarios": ["highway", "merge", "intersection"],
    "modalities": ["lidar", "grayscale", "both"],
    "episodes_per_test": 10,
    "max_steps_per_episode": 300,
    "random_seed": 42
}
```

### Performance Thresholds
```python
PERFORMANCE_THRESHOLDS = {
    "min_success_rate": 0.70,
    "max_crash_rate": 0.30,
    "min_avg_reward": 10.0,
    "min_safety_score": 0.80
}
```

## 📈 Continuous Integration

### Automated Testing
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: python -m pytest tests/unit/ -v
      - name: Run integration tests
        run: python -m pytest tests/integration/ -v --tb=short
```

### Test Coverage
```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

## 🐛 Debugging Tests

### Common Issues

**Environment Import Errors:**
```bash
# Ensure proper Python path
export PYTHONPATH=/path/to/rl_final:$PYTHONPATH
python tests/unit/test_env_initialization.py
```

**Model Loading Failures:**
```bash
# Check model file exists
ls results/models/*.zip

# Test model loading
python -c "from stable_baselines3 import PPO; PPO.load('model.zip')"
```

**Visualization Errors:**
```bash
# Install matplotlib backend
export MPLBACKEND=Agg
python tests/analysis/analyze_exploration.py
```

### Test Debugging Tools
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Run single test with debug
pytest tests/unit/test_smoke.py::TestTrainingPipeline::test_config_factory -s -v
```

## 📋 Test Development Guidelines

### Writing New Tests
1. **Categorize**: Place in appropriate subdirectory (unit/integration/analysis/demonstration)
2. **Document**: Clear docstrings explaining test purpose and expected behavior
3. **Isolate**: Tests should be independent and not rely on external state
4. **Parameterize**: Use pytest.mark.parametrize for multiple test cases
5. **Assert Clearly**: Meaningful assertion messages for debugging

### Test Naming Convention
```python
def test_feature_behavior_condition():
    """Test that feature works correctly under specific condition."""
    pass

def test_error_handling_edge_case():
    """Test error handling for edge cases."""
    pass
```

### Performance Benchmarking
```python
def test_performance_baseline(benchmark):
    """Benchmark critical path performance."""
    result = benchmark(test_function)
    assert result < PERFORMANCE_THRESHOLD
```

## 🎯 Quality Assurance

### Test Quality Metrics
- **Coverage**: >90% code coverage target
- **Reliability**: Tests pass consistently across runs
- **Speed**: Unit tests complete in <30 seconds
- **Maintainability**: Clear, well-documented test code

### Regression Prevention
- All tests must pass before merging
- Performance regressions trigger alerts
- Safety constraint violations block deployment

## 🔗 Integration with CI/CD

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Run tests before commit
pre-commit run --all-files
```

### Automated Reporting
- Test results posted to Slack/Teams
- Coverage reports uploaded to Codecov
- Performance metrics tracked in MLflow

This comprehensive test suite ensures the reliability, safety, and performance of the enhanced autonomous driving RL system across all scenarios and modalities.
