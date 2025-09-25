# Contributing to SatMon
## Development Guidelines and Standards

Thank you for your interest in contributing to SatMon! This document provides guidelines for maintaining the high code quality and professional standards that make this project production-ready.

## 🎯 Project Vision

SatMon aims to be the gold standard for satellite telemetry anomaly detection, demonstrating enterprise-level software engineering practices while solving real aerospace challenges.

## 🏗 Development Environment Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git
- PostgreSQL (optional, SQLite fallback available)

### Initial Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/satmon.git
cd satmon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r test_requirements.txt

# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Initialize database
python -c "from services.api.db import create_tables; create_tables()"

# Run initial tests
pytest tests/ -v
```

## 📋 Development Workflow

### Branch Strategy
- `main`: Production-ready code, protected branch
- `develop`: Integration branch for features
- `feature/*`: Individual feature development
- `hotfix/*`: Critical production fixes

### Feature Development Process

1. **Create Feature Branch**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **Development Cycle**
```bash
# Make your changes
# Write comprehensive tests
# Update documentation

# Run quality checks
black .                    # Code formatting
isort .                   # Import sorting
flake8 .                  # Linting
mypy scripts/ services/   # Type checking
bandit -r scripts/ services/  # Security check
pytest tests/ -v --cov=scripts --cov=services  # Tests with coverage
```

3. **Commit Standards**
```bash
# Use conventional commits
git commit -m "feat: add LSTM autoencoder algorithm"
git commit -m "fix: resolve memory leak in data processing"
git commit -m "docs: update API documentation"
git commit -m "test: add integration tests for anomaly detection"
```

4. **Pull Request Process**
- Ensure all CI checks pass
- Add detailed description of changes
- Include performance impact analysis
- Link to relevant issues
- Request review from maintainers

## 🧪 Testing Standards

### Test Categories

#### Unit Tests
```python
# Example: Test individual algorithm components
def test_zscore_detection_accuracy():
    detector = ZScoreDetector(window_size=50, threshold=3.0)
    test_data = generate_test_signal_with_anomalies()
    anomalies, scores = detector.detect(test_data)
    
    precision, recall, f1 = calculate_metrics(test_data.ground_truth, anomalies)
    assert recall >= 0.8, f"Recall too low: {recall}"
    assert precision >= 0.2, f"Precision too low: {precision}"
```

#### Integration Tests
```python
# Example: Test complete API workflows
@pytest.mark.asyncio
async def test_end_to_end_anomaly_detection():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test data ingestion
        response = await client.post("/timeseries", json=test_telemetry_data)
        assert response.status_code == 201
        
        # Test anomaly detection
        response = await client.post("/detect", json={
            "channel": "test_channel",
            "algorithm": "zscore"
        })
        assert response.status_code == 200
        assert "anomalies" in response.json()
```

#### Performance Tests
```python
# Example: Ensure performance requirements are met
def test_algorithm_performance_requirements():
    large_dataset = generate_large_test_dataset(10000)
    
    start_time = time.time()
    anomalies = detect_anomalies_zscore(large_dataset)
    execution_time = time.time() - start_time
    
    throughput = len(large_dataset) / execution_time
    assert throughput >= 1000, f"Throughput too low: {throughput:.0f} points/s"
    assert execution_time <= 10, f"Execution too slow: {execution_time:.2f}s"
```

### Coverage Requirements
- **Minimum coverage**: 80% overall
- **Critical components**: 95% coverage required
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests

## 📊 Performance Standards

### Algorithm Performance Requirements
| Metric | Z-Score | Isolation Forest | LSTM |
|--------|---------|------------------|------|
| Min Throughput | 10,000 pts/s | 5,000 pts/s | 8,000 pts/s |
| Max Execution Time | 1s per 1000 pts | 2s per 1000 pts | 1.5s per 1000 pts |
| Memory Usage | <50MB peak | <100MB peak | <200MB peak |
| Min Recall | 80% | 40% | 70% |

### API Performance Requirements
- **Response Time**: <100ms for 95% of requests
- **Throughput**: 500 requests/second
- **Memory Usage**: <500MB under normal load
- **Error Rate**: <0.1% under normal conditions

## 🎨 Code Style Guidelines

### Python Code Standards
```python
# Follow PEP 8 with these specific additions:

# 1. Use type hints for all function signatures
def detect_anomalies(data: List[float], threshold: float = 3.0) -> Tuple[List[bool], List[float]]:
    """Detect anomalies in time series data.
    
    Args:
        data: Time series values
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Tuple of (anomaly indicators, anomaly scores)
        
    Raises:
        ValueError: If data is empty or threshold is invalid
    """
    pass

# 2. Use comprehensive docstrings
class AnomalyDetector:
    """Base class for anomaly detection algorithms.
    
    This class provides the interface for all anomaly detection
    implementations in SatMon, ensuring consistent behavior
    across different algorithms.
    
    Attributes:
        name: Human-readable algorithm name
        parameters: Algorithm configuration parameters
        
    Example:
        detector = AnomalyDetector("Custom Algorithm")
        anomalies = detector.detect(telemetry_data)
    """
    pass

# 3. Use meaningful variable names
satellite_telemetry_data = load_telemetry_channel("GOES-16")
anomaly_detection_results = run_detection_pipeline(satellite_telemetry_data)
performance_metrics = calculate_algorithm_performance(anomaly_detection_results)
```

### Database Conventions
```python
# Use SQLAlchemy models with proper relationships
class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    telemetry_points = relationship("Telemetry", back_populates="channel")
    anomalies = relationship("Anomaly", back_populates="channel")
    
    def __repr__(self):
        return f"<Channel(name='{self.name}')>"
```

### API Design Patterns
```python
# Follow REST conventions with comprehensive validation
@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: Session = Depends(get_db)
) -> AnomalyDetectionResponse:
    """Run anomaly detection on specified channel data.
    
    This endpoint provides real-time anomaly detection capabilities
    for satellite telemetry data using multiple algorithms.
    """
    # Input validation
    if not request.channel:
        raise HTTPException(400, "Channel name required")
    
    # Business logic
    detector = get_algorithm_detector(request.algorithm)
    results = await detector.detect_anomalies(request.channel, db)
    
    # Structured response
    return AnomalyDetectionResponse(
        channel=request.channel,
        algorithm=request.algorithm,
        anomalies=results.anomalies,
        performance_metrics=results.metrics,
        execution_time=results.execution_time
    )
```

## 📚 Documentation Standards

### Code Documentation
- **All public functions**: Comprehensive docstrings with examples
- **Complex algorithms**: Inline comments explaining logic
- **Configuration**: Document all parameters and defaults
- **Error handling**: Document all exceptions that can be raised

### README Files
- **Clear purpose statement**: What the component does
- **Installation instructions**: Step-by-step setup
- **Usage examples**: Real-world use cases
- **API reference**: If applicable
- **Performance notes**: Known limitations or optimizations

### API Documentation
- **OpenAPI/Swagger**: Auto-generated from FastAPI
- **Request/response examples**: Real JSON samples
- **Error codes**: All possible HTTP status codes
- **Rate limiting**: If applicable
- **Authentication**: Security requirements

## 🔒 Security Guidelines

### General Security Practices
- **Input validation**: Sanitize all external inputs
- **SQL injection prevention**: Use parameterized queries
- **XSS prevention**: Escape output in templates
- **CSRF protection**: For state-changing operations
- **Rate limiting**: Prevent abuse of API endpoints

### Secrets Management
```python
# Use environment variables for sensitive data
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///satmon.db")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# Never commit secrets to version control
# Use .env files for local development (in .gitignore)
# Use proper secrets management in production
```

### Docker Security
```dockerfile
# Use non-root users
RUN groupadd -r satmon && useradd -r -g satmon satmon
USER satmon

# Minimal attack surface
FROM python:3.11-slim
# Install only required packages

# Health checks
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health
```

## 📈 Performance Optimization

### Algorithm Optimization
- **Vectorization**: Use NumPy operations instead of loops
- **Memory efficiency**: Process data in batches
- **Caching**: Cache expensive computations
- **Profiling**: Use cProfile for performance analysis

### Database Optimization
- **Indexing**: Index frequently queried columns
- **Connection pooling**: Reuse database connections
- **Query optimization**: Use efficient SQL patterns
- **Bulk operations**: Batch inserts/updates

### API Optimization
- **Async processing**: Use asyncio for I/O operations
- **Response compression**: Enable gzip compression
- **Caching headers**: Set appropriate cache headers
- **Pagination**: Limit large result sets

## 🚀 Deployment Guidelines

### Environment Configuration
```bash
# Production environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/satmon"
export LOG_LEVEL="INFO"
export WORKERS=4
export MAX_CONNECTIONS=100
```

### Health Checks
```python
# Implement comprehensive health checks
@router.get("/health")
async def health_check():
    """Comprehensive system health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": get_version(),
        "components": {
            "database": await check_database_health(),
            "algorithms": await check_algorithm_health(),
            "memory": get_memory_usage(),
            "disk": get_disk_usage()
        }
    }
```

### Monitoring Integration
- **Metrics collection**: Prometheus-compatible metrics
- **Logging**: Structured JSON logging
- **Alerting**: Configure alerts for critical thresholds
- **Tracing**: Distributed tracing for complex operations

## 🤝 Community Guidelines

### Code Reviews
- **Be constructive**: Focus on improving the code
- **Be thorough**: Review logic, tests, documentation
- **Be timely**: Respond to reviews within 48 hours
- **Be respectful**: Maintain professional tone

### Issue Reporting
- **Use templates**: Follow issue templates
- **Provide context**: Include system info, steps to reproduce
- **Be specific**: Clear, actionable descriptions
- **Include examples**: Code samples, error messages

### Communication
- **Professional tone**: Maintain respect in all interactions
- **Clear communication**: Use precise, technical language
- **Collaborative approach**: Work together toward solutions
- **Knowledge sharing**: Help others learn and grow

## 📋 Release Process

### Version Numbering
- **Semantic versioning**: MAJOR.MINOR.PATCH
- **Pre-release tags**: alpha, beta, rc
- **Tag format**: v1.2.3

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks validated
- [ ] Security scan completed
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes prepared

---

## 🙋‍♂️ Getting Help

### Resources
- **Documentation**: Check existing docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Code Examples**: Review test files for usage patterns

### Contact
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: For security-related concerns

---

Thank you for contributing to SatMon! Your efforts help make satellite telemetry analysis more reliable and accessible for the aerospace community.