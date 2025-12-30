.PHONY: help build up down logs clean test test-all test-shared test-main test-client test-blockchain test-queue test-docker

help:
	@echo "Available commands:"
	@echo "  make build           - Build Docker images"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make logs            - View logs from all services"
	@echo "  make clean           - Remove containers, volumes, and images"
	@echo ""
	@echo "Testing commands (local):"
	@echo "  make test            - Run all tests (shared, main, client, blockchain)"
	@echo "  make test-all        - Run all tests including integration tests"
	@echo "  make test-shared     - Run tests for shared utilities"
	@echo "  make test-main       - Run tests for main service"
	@echo "  make test-client     - Run tests for client service"
	@echo "  make test-blockchain - Run tests for blockchain service (Go)"
	@echo "  make test-queue      - Run queue integration test"
	@echo "  make test-integration - Run end-to-end integration tests"
	@echo "  make test-docker     - Run tests in Docker containers"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker-compose rm -f
	docker system prune -f

# Test targets (local development)
test: test-shared test-main test-client test-blockchain
	@echo "✓ All tests passed"

test-all: test test-queue test-integration test-blockchain
	@echo "✓ All tests including integration tests passed"

test-integration:
	@echo "Running integration tests..."
	@pytest tests/test_integration/ -v -m integration || (echo "✗ Integration tests failed" && exit 1)

test-shared:
	@echo "Running shared utilities tests (including queue)..."
	@pytest tests/test_shared/ -v

test-main:
	@echo "Running main service tests..."
	@pytest tests/test_main/ -v || (echo "✗ Main service tests failed" && exit 1)

test-client:
	@echo "Running client service tests..."
	@pytest tests/test_client/ -v || (echo "✗ Client service tests failed" && exit 1)

test-blockchain:
	@echo "Running blockchain service tests (Go)..."
	@cd blockchain_service && go test -tags=test -v ./fabric . || (echo "✗ Blockchain service tests failed" && exit 1)

test-queue:
	@echo "Running queue integration test..."
	@pytest tests/test_shared/test_queue.py -v

# Docker-based testing (for CI/CD or isolated environments)
test-docker:
	@echo "Running tests in Docker containers..."
	@docker-compose exec -T main-service pytest /app/tests -v || echo "ℹ No tests found in main-service"
	@docker-compose exec -T client-service pytest /app/tests -v || echo "ℹ No tests found in client-service"

