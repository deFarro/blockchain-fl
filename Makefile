.PHONY: help build up down logs clean test

help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make clean       - Remove containers, volumes, and images"
	@echo "  make test        - Run tests"

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

test:
	docker-compose exec main-service pytest /app/tests

