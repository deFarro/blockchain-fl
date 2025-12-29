# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY blockchain_service/go.mod blockchain_service/go.sum ./
RUN go mod download

# Copy source code
COPY blockchain_service/ .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o blockchain-service .

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates wget curl

WORKDIR /root/

# Copy the binary from builder
COPY --from=builder /app/blockchain-service .

EXPOSE 8080

CMD ["./blockchain-service"]
