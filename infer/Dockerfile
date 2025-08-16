# Build stage
FROM golang:1.22-bookworm AS builder
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o server main.go

# Run stage
FROM gcr.io/distroless/base-debian12
WORKDIR /app
ENV INFER_PORT=8080
COPY --from=builder /app/server /app/server
EXPOSE 8080
CMD ["/app/server"]
