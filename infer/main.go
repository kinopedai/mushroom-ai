package main

import (
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

func getenv(k, d string) string {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	return v
}

func main() {
	addr := getenv("ADDR", ":"+getenv("INFER_PORT", "8080"))

	r := gin.Default()
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
	r.GET("/hello", func(c *gin.Context) {
		c.String(http.StatusOK, "infer-go: hello world 👋")
	})

	r.Run(addr)
}
