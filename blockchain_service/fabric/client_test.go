package fabric

import (
	"os"
	"testing"
)

func TestNewFabricClient(t *testing.T) {
	client := NewFabricClient()

	if client == nil {
		t.Fatal("NewFabricClient returned nil")
	}

	if client.initialized {
		t.Error("Fabric client should not be initialized by default")
	}
}

func TestFabricClientInitializationWithoutConfig(t *testing.T) {
	// Save original environment variables
	originalProfile := os.Getenv("FABRIC_NETWORK_PROFILE")
	originalWallet := os.Getenv("FABRIC_WALLET_PATH")

	// Clear environment variables to simulate no Fabric configuration
	os.Unsetenv("FABRIC_NETWORK_PROFILE")
	os.Unsetenv("FABRIC_WALLET_PATH")

	// Restore after test
	defer func() {
		if originalProfile != "" {
			os.Setenv("FABRIC_NETWORK_PROFILE", originalProfile)
		}
		if originalWallet != "" {
			os.Setenv("FABRIC_WALLET_PATH", originalWallet)
		}
	}()

	client := NewFabricClient()
	err := client.Initialize()

	// Should fail because Fabric is not configured
	if err == nil {
		t.Error("Initialize should fail when Fabric is not configured")
	}

	if client.IsInitialized() {
		t.Error("Client should not be initialized when Fabric is not configured")
	}
}

func TestFabricClientIsInitialized(t *testing.T) {
	client := NewFabricClient()

	if client.IsInitialized() {
		t.Error("New client should not be initialized")
	}

	// Try to initialize (will fail without proper config, but tests the method)
	err := client.Initialize()
	if err == nil {
		// If initialization succeeded, check that IsInitialized returns true
		if !client.IsInitialized() {
			t.Error("IsInitialized should return true after successful initialization")
		}
	} else {
		// If initialization failed, check that IsInitialized returns false
		if client.IsInitialized() {
			t.Error("IsInitialized should return false after failed initialization")
		}
	}
}

func TestFabricClientClose(t *testing.T) {
	client := NewFabricClient()

	// Close should not panic even if client is not initialized
	client.Close()

	// Should be safe to call multiple times
	client.Close()
}

