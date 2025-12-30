package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/blockchain-fl/blockchain-service/fabric"
)

// ModelVersion represents a model version record
type ModelVersion struct {
	VersionID      string                 `json:"version_id"`
	ParentVersionID string                `json:"parent_version_id,omitempty"`
	Hash           string                 `json:"hash"`
	Metadata       map[string]interface{} `json:"metadata"`
	Timestamp      string                 `json:"timestamp"`
}

// ValidationRecord represents validation results
type ValidationRecord struct {
	VersionID string             `json:"version_id"`
	Accuracy  float64            `json:"accuracy"`
	Metrics   map[string]float64 `json:"metrics"`
	Timestamp string             `json:"timestamp"`
}

// RollbackEvent represents a rollback event
type RollbackEvent struct {
	TargetVersionID string `json:"target_version_id"`
	Reason          string `json:"reason"`
	Timestamp       string `json:"timestamp"`
}

// BlockchainService handles blockchain operations
type BlockchainService struct {
	fabricClient *fabric.FabricClient
	// In-memory storage for development mode (when Fabric is not configured)
	records map[string]ModelVersion
	useFabric bool
}

// NewBlockchainService creates a new blockchain service
func NewBlockchainService() *BlockchainService {
	service := &BlockchainService{
		records: make(map[string]ModelVersion),
		useFabric: false,
	}

	// Try to initialize Fabric client
	fabricClient := fabric.NewFabricClient()
	if err := fabricClient.Initialize(); err == nil {
		service.fabricClient = fabricClient
		service.useFabric = true
		log.Println("Using Hyperledger Fabric for blockchain operations")
	} else {
		log.Println("Fabric not configured, using development mode (in-memory storage)")
		log.Printf("Fabric initialization error: %v", err)
	}

	return service
}

// RegisterModelUpdateRequest represents a request to register a model update
type RegisterModelUpdateRequest struct {
	ModelVersionID string                 `json:"model_version_id"`
	ParentVersionID string                `json:"parent_version_id,omitempty"`
	Hash           string                 `json:"hash"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// RegisterModelUpdateResponse represents the response
type RegisterModelUpdateResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// RecordValidationRequest represents a validation request
type RecordValidationRequest struct {
	ModelVersionID string             `json:"model_version_id"`
	Accuracy       float64            `json:"accuracy"`
	Metrics        map[string]float64 `json:"metrics"`
}

// RecordValidationResponse represents the response
type RecordValidationResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// RollbackModelRequest represents a rollback request
type RollbackModelRequest struct {
	TargetVersionID string `json:"target_version_id"`
	Reason          string `json:"reason"`
}

// RollbackModelResponse represents the response
type RollbackModelResponse struct {
	TransactionID string `json:"transaction_id"`
	Status        string `json:"status"`
}

// GetProvenanceResponse represents provenance information
type GetProvenanceResponse struct {
	VersionID      string                 `json:"version_id"`
	ParentVersionID string                `json:"parent_version_id,omitempty"`
	Hash           string                 `json:"hash"`
	Metadata       map[string]interface{} `json:"metadata"`
	Timestamp      string                 `json:"timestamp"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status string `json:"status"`
}

func (bs *BlockchainService) registerModelUpdate(w http.ResponseWriter, r *http.Request) {
	var req RegisterModelUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		// Extract metadata from request
		metadataJSON, _ := json.Marshal(req.Metadata)
		
		// Extract iteration and num_clients from metadata if present
		iteration := 0
		numClients := 0
		clientIDs := []string{}
		if iter, ok := req.Metadata["iteration"].(float64); ok {
			iteration = int(iter)
		}
		if num, ok := req.Metadata["num_clients"].(float64); ok {
			numClients = int(num)
		}
		if ids, ok := req.Metadata["client_ids"].([]interface{}); ok {
			for _, id := range ids {
				if str, ok := id.(string); ok {
					clientIDs = append(clientIDs, str)
				}
			}
		}
		clientIDsJSON, _ := json.Marshal(clientIDs)
		
		// Get diff_hash and ipfs_cid from metadata
		diffHash := ""
		ipfsCID := ""
		if dh, ok := req.Metadata["diff_hash"].(string); ok {
			diffHash = dh
		}
		if cid, ok := req.Metadata["ipfs_cid"].(string); ok {
			ipfsCID = cid
		}

		txID, err = bs.fabricClient.RegisterModelUpdate(
			req.ModelVersionID,
			req.ParentVersionID,
			req.Hash,
			diffHash,
			ipfsCID,
			string(metadataJSON),
			iteration,
			numClients,
			string(clientIDsJSON),
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false // Fallback to in-memory
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		version := ModelVersion{
			VersionID:      req.ModelVersionID,
			ParentVersionID: req.ParentVersionID,
			Hash:           req.Hash,
			Metadata:       req.Metadata,
			Timestamp:      fmt.Sprintf("%d", time.Now().Unix()),
		}
		bs.records[req.ModelVersionID] = version
		txID = fmt.Sprintf("tx_%s", req.ModelVersionID)
	}

	response := RegisterModelUpdateResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) recordValidation(w http.ResponseWriter, r *http.Request) {
	var req RecordValidationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		metricsJSON, _ := json.Marshal(req.Metrics)
		txID, err = bs.fabricClient.RecordValidation(
			req.ModelVersionID,
			req.Accuracy,
			string(metricsJSON),
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		log.Printf("Validation recorded: version=%s, accuracy=%.4f", req.ModelVersionID, req.Accuracy)
		txID = fmt.Sprintf("tx_validation_%s", req.ModelVersionID)
	}

	response := RecordValidationResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) rollbackModel(w http.ResponseWriter, r *http.Request) {
	var req RollbackModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var txID string
	var err error

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		// Determine current version (would need to track this, for now use empty)
		fromVersionID := "" // In a real implementation, track current version
		triggeredBy := "automatic" // Could be extracted from request metadata
		
		txID, err = bs.fabricClient.RollbackModel(
			fromVersionID,
			req.TargetVersionID,
			req.Reason,
			triggeredBy,
		)
		if err != nil {
			log.Printf("Fabric transaction failed: %v, falling back to in-memory storage", err)
			bs.useFabric = false
		}
	}

	if !bs.useFabric {
		// Fallback to in-memory storage
		log.Printf("Rollback requested: target_version=%s, reason=%s", req.TargetVersionID, req.Reason)
		txID = fmt.Sprintf("tx_rollback_%s", req.TargetVersionID)
	}

	response := RollbackModelResponse{
		TransactionID: txID,
		Status:        "success",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) getProvenance(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	versionID := vars["version_id"]

	if bs.useFabric && bs.fabricClient != nil {
		// Use Fabric SDK
		provenanceJSON, err := bs.fabricClient.GetModelProvenance(versionID)
		if err == nil {
			// Return the JSON directly from Fabric
			w.Header().Set("Content-Type", "application/json")
			w.Write(provenanceJSON)
			return
		}
		log.Printf("Fabric query failed: %v, falling back to in-memory storage", err)
		bs.useFabric = false
	}

	// Fallback to in-memory storage
	version, exists := bs.records[versionID]
	if !exists {
		http.Error(w, "Version not found", http.StatusNotFound)
		return
	}

	response := GetProvenanceResponse{
		VersionID:      version.VersionID,
		ParentVersionID: version.ParentVersionID,
		Hash:           version.Hash,
		Metadata:       version.Metadata,
		Timestamp:      version.Timestamp,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (bs *BlockchainService) health(w http.ResponseWriter, r *http.Request) {
	response := HealthResponse{
		Status: "healthy",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	port := os.Getenv("BLOCKCHAIN_SERVICE_PORT")
	if port == "" {
		port = "8080"
	}

	service := NewBlockchainService()
	
	// Cleanup on exit
	defer func() {
		if service.fabricClient != nil {
			service.fabricClient.Close()
		}
	}()

	r := mux.NewRouter()
	r.HandleFunc("/health", service.health).Methods("GET")
	r.HandleFunc("/api/v1/model/register", service.registerModelUpdate).Methods("POST")
	r.HandleFunc("/api/v1/model/validate", service.recordValidation).Methods("POST")
	r.HandleFunc("/api/v1/model/rollback", service.rollbackModel).Methods("POST")
	r.HandleFunc("/api/v1/model/provenance/{version_id}", service.getProvenance).Methods("GET")

	log.Printf("Blockchain service starting on port %s", port)
	if service.useFabric {
		log.Println("✓ Connected to Hyperledger Fabric network")
	} else {
		log.Println("⚠ Running in development mode (in-memory storage)")
		log.Println("  To use Fabric, configure FABRIC_NETWORK_PROFILE and FABRIC_WALLET_PATH")
	}
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), r))
}

