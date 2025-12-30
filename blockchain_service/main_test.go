package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
)

func TestHealthEndpoint(t *testing.T) {
	service := NewBlockchainService()
	req, err := http.NewRequest("GET", "/health", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(service.health)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var response HealthResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}

	if response.Status != "healthy" {
		t.Errorf("handler returned wrong status: got %v want %v", response.Status, "healthy")
	}
}

func TestRegisterModelUpdate(t *testing.T) {
	service := NewBlockchainService()

	requestBody := RegisterModelUpdateRequest{
		ModelVersionID: "test_version_1",
		ParentVersionID: "",
		Hash:           "test_hash_123",
		Metadata: map[string]interface{}{
			"iteration":    float64(1),
			"num_clients":  float64(2),
			"client_ids":   []interface{}{"client_0", "client_1"},
			"diff_hash":    "diff_hash_123",
			"ipfs_cid":     "QmTest123",
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "/api/v1/model/register", bytes.NewBuffer(jsonBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(service.registerModelUpdate)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var response RegisterModelUpdateResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}

	if response.Status != "success" {
		t.Errorf("handler returned wrong status: got %v want %v", response.Status, "success")
	}

	if response.TransactionID == "" {
		t.Error("handler returned empty transaction ID")
	}

	// Verify record was stored (in development mode)
	if !service.useFabric {
		record, exists := service.records["test_version_1"]
		if !exists {
			t.Error("record was not stored in memory")
		}
		if record.Hash != "test_hash_123" {
			t.Errorf("record hash mismatch: got %v want %v", record.Hash, "test_hash_123")
		}
	}
}

func TestRegisterModelUpdateInvalidJSON(t *testing.T) {
	service := NewBlockchainService()

	req, err := http.NewRequest("POST", "/api/v1/model/register", bytes.NewBuffer([]byte("invalid json")))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(service.registerModelUpdate)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusBadRequest {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusBadRequest)
	}
}

func TestRecordValidation(t *testing.T) {
	service := NewBlockchainService()

	// First register a model version
	registerReq := RegisterModelUpdateRequest{
		ModelVersionID: "test_version_1",
		Hash:           "test_hash_123",
		Metadata:       make(map[string]interface{}),
	}
	jsonBody, _ := json.Marshal(registerReq)
	req, _ := http.NewRequest("POST", "/api/v1/model/register", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	service.registerModelUpdate(rr, req)

	// Now record validation
	requestBody := RecordValidationRequest{
		ModelVersionID: "test_version_1",
		Accuracy:       0.95,
		Metrics: map[string]float64{
			"loss": 0.05,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("POST", "/api/v1/model/validate", bytes.NewBuffer(jsonBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr = httptest.NewRecorder()
	handler := http.HandlerFunc(service.recordValidation)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var response RecordValidationResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}

	if response.Status != "success" {
		t.Errorf("handler returned wrong status: got %v want %v", response.Status, "success")
	}

	if response.TransactionID == "" {
		t.Error("handler returned empty transaction ID")
	}
}

func TestRecordValidationInvalidJSON(t *testing.T) {
	service := NewBlockchainService()

	req, err := http.NewRequest("POST", "/api/v1/model/validate", bytes.NewBuffer([]byte("invalid json")))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(service.recordValidation)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusBadRequest {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusBadRequest)
	}
}

func TestRollbackModel(t *testing.T) {
	service := NewBlockchainService()

	// First register a model version to rollback to
	registerReq := RegisterModelUpdateRequest{
		ModelVersionID: "target_version",
		Hash:           "target_hash",
		Metadata:       make(map[string]interface{}),
	}
	jsonBody, _ := json.Marshal(registerReq)
	req, _ := http.NewRequest("POST", "/api/v1/model/register", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	service.registerModelUpdate(rr, req)

	// Now test rollback
	requestBody := RollbackModelRequest{
		TargetVersionID: "target_version",
		Reason:          "Test rollback",
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		t.Fatal(err)
	}

	req, err = http.NewRequest("POST", "/api/v1/model/rollback", bytes.NewBuffer(jsonBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr = httptest.NewRecorder()
	handler := http.HandlerFunc(service.rollbackModel)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var response RollbackModelResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}

	if response.Status != "success" {
		t.Errorf("handler returned wrong status: got %v want %v", response.Status, "success")
	}

	if response.TransactionID == "" {
		t.Error("handler returned empty transaction ID")
	}
}

func TestRollbackModelInvalidJSON(t *testing.T) {
	service := NewBlockchainService()

	req, err := http.NewRequest("POST", "/api/v1/model/rollback", bytes.NewBuffer([]byte("invalid json")))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(service.rollbackModel)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusBadRequest {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusBadRequest)
	}
}

func TestGetProvenance(t *testing.T) {
	service := NewBlockchainService()

	// First register a model version
	registerReq := RegisterModelUpdateRequest{
		ModelVersionID: "test_version_1",
		ParentVersionID: "",
		Hash:           "test_hash_123",
		Metadata:       map[string]interface{}{"test": "data"},
	}
	jsonBody, _ := json.Marshal(registerReq)
	req, _ := http.NewRequest("POST", "/api/v1/model/register", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	service.registerModelUpdate(rr, req)

	// Now get provenance
	req, err := http.NewRequest("GET", "/api/v1/model/provenance/test_version_1", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr = httptest.NewRecorder()
	// Need to use mux router for path variables
	router := mux.NewRouter()
	router.HandleFunc("/api/v1/model/provenance/{version_id}", service.getProvenance).Methods("GET")
	router.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var response GetProvenanceResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}

	if response.VersionID != "test_version_1" {
		t.Errorf("handler returned wrong version ID: got %v want %v", response.VersionID, "test_version_1")
	}

	if response.Hash != "test_hash_123" {
		t.Errorf("handler returned wrong hash: got %v want %v", response.Hash, "test_hash_123")
	}
}

func TestGetProvenanceNotFound(t *testing.T) {
	service := NewBlockchainService()

	req, err := http.NewRequest("GET", "/api/v1/model/provenance/nonexistent", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	// Need to use mux router for path variables
	router := mux.NewRouter()
	router.HandleFunc("/api/v1/model/provenance/{version_id}", service.getProvenance).Methods("GET")
	router.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusNotFound {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusNotFound)
	}
}

func TestNewBlockchainService(t *testing.T) {
	service := NewBlockchainService()

	if service == nil {
		t.Fatal("NewBlockchainService returned nil")
	}

	if service.records == nil {
		t.Error("records map was not initialized")
	}

	// In development mode (no Fabric configured), useFabric should be false
	// This is expected behavior when running tests
	if service.useFabric {
		t.Log("Note: Fabric is configured, running in blockchain mode")
	} else {
		t.Log("Note: Running in development mode (in-memory storage)")
	}
}

