package chain

import (
	"testing"
)

func TestLocalChain_AddBlock(t *testing.T) {
	c := NewLocalChain()
	if c.Len() != 1 {
		t.Fatalf("expected 1 block (genesis), got %d", c.Len())
	}
	if !c.Validate() {
		t.Fatal("genesis chain should be valid")
	}

	hash1, b1 := c.AddBlock("v1", []byte("payload1"))
	if hash1 == "" || b1 == nil {
		t.Fatal("AddBlock should return hash and block")
	}
	if b1.PrevHash != c.Blocks[0].Hash {
		t.Error("block should link to previous hash")
	}
	if c.Len() != 2 {
		t.Fatalf("expected 2 blocks, got %d", c.Len())
	}
	if !c.Validate() {
		t.Fatal("chain with one block should be valid")
	}

	c.AddBlock("v2", nil)
	if c.Len() != 3 {
		t.Fatalf("expected 3 blocks, got %d", c.Len())
	}
	if c.ByID["v1"] != b1 {
		t.Error("ByID should return same block for v1")
	}
	if c.BlockByVersionID("v2") == nil {
		t.Error("BlockByVersionID(v2) should be non-nil")
	}
	if c.BlockByVersionID("missing") != nil {
		t.Error("BlockByVersionID(missing) should be nil")
	}
}

func TestLocalChain_RollbackBlocks(t *testing.T) {
	c := NewLocalChain()
	c.AddBlock("v1", []byte("version1"))

	if c.GetMostRecentRollbackPayload() != nil {
		t.Fatal("no rollback yet, should be nil")
	}

	hashR1, _ := c.AddRollbackBlock([]byte(`{"to_version_id":"v0","reason":"test"}`))
	if hashR1 == "" {
		t.Fatal("AddRollbackBlock should return hash")
	}
	if c.BlockByVersionID(RollbackBlockVersionID) != nil {
		t.Error("rollback blocks should not be in ByID")
	}
	if c.Len() != 3 {
		t.Fatalf("expected 3 blocks (genesis, v1, rollback), got %d", c.Len())
	}
	payload := c.GetMostRecentRollbackPayload()
	if payload == nil || string(payload) != `{"to_version_id":"v0","reason":"test"}` {
		t.Errorf("GetMostRecentRollbackPayload: got %q", payload)
	}

	c.AddRollbackBlock([]byte(`{"to_version_id":"v0","reason":"second"}`))
	payload = c.GetMostRecentRollbackPayload()
	if payload == nil || string(payload) != `{"to_version_id":"v0","reason":"second"}` {
		t.Errorf("GetMostRecentRollbackPayload (second): got %q", payload)
	}
}
