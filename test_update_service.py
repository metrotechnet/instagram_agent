#!/usr/bin/env python3
"""
Test script for Instagram Agent Update Service

This script tests the update service endpoint and various pipeline functions
to ensure the system is working correctly.
"""

import json
import time
import os
import sys
from typing import Dict, Any
import requests

import chromadb
from chromadb.utils import embedding_functions
from config import *


class UpdateServiceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test results"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   └── {message}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def test_service_health(self) -> bool:
        """Test if the service is running and responsive"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                if "Instagram AI Agent" in data.get("message", ""):
                    self.log_test("Service Health Check", True, "Service is running")
                    return True
                else:
                    self.log_test("Service Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Service Health Check", False, f"HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.log_test("Service Health Check", False, "Service not running or unreachable")
            return False
        except Exception as e:
            self.log_test("Service Health Check", False, f"Error: {str(e)}")
            return False

    def test_update_endpoint_dry_run(self) -> bool:
        """Test the update endpoint with a small limit"""
        try:
            # Test with limit=1 to minimize API calls and processing time
            response = self.session.post(
                f"{self.base_url}/update",
                params={"limit": 1},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if "status" in data and "exécuté" in data["status"]:
                    self.log_test("Update Endpoint (dry run)", True, "Update completed successfully")
                    return True
                else:
                    self.log_test("Update Endpoint (dry run)", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Update Endpoint (dry run)", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.log_test("Update Endpoint (dry run)", False, "Request timeout (>60s)")
            return False
        except Exception as e:
            self.log_test("Update Endpoint (dry run)", False, f"Error: {str(e)}")
            return False

    def test_chroma_db_connection(self) -> bool:
        """Test ChromaDB connection and collection access"""
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY, 
                model_name="text-embedding-3-large"
            )
            
            chroma_client = chromadb.Client(chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet", 
                persist_directory=CHROMA_DB_DIR
            ))
            
            # Check if collection exists
            collections = chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if "instagram_transcripts" in collection_names:
                collection = chroma_client.get_collection("instagram_transcripts")
                count = collection.count()
                self.log_test("ChromaDB Connection", True, f"Collection found with {count} documents")
                return True
            else:
                self.log_test("ChromaDB Connection", False, "Collection 'instagram_transcripts' not found")
                return False
                
        except Exception as e:
            self.log_test("ChromaDB Connection", False, f"Error: {str(e)}")
            return False

    def test_query_endpoint(self) -> bool:
        """Test the query endpoint functionality"""
        try:
            test_query = "test query"
            response = self.session.post(
                f"{self.base_url}/query",
                params={"question": test_query, "top_k": 1},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    self.log_test("Query Endpoint", True, "Query executed successfully")
                    return True
                else:
                    self.log_test("Query Endpoint", False, f"No 'answer' in response: {data}")
                    return False
            else:
                self.log_test("Query Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.log_test("Query Endpoint", False, "Request timeout (>30s)")
            return False
        except Exception as e:
            self.log_test("Query Endpoint", False, f"Error: {str(e)}")
            return False

    def test_directories_exist(self) -> bool:
        """Test that required directories exist"""
        directories = [VIDEO_DIR, TRANSCRIPTS_DIR, CHROMA_DB_DIR]
        all_exist = True
        
        for directory in directories:
            if os.path.exists(directory):
                self.log_test(f"Directory {directory}", True, "Directory exists")
            else:
                self.log_test(f"Directory {directory}", False, "Directory missing")
                all_exist = False
                
        return all_exist

    def test_config_validation(self) -> bool:
        """Test configuration validation"""
        config_issues = []
        
        # Check Instagram credentials (warn if default values)
        if INSTAGRAM_USER == "ton_user":
            config_issues.append("Instagram username is default value")
        if INSTAGRAM_PASS == "ton_mdp":
            config_issues.append("Instagram password is default value")
        if TARGET_ACCOUNT == "compte_cible":
            config_issues.append("Target account is default value")
            
        # Check OpenAI API key
        if OPENAI_API_KEY == "TA_CLE_OPENAI":
            config_issues.append("OpenAI API key is default value")
        elif not OPENAI_API_KEY or len(OPENAI_API_KEY) < 10:
            config_issues.append("OpenAI API key appears invalid")
            
        if config_issues:
            self.log_test("Configuration Validation", False, "; ".join(config_issues))
            return False
        else:
            self.log_test("Configuration Validation", True, "Configuration appears valid")
            return True

    def run_all_tests(self, include_update_test: bool = True) -> Dict[str, Any]:
        """Run all tests and return summary"""
        print("🧪 Starting Instagram Agent Update Service Tests\n")
        
        # Basic tests
        self.test_config_validation()
        self.test_directories_exist()
        self.test_chroma_db_connection()
        
        # Service tests (require running service)
        service_running = self.test_service_health()
        
        if service_running:
            self.test_query_endpoint()
            
            if include_update_test:
                print("\n⚠️  Running update test - this will make actual API calls and process videos")
                confirm = input("Continue? (y/N): ").lower().strip()
                if confirm == 'y':
                    self.test_update_endpoint_dry_run()
                else:
                    self.log_test("Update Endpoint (skipped)", True, "Skipped by user choice")
        else:
            print("\n⚠️  Service tests skipped - service not running")
            print("   Start the service with: uvicorn app:app --reload")
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.test_results
        }

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Instagram Agent Update Service")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the service (default: http://localhost:8000)")
    parser.add_argument("--no-update", action="store_true",
                       help="Skip the update endpoint test")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    
    args = parser.parse_args()
    
    tester = UpdateServiceTester(base_url=args.url)
    results = tester.run_all_tests(include_update_test=not args.no_update)
    
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results["success_rate"] == 1.0 else 1)

if __name__ == "__main__":
    main()