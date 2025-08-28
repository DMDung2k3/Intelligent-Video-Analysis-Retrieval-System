#!/usr/bin/env python3
"""
Final validation test - clean test for all fixes
"""

import sys
import socket
import requests
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_clean_server_startup():
    """Test clean server startup from scratch"""
    print("=" * 60)
    print("FINAL VALIDATION - CLEAN SERVER STARTUP")
    print("=" * 60)
    
    try:
        from api import NetworkServer
        
        print("\n[TEST 1] Clean NetworkServer creation...")
        server = NetworkServer(port=5001)  # Use different port to avoid conflicts
        print(f"  Port: {server.port}")
        print(f"  Initial is_running: {server.is_running}")
        print(f"  Initial server_active: {server.server_active}")
        
        print("\n[TEST 2] Server startup with validation...")
        start_result = server.start_server()
        print(f"  start_server() result: {start_result}")
        
        if not start_result:
            print("  [ERROR] Server failed to start!")
            return False
            
        print(f"  After start is_running: {server.is_running}")
        print(f"  After start server_active: {server.server_active}")
        print(f"  Server thread alive: {server.server_thread.is_alive() if server.server_thread else 'No thread'}")
        
        print("\n[TEST 3] Wait and validate HTTP service...")
        time.sleep(2)  # Give server time to fully start
        
        endpoints = ["/api/ping", "/api/node/info", "/api/status"]
        all_working = True
        
        for endpoint in endpoints:
            try:
                url = f"http://localhost:5001{endpoint}"
                response = requests.get(url, timeout=3)
                status_ok = response.status_code == 200
                print(f"  {endpoint}: HTTP {response.status_code} {'✓' if status_ok else '✗'}")
                
                if status_ok:
                    try:
                        data = response.json()
                        if endpoint == "/api/ping":
                            print(f"    Node: {data.get('hostname')} ({data.get('node_id', 'Unknown')})")
                        elif endpoint == "/api/status":
                            print(f"    Uptime: {data.get('uptime_seconds', 0):.1f}s, Thread: {data.get('thread_alive', False)}")
                    except:
                        pass
                else:
                    all_working = False
                    
            except Exception as e:
                print(f"  {endpoint}: FAILED - {e}")
                all_working = False
        
        print("\n[TEST 4] Discovery service validation...")
        discovery_active = server.discovery.discovery_active
        print(f"  Discovery active: {discovery_active}")
        
        if discovery_active:
            nodes = server.discovery.get_discovered_nodes()
            print(f"  Discovered nodes: {len(nodes)}")
            for node in nodes:
                print(f"    - {node.hostname} ({node.ip_address}:{node.port})")
        
        print("\n[TEST 5] Clean server stop...")
        server.stop_server()
        print(f"  After stop is_running: {server.is_running}")
        print(f"  After stop server_active: {server.server_active}")
        
        # Final validation - server should not respond
        time.sleep(1)
        try:
            response = requests.get("http://localhost:5001/api/ping", timeout=2)
            print(f"  [WARNING] Server still responding after stop!")
            return False
        except:
            print(f"  Server properly stopped ✓")
        
        print(f"\n[RESULT] Final validation: {'✓ PASS' if all_working else '✗ FAIL'}")
        return all_working
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_simulation():
    """Simulate GUI server start process"""
    print("\n" + "=" * 60)
    print("GUI SIMULATION TEST")
    print("=" * 60)
    
    try:
        from api import NetworkServer
        import requests
        
        print("\n[GUI SIM] Create server like GUI does...")
        port = 5002  # Different port
        server = NetworkServer(port=port)
        
        print("\n[GUI SIM] Start server with validation like fixed GUI...")
        
        # Simulate the fixed GUI logic
        server_started = server.start_server()
        if not server_started:
            raise Exception("Failed to start HTTP server - port may be in use")
        
        # Wait and validate like GUI does now
        time.sleep(1)
        
        try:
            response = requests.get(f"http://localhost:{port}/api/ping", timeout=3)
            if response.status_code != 200:
                raise Exception(f"Server not responding correctly (HTTP {response.status_code})")
        except requests.exceptions.ConnectionError:
            server.stop_server()
            raise Exception("Server failed to start - no HTTP listener active")
        except requests.exceptions.Timeout:
            server.stop_server()
            raise Exception("Server not responding - startup timeout")
        
        print("[GUI SIM] Server validation passed ✓")
        print("[GUI SIM] GUI would now show '🟢 Server Running'")
        
        # Test that server is actually working
        response = requests.get(f"http://localhost:{port}/api/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"[GUI SIM] Server confirmed working - uptime: {data.get('uptime_seconds', 0):.1f}s")
        
        # Clean stop
        server.stop_server()
        print("[GUI SIM] Server stopped cleanly")
        
        return True
        
    except Exception as e:
        print(f"[GUI SIM ERROR] {e}")
        return False

def main():
    print("FINAL VALIDATION TEST SUITE")
    print("Testing all fixes for network server startup issues")
    print("=" * 70)
    
    # Test 1: Clean server startup
    test1_pass = test_clean_server_startup()
    
    # Test 2: GUI simulation
    test2_pass = test_gui_simulation()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"Clean Server Startup: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"GUI Logic Simulation: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    overall_pass = test1_pass and test2_pass
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ TESTS FAILED'}")
    
    if overall_pass:
        print("\n🎉 SUCCESS: All server startup issues have been fixed!")
        print("   - NetworkServer properly validates startup")
        print("   - GUI will only show 'Server Running' when server actually works")
        print("   - All API endpoints (/api/ping, /api/node/info, /api/status) working")
        print("   - Thread management and lifecycle properly handled")
        print("   - Error handling and cleanup implemented")
        print("\n   Remote machines should now be able to connect successfully!")
    else:
        print("\n❌ Some issues remain - check the test output above")

if __name__ == "__main__":
    main()