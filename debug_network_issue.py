#!/usr/bin/env python3
"""
Debug network connection issues between machines running same app
"""

import sys
import socket
import requests
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_udp_discovery():
    """Listen for UDP broadcasts to see what other machines are announcing"""
    print("=" * 60)
    print("1. ANALYZING UDP DISCOVERY BROADCASTS")
    print("=" * 60)
    
    test_ports = [5001, 5556]  # New and legacy broadcast ports
    
    for port in test_ports:
        print(f"\n[LISTEN] Listening on UDP port {port} for 10 seconds...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            sock.settimeout(2.0)
            
            messages = []
            start_time = time.time()
            
            while time.time() - start_time < 10:
                try:
                    data, addr = sock.recvfrom(8192)
                    try:
                        message = json.loads(data.decode('utf-8'))
                        messages.append((addr, message))
                        
                        node_data = message.get('node', {})
                        hostname = node_data.get('hostname', 'Unknown')
                        ip = node_data.get('ip_address', addr[0])
                        announced_port = node_data.get('port', 'Unknown')
                        
                        print(f"  [BROADCAST] From {addr[0]}: {hostname} announces service on port {announced_port}")
                        
                    except json.JSONDecodeError:
                        print(f"  [DATA] Non-JSON data from {addr[0]}")
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"  [ERROR] Error: {e}")
            
            sock.close()
            
            if messages:
                print(f"\n[ANALYSIS] ANALYSIS for port {port}:")
                for addr, message in messages:
                    node = message.get('node', {})
                    print(f"  Machine: {node.get('hostname', 'Unknown')} ({addr[0]})")
                    print(f"  Announced service port: {node.get('port', 'Unknown')}")
                    print(f"  Node ID: {node.get('node_id', 'Unknown')}")
                    print(f"  Last seen: {node.get('last_seen', 'Unknown')}")
                    print(f"  Shared folders: {len(node.get('shared_folders', []))}")
                    print()
            else:
                print(f"  [NONE] No broadcasts received on port {port}")
                
        except Exception as e:
            print(f"[ERROR] Failed to listen on port {port}: {e}")

def test_http_services():
    """Test HTTP services on discovered machines"""
    print("=" * 60)  
    print("2. TESTING HTTP SERVICES")
    print("=" * 60)
    
    # Test different combinations of IP and port
    test_targets = [
        ("10.28.130.48", 5000),  # Current config
        ("10.28.130.48", 5555),  # Legacy config
        ("10.28.130.48", 8000),  # Alternative ports
        ("10.28.130.48", 3000),
    ]
    
    for ip, port in test_targets:
        print(f"\n[TEST] Testing {ip}:{port}")
        
        # Test TCP connection first
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                print(f"  [OPEN] TCP port {port} is OPEN")
                
                # Test HTTP endpoints
                endpoints = ["/api/ping", "/api/node/info", "/", "/api/status"]
                
                for endpoint in endpoints:
                    try:
                        url = f"http://{ip}:{port}{endpoint}"
                        response = requests.get(url, timeout=3)
                        print(f"  [HTTP] {endpoint}: HTTP {response.status_code}")
                        
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                if endpoint == "/api/ping":
                                    print(f"    Node ID: {data.get('node_id', 'Unknown')}")
                                    print(f"    Status: {data.get('status', 'Unknown')}")
                                    print(f"    Hostname: {data.get('hostname', 'Unknown')}")
                            except:
                                print(f"    Response: {response.text[:100]}...")
                                
                    except Exception as e:
                        print(f"  [FAIL] {endpoint}: {e}")
                        
            else:
                print(f"  [CLOSED] TCP port {port} is CLOSED")
                
        except Exception as e:
            print(f"  [ERROR] Connection test failed: {e}")

def analyze_local_config():
    """Analyze local configuration"""
    print("=" * 60)
    print("3. ANALYZING LOCAL CONFIGURATION") 
    print("=" * 60)
    
    try:
        # Read config
        with open("config.json", "r") as f:
            config = json.load(f)
            
        print("[CONFIG] Local Configuration:")
        print(f"  API Port: {config.get('api', {}).get('port', 'Unknown')}")
        print(f"  System Version: {config.get('system', {}).get('version', 'Unknown')}")
        print(f"  Debug Mode: {config.get('system', {}).get('debug', 'Unknown')}")
        
        # Check if NetworkServer should be running
        api_config = config.get('api', {})
        print(f"\n[SERVER] Expected API Server:")
        print(f"  Host: {api_config.get('host', 'localhost')}")
        print(f"  Port: {api_config.get('port', 5000)}")
        print(f"  CORS: {api_config.get('cors_enabled', True)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")
    
    # Check local network info
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0] 
        s.close()
        
        print(f"\n[LOCAL] Local Machine:")
        print(f"  Hostname: {hostname}")
        print(f"  IP Address: {local_ip}")
        
    except Exception as e:
        print(f"[ERROR] Failed to get local info: {e}")

def main():
    print("NETWORK CONNECTION DEBUG TOOL")
    print("Analyzing why machines can discover but not connect...")
    print()
    
    # Step 1: Analyze UDP discovery
    test_udp_discovery()
    
    # Step 2: Test HTTP services  
    test_http_services()
    
    # Step 3: Check local config
    analyze_local_config()
    
    print("=" * 60)
    print("4. SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
LIKELY CAUSES:
1. Port mismatch - Discovery announces one port, HTTP runs on another
2. HTTP server not started - Only UDP discovery running  
3. Version mismatch - Different app versions using different ports
4. Firewall - Blocks HTTP but allows UDP
5. Config difference - Different port configurations

SOLUTIONS TO TRY:
1. Check if remote machine has HTTP server running
2. Verify port configuration matches between machines
3. Check firewall settings on remote machine  
4. Ensure both machines run same app version
5. Try manual connection with discovered port from broadcast
""")

if __name__ == "__main__":
    main()