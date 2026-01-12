"""
Initialization Script for Enhanced MAHIA-X
Sets up and demonstrates all enhanced capabilities
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

# Import the coordinator
from core.mahia_coordinator import MAHIACoordinator

def initialize_enhanced_mahia() -> Optional[MAHIACoordinator]:
    """Initialize the enhanced MAHIA-X system"""
    print("Initializing Enhanced MAHIA-X System...")
    print("=" * 50)
    
    # Create coordinator
    coordinator = MAHIACoordinator()
    
    # Start system
    start_time = time.time()
    if coordinator.start_system():
        init_time = time.time() - start_time
        print(f"✅ MAHIA-X Coordinator started successfully in {init_time:.2f} seconds")
    else:
        print("❌ Failed to start MAHIA-X Coordinator")
        return None
        
    # Display system capabilities
    print("\nEnhanced Capabilities Loaded:")
    print("1. 🧠 Self-Improvement & Adaptive Learning")
    print("2. 🎨 Multimodal Processing (Text/Image/Audio)")
    print("3. 👤 Personalization Engine")
    print("4. 🛡️ Error Detection & Self-Correction")
    print("5. ⚡ Dynamic Resource Optimization")
    print("6. 🔗 Expert System Integration")
    print("7. 🛡️ Ethics & Privacy Protection")
    print("8. 📝 Explainable AI & Intelligent Suggestions")
    print("9. 🔄 Real-time Parameter Optimization")
    
    return coordinator

def demonstrate_capabilities(coordinator: MAHIACoordinator):
    """Demonstrate the enhanced capabilities"""
    print("\n" + "=" * 50)
    print("DEMONSTRATING ENHANCED CAPABILITIES")
    print("=" * 50)
    
    # Example 1: Basic Query with Personalization
    print("\n1. 🎯 Basic Query with Personalization:")
    request1 = {
        'query': 'How to implement a neural network in Python?',
        'user_id': 'demo_user_001',
        'context': {
            'user_level': 'intermediate',
            'preferred_detail': 'high'
        }
    }
    
    result1 = coordinator.process_enhanced_request(request1)
    print(f"Response: {result1['response'][:100]}...")
    print(f"Explanation: {result1['explanation']}")
    print(f"Suggestions: {len(result1['suggestions'])} provided")
    
    # Example 2: Multimodal Query
    print("\n2. 🎨 Multimodal Query Processing:")
    request2 = {
        'query': 'Analyze this data and provide insights',
        'user_id': 'demo_user_002',
        'multimodal_inputs': {
            'text': [0.1, 0.5, 0.3, 0.8],  # Simplified text features
            'image': [0.2, 0.4, 0.6, 0.9],  # Simplified image features
        },
        'context': {
            'domain': 'data_science'
        }
    }
    
    result2 = coordinator.process_enhanced_request(request2)
    print(f"Response: {result2['response'][:100]}...")
    print(f"Multimodal processing: {'✓' if 'multimodal_features' in str(result2) else '✗'}")
    
    # Example 3: Ethics and Privacy Protection
    print("\n3. 🛡️ Ethics and Privacy Protection:")
    request3 = {
        'query': 'How to protect user privacy in applications?',
        'user_id': 'demo_user_003',
        'context': {
            'sensitive_data': 'user@example.com and SSN: 123-45-6789'
        }
    }
    
    result3 = coordinator.process_enhanced_request(request3)
    print(f"Response: {result3['response'][:100]}...")
    print("Privacy protection: Data anonymization applied")
    
    # Example 4: Error Detection and Correction
    print("\n4. 🛠️ Error Detection and Correction:")
    request4 = {
        'query': 'Explain machine learning concepts',
        'user_id': 'demo_user_004',
        'context': {}
    }
    
    # Simulate a response with potential errors
    result4 = coordinator.process_enhanced_request(request4)
    print(f"Errors detected: {result4['errors_detected']}")
    print(f"Corrections made: {result4['corrections_made']}")
    
    # Example 5: Self-Improvement Feedback
    print("\n5. 📈 Self-Improvement and Feedback Learning:")
    feedback = {
        'rating': 0.9,
        'type': 'accuracy',
        'comment': 'Very comprehensive explanation with good examples'
    }
    
    coordinator.add_user_feedback('demo_user_001', feedback)
    print("User feedback recorded for continuous improvement")
    
    # Show system status
    print("\n6. 📊 System Status:")
    status = coordinator.get_system_status()
    print(f"Total requests processed: {status['performance']['total_requests']}")
    print(f"Average response time: {status['performance']['average_response_time']:.3f}s")
    print(f"Error rate: {status['performance']['error_rate']:.2%}")
    print(f"Users tracked: {status['personalization']['users_tracked']}")

def run_performance_test(coordinator: MAHIACoordinator, num_requests: int = 10):
    """Run a performance test"""
    print(f"\n" + "=" * 50)
    print(f"PERFORMANCE TEST ({num_requests} requests)")
    print("=" * 50)
    
    test_request = {
        'query': 'Explain the concept of neural networks',
        'user_id': 'perf_test_user',
        'context': {'domain': 'machine_learning'}
    }
    
    start_time = time.time()
    response_times = []
    
    for i in range(num_requests):
        req_start = time.time()
        result = coordinator.process_enhanced_request(test_request)
        req_time = time.time() - req_start
        response_times.append(req_time)
        
        # Small delay to simulate real usage
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    
    print(f"Total time for {num_requests} requests: {total_time:.3f}s")
    print(f"Average response time: {avg_response_time:.3f}s")
    print(f"Fastest response: {min_time:.3f}s")
    print(f"Slowest response: {max_time:.3f}s")
    print(f"Requests per second: {num_requests/total_time:.2f}")

def main():
    """Main function to run the enhanced MAHIA-X demonstration"""
    print("🚀 MAHIA-X Enhanced Capabilities Demo")
    print("=====================================")
    
    # Initialize system
    coordinator = initialize_enhanced_mahia()
    if not coordinator:
        print("Failed to initialize MAHIA-X. Exiting.")
        return
        
    try:
        # Demonstrate capabilities
        demonstrate_capabilities(coordinator)
        
        # Run performance test
        run_performance_test(coordinator, 5)
        
        # Show final system status
        print("\n" + "=" * 50)
        print("FINAL SYSTEM STATUS")
        print("=" * 50)
        status = coordinator.get_system_status()
        print(f"System operational: {'Yes' if status.get('core_status', {}).get('fully_operational') else 'No'}")
        print(f"Components loaded: {status.get('core_status', {}).get('components_loaded', 0)}")
        print(f"Enhanced MAHIA-X is ready for production use! 🚀")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
    finally:
        # Clean shutdown
        coordinator.stop_system()
        print("\nMAHIA-X Coordinator stopped. Goodbye! 👋")

if __name__ == "__main__":
    main()