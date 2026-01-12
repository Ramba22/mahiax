"""
Expert Engine for MAHIA-X
Coordinates specialized sub-models and external knowledge sources
"""

import json
import time
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from datetime import datetime
import threading
import queue

class ExpertRouter:
    """Routes queries to appropriate expert modules"""
    
    def __init__(self):
        self.experts = {}  # expert_name -> expert_instance
        self.routing_rules = {}  # query_pattern -> expert_name
        self.performance_stats = defaultdict(lambda: {
            'requests': 0,
            'avg_response_time': 0.0,
            'success_rate': 1.0
        })
        
    def register_expert(self, expert_name: str, expert_instance: Any, 
                       capabilities: List[str], priority: int = 1):
        """Register an expert module"""
        self.experts[expert_name] = {
            'instance': expert_instance,
            'capabilities': capabilities,
            'priority': priority,
            'registered_at': datetime.now().isoformat()
        }
        
    def register_routing_rule(self, pattern: str, expert_name: str):
        """Register a routing rule"""
        self.routing_rules[pattern] = expert_name
        
    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route query to appropriate expert"""
        start_time = time.time()
        
        # Find matching expert
        expert_name = self._find_best_expert(query, context)
        
        if expert_name and expert_name in self.experts:
            expert_info = self.experts[expert_name]
            try:
                # Call expert
                if hasattr(expert_info['instance'], 'process_query'):
                    result = expert_info['instance'].process_query(query, context)
                else:
                    result = {"error": f"Expert {expert_name} has no process_query method"}
                    
                # Update performance stats
                response_time = time.time() - start_time
                self._update_performance_stats(expert_name, response_time, True)
                
                return {
                    'expert': expert_name,
                    'result': result,
                    'response_time': response_time,
                    'status': 'success'
                }
                
            except Exception as e:
                # Update performance stats for failure
                response_time = time.time() - start_time
                self._update_performance_stats(expert_name, response_time, False)
                
                return {
                    'expert': expert_name,
                    'result': {"error": str(e)},
                    'response_time': response_time,
                    'status': 'error'
                }
        else:
            return {
                'expert': 'none',
                'result': {"error": "No suitable expert found"},
                'response_time': time.time() - start_time,
                'status': 'no_expert'
            }
            
    def _find_best_expert(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Find the best expert for a query"""
        # Check routing rules first
        for pattern, expert_name in self.routing_rules.items():
            if pattern.lower() in query.lower():
                return expert_name
                
        # Find expert based on capabilities
        query_keywords = set(query.lower().split())
        best_expert = None
        best_score = 0
        
        for expert_name, expert_info in self.experts.items():
            capabilities = set(expert_info['capabilities'])
            overlap = len(query_keywords.intersection(capabilities))
            score = overlap * expert_info['priority']
            
            if score > best_score:
                best_score = score
                best_expert = expert_name
                
        return best_expert
        
    def _update_performance_stats(self, expert_name: str, response_time: float, success: bool):
        """Update performance statistics for an expert"""
        stats = self.performance_stats[expert_name]
        stats['requests'] += 1
        
        # Update average response time
        current_avg = stats['avg_response_time']
        requests = stats['requests']
        stats['avg_response_time'] = (current_avg * (requests - 1) + response_time) / requests
        
        # Update success rate
        current_success_rate = stats['success_rate']
        if success:
            stats['success_rate'] = (current_success_rate * (requests - 1) + 1) / requests
        else:
            stats['success_rate'] = current_success_rate * (requests - 1) / requests
            
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics for all experts"""
        return dict(self.performance_stats)

class KnowledgeDatabase:
    """Manages external knowledge sources"""
    
    def __init__(self):
        self.knowledge_sources = {}
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def add_knowledge_source(self, name: str, source_instance: Any, 
                           query_method: str = 'query'):
        """Add a knowledge source"""
        self.knowledge_sources[name] = {
            'instance': source_instance,
            'query_method': query_method,
            'added_at': datetime.now().isoformat()
        }
        
    def query_knowledge(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query knowledge sources"""
        if sources is None:
            sources = list(self.knowledge_sources.keys())
            
        results = {}
        
        for source_name in sources:
            if source_name in self.knowledge_sources:
                source_info = self.knowledge_sources[source_name]
                
                # Check cache first
                cache_key = f"{source_name}:{query}"
                if cache_key in self.cache:
                    cached_result, timestamp = self.cache[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        results[source_name] = cached_result
                        continue
                        
                # Query source
                try:
                    method = getattr(source_info['instance'], source_info['query_method'])
                    result = method(query)
                    
                    # Cache result
                    self.cache[cache_key] = (result, time.time())
                    results[source_name] = result
                    
                except Exception as e:
                    results[source_name] = {"error": str(e)}
                    
        return results
        
    def clear_cache(self):
        """Clear the knowledge cache"""
        self.cache.clear()

class ExpertEngine:
    """Main expert engine coordinating routing and knowledge"""
    
    def __init__(self):
        self.router = ExpertRouter()
        self.knowledge_db = KnowledgeDatabase()
        self.integration_queue = queue.Queue()
        self.integration_thread = None
        self.running = False
        
    def start_integration_thread(self):
        """Start background integration thread"""
        if not self.running:
            self.running = True
            self.integration_thread = threading.Thread(
                target=self._integration_loop,
                daemon=True
            )
            self.integration_thread.start()
            
    def stop_integration_thread(self):
        """Stop background integration thread"""
        self.running = False
        if self.integration_thread:
            self.integration_thread.join()
            
    def _integration_loop(self):
        """Main integration loop"""
        while self.running:
            try:
                # Process integration requests
                if not self.integration_queue.empty():
                    request = self.integration_queue.get(timeout=1.0)
                    self._process_integration_request(request)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in integration loop: {e}")
                time.sleep(1.0)
                
    def _process_integration_request(self, request: Dict[str, Any]):
        """Process an integration request"""
        request_type = request.get('type')
        if request_type == 'expert_query':
            query = request.get('query', '')
            context = request.get('context', {})
            result = self.router.route_query(query, context)
            # Store result or send to callback
        elif request_type == 'knowledge_query':
            query = request.get('query', '')
            sources = request.get('sources')
            result = self.knowledge_db.query_knowledge(query, sources or [])
            # Store result or send to callback
            
    def query_expert(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the expert system"""
        return self.router.route_query(query, context)
        
    def query_knowledge(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query external knowledge sources"""
        return self.knowledge_db.query_knowledge(query, sources)
        
    def async_query(self, query: str, context: Optional[Dict[str, Any]] = None, 
                   callback: Optional[Callable] = None) -> str:
        """Submit query for asynchronous processing"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request = {
            'id': request_id,
            'type': 'expert_query',
            'query': query,
            'context': context,
            'callback': callback,
            'timestamp': datetime.now().isoformat()
        }
        self.integration_queue.put(request)
        return request_id
        
    def register_expert(self, expert_name: str, expert_instance: Any, 
                       capabilities: List[str], priority: int = 1):
        """Register an expert module"""
        self.router.register_expert(expert_name, expert_instance, capabilities, priority)
        
    def register_knowledge_source(self, name: str, source_instance: Any, 
                                query_method: str = 'query'):
        """Register a knowledge source"""
        self.knowledge_db.add_knowledge_source(name, source_instance, query_method)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'experts': list(self.router.experts.keys()),
            'knowledge_sources': list(self.knowledge_db.knowledge_sources.keys()),
            'expert_stats': self.router.get_expert_stats(),
            'running': self.running
        }

# Example expert implementations
class TechnicalExpert:
    """Example technical expert"""
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process technical queries"""
        return {
            'response': f"Technical response to: {query}",
            'confidence': 0.9,
            'sources': ['technical_database']
        }

class CreativeExpert:
    """Example creative expert"""
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process creative queries"""
        return {
            'response': f"Creative response to: {query}",
            'confidence': 0.8,
            'sources': ['creative_knowledge_base']
        }

# Example knowledge source
class ExternalKnowledgeAPI:
    """Example external knowledge source"""
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query external knowledge"""
        return {
            'results': [f"Knowledge result 1 for: {query}", f"Knowledge result 2 for: {query}"],
            'source': 'external_api',
            'timestamp': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Initialize expert engine
    engine = ExpertEngine()
    
    # Register experts
    tech_expert = TechnicalExpert()
    creative_expert = CreativeExpert()
    
    engine.register_expert('technical', tech_expert, ['code', 'programming', 'technical'], 2)
    engine.register_expert('creative', creative_expert, ['idea', 'creative', 'design'], 1)
    
    # Register knowledge sources
    external_knowledge = ExternalKnowledgeAPI()
    engine.register_knowledge_source('external_api', external_knowledge)
    
    # Start integration thread
    engine.start_integration_thread()
    
    # Query experts
    result1 = engine.query_expert("How to write Python code?")
    print("Technical Query Result:", result1)
    
    result2 = engine.query_expert("Give me creative ideas for a project")
    print("Creative Query Result:", result2)
    
    # Query knowledge
    knowledge_result = engine.query_knowledge("What is machine learning?")
    print("Knowledge Query Result:", knowledge_result)
    
    # Get system status
    status = engine.get_system_status()
    print("System Status:", status)
    
    # Stop integration thread
    engine.stop_integration_thread()