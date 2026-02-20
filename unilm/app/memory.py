import json
from typing import Optional

import redis

from agents.base import BaseModel


class RedisMemoryManager:
    """Simple Redis memory with threshold-based summarization."""
    
    def __init__(self, agent: BaseModel, redis_host: str = "localhost", redis_port: int = 6379,
                 history_threshold: int = 10):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.threshold = history_threshold
        self.summerize_agent = agent
    
    async def add_message(self, session_id: str, role: str, content: str, agent: Optional[str] = None):
        """Add message and trigger summarization if threshold exceeded."""
        key = f"session:{session_id}:history"
        self.redis.rpush(key, json.dumps({"role": role, "content": content, "agent": agent}))
        self.redis.expire(key, 86400)
        
        if self.redis.llen(key) >= self.threshold:
            await self._summarize(session_id)
    
    def get_history(self, session_id: str) -> list[dict]:
        """Get recent history (after summarization keeps only last 3)."""
        key = f"session:{session_id}:history"
        return [json.loads(m) for m in self.redis.lrange(key, 0, -1)]
    
    def get_context(self, session_id: str) -> str:
        """Get summary + recent history for agents."""
        summary_key = f"session:{session_id}:summary"
        summary = self.redis.get(summary_key)
        history = self.get_history(session_id)
        
        context = ""
        if summary:
            context += f"Summary: {summary}\n\n"
        
        if history:
            context += "Recent messages:\n"
            for msg in history:
                role = msg["agent"] if msg["agent"] else msg["role"]
                context += f"- {role}: {msg['content'][:100]}\n"
        
        return context or "No context available."
    
    async def _summarize(self, session_id: str):
        """Summarize history when threshold exceeded."""
        key = f"session:{session_id}:history"
        summary_key = f"session:{session_id}:summary"
        
        messages = self.redis.lrange(key, 0, -1)
        if not messages:
            return
        
        history_text = "\n".join([f"{json.loads(m)['role']}: {json.loads(m)['content']}" for m in messages])
        
        summerizer = self.summerize_agent
        res = await summerizer.run_agent(f"Summarize this conversation:\n{history_text}")
            
        self.redis.set(summary_key, res.content, ex=86400)
        self.redis.delete(key)
        for msg in messages[-3:]:
            self.redis.rpush(key, msg)
