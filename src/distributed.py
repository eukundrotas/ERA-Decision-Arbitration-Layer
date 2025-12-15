"""
Distributed Execution Module
Level 2 Upgrade: Parallel and distributed processing

Features:
- Async execution with asyncio
- Thread pool for I/O-bound tasks
- Process pool for CPU-bound tasks
- Rate limiting
- Retry with backoff
- Task queue management
"""

import asyncio
import logging
import time
import random
import threading
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps
from collections import deque
import queue

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TaskResult(Generic[T]):
    """Result of an async task"""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[str] = None
    duration_ms: int = 0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 10.0
    burst_size: int = 20
    retry_after_seconds: float = 1.0


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """Acquire tokens from bucket"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            elif blocking:
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0
                return True
            else:
                return False
    
    def available(self) -> int:
        """Get available tokens"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.capacity, int(self.tokens + elapsed * self.rate))


class RateLimiter:
    """Rate limiter with multiple strategies"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.bucket = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.burst_size
        )
        self._request_times: deque = deque(maxlen=100)
    
    def acquire(self, blocking: bool = True) -> bool:
        """Acquire permission to make a request"""
        result = self.bucket.acquire(blocking=blocking)
        if result:
            self._request_times.append(time.time())
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        now = time.time()
        recent_requests = sum(1 for t in self._request_times if now - t < 60)
        return {
            "available_tokens": self.bucket.available(),
            "requests_last_minute": recent_requests,
            "max_rate": self.config.requests_per_second,
            "burst_capacity": self.config.burst_size
        }


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise
                    
                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class ParallelExecutor:
    """
    Parallel task executor with thread and process pools.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        rate_limit: Optional[RateLimitConfig] = None,
        use_processes: bool = False
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        
        self._executor = None
        self._task_counter = 0
        self._lock = threading.Lock()
    
    def _get_executor(self):
        """Get or create executor"""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter:06d}"
    
    def execute_batch(
        self,
        func: Callable,
        args_list: List[tuple],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Execute function with multiple argument sets in parallel.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            timeout: Timeout per task in seconds
            
        Returns:
            List of TaskResult objects
        """
        executor = self._get_executor()
        results = []
        
        # Submit all tasks
        futures = {}
        for args in args_list:
            task_id = self._generate_task_id()
            
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.acquire()
            
            future = executor.submit(self._execute_task, func, args, task_id)
            futures[future] = task_id
        
        # Collect results
        for future in as_completed(futures, timeout=timeout):
            task_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def _execute_task(
        self,
        func: Callable,
        args: tuple,
        task_id: str
    ) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()
        
        try:
            result = func(*args)
            duration_ms = int((time.time() - start_time) * 1000)
            
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Task {task_id} failed: {e}")
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Map function over items in parallel.
        """
        args_list = [(item,) for item in items]
        return self.execute_batch(func, args_list, timeout)
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor"""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


class AsyncExecutor:
    """
    Async task executor using asyncio.
    """
    
    def __init__(
        self,
        max_concurrency: int = 10,
        rate_limit: Optional[RateLimitConfig] = None
    ):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limit = rate_limit
        
        self._task_counter = 0
        self._lock = asyncio.Lock()
        
        # Async rate limiter
        self._request_times: deque = deque(maxlen=100)
    
    async def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        async with self._lock:
            self._task_counter += 1
            return f"async_task_{self._task_counter:06d}"
    
    async def _rate_limit_acquire(self):
        """Async rate limiting"""
        if not self.rate_limit:
            return
        
        now = time.time()
        
        # Remove old timestamps
        while self._request_times and now - self._request_times[0] > 1.0:
            self._request_times.popleft()
        
        # Wait if over limit
        if len(self._request_times) >= self.rate_limit.requests_per_second:
            wait_time = 1.0 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self._request_times.append(time.time())
    
    async def execute_task(
        self,
        coro: Callable,
        *args,
        **kwargs
    ) -> TaskResult:
        """Execute a single async task"""
        task_id = await self._generate_task_id()
        start_time = time.time()
        
        async with self.semaphore:
            await self._rate_limit_acquire()
            
            try:
                result = await coro(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)
                
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    duration_ms=duration_ms
                )
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error(f"Async task {task_id} failed: {e}")
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )
    
    async def execute_batch(
        self,
        coro: Callable,
        args_list: List[tuple],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Execute coroutine with multiple argument sets concurrently.
        """
        tasks = [
            self.execute_task(coro, *args)
            for args in args_list
        ]
        
        if timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Batch execution timed out")
                results = []
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to TaskResult
        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final_results.append(TaskResult(
                    task_id=f"error_{i}",
                    success=False,
                    error=str(r)
                ))
            else:
                final_results.append(r)
        
        return final_results
    
    async def map(
        self,
        coro: Callable,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Map async function over items concurrently.
        """
        args_list = [(item,) for item in items]
        return await self.execute_batch(coro, args_list, timeout)


class TaskQueue:
    """
    Simple task queue for background processing.
    """
    
    def __init__(self, max_size: int = 1000):
        self.queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._results: Dict[str, TaskResult] = {}
        self._results_lock = threading.Lock()
        self._task_counter = 0
    
    def start_workers(self, num_workers: int = 4):
        """Start worker threads"""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {num_workers} task queue workers")
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self._stop_event.is_set():
            try:
                task_id, func, args, kwargs = self.queue.get(timeout=1.0)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    task_result = TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_ms=int((time.time() - start_time) * 1000)
                    )
                except Exception as e:
                    task_result = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        duration_ms=int((time.time() - start_time) * 1000)
                    )
                
                with self._results_lock:
                    self._results[task_id] = task_result
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
    
    def submit(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """Submit a task to the queue"""
        self._task_counter += 1
        task_id = f"queued_{self._task_counter:06d}"
        
        self.queue.put((task_id, func, args, kwargs))
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of a completed task"""
        start = time.time()
        
        while True:
            with self._results_lock:
                if task_id in self._results:
                    return self._results.pop(task_id)
            
            if timeout and time.time() - start > timeout:
                return None
            
            time.sleep(0.1)
    
    def stop(self):
        """Stop all workers"""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._workers.clear()
    
    def get_queue_size(self) -> int:
        """Get number of pending tasks"""
        return self.queue.qsize()


# Singleton instances
parallel_executor = ParallelExecutor(max_workers=10)
async_executor = AsyncExecutor(max_concurrency=10)
task_queue = TaskQueue(max_size=1000)


def run_parallel(
    func: Callable,
    args_list: List[tuple],
    max_workers: int = 10,
    timeout: Optional[float] = None
) -> List[TaskResult]:
    """
    Convenience function for parallel execution.
    """
    executor = ParallelExecutor(max_workers=max_workers)
    try:
        return executor.execute_batch(func, args_list, timeout)
    finally:
        executor.shutdown()


async def run_async_parallel(
    coro: Callable,
    args_list: List[tuple],
    max_concurrency: int = 10,
    timeout: Optional[float] = None
) -> List[TaskResult]:
    """
    Convenience function for async parallel execution.
    """
    executor = AsyncExecutor(max_concurrency=max_concurrency)
    return await executor.execute_batch(coro, args_list, timeout)
