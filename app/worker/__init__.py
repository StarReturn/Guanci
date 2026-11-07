"""Async build worker package.

Public entrypoints:
  - ``app.worker.queue.enqueue_build``  — called by the API to enqueue a task
  - ``app.worker.queue.queue_depth``    — used by /health to report depth
  - ``python -m app.worker``            — starts the RQ worker process
"""
