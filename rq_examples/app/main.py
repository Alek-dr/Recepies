import time

from app import a_queue, b_queue, redis_queue
from jobs import job_a, job_b, super_job

if __name__ == '__main__':
    j1 = a_queue.enqueue(job_a)
    j2 = b_queue.enqueue(job_b)
    job = redis_queue.enqueue(super_job, j1.id, j2.id, depends_on=[j1, j2])
    time.sleep(0.1)
    print(job.result)
