from rq import Queue
from redis import Redis

redis_conn = Redis(host="127.0.0.1", port="6379")

# Создать очереди с именами
redis_queue = Queue(name="super_job", connection=redis_conn)
a_queue = Queue(name="jobs_a", connection=redis_conn)
b_queue = Queue(name="jobs_b", connection=redis_conn)
