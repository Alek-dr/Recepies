from app import a_queue, b_queue


def super_job(a_id, b_id):
    job_a = a_queue.fetch_job(a_id)
    job_b = b_queue.fetch_job(b_id)
    print(f"Super job: {job_a.result + job_b.result}")
    return job_a.result + job_b.result


def job_a():
    print("1")
    return 1


def job_b():
    print("3")
    return 3
