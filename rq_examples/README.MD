### RQ Examples

* Установить rq, rq-monitor

```
pip install rq rq-dashboard
```

### Пример
Смысл примера: есть super_job, которая получает результаты от двух других
job (job_a, job_b).  

Запускаем:
```
rq-dashboard
```
Воркер, который обрабатыват основную джобу
(W1 - имя воркера, super_job - обрабатываемая очередь)

```
rq worker --with-scheduler --name W1 super_job
```
Воркер, который обрабатыват job_a в очереди jobs_a
```
rq worker --with-scheduler --name W2 jobs_a
```
Воркер, который обрабатыват job_b в очереди jobs_b
```
rq worker --with-scheduler --name W3 jobs_b
```
