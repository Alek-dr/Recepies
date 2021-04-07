### RQ Examples

* Установить rq, rq-monitor

```
pip install rq rq-dashboard
```

* Запустить worker Запускает worker с именем W1, который обрабатывает очередь super_job

```
rq worker --with-scheduler --name W1 super_job
```

