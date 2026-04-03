# Monitoring

Prometheus and Grafana can provide visibility into VQE performance, batch
generation progress, resource utilization, and pipeline health. Simulation
containers push metrics to the Prometheus PushGateway; Prometheus scrapes
the gateway and a set of exporters; Grafana renders dashboards from the
collected data.

Scrape targets are configured in
[`monitoring/prometheus.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/prometheus.yml).
The exporting services themselves are defined in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml).

## Section Guide

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Performance Metrics**

    ---

    Metric names, labels, collection configuration, PromQL query examples,
    and performance baselines from the thesis experiments.

    [:octicons-arrow-right-24: Performance Metrics](performance-metrics.md)

-   :material-view-dashboard-outline:{ .lg .middle } **Grafana Dashboards**

    ---

    Dashboard layout, template variables,
    and custom query examples.

    [:octicons-arrow-right-24: Grafana Dashboards](grafana-dashboards.md)

</div>
