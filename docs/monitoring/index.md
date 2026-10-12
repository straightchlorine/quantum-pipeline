---
title: "Monitoring Overview: Prometheus, Grafana"
description: "How Prometheus and Grafana track VQE performance, batch progress, resource use and pipeline health while simulations run, with links to metrics and dashboards."
---

# Monitoring

Prometheus and Grafana can provide visibility into VQE performance, batch
generation progress, resource utilization, and pipeline health. Simulation
containers push metrics to the Prometheus PushGateway; Prometheus scrapes
the gateway and a set of exporters; Grafana renders dashboards from the
collected data. The Docker Compose stack ships the exporters but not
PushGateway, Prometheus or Grafana.

Scrape targets are configured in
[`monitoring/prometheus.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/prometheus.yml).
The exporting services themselves are defined in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml).

## Section Guide

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Performance Metrics**

    ---

    Metric names, types and labels, collection settings and how to run a
    PushGateway.

    [:octicons-arrow-right-24: Performance Metrics](performance-metrics.md)

-   :material-view-dashboard-outline:{ .lg .middle } **Grafana Dashboards**

    ---

    Dashboard import, template variables and a description of each row of
    panels.

    [:octicons-arrow-right-24: Grafana Dashboards](grafana-dashboards.md)

</div>
