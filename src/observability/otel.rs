//! OpenTelemetry metrics provider.
//!
//! This module provides [`OtelMetrics`], an implementation of [`MetricsProvider`]
//! that exports Asupersync runtime metrics via OpenTelemetry.
//!
//! # Feature
//!
//! Enable the `metrics` feature to compile this module.
//!
//! # Example
//!
//! ```ignore
//! use opentelemetry::global;
//! use opentelemetry_prometheus::exporter;
//! use prometheus::Registry;
//! use asupersync::observability::OtelMetrics;
//!
//! let registry = Registry::new();
//! let exporter = exporter().with_registry(registry.clone()).build().unwrap();
//! let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
//!     .with_reader(opentelemetry_sdk::metrics::PeriodicReader::builder(exporter).build())
//!     .build();
//! opentelemetry::global::set_meter_provider(provider);
//!
//! let metrics = OtelMetrics::new(global::meter("asupersync"));
//! // RuntimeBuilder::new().metrics(metrics).build();
//! ```

use crate::observability::metrics::{MetricsProvider, OutcomeKind};
use crate::types::{CancelKind, RegionId, TaskId};
use opentelemetry::metrics::{Counter, Histogram, Meter, ObservableGauge};
use opentelemetry::KeyValue;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// OpenTelemetry metrics provider for Asupersync.
#[derive(Debug, Clone)]
pub struct OtelMetrics {
    // Task metrics
    tasks_active: ObservableGauge<u64>,
    tasks_spawned: Counter<u64>,
    tasks_completed: Counter<u64>,
    task_duration: Histogram<f64>,
    // Region metrics
    regions_active: ObservableGauge<u64>,
    regions_created: Counter<u64>,
    regions_closed: Counter<u64>,
    region_lifetime: Histogram<f64>,
    // Cancellation metrics
    cancellations: Counter<u64>,
    drain_duration: Histogram<f64>,
    // Budget metrics
    deadlines_set: Counter<u64>,
    deadlines_exceeded: Counter<u64>,
    // Obligation metrics
    obligations_active: ObservableGauge<u64>,
    obligations_created: Counter<u64>,
    obligations_discharged: Counter<u64>,
    obligations_leaked: Counter<u64>,
    // Scheduler metrics
    scheduler_poll_time: Histogram<f64>,
    scheduler_tasks_polled: Histogram<f64>,
    // Shared gauge state
    state: Arc<MetricsState>,
}

#[derive(Debug, Default)]
struct MetricsState {
    active_tasks: AtomicU64,
    active_regions: AtomicU64,
    active_obligations: AtomicU64,
}

impl MetricsState {
    fn inc_tasks(&self) {
        self.active_tasks.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_tasks(&self) {
        let _ = self
            .active_tasks
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    fn inc_regions(&self) {
        self.active_regions.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_regions(&self) {
        let _ = self
            .active_regions
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    fn inc_obligations(&self) {
        self.active_obligations.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_obligations(&self) {
        let _ = self
            .active_obligations
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }
}

impl OtelMetrics {
    /// Constructs a new OpenTelemetry metrics provider from a [`Meter`].
    #[must_use]
    pub fn new(meter: Meter) -> Self {
        let state = Arc::new(MetricsState::default());

        let tasks_active = meter
            .u64_observable_gauge("asupersync.tasks.active")
            .with_description("Currently running tasks")
            .with_callback({
                let state = Arc::clone(&state);
                move |observer| {
                    observer.observe(state.active_tasks.load(Ordering::Relaxed), &[]);
                }
            })
            .build();

        let regions_active = meter
            .u64_observable_gauge("asupersync.regions.active")
            .with_description("Currently active regions")
            .with_callback({
                let state = Arc::clone(&state);
                move |observer| {
                    observer.observe(state.active_regions.load(Ordering::Relaxed), &[]);
                }
            })
            .build();

        let obligations_active = meter
            .u64_observable_gauge("asupersync.obligations.active")
            .with_description("Currently active obligations")
            .with_callback({
                let state = Arc::clone(&state);
                move |observer| {
                    observer.observe(state.active_obligations.load(Ordering::Relaxed), &[]);
                }
            })
            .build();

        Self {
            tasks_active,
            tasks_spawned: meter
                .u64_counter("asupersync.tasks.spawned")
                .with_description("Total tasks spawned")
                .build(),
            tasks_completed: meter
                .u64_counter("asupersync.tasks.completed")
                .with_description("Total tasks completed")
                .build(),
            task_duration: meter
                .f64_histogram("asupersync.tasks.duration")
                .with_description("Task execution duration in seconds")
                .build(),
            regions_active,
            regions_created: meter
                .u64_counter("asupersync.regions.created")
                .with_description("Total regions created")
                .build(),
            regions_closed: meter
                .u64_counter("asupersync.regions.closed")
                .with_description("Total regions closed")
                .build(),
            region_lifetime: meter
                .f64_histogram("asupersync.regions.lifetime")
                .with_description("Region lifetime in seconds")
                .build(),
            cancellations: meter
                .u64_counter("asupersync.cancellations")
                .with_description("Cancellation requests")
                .build(),
            drain_duration: meter
                .f64_histogram("asupersync.cancellation.drain_duration")
                .with_description("Cancellation drain duration in seconds")
                .build(),
            deadlines_set: meter
                .u64_counter("asupersync.deadlines.set")
                .with_description("Deadlines configured")
                .build(),
            deadlines_exceeded: meter
                .u64_counter("asupersync.deadlines.exceeded")
                .with_description("Deadline exceeded events")
                .build(),
            obligations_active,
            obligations_created: meter
                .u64_counter("asupersync.obligations.created")
                .with_description("Obligations created")
                .build(),
            obligations_discharged: meter
                .u64_counter("asupersync.obligations.discharged")
                .with_description("Obligations discharged")
                .build(),
            obligations_leaked: meter
                .u64_counter("asupersync.obligations.leaked")
                .with_description("Obligations leaked")
                .build(),
            scheduler_poll_time: meter
                .f64_histogram("asupersync.scheduler.poll_time")
                .with_description("Scheduler poll duration in seconds")
                .build(),
            scheduler_tasks_polled: meter
                .f64_histogram("asupersync.scheduler.tasks_polled")
                .with_description("Tasks polled per scheduler tick")
                .build(),
            state,
        }
    }
}

impl MetricsProvider for OtelMetrics {
    fn task_spawned(&self, _region_id: RegionId, _task_id: TaskId) {
        self.state.inc_tasks();
        self.tasks_spawned.add(1, &[]);
    }

    fn task_completed(&self, _task_id: TaskId, outcome: OutcomeKind, duration: Duration) {
        self.state.dec_tasks();
        self.tasks_completed
            .add(1, &[KeyValue::new("outcome", outcome_label(outcome))]);
        self.task_duration.record(
            duration.as_secs_f64(),
            &[KeyValue::new("outcome", outcome_label(outcome))],
        );
    }

    fn region_created(&self, _region_id: RegionId, _parent: Option<RegionId>) {
        self.state.inc_regions();
        self.regions_created.add(1, &[]);
    }

    fn region_closed(&self, _region_id: RegionId, lifetime: Duration) {
        self.state.dec_regions();
        self.regions_closed.add(1, &[]);
        self.region_lifetime.record(lifetime.as_secs_f64(), &[]);
    }

    fn cancellation_requested(&self, _region_id: RegionId, kind: CancelKind) {
        self.cancellations
            .add(1, &[KeyValue::new("kind", cancel_kind_label(kind))]);
    }

    fn drain_completed(&self, _region_id: RegionId, duration: Duration) {
        self.drain_duration.record(duration.as_secs_f64(), &[]);
    }

    fn deadline_set(&self, _region_id: RegionId, _deadline: Duration) {
        self.deadlines_set.add(1, &[]);
    }

    fn deadline_exceeded(&self, _region_id: RegionId) {
        self.deadlines_exceeded.add(1, &[]);
    }

    fn obligation_created(&self, _region_id: RegionId) {
        self.state.inc_obligations();
        self.obligations_created.add(1, &[]);
    }

    fn obligation_discharged(&self, _region_id: RegionId) {
        self.state.dec_obligations();
        self.obligations_discharged.add(1, &[]);
    }

    fn obligation_leaked(&self, _region_id: RegionId) {
        self.state.dec_obligations();
        self.obligations_leaked.add(1, &[]);
    }

    fn scheduler_tick(&self, tasks_polled: usize, duration: Duration) {
        self.scheduler_poll_time.record(duration.as_secs_f64(), &[]);
        self.scheduler_tasks_polled.record(tasks_polled as f64, &[]);
    }
}

const fn outcome_label(outcome: OutcomeKind) -> &'static str {
    match outcome {
        OutcomeKind::Ok => "ok",
        OutcomeKind::Err => "err",
        OutcomeKind::Cancelled => "cancelled",
        OutcomeKind::Panicked => "panicked",
    }
}

const fn cancel_kind_label(kind: CancelKind) -> &'static str {
    match kind {
        CancelKind::User => "user",
        CancelKind::Timeout => "timeout",
        CancelKind::Deadline => "deadline",
        CancelKind::PollQuota => "poll_quota",
        CancelKind::CostBudget => "cost_budget",
        CancelKind::FailFast => "fail_fast",
        CancelKind::RaceLost => "race_lost",
        CancelKind::ParentCancelled => "parent_cancelled",
        CancelKind::ResourceUnavailable => "resource_unavailable",
        CancelKind::Shutdown => "shutdown",
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    #[test]
    fn otel_metrics_exports_in_memory() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("asupersync");

        let metrics = OtelMetrics::new(meter);

        metrics.task_spawned(RegionId::testing_default(), TaskId::testing_default());
        metrics.task_completed(
            TaskId::testing_default(),
            OutcomeKind::Ok,
            Duration::from_millis(10),
        );
        metrics.region_created(RegionId::testing_default(), None);
        metrics.region_closed(RegionId::testing_default(), Duration::from_secs(1));
        metrics.cancellation_requested(RegionId::testing_default(), CancelKind::User);
        metrics.drain_completed(RegionId::testing_default(), Duration::from_millis(5));
        metrics.deadline_set(RegionId::testing_default(), Duration::from_secs(2));
        metrics.deadline_exceeded(RegionId::testing_default());
        metrics.obligation_created(RegionId::testing_default());
        metrics.obligation_discharged(RegionId::testing_default());
        metrics.obligation_leaked(RegionId::testing_default());
        metrics.scheduler_tick(3, Duration::from_millis(1));

        provider.force_flush().expect("force_flush");
        let finished = exporter.get_finished_metrics().expect("finished metrics");
        assert!(!finished.is_empty());

        provider.shutdown().expect("shutdown");
    }
}
